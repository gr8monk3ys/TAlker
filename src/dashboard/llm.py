"""
Production-grade RAG implementation with:
- Multi-provider LLM support (OpenAI, Anthropic, Google, Cohere, Ollama)
- Multi-provider embeddings (OpenAI, Cohere, Ollama, HuggingFace, FastEmbed)
- ChromaDB for persistent vector storage
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking
- Query expansion for improved retrieval
- Source citations and confidence scores
- Token tracking and cost estimation
- Proper logging and error handling
"""

import os
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import glob
import zipfile

from src.dashboard.providers import (
    ProviderConfig,
    LLMFactory,
    EmbeddingFactory,
    TokenTracker,
    LLM_MODELS,
    EMBEDDING_MODELS,
    LLMProvider,
    EmbeddingProvider,
    check_ollama_availability,
    get_available_ollama_models,
    validate_api_keys,
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Provider configuration
    provider_config: ProviderConfig = field(default_factory=ProviderConfig)

    # Reranker settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    initial_k: int = 20  # Initial retrieval count
    final_k: int = 5     # After reranking
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    similarity_threshold: float = 0.3

    # Query expansion
    use_query_expansion: bool = True
    num_expanded_queries: int = 3

    # Memory settings
    memory_window: int = 5  # Keep last 5 conversation turns

    # Convenience properties for backward compatibility
    @property
    def llm_model(self) -> str:
        return self.provider_config.llm_model

    @llm_model.setter
    def llm_model(self, value: str):
        self.provider_config.llm_model = value

    @property
    def embedding_model(self) -> str:
        return self.provider_config.embedding_model

    @embedding_model.setter
    def embedding_model(self, value: str):
        self.provider_config.embedding_model = value

    @property
    def temperature(self) -> float:
        return self.provider_config.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.provider_config.temperature = value


@dataclass
class RetrievalResult:
    """Structured retrieval result with metadata."""
    content: str
    source: str
    page: Optional[int]
    relevance_score: float
    chunk_id: str


@dataclass
class RAGResponse:
    """Structured RAG response with citations."""
    answer: str
    sources: list[RetrievalResult]
    confidence: float
    tokens_used: int
    cost: float = 0.0
    model_used: str = ""


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""

    def __init__(self):
        self.tokens: list[str] = []
        self.finished = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.finished = True

    def get_tokens(self) -> Generator[str, None, None]:
        """Yield tokens as they become available."""
        idx = 0
        while not self.finished or idx < len(self.tokens):
            if idx < len(self.tokens):
                yield self.tokens[idx]
                idx += 1


class LlmChain:
    """
    Production-grade RAG implementation with multi-provider support,
    hybrid search, reranking, and source citations.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.persist_dir = self.data_dir / ".chroma_db"

        # Initialize components
        self.conversation_chain = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.reranker = None
        self.documents = []
        self._last_sources: list[RetrievalResult] = []

        # Token tracking
        self.token_tracker = TokenTracker()

        # Text splitter with improved settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Initialize embeddings using factory
        self.embeddings = EmbeddingFactory.create(self.config.provider_config)
        logger.info(f"Initialized embeddings: {self.config.embedding_model}")

        # Initialize reranker
        if self.config.use_reranker:
            self._init_reranker()

        self._setup_chain()

    def _init_reranker(self) -> None:
        """Initialize the cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.config.reranker_model)
            logger.info(f"Loaded reranker: {self.config.reranker_model}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Proceeding without reranking.")
            self.reranker = None

    def _compute_content_hash(self) -> str:
        """Compute hash of all documents to detect changes."""
        content_hash = hashlib.md5()
        # Also include embedding model in hash to rebuild if model changes
        content_hash.update(self.config.embedding_model.encode())

        for file_path in sorted(glob.glob(str(self.data_dir / "**/*"), recursive=True)):
            if os.path.isfile(file_path) and not file_path.startswith(str(self.persist_dir)):
                try:
                    stat = os.stat(file_path)
                    content_hash.update(f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode())
                except Exception:
                    continue
        return content_hash.hexdigest()

    def _safe_extract_zip(self, zip_path: str, extract_dir: Path) -> bool:
        """Safely extract a zip file, preventing zip slip attacks."""
        extract_dir_resolved = extract_dir.resolve()

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Get the target path and resolve it
                member_path = (extract_dir / member).resolve()

                # Security check: ensure path is within extract_dir
                try:
                    member_path.relative_to(extract_dir_resolved)
                except ValueError:
                    logger.warning(f"Skipping potentially malicious zip member: {member}")
                    continue

                # Extract safely
                zip_ref.extract(member, extract_dir)

        return True

    def _extract_zip_if_needed(self) -> None:
        """Extract zip files in the data directory if they haven't been extracted."""
        for file_path in glob.glob(str(self.data_dir / "*.zip")):
            try:
                zip_name = os.path.splitext(os.path.basename(file_path))[0]
                extract_dir = self.data_dir / zip_name

                if not extract_dir.exists():
                    logger.info(f"Extracting {file_path} to {extract_dir}")
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    self._safe_extract_zip(file_path, extract_dir)
                    logger.info(f"Successfully extracted {file_path}")
            except Exception as e:
                logger.error(f"Error extracting {file_path}: {e}")

    def _load_documents(self) -> list[Document]:
        """Load all documents from the data directory with enhanced metadata."""
        documents = []
        self._extract_zip_if_needed()

        supported_extensions = {'.txt', '.pdf', '.csv', '.md'}

        for file_path in glob.glob(str(self.data_dir / "**/*"), recursive=True):
            if not os.path.isfile(file_path):
                continue
            if str(self.persist_dir) in file_path:
                continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in supported_extensions:
                continue

            try:
                loader = self._get_loader(file_path, ext)
                if loader is None:
                    continue

                logger.info(f"Loading {file_path}...")
                docs = loader.load()

                # Enhance metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_file': os.path.basename(file_path),
                        'file_path': file_path,
                        'file_type': ext[1:],
                        'chunk_id': hashlib.md5(
                            f"{file_path}:{doc.page_content[:100]}".encode()
                        ).hexdigest()[:8]
                    })

                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} document(s) from {file_path}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        return documents

    def _get_loader(self, file_path: str, ext: str):
        """Get appropriate document loader for file type."""
        loaders = {
            '.txt': lambda: TextLoader(file_path, encoding='utf-8'),
            '.md': lambda: TextLoader(file_path, encoding='utf-8'),
            '.pdf': lambda: PyPDFLoader(file_path),
            '.csv': lambda: CSVLoader(
                file_path,
                csv_args={'delimiter': ',', 'quotechar': '"'}
            )
        }
        loader_factory = loaders.get(ext)
        return loader_factory() if loader_factory else None

    def _should_rebuild_index(self) -> bool:
        """Check if vector store needs rebuilding."""
        hash_file = self.persist_dir / "content_hash.txt"
        current_hash = self._compute_content_hash()

        if not self.persist_dir.exists() or not hash_file.exists():
            return True

        try:
            stored_hash = hash_file.read_text().strip()
            return stored_hash != current_hash
        except Exception:
            return True

    def _save_content_hash(self) -> None:
        """Save current content hash for future comparisons."""
        hash_file = self.persist_dir / "content_hash.txt"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(self._compute_content_hash())

    def create_vectorstore(self, documents: list[Document]) -> Chroma:
        """Create or load ChromaDB vectorstore with persistence."""
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks")

        # Store documents for BM25
        self.documents = texts

        # Create persistent ChromaDB
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistence
        client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Delete existing collection if exists
        try:
            client.delete_collection("course_materials")
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            client=client,
            collection_name="course_materials",
            collection_metadata={"hnsw:space": "cosine"}
        )

        self._save_content_hash()
        logger.info(f"Created ChromaDB vectorstore at {self.persist_dir}")

        return vectorstore

    def _load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing ChromaDB vectorstore if available."""
        try:
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            vectorstore = Chroma(
                client=client,
                collection_name="course_materials",
                embedding_function=self.embeddings
            )

            # Load documents for BM25 from vectorstore
            collection = client.get_collection("course_materials")
            if collection.count() > 0:
                results = collection.get(include=['documents', 'metadatas'])
                self.documents = [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in zip(results['documents'], results['metadatas'])
                ]
                logger.info(f"Loaded {len(self.documents)} documents from existing vectorstore")
                return vectorstore

        except Exception as e:
            logger.warning(f"Failed to load existing vectorstore: {e}")

        return None

    def _create_hybrid_retriever(self) -> EnsembleRetriever:
        """Create hybrid retriever combining BM25 and semantic search."""
        if not self.documents:
            raise ValueError("No documents available for retrieval")

        # BM25 retriever for keyword matching
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=self.config.initial_k
        )

        # Semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config.initial_k,
                "score_threshold": self.config.similarity_threshold
            }
        )

        # Ensemble retriever with weighted combination
        hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, semantic_retriever],
            weights=[self.config.bm25_weight, self.config.semantic_weight]
        )

        logger.info("Created hybrid retriever (BM25 + Semantic)")
        return hybrid_retriever

    def _expand_query(self, query: str) -> list[str]:
        """Expand query into multiple related queries for better retrieval."""
        if not self.config.use_query_expansion:
            return [query]

        try:
            llm = LLMFactory.create(self.config.provider_config)

            expansion_prompt = f"""Generate {self.config.num_expanded_queries} alternative phrasings of this question for better document retrieval.
Return only the questions, one per line, without numbering or explanations.

Original question: {query}

Alternative questions:"""

            response = llm.invoke(expansion_prompt)
            expanded = response.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip()][:self.config.num_expanded_queries]

            return [query] + expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def _rerank_documents(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank documents using cross-encoder."""
        if not self.reranker or not documents:
            return documents[:self.config.final_k]

        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in documents]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Sort by score and return top k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Store scores in metadata for citation
        reranked = []
        for doc, score in scored_docs[:self.config.final_k]:
            doc.metadata['relevance_score'] = float(score)
            reranked.append(doc)

        logger.debug(f"Reranked {len(documents)} docs, kept top {len(reranked)}")
        return reranked

    def get_conversation_chain(
        self,
        vectorstore: Chroma,
        memory: ConversationBufferWindowMemory = None
    ) -> ConversationalRetrievalChain:
        """Create conversation chain with hybrid retrieval and reranking."""

        llm = LLMFactory.create(self.config.provider_config)
        logger.info(f"Initialized LLM: {self.config.llm_model}")

        if memory is None:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.config.memory_window
            )

        # Enhanced prompt template with citation instructions
        prompt_template = """You are a knowledgeable teaching assistant helping students with course materials.

Use the following context from course materials to answer the student's question.
Always cite your sources by referencing the document name and page number when available.

Guidelines:
1. If the answer is found in the context, provide a clear, educational response
2. If the context doesn't contain the answer, say "I couldn't find this in the course materials" and suggest relevant topics to explore
3. Use bullet points and clear formatting for complex explanations
4. When citing sources, use the format: [Source: filename, Page X]

Context from course materials:
{context}

Student's Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create hybrid retriever
        hybrid_retriever = self._create_hybrid_retriever()

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def _setup_chain(self) -> None:
        """Initialize the conversation chain with smart caching."""
        try:
            # Check if we can use existing vectorstore
            if not self._should_rebuild_index():
                logger.info("Loading existing vectorstore...")
                self.vectorstore = self._load_existing_vectorstore()

            if self.vectorstore is None:
                logger.info("Building new vectorstore...")
                documents = self._load_documents()
                if documents:
                    logger.info(f"Loaded {len(documents)} documents")
                    self.vectorstore = self.create_vectorstore(documents)
                else:
                    logger.warning("No documents found in data directory")
                    return

            self.conversation_chain = self.get_conversation_chain(self.vectorstore)
            logger.info("Successfully initialized RAG pipeline")

        except Exception as e:
            logger.error(f"Error setting up chain: {e}", exc_info=True)

    def get_response(self, question: str) -> str:
        """Generate an answer with source citations."""
        if not self.conversation_chain:
            return """I don't have access to any course materials yet. Please:
1. Go to the Upload page
2. Upload relevant documents (syllabus, assignments, lecture slides)
3. Come back and ask your question again

This will help me provide accurate answers about your course."""

        try:
            # Get response with source documents
            response = self.conversation_chain.invoke({"question": question})
            answer = response["answer"]
            source_docs = response.get("source_documents", [])

            # Apply reranking to source documents
            if source_docs:
                source_docs = self._rerank_documents(question, source_docs)

            # Store sources for retrieval
            self._last_sources = [
                RetrievalResult(
                    content=doc.page_content[:200] + "...",
                    source=doc.metadata.get('source_file', 'Unknown'),
                    page=doc.metadata.get('page'),
                    relevance_score=doc.metadata.get('relevance_score', 0.0),
                    chunk_id=doc.metadata.get('chunk_id', 'unknown')
                )
                for doc in source_docs
            ]

            return answer

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating response: {error_msg}", exc_info=True)

            if "api" in error_msg.lower() or "key" in error_msg.lower():
                return f"Error: API connection failed. Please check your API key for {self.config.llm_model}."
            if "ollama" in error_msg.lower():
                return "Error: Could not connect to Ollama. Make sure Ollama is running locally."
            return f"Error generating response: {error_msg}"

    def get_last_sources(self) -> list[RetrievalResult]:
        """Get source documents from the last query."""
        return self._last_sources

    def get_structured_response(self, question: str) -> RAGResponse:
        """Get a fully structured response with metadata."""
        answer = self.get_response(question)
        sources = self.get_last_sources()

        # Calculate confidence based on relevance scores
        if sources:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
            confidence = min(1.0, max(0.0, avg_relevance))
        else:
            confidence = 0.0

        # Get cost info
        model_info = LLM_MODELS.get(self.config.llm_model)
        cost = 0.0
        if model_info:
            # Rough estimate: ~500 tokens per response
            cost = model_info.output_cost_per_1k * 0.5

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            tokens_used=0,
            cost=cost,
            model_used=self.config.llm_model
        )

    def rebuild_index(self) -> bool:
        """Force rebuild of the vector index."""
        try:
            # Clear existing data
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)

            # Rebuild
            documents = self._load_documents()
            if documents:
                self.vectorstore = self.create_vectorstore(documents)
                self.conversation_chain = self.get_conversation_chain(self.vectorstore)
                logger.info("Successfully rebuilt index")
                return True
            return False

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}", exc_info=True)
            return False

    def switch_provider(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> bool:
        """
        Switch to a different LLM or embedding provider.
        Rebuilds index if embedding model changes.
        """
        rebuild_needed = False

        if embedding_model and embedding_model != self.config.embedding_model:
            self.config.provider_config.embedding_model = embedding_model
            self.embeddings = EmbeddingFactory.create(self.config.provider_config)
            rebuild_needed = True
            logger.info(f"Switched embeddings to: {embedding_model}")

        if llm_model:
            self.config.provider_config.llm_model = llm_model
            logger.info(f"Switched LLM to: {llm_model}")

        if rebuild_needed:
            return self.rebuild_index()
        elif self.vectorstore:
            # Just rebuild the chain with new LLM
            self.conversation_chain = self.get_conversation_chain(self.vectorstore)
            return True

        return True

    def get_provider_status(self) -> dict:
        """Get status of available providers."""
        return {
            "api_keys": validate_api_keys(),
            "ollama_models": get_available_ollama_models() if check_ollama_availability() else [],
            "current_llm": self.config.llm_model,
            "current_embeddings": self.config.embedding_model,
            "token_usage": self.token_tracker.get_summary()
        }


if __name__ == "__main__":
    # Example: Using different providers
    print("Available providers:")
    status = validate_api_keys()
    for provider, available in status.items():
        print(f"  {provider}: {'✓' if available else '✗'}")

    # Default configuration
    llm_chain = LlmChain()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.startswith("/switch "):
            model = user_input.split(" ", 1)[1]
            llm_chain.switch_provider(llm_model=model)
            print(f"Switched to {model}")
            continue

        response = llm_chain.get_structured_response(user_input)
        print(f"\nBot ({response.model_used}): {response.answer}")
        print(f"\nConfidence: {response.confidence:.2%}")

        if response.sources:
            print("\nSources:")
            for src in response.sources:
                print(f"  - {src.source}" + (f" (Page {src.page})" if src.page else ""))
