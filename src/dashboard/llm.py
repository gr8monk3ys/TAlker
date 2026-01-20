"""
Production-grade RAG implementation with:
- ChromaDB for persistent vector storage
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking
- GPT-4o with text-embedding-3-large
- Source citations and confidence scores
- Proper logging and error handling
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import glob
import zipfile

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
    # Model settings
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    temperature: float = 0.1

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    initial_k: int = 20  # Initial retrieval count
    final_k: int = 5     # After reranking
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    similarity_threshold: float = 0.3

    # Memory settings
    memory_window: int = 5  # Keep last 5 conversation turns


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


class LlmChain:
    """
    Production-grade RAG implementation with hybrid search,
    reranking, and source citations.
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

        # Text splitter with improved settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Initialize embeddings with latest model
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            dimensions=1536  # Optimal for text-embedding-3-large
        )

        # Initialize reranker
        try:
            self.reranker = CrossEncoder(self.config.reranker_model)
            logger.info(f"Loaded reranker: {self.config.reranker_model}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Proceeding without reranking.")
            self.reranker = None

        self._setup_chain()

    def _compute_content_hash(self) -> str:
        """Compute hash of all documents to detect changes."""
        content_hash = hashlib.md5()
        for file_path in sorted(glob.glob(str(self.data_dir / "**/*"), recursive=True)):
            if os.path.isfile(file_path) and not file_path.startswith(str(self.persist_dir)):
                try:
                    stat = os.stat(file_path)
                    content_hash.update(f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode())
                except Exception:
                    continue
        return content_hash.hexdigest()

    def _extract_zip_if_needed(self) -> None:
        """Extract zip files in the data directory if they haven't been extracted."""
        for file_path in glob.glob(str(self.data_dir / "*.zip")):
            try:
                zip_name = os.path.splitext(os.path.basename(file_path))[0]
                extract_dir = self.data_dir / zip_name

                if not extract_dir.exists():
                    logger.info(f"Extracting {file_path} to {extract_dir}")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
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
                        'file_type': ext[1:],  # Remove the dot
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

        llm = ChatOpenAI(
            temperature=self.config.temperature,
            model=self.config.llm_model,
        )

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

            if "openai" in error_msg.lower() or "api" in error_msg.lower():
                return "Error: Could not connect to OpenAI. Please check if your API key is properly set in the .env file."
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

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            tokens_used=0  # Would need token counting implementation
        )

    def rebuild_index(self) -> bool:
        """Force rebuild of the vector index."""
        try:
            # Clear existing data
            if self.persist_dir.exists():
                import shutil
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


if __name__ == "__main__":
    llm_chain = LlmChain()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = llm_chain.get_structured_response(user_input)
        print(f"\nBot: {response.answer}")
        print(f"\nConfidence: {response.confidence:.2%}")

        if response.sources:
            print("\nSources:")
            for src in response.sources:
                print(f"  - {src.source}" + (f" (Page {src.page})" if src.page else ""))
