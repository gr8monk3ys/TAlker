from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import glob
import zipfile

load_dotenv()

class LlmChain:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.conversation_chain = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.embeddings = OpenAIEmbeddings()
        self._setup_chain()

    def _extract_zip_if_needed(self):
        """Extract zip files in the data directory if they haven't been extracted."""
        for file_path in glob.glob(os.path.join(self.data_dir, "*.zip")):
            try:
                zip_name = os.path.splitext(os.path.basename(file_path))[0]
                extract_dir = os.path.join(self.data_dir, zip_name)
                
                if not os.path.exists(extract_dir):
                    print(f"Extracting {file_path} to {extract_dir}")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"Successfully extracted {file_path}")
            except Exception as e:
                print(f"Error extracting {file_path}: {str(e)}")

    def _load_documents(self):
        """Load all documents from the data directory."""
        documents = []
        
        # Extract any zip files first
        self._extract_zip_if_needed()
        
        # Get all files in data directory and subdirectories
        for file_path in glob.glob(os.path.join(self.data_dir, "**/*"), recursive=True):
            if not os.path.isfile(file_path):
                continue
                
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif ext == '.csv':
                    loader = CSVLoader(
                        file_path,
                        csv_args={
                            'delimiter': ',',
                            'quotechar': '"'
                        }
                    )
                else:
                    continue
                
                print(f"Loading {file_path}...")
                documents.extend(loader.load())
                print(f"Successfully loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
        
        return documents

    def create_vectorstore(self, documents) -> FAISS:
        """Create a FAISS vectorstore from documents."""
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")
        return FAISS.from_documents(texts, self.embeddings)

    def get_conversation_chain(
        self,
        vectorstore: FAISS,
        memory: ConversationBufferMemory = None
    ) -> ConversationalRetrievalChain:
        """Create a conversation chain with the given vectorstore."""
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-1106-preview",
            verbose=True
        )
        
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the context seems irrelevant to the question, politely ask the user to rephrase their question 
        or suggest uploading relevant course materials.

        Context: {context}

        Question: {question}
        
        Helpful Answer: Let me help you understand what I found in the course materials."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            ),
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def _setup_chain(self) -> None:
        """Initialize the conversation chain."""
        try:
            documents = self._load_documents()
            if documents:
                print(f"Loaded {len(documents)} documents")
                self.vectorstore = self.create_vectorstore(documents)
                self.conversation_chain = self.get_conversation_chain(self.vectorstore)
                print("Successfully set up conversation chain")
            else:
                print("No documents found in data directory")
        except Exception as e:
            print(f"Error setting up chain: {str(e)}")

    def get_response(self, question: str) -> str:
        """Generate an answer for the given question."""
        if not self.conversation_chain:
            return """I don't have access to any course materials yet. Please:
            1. Go to the Upload page
            2. Upload relevant documents (syllabus, assignments, lecture slides)
            3. Come back and ask your question again
            
            This will help me provide accurate answers about your course."""
        
        try:
            response = self.conversation_chain({"question": question})
            return response["answer"]
        except Exception as e:
            error_msg = str(e)
            if "openai" in error_msg.lower():
                return "Error: Could not connect to OpenAI. Please check if your API key is properly set in the .env file."
            return f"Error generating response: {error_msg}"

if __name__ == "__main__":
    llm_chain = LlmChain()
    user_input = input("You: ")
    print("Bot:", llm_chain.get_response(user_input))
