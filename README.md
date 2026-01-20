# TAlker - Production-Grade RAG Teaching Assistant

A multi-provider RAG (Retrieval-Augmented Generation) system for automating teaching assistant tasks. Features hybrid search, cross-encoder reranking, RAGAS evaluation, and support for both cloud and local LLMs.

## Features

### Multi-Provider LLM Support
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Cohere**: Command R+, Command R
- **Ollama** (Local): Llama 3.1, Mistral, Mixtral, Phi-3, Qwen, DeepSeek

### Advanced RAG Pipeline
- **ChromaDB** for persistent vector storage with automatic caching
- **Hybrid Search**: BM25 + semantic search with configurable weights
- **Cross-Encoder Reranking** using ms-marco-MiniLM for improved relevance
- **Query Expansion** for better retrieval coverage
- **Source Citations** with confidence scores

### Multi-Provider Embeddings
- OpenAI (text-embedding-3-large/small)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- Ollama (nomic-embed-text, mxbai-embed-large)
- HuggingFace (BGE, MPNet)
- FastEmbed (optimized local embeddings)

### RAGAS Evaluation Framework
- Faithfulness scoring
- Answer relevancy metrics
- Context precision/recall/relevancy
- Automated evaluation reports

### Additional Features
- Piazza integration for course Q&A management
- Token usage and cost tracking
- Modern Streamlit web interface
- Comprehensive test suite

## Prerequisites

- Python 3.10 or higher
- Poetry (Python package manager)
- At least one LLM provider API key (or Ollama for local models)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gr8monk3ys/TAlker.git
   cd TAlker
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```bash
   # At least one of these is required
   OPENAI_API_KEY=sk-your-key
   ANTHROPIC_API_KEY=sk-ant-your-key
   GOOGLE_API_KEY=your-key
   COHERE_API_KEY=your-key

   # Optional: Piazza integration
   PIAZZA_EMAIL=your-email
   PIAZZA_PASSWORD=your-password
   PIAZZA_COURSE_ID=your-course-id
   ```

3. Install dependencies:
   ```bash
   make setup
   ```

4. (Optional) For local models, install Ollama:
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull recommended models
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

## Usage

1. Start the application:
   ```bash
   make run
   ```

2. Access the web interface at `http://localhost:8501`

3. Navigate to:
   - **Upload**: Add course materials (PDF, TXT, CSV, MD)
   - **Test**: Chat with the RAG system
   - **Analysis**: View course analytics
   - **Evaluation**: Run RAGAS evaluation
   - **Settings**: Configure providers and parameters

## Project Structure

```
TAlker/
├── src/
│   ├── dashboard/              # Streamlit web interface
│   │   ├── Home.py            # Main application
│   │   ├── llm.py             # RAG pipeline implementation
│   │   ├── providers.py       # Multi-provider LLM/embedding support
│   │   ├── evaluation.py      # RAGAS evaluation framework
│   │   └── pages/
│   │       ├── 1_Upload.py    # Document upload
│   │       ├── 2_Test.py      # Chat interface
│   │       ├── 3_Analysis.py  # Analytics dashboard
│   │       ├── 4_Evaluation.py # RAGAS evaluation UI
│   │       └── 5_Settings.py  # Provider configuration
│   └── piazza_bot/            # Piazza API integration
│       ├── bot.py             # Piazza bot logic
│       ├── profile.py         # User profiles
│       └── responses.py       # Response types
├── tests/                     # Test suite
│   ├── test_llm.py           # RAG pipeline tests
│   ├── test_evaluation.py    # Evaluation tests
│   └── test_providers.py     # Provider tests
├── data/                      # Document storage
├── pyproject.toml            # Dependencies
└── Makefile                  # Development commands
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install Poetry and dependencies |
| `make run` | Start the Streamlit application |
| `make dev` | Start with hot reload |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage |
| `make lint` | Run linter |
| `make format` | Format code with Black |
| `make check` | Run all code quality checks |
| `make evaluate` | Run RAGAS evaluation |
| `make rebuild-index` | Rebuild vector index |
| `make clean` | Clean cache files |

## Configuration

### RAG Parameters

Configure in Settings page or via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | gpt-4o | LLM model to use |
| `EMBEDDING_MODEL` | text-embedding-3-large | Embedding model |
| `CHUNK_SIZE` | 1000 | Document chunk size |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `INITIAL_K` | 20 | Documents to retrieve |
| `FINAL_K` | 5 | Documents after reranking |
| `BM25_WEIGHT` | 0.3 | Weight for keyword search |

### Running Locally (Offline)

For fully offline operation:

1. Install Ollama and pull models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

2. In Settings, select:
   - LLM: `llama3.1:8b` (Ollama)
   - Embeddings: `nomic-embed-text` (Ollama)

3. No API keys required!

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Format and lint
make format && make lint

# Full check before commit
make all
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests and linting (`make all`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM orchestration
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation
- [Streamlit](https://streamlit.io/) - Web interface
- [Piazza API](https://github.com/hfaran/piazza-api) - Course integration
