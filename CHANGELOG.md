# Changelog

All notable changes to TAlker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-20

### Added
- **Multi-provider LLM support**: OpenAI, Anthropic, Google, Cohere, and Ollama
- **Multi-provider embeddings**: OpenAI, Cohere, Ollama, HuggingFace, FastEmbed
- **Offline/local model support** via Ollama integration
- **Settings page** (5_Settings.py) for UI-based provider configuration
- **Query expansion** for improved retrieval
- **Token tracking** and cost estimation
- **Provider status validation** for API keys
- New test suite for providers (`test_providers.py`)

### Changed
- Refactored `llm.py` to use provider factories
- Updated dependencies in `pyproject.toml` for all providers
- Enhanced `.env.example` with all provider configurations

## [0.2.0] - 2025-01-20

### Added
- **ChromaDB** for persistent vector storage (replaced FAISS)
- **Hybrid search**: BM25 + semantic search with configurable weights
- **Cross-encoder reranking** using ms-marco-MiniLM-L-6-v2
- **RAGAS evaluation framework** with 5 metrics:
  - Faithfulness
  - Answer relevancy
  - Context precision
  - Context recall
  - Context relevancy
- **Evaluation page** (4_Evaluation.py) with visualizations
- **Source citations** with confidence scores in chat UI
- **Content hashing** for smart cache invalidation
- Comprehensive test suite (`test_llm.py`, `test_evaluation.py`)
- Enhanced logging throughout the codebase

### Changed
- Upgraded LLM from GPT-4-1106-preview to GPT-4o
- Upgraded embeddings from default to text-embedding-3-large
- Increased chunk size from 500 to 1000 characters
- Improved prompt template with citation instructions
- Updated Test page (2_Test.py) with metrics sidebar

### Removed
- FAISS dependency (replaced by ChromaDB)
- In-memory vector store (now persistent)

## [0.1.0] - Initial Release

### Added
- Basic RAG implementation with FAISS and OpenAI
- Streamlit web interface
- Piazza API integration
- Document upload (PDF, TXT, CSV)
- Chat interface for Q&A
- Course analytics dashboard
- User authentication

### Technical Stack
- LangChain for LLM orchestration
- FAISS for vector similarity search
- OpenAI GPT-4 for generation
- Streamlit for web UI
- Poetry for dependency management
