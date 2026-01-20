"""
Tests for the LlmChain RAG implementation.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain.schema import Document


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.dashboard.llm import RAGConfig

        config = RAGConfig()

        assert config.llm_model == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.initial_k == 20
        assert config.final_k == 5

    def test_custom_config(self):
        """Test custom configuration."""
        from src.dashboard.llm import RAGConfig

        config = RAGConfig(
            llm_model="gpt-4o-mini",
            chunk_size=500,
            final_k=3
        )

        assert config.llm_model == "gpt-4o-mini"
        assert config.chunk_size == 500
        assert config.final_k == 3


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        from src.dashboard.llm import RetrievalResult

        result = RetrievalResult(
            content="Test content",
            source="test.pdf",
            page=1,
            relevance_score=0.85,
            chunk_id="abc123"
        )

        assert result.content == "Test content"
        assert result.source == "test.pdf"
        assert result.page == 1
        assert result.relevance_score == 0.85


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_rag_response_creation(self):
        """Test creating a RAG response."""
        from src.dashboard.llm import RAGResponse, RetrievalResult

        sources = [
            RetrievalResult("content1", "file1.pdf", 1, 0.9, "id1"),
            RetrievalResult("content2", "file2.pdf", 2, 0.7, "id2"),
        ]

        response = RAGResponse(
            answer="Test answer",
            sources=sources,
            confidence=0.8,
            tokens_used=100
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 2
        assert response.confidence == 0.8


class TestTextSplitter:
    """Tests for text splitting functionality."""

    def test_text_splitter_configuration(self, rag_config):
        """Test that text splitter is configured correctly."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain(config=rag_config)

                    assert chain.text_splitter._chunk_size == 500
                    assert chain.text_splitter._chunk_overlap == 50

    def test_text_splitting(self, sample_documents):
        """Test that documents are split correctly."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )

        # Create a long document
        long_doc = Document(
            page_content="This is a test. " * 50,
            metadata={"source": "test.txt"}
        )

        chunks = splitter.split_documents([long_doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 50  # Allow some flexibility


class TestDocumentLoading:
    """Tests for document loading functionality."""

    def test_get_loader_txt(self):
        """Test getting loader for txt files."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    loader = chain._get_loader("/path/to/file.txt", ".txt")
                    assert loader is not None

    def test_get_loader_pdf(self):
        """Test getting loader for pdf files."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    loader = chain._get_loader("/path/to/file.pdf", ".pdf")
                    assert loader is not None

    def test_get_loader_unsupported(self):
        """Test getting loader for unsupported files."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    loader = chain._get_loader("/path/to/file.xyz", ".xyz")
                    assert loader is None

    def test_load_documents_from_directory(self, sample_text_files, temp_data_dir):
        """Test loading documents from a directory."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    chain.data_dir = temp_data_dir
                    chain.persist_dir = temp_data_dir / ".chroma_db"

                    documents = chain._load_documents()

                    assert len(documents) == 2
                    # Check metadata is added
                    for doc in documents:
                        assert 'source_file' in doc.metadata
                        assert 'file_type' in doc.metadata


class TestReranking:
    """Tests for document reranking."""

    def test_rerank_documents(self, sample_documents):
        """Test document reranking with cross-encoder."""
        from src.dashboard.llm import LlmChain, RAGConfig

        config = RAGConfig(final_k=3)

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder') as mock_ce:
                mock_ce_instance = MagicMock()
                mock_ce_instance.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]
                mock_ce.return_value = mock_ce_instance

                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain(config=config)

                    reranked = chain._rerank_documents(
                        "What is machine learning?",
                        sample_documents
                    )

                    # Should return top 3
                    assert len(reranked) == 3
                    # Should be sorted by score
                    scores = [doc.metadata.get('relevance_score', 0) for doc in reranked]
                    assert scores == sorted(scores, reverse=True)

    def test_rerank_without_reranker(self, sample_documents):
        """Test reranking falls back gracefully without reranker."""
        from src.dashboard.llm import LlmChain, RAGConfig

        config = RAGConfig(final_k=3)

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder', side_effect=Exception("Load failed")):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain(config=config)

                    # Should return first k documents without reranking
                    result = chain._rerank_documents("test query", sample_documents)
                    assert len(result) == 3


class TestContentHashing:
    """Tests for content hashing and cache invalidation."""

    def test_compute_content_hash(self, sample_text_files, temp_data_dir):
        """Test content hash computation."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    chain.data_dir = temp_data_dir
                    chain.persist_dir = temp_data_dir / ".chroma_db"

                    hash1 = chain._compute_content_hash()

                    # Hash should be consistent
                    hash2 = chain._compute_content_hash()
                    assert hash1 == hash2

    def test_hash_changes_with_content(self, sample_text_files, temp_data_dir):
        """Test that hash changes when content changes."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    chain.data_dir = temp_data_dir
                    chain.persist_dir = temp_data_dir / ".chroma_db"

                    hash1 = chain._compute_content_hash()

                    # Modify a file
                    (temp_data_dir / "new_file.txt").write_text("New content")

                    hash2 = chain._compute_content_hash()
                    assert hash1 != hash2


class TestResponseGeneration:
    """Tests for response generation."""

    def test_get_response_no_chain(self):
        """Test response when no chain is initialized."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()
                    chain.conversation_chain = None

                    response = chain.get_response("Test question")

                    assert "don't have access" in response.lower()

    def test_get_structured_response(self):
        """Test structured response generation."""
        from src.dashboard.llm import LlmChain, RetrievalResult

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()

                    # Mock the get_response method
                    chain.get_response = MagicMock(return_value="Test answer")
                    chain._last_sources = [
                        RetrievalResult("content", "file.pdf", 1, 0.8, "id1")
                    ]

                    response = chain.get_structured_response("Test question")

                    assert response.answer == "Test answer"
                    assert len(response.sources) == 1
                    assert response.confidence == 0.8


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_openai_error(self):
        """Test handling of OpenAI API errors."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()

                    # Mock conversation chain to raise OpenAI error
                    mock_chain = MagicMock()
                    mock_chain.invoke.side_effect = Exception("OpenAI API error")
                    chain.conversation_chain = mock_chain

                    response = chain.get_response("Test question")

                    assert "error" in response.lower()
                    assert "openai" in response.lower()

    def test_handles_general_error(self):
        """Test handling of general errors."""
        from src.dashboard.llm import LlmChain

        with patch('src.dashboard.llm.OpenAIEmbeddings'):
            with patch('src.dashboard.llm.CrossEncoder'):
                with patch.object(LlmChain, '_setup_chain'):
                    chain = LlmChain()

                    # Mock conversation chain to raise general error
                    mock_chain = MagicMock()
                    mock_chain.invoke.side_effect = Exception("Some random error")
                    chain.conversation_chain = mock_chain

                    response = chain.get_response("Test question")

                    assert "error" in response.lower()
