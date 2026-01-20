"""
Pytest configuration and fixtures for TAlker tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        yield data_dir


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "lecture1.pdf", "page": 1}
        ),
        Document(
            page_content="The final exam is worth 40% of your grade.",
            metadata={"source": "syllabus.pdf", "page": 5}
        ),
        Document(
            page_content="Office hours are on Tuesdays from 2-4 PM.",
            metadata={"source": "syllabus.pdf", "page": 2}
        ),
        Document(
            page_content="Neural networks consist of layers of interconnected nodes.",
            metadata={"source": "lecture2.pdf", "page": 3}
        ),
        Document(
            page_content="Assignments must be submitted through the online portal.",
            metadata={"source": "syllabus.pdf", "page": 6}
        ),
    ]


@pytest.fixture
def sample_text_files(temp_data_dir):
    """Create sample text files for testing document loading."""
    files = []

    # Create syllabus file
    syllabus = temp_data_dir / "syllabus.txt"
    syllabus.write_text("""
Course: Introduction to Machine Learning
Instructor: Dr. Smith
Office Hours: Tuesday 2-4 PM

Grading:
- Assignments: 30%
- Midterm: 30%
- Final Exam: 40%

Late Policy: 10% penalty per day
""")
    files.append(syllabus)

    # Create lecture notes
    lecture = temp_data_dir / "lecture1.txt"
    lecture.write_text("""
Lecture 1: Introduction to Machine Learning

Machine learning is a field of artificial intelligence that uses
statistical techniques to give computer systems the ability to learn.

Key concepts:
- Supervised learning
- Unsupervised learning
- Reinforcement learning
""")
    files.append(lecture)

    return files


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for testing without API calls."""
    with patch('src.dashboard.llm.OpenAIEmbeddings') as mock:
        mock_instance = MagicMock()
        # Return consistent fake embeddings
        mock_instance.embed_documents.return_value = [[0.1] * 1536 for _ in range(10)]
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai_chat():
    """Mock OpenAI chat for testing without API calls."""
    with patch('src.dashboard.llm.ChatOpenAI') as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(
            content="This is a test response about the course materials."
        )
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_cross_encoder():
    """Mock cross-encoder for testing without model loading."""
    with patch('src.dashboard.llm.CrossEncoder') as mock:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def rag_config():
    """Create a test RAG configuration."""
    from src.dashboard.llm import RAGConfig

    return RAGConfig(
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=50,
        initial_k=10,
        final_k=3,
    )


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
    yield
