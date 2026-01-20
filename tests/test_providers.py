"""
Tests for the multi-provider system.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLLMModels:
    """Tests for LLM model registry."""

    def test_llm_models_exist(self):
        """Test that LLM models are registered."""
        from src.dashboard.providers import LLM_MODELS

        assert len(LLM_MODELS) > 0
        assert "gpt-4o" in LLM_MODELS
        assert "claude-3-5-sonnet-20241022" in LLM_MODELS

    def test_model_info_structure(self):
        """Test ModelInfo structure."""
        from src.dashboard.providers import LLM_MODELS, LLMProvider

        gpt4o = LLM_MODELS["gpt-4o"]

        assert gpt4o.name == "gpt-4o"
        assert gpt4o.provider == LLMProvider.OPENAI
        assert gpt4o.context_window > 0
        assert gpt4o.input_cost_per_1k >= 0
        assert gpt4o.output_cost_per_1k >= 0

    def test_local_models_are_free(self):
        """Test that local models have zero cost."""
        from src.dashboard.providers import LLM_MODELS

        for name, model in LLM_MODELS.items():
            if model.is_local:
                assert model.input_cost_per_1k == 0.0
                assert model.output_cost_per_1k == 0.0


class TestEmbeddingModels:
    """Tests for embedding model registry."""

    def test_embedding_models_exist(self):
        """Test that embedding models are registered."""
        from src.dashboard.providers import EMBEDDING_MODELS

        assert len(EMBEDDING_MODELS) > 0
        assert "text-embedding-3-large" in EMBEDDING_MODELS
        assert "nomic-embed-text" in EMBEDDING_MODELS

    def test_embedding_info_structure(self):
        """Test EmbeddingInfo structure."""
        from src.dashboard.providers import EMBEDDING_MODELS, EmbeddingProvider

        openai_embed = EMBEDDING_MODELS["text-embedding-3-large"]

        assert openai_embed.provider == EmbeddingProvider.OPENAI
        assert openai_embed.dimensions > 0
        assert openai_embed.cost_per_1k >= 0


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from src.dashboard.providers import ProviderConfig

        config = ProviderConfig()

        assert config.llm_model == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.temperature == 0.1
        assert config.streaming is True

    def test_custom_config(self):
        """Test custom configuration."""
        from src.dashboard.providers import ProviderConfig

        config = ProviderConfig(
            llm_model="claude-3-5-sonnet-20241022",
            embedding_model="embed-english-v3.0",
            temperature=0.5
        )

        assert config.llm_model == "claude-3-5-sonnet-20241022"
        assert config.embedding_model == "embed-english-v3.0"
        assert config.temperature == 0.5

    def test_get_model_info(self):
        """Test getting model info from config."""
        from src.dashboard.providers import ProviderConfig

        config = ProviderConfig(llm_model="gpt-4o")
        info = config.get_llm_info()

        assert info.name == "gpt-4o"


class TestTokenTracker:
    """Tests for TokenTracker."""

    def test_track_usage(self):
        """Test tracking token usage."""
        from src.dashboard.providers import TokenTracker

        tracker = TokenTracker()

        result = tracker.track("gpt-4o", input_tokens=100, output_tokens=50)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cost"] > 0

    def test_cumulative_tracking(self):
        """Test cumulative usage tracking."""
        from src.dashboard.providers import TokenTracker

        tracker = TokenTracker()

        tracker.track("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.track("gpt-4o", input_tokens=200, output_tokens=100)

        summary = tracker.get_summary()

        assert summary["by_model"]["gpt-4o"]["input_tokens"] == 300
        assert summary["by_model"]["gpt-4o"]["output_tokens"] == 150
        assert summary["by_model"]["gpt-4o"]["calls"] == 2

    def test_total_cost(self):
        """Test total cost calculation."""
        from src.dashboard.providers import TokenTracker

        tracker = TokenTracker()

        tracker.track("gpt-4o", input_tokens=1000, output_tokens=500)

        total = tracker.get_total_cost()
        assert total > 0

    def test_reset_tracker(self):
        """Test resetting the tracker."""
        from src.dashboard.providers import TokenTracker

        tracker = TokenTracker()

        tracker.track("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.reset()

        assert tracker.get_total_cost() == 0


class TestLLMFactory:
    """Tests for LLMFactory."""

    def test_create_openai_llm(self):
        """Test creating OpenAI LLM."""
        from src.dashboard.providers import LLMFactory, ProviderConfig

        config = ProviderConfig(llm_model="gpt-4o")

        with patch('src.dashboard.providers.ChatOpenAI') as mock:
            mock.return_value = MagicMock()
            llm = LLMFactory.create(config)
            mock.assert_called_once()

    def test_create_anthropic_llm(self):
        """Test creating Anthropic LLM."""
        from src.dashboard.providers import LLMFactory, ProviderConfig

        config = ProviderConfig(llm_model="claude-3-5-sonnet-20241022")

        with patch('src.dashboard.providers.ChatAnthropic') as mock:
            mock.return_value = MagicMock()
            llm = LLMFactory.create(config)
            mock.assert_called_once()

    def test_create_ollama_llm(self):
        """Test creating Ollama LLM."""
        from src.dashboard.providers import LLMFactory, ProviderConfig

        config = ProviderConfig(llm_model="llama3.1:8b")

        with patch('src.dashboard.providers.ChatOllama') as mock:
            mock.return_value = MagicMock()
            llm = LLMFactory.create(config)
            mock.assert_called_once()


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory."""

    def test_create_openai_embeddings(self):
        """Test creating OpenAI embeddings."""
        from src.dashboard.providers import EmbeddingFactory, ProviderConfig

        config = ProviderConfig(embedding_model="text-embedding-3-large")

        with patch('src.dashboard.providers.OpenAIEmbeddings') as mock:
            mock.return_value = MagicMock()
            embeddings = EmbeddingFactory.create(config)
            mock.assert_called_once()

    def test_create_ollama_embeddings(self):
        """Test creating Ollama embeddings."""
        from src.dashboard.providers import EmbeddingFactory, ProviderConfig

        config = ProviderConfig(embedding_model="nomic-embed-text")

        with patch('src.dashboard.providers.OllamaEmbeddings') as mock:
            mock.return_value = MagicMock()
            embeddings = EmbeddingFactory.create(config)
            mock.assert_called_once()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_models_by_provider(self):
        """Test filtering models by provider."""
        from src.dashboard.providers import get_models_by_provider, LLMProvider

        openai_models = get_models_by_provider(LLMProvider.OPENAI)

        assert len(openai_models) > 0
        for model in openai_models:
            assert model.provider == LLMProvider.OPENAI

    def test_get_local_models(self):
        """Test getting local models."""
        from src.dashboard.providers import get_local_models

        local = get_local_models()

        assert len(local) > 0
        for model in local:
            assert model.is_local is True

    def test_get_local_embeddings(self):
        """Test getting local embedding models."""
        from src.dashboard.providers import get_local_embeddings

        local = get_local_embeddings()

        assert len(local) > 0
        for embed in local:
            assert embed.is_local is True

    def test_validate_api_keys(self):
        """Test API key validation."""
        from src.dashboard.providers import validate_api_keys

        # Should return dict with provider status
        status = validate_api_keys()

        assert "openai" in status
        assert "anthropic" in status
        assert "google" in status
        assert "cohere" in status
        assert "ollama" in status

    def test_check_ollama_availability(self):
        """Test Ollama availability check."""
        from src.dashboard.providers import check_ollama_availability

        with patch('src.dashboard.providers.requests.get') as mock:
            mock.return_value = MagicMock(status_code=200)
            result = check_ollama_availability()
            assert result is True

        with patch('src.dashboard.providers.requests.get') as mock:
            mock.side_effect = Exception("Connection failed")
            result = check_ollama_availability()
            assert result is False
