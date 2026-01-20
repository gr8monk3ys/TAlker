"""
Multi-provider LLM and Embedding support.

Supports:
- OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Cohere (Command R+, Command R)
- Ollama (Local: Llama 3, Mistral, Mixtral, Phi-3, etc.)
- HuggingFace (Local transformers)
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Iterator

from langchain.schema import BaseMessage
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    COHERE = "cohere"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    FASTEMBED = "fastembed"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    provider: LLMProvider
    context_window: int
    input_cost_per_1k: float  # USD per 1K tokens
    output_cost_per_1k: float
    supports_streaming: bool = True
    supports_functions: bool = False
    is_local: bool = False
    description: str = ""


@dataclass
class EmbeddingInfo:
    """Information about an embedding model."""
    name: str
    provider: EmbeddingProvider
    dimensions: int
    cost_per_1k: float  # USD per 1K tokens
    is_local: bool = False
    description: str = ""


# Available models registry
LLM_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=LLMProvider.OPENAI,
        context_window=128000,
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        supports_functions=True,
        description="Most capable OpenAI model, multimodal"
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        context_window=128000,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        supports_functions=True,
        description="Fast and affordable, great for most tasks"
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=LLMProvider.OPENAI,
        context_window=16385,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        supports_functions=True,
        description="Legacy model, fast and cheap"
    ),

    # Anthropic
    "claude-3-5-sonnet-20241022": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=LLMProvider.ANTHROPIC,
        context_window=200000,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        supports_functions=True,
        description="Best balance of intelligence and speed"
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        name="claude-3-5-haiku-20241022",
        provider=LLMProvider.ANTHROPIC,
        context_window=200000,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
        supports_functions=True,
        description="Fast and affordable Claude model"
    ),

    # Google
    "gemini-1.5-pro": ModelInfo(
        name="gemini-1.5-pro",
        provider=LLMProvider.GOOGLE,
        context_window=2000000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        supports_functions=True,
        description="Google's most capable model, 2M context"
    ),
    "gemini-1.5-flash": ModelInfo(
        name="gemini-1.5-flash",
        provider=LLMProvider.GOOGLE,
        context_window=1000000,
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        supports_functions=True,
        description="Fast and efficient Gemini model"
    ),

    # Cohere
    "command-r-plus": ModelInfo(
        name="command-r-plus",
        provider=LLMProvider.COHERE,
        context_window=128000,
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        supports_functions=True,
        description="Cohere's most capable model for RAG"
    ),
    "command-r": ModelInfo(
        name="command-r",
        provider=LLMProvider.COHERE,
        context_window=128000,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        description="Efficient Cohere model optimized for RAG"
    ),

    # Ollama (Local)
    "llama3.1:8b": ModelInfo(
        name="llama3.1:8b",
        provider=LLMProvider.OLLAMA,
        context_window=128000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Meta's Llama 3.1 8B - great local model"
    ),
    "llama3.1:70b": ModelInfo(
        name="llama3.1:70b",
        provider=LLMProvider.OLLAMA,
        context_window=128000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Meta's Llama 3.1 70B - powerful local model"
    ),
    "mistral:7b": ModelInfo(
        name="mistral:7b",
        provider=LLMProvider.OLLAMA,
        context_window=32000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Mistral 7B - efficient local model"
    ),
    "mixtral:8x7b": ModelInfo(
        name="mixtral:8x7b",
        provider=LLMProvider.OLLAMA,
        context_window=32000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Mixtral 8x7B MoE - powerful local model"
    ),
    "phi3:medium": ModelInfo(
        name="phi3:medium",
        provider=LLMProvider.OLLAMA,
        context_window=128000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Microsoft Phi-3 Medium - efficient for RAG"
    ),
    "qwen2.5:7b": ModelInfo(
        name="qwen2.5:7b",
        provider=LLMProvider.OLLAMA,
        context_window=128000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="Alibaba Qwen 2.5 7B - strong multilingual"
    ),
    "deepseek-r1:8b": ModelInfo(
        name="deepseek-r1:8b",
        provider=LLMProvider.OLLAMA,
        context_window=64000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        is_local=True,
        description="DeepSeek R1 8B - reasoning focused"
    ),
}

EMBEDDING_MODELS: dict[str, EmbeddingInfo] = {
    # OpenAI
    "text-embedding-3-large": EmbeddingInfo(
        name="text-embedding-3-large",
        provider=EmbeddingProvider.OPENAI,
        dimensions=3072,
        cost_per_1k=0.00013,
        description="Best OpenAI embeddings"
    ),
    "text-embedding-3-small": EmbeddingInfo(
        name="text-embedding-3-small",
        provider=EmbeddingProvider.OPENAI,
        dimensions=1536,
        cost_per_1k=0.00002,
        description="Fast and affordable OpenAI embeddings"
    ),

    # Cohere
    "embed-english-v3.0": EmbeddingInfo(
        name="embed-english-v3.0",
        provider=EmbeddingProvider.COHERE,
        dimensions=1024,
        cost_per_1k=0.0001,
        description="Cohere English embeddings"
    ),
    "embed-multilingual-v3.0": EmbeddingInfo(
        name="embed-multilingual-v3.0",
        provider=EmbeddingProvider.COHERE,
        dimensions=1024,
        cost_per_1k=0.0001,
        description="Cohere multilingual embeddings"
    ),

    # Ollama (Local)
    "nomic-embed-text": EmbeddingInfo(
        name="nomic-embed-text",
        provider=EmbeddingProvider.OLLAMA,
        dimensions=768,
        cost_per_1k=0.0,
        is_local=True,
        description="Nomic AI embeddings via Ollama"
    ),
    "mxbai-embed-large": EmbeddingInfo(
        name="mxbai-embed-large",
        provider=EmbeddingProvider.OLLAMA,
        dimensions=1024,
        cost_per_1k=0.0,
        is_local=True,
        description="MixedBread AI large embeddings"
    ),
    "all-minilm": EmbeddingInfo(
        name="all-minilm",
        provider=EmbeddingProvider.OLLAMA,
        dimensions=384,
        cost_per_1k=0.0,
        is_local=True,
        description="Sentence Transformers MiniLM via Ollama"
    ),

    # HuggingFace (Local)
    "BAAI/bge-large-en-v1.5": EmbeddingInfo(
        name="BAAI/bge-large-en-v1.5",
        provider=EmbeddingProvider.HUGGINGFACE,
        dimensions=1024,
        cost_per_1k=0.0,
        is_local=True,
        description="BGE Large - top performing open embeddings"
    ),
    "sentence-transformers/all-mpnet-base-v2": EmbeddingInfo(
        name="sentence-transformers/all-mpnet-base-v2",
        provider=EmbeddingProvider.HUGGINGFACE,
        dimensions=768,
        cost_per_1k=0.0,
        is_local=True,
        description="MPNet - great general purpose embeddings"
    ),

    # FastEmbed (Local, optimized)
    "BAAI/bge-small-en-v1.5": EmbeddingInfo(
        name="BAAI/bge-small-en-v1.5",
        provider=EmbeddingProvider.FASTEMBED,
        dimensions=384,
        cost_per_1k=0.0,
        is_local=True,
        description="BGE Small via FastEmbed - fast and efficient"
    ),
}


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    streaming: bool = True

    # Provider-specific settings
    ollama_base_url: str = "http://localhost:11434"
    huggingface_device: str = "cpu"  # or "cuda", "mps"

    def get_llm_info(self) -> ModelInfo:
        """Get info about the configured LLM."""
        return LLM_MODELS.get(self.llm_model, LLM_MODELS["gpt-4o"])

    def get_embedding_info(self) -> EmbeddingInfo:
        """Get info about the configured embedding model."""
        return EMBEDDING_MODELS.get(
            self.embedding_model,
            EMBEDDING_MODELS["text-embedding-3-large"]
        )


class TokenTracker:
    """Track token usage and costs across providers."""

    def __init__(self):
        self.usage: dict[str, dict] = {}

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> dict:
        """Track token usage for a model."""
        if model not in self.usage:
            self.usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0,
                "calls": 0
            }

        model_info = LLM_MODELS.get(model)
        cost = 0.0
        if model_info:
            cost = (
                (input_tokens / 1000) * model_info.input_cost_per_1k +
                (output_tokens / 1000) * model_info.output_cost_per_1k
            )

        self.usage[model]["input_tokens"] += input_tokens
        self.usage[model]["output_tokens"] += output_tokens
        self.usage[model]["total_cost"] += cost
        self.usage[model]["calls"] += 1

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(m["total_cost"] for m in self.usage.values())

    def get_summary(self) -> dict:
        """Get usage summary."""
        return {
            "by_model": self.usage,
            "total_cost": self.get_total_cost(),
            "total_calls": sum(m["calls"] for m in self.usage.values())
        }

    def reset(self):
        """Reset usage tracking."""
        self.usage = {}


class LLMFactory:
    """Factory for creating LLM instances from different providers."""

    @staticmethod
    def create(
        config: ProviderConfig,
        callbacks: Optional[list[BaseCallbackHandler]] = None
    ):
        """Create an LLM instance based on configuration."""
        model_info = config.get_llm_info()
        provider = model_info.provider

        common_kwargs = {
            "temperature": config.temperature,
            "streaming": config.streaming and model_info.supports_streaming,
        }
        if config.max_tokens:
            common_kwargs["max_tokens"] = config.max_tokens
        if callbacks:
            common_kwargs["callbacks"] = callbacks

        if provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.llm_model,
                **common_kwargs
            )

        elif provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.llm_model,
                **common_kwargs
            )

        elif provider == LLMProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.llm_model,
                **common_kwargs
            )

        elif provider == LLMProvider.COHERE:
            from langchain_cohere import ChatCohere
            return ChatCohere(
                model=config.llm_model,
                **common_kwargs
            )

        elif provider == LLMProvider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm_model,
                base_url=config.ollama_base_url,
                **common_kwargs
            )

        elif provider == LLMProvider.HUGGINGFACE:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                device_map=config.huggingface_device
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_tokens or 512
            )
            hf_pipeline = HuggingFacePipeline(pipeline=pipe)
            return ChatHuggingFace(llm=hf_pipeline)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class EmbeddingFactory:
    """Factory for creating embedding instances from different providers."""

    @staticmethod
    def create(config: ProviderConfig):
        """Create an embedding instance based on configuration."""
        embedding_info = config.get_embedding_info()
        provider = embedding_info.provider

        if provider == EmbeddingProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=config.embedding_model,
                dimensions=embedding_info.dimensions
            )

        elif provider == EmbeddingProvider.COHERE:
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(
                model=config.embedding_model
            )

        elif provider == EmbeddingProvider.OLLAMA:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=config.embedding_model,
                base_url=config.ollama_base_url
            )

        elif provider == EmbeddingProvider.HUGGINGFACE:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": config.huggingface_device},
                encode_kwargs={"normalize_embeddings": True}
            )

        elif provider == EmbeddingProvider.FASTEMBED:
            from langchain_community.embeddings import FastEmbedEmbeddings
            return FastEmbedEmbeddings(
                model_name=config.embedding_model
            )

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


def check_ollama_availability(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and available."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_available_ollama_models(
    base_url: str = "http://localhost:11434"
) -> list[str]:
    """Get list of models available in local Ollama instance."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        logger.warning(f"Failed to get Ollama models: {e}")
    return []


def get_models_by_provider(provider: LLMProvider) -> list[ModelInfo]:
    """Get all models for a specific provider."""
    return [m for m in LLM_MODELS.values() if m.provider == provider]


def get_embeddings_by_provider(provider: EmbeddingProvider) -> list[EmbeddingInfo]:
    """Get all embedding models for a specific provider."""
    return [e for e in EMBEDDING_MODELS.values() if e.provider == provider]


def get_local_models() -> list[ModelInfo]:
    """Get all local (offline) models."""
    return [m for m in LLM_MODELS.values() if m.is_local]


def get_local_embeddings() -> list[EmbeddingInfo]:
    """Get all local (offline) embedding models."""
    return [e for e in EMBEDDING_MODELS.values() if e.is_local]


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "cohere": bool(os.getenv("COHERE_API_KEY")),
        "ollama": check_ollama_availability(),
    }
