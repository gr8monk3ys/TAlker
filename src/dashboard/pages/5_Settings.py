"""
Settings page for configuring RAG providers and parameters.
"""

import streamlit as st
from src.dashboard.llm import LlmChain, RAGConfig
from src.dashboard.providers import (
    LLM_MODELS,
    EMBEDDING_MODELS,
    LLMProvider,
    EmbeddingProvider,
    ProviderConfig,
    validate_api_keys,
    check_ollama_availability,
    get_available_ollama_models,
    get_models_by_provider,
    get_embeddings_by_provider,
    get_local_models,
    get_local_embeddings,
)

st.set_page_config(
    page_title="Settings",
    page_icon="cog:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Settings")
st.markdown("Configure your RAG system providers, models, and parameters.")

# Initialize LLM chain if not exists
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = LlmChain()

chain = st.session_state.llm_chain

# Provider status
st.markdown("## Provider Status")

api_status = validate_api_keys()
ollama_available = api_status.get("ollama", False)

cols = st.columns(5)
providers_display = [
    ("OpenAI", api_status.get("openai", False), "https://platform.openai.com/api-keys"),
    ("Anthropic", api_status.get("anthropic", False), "https://console.anthropic.com/"),
    ("Google", api_status.get("google", False), "https://makersuite.google.com/app/apikey"),
    ("Cohere", api_status.get("cohere", False), "https://dashboard.cohere.com/api-keys"),
    ("Ollama", ollama_available, "https://ollama.ai"),
]

for col, (name, available, url) in zip(cols, providers_display):
    with col:
        if available:
            st.success(f"**{name}**")
        else:
            st.error(f"**{name}**")
            st.caption(f"[Get API Key]({url})")

# LLM Configuration
st.markdown("---")
st.markdown("## LLM Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Select Provider")

    # Group models by provider
    provider_options = ["OpenAI", "Anthropic", "Google", "Cohere", "Ollama (Local)"]
    available_providers = []

    for p in provider_options:
        p_key = p.lower().replace(" (local)", "")
        if api_status.get(p_key, False):
            available_providers.append(p)

    if not available_providers:
        available_providers = ["Ollama (Local)"]  # Fallback

    selected_provider = st.selectbox(
        "Provider",
        available_providers,
        help="Select the LLM provider. Only providers with valid API keys are shown."
    )

    # Get models for selected provider
    provider_map = {
        "OpenAI": LLMProvider.OPENAI,
        "Anthropic": LLMProvider.ANTHROPIC,
        "Google": LLMProvider.GOOGLE,
        "Cohere": LLMProvider.COHERE,
        "Ollama (Local)": LLMProvider.OLLAMA,
    }

    provider_enum = provider_map.get(selected_provider, LLMProvider.OPENAI)
    provider_models = get_models_by_provider(provider_enum)

    # For Ollama, also show locally available models
    if provider_enum == LLMProvider.OLLAMA and ollama_available:
        local_models = get_available_ollama_models()
        st.info(f"Ollama models available: {', '.join(local_models) if local_models else 'None found'}")

    model_options = {m.name: m for m in provider_models}
    current_model = chain.config.llm_model

    # Select model
    selected_model = st.selectbox(
        "Model",
        list(model_options.keys()),
        index=list(model_options.keys()).index(current_model) if current_model in model_options else 0,
        help="Select the LLM model to use"
    )

    # Show model info
    if selected_model in model_options:
        model_info = model_options[selected_model]
        st.caption(model_info.description)
        st.caption(f"Context: {model_info.context_window:,} tokens")
        if not model_info.is_local:
            st.caption(f"Cost: ${model_info.input_cost_per_1k:.4f}/1K in, ${model_info.output_cost_per_1k:.4f}/1K out")

with col2:
    st.markdown("### Model Parameters")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=chain.config.temperature,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )

    use_query_expansion = st.checkbox(
        "Enable Query Expansion",
        value=chain.config.use_query_expansion,
        help="Generate multiple query variations for better retrieval"
    )

    use_reranker = st.checkbox(
        "Enable Cross-Encoder Reranking",
        value=chain.config.use_reranker,
        help="Use cross-encoder to rerank retrieved documents for better relevance"
    )

# Embedding Configuration
st.markdown("---")
st.markdown("## Embedding Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Select Embedding Provider")

    embedding_provider_options = ["OpenAI", "Cohere", "Ollama (Local)", "HuggingFace (Local)", "FastEmbed (Local)"]
    available_embed_providers = []

    embed_provider_map = {
        "OpenAI": (EmbeddingProvider.OPENAI, "openai"),
        "Cohere": (EmbeddingProvider.COHERE, "cohere"),
        "Ollama (Local)": (EmbeddingProvider.OLLAMA, "ollama"),
        "HuggingFace (Local)": (EmbeddingProvider.HUGGINGFACE, None),
        "FastEmbed (Local)": (EmbeddingProvider.FASTEMBED, None),
    }

    for p in embedding_provider_options:
        enum, key = embed_provider_map[p]
        if key is None or api_status.get(key, False) or "(Local)" in p:
            available_embed_providers.append(p)

    selected_embed_provider = st.selectbox(
        "Embedding Provider",
        available_embed_providers,
        help="Select the embedding provider"
    )

    embed_enum, _ = embed_provider_map.get(selected_embed_provider, (EmbeddingProvider.OPENAI, None))
    embed_models = get_embeddings_by_provider(embed_enum)

    embed_model_options = {m.name: m for m in embed_models}
    current_embed = chain.config.embedding_model

    selected_embed = st.selectbox(
        "Embedding Model",
        list(embed_model_options.keys()),
        index=list(embed_model_options.keys()).index(current_embed) if current_embed in embed_model_options else 0,
        help="Select the embedding model. Changing this will rebuild the vector index."
    )

    # Show embedding info
    if selected_embed in embed_model_options:
        embed_info = embed_model_options[selected_embed]
        st.caption(embed_info.description)
        st.caption(f"Dimensions: {embed_info.dimensions}")
        if not embed_info.is_local:
            st.caption(f"Cost: ${embed_info.cost_per_1k:.5f}/1K tokens")

with col2:
    st.markdown("### Retrieval Parameters")

    initial_k = st.number_input(
        "Initial K (before reranking)",
        min_value=5,
        max_value=50,
        value=chain.config.initial_k,
        help="Number of documents to retrieve before reranking"
    )

    final_k = st.number_input(
        "Final K (after reranking)",
        min_value=1,
        max_value=20,
        value=chain.config.final_k,
        help="Number of documents to keep after reranking"
    )

    bm25_weight = st.slider(
        "BM25 Weight",
        min_value=0.0,
        max_value=1.0,
        value=chain.config.bm25_weight,
        step=0.1,
        help="Weight for keyword-based BM25 retrieval (semantic = 1 - BM25)"
    )

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=chain.config.similarity_threshold,
        step=0.05,
        help="Minimum similarity score for retrieved documents"
    )

# Chunking Configuration
st.markdown("---")
st.markdown("## Document Chunking")

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=4000,
        value=chain.config.chunk_size,
        step=100,
        help="Size of text chunks in characters"
    )

with col2:
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=chain.config.chunk_overlap,
        step=50,
        help="Overlap between consecutive chunks"
    )

# Apply Changes
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("Apply Changes", type="primary"):
        with st.spinner("Applying configuration..."):
            try:
                # Check if embedding model changed (requires rebuild)
                embed_changed = selected_embed != chain.config.embedding_model

                # Update configuration
                chain.config.provider_config.llm_model = selected_model
                chain.config.provider_config.temperature = temperature
                chain.config.use_query_expansion = use_query_expansion
                chain.config.use_reranker = use_reranker
                chain.config.initial_k = initial_k
                chain.config.final_k = final_k
                chain.config.bm25_weight = bm25_weight
                chain.config.semantic_weight = 1.0 - bm25_weight
                chain.config.similarity_threshold = similarity_threshold
                chain.config.chunk_size = chunk_size
                chain.config.chunk_overlap = chunk_overlap

                # Switch provider (handles embedding changes)
                if embed_changed:
                    st.warning("Embedding model changed. Rebuilding vector index...")
                    success = chain.switch_provider(
                        llm_model=selected_model,
                        embedding_model=selected_embed
                    )
                else:
                    success = chain.switch_provider(llm_model=selected_model)

                if success:
                    st.success("Configuration applied successfully!")
                else:
                    st.error("Failed to apply configuration")

            except Exception as e:
                st.error(f"Error applying configuration: {e}")

with col2:
    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding vector index..."):
            if chain.rebuild_index():
                st.success("Index rebuilt successfully!")
            else:
                st.error("Failed to rebuild index")

# Current Configuration Summary
st.markdown("---")
st.markdown("## Current Configuration")

config_data = {
    "LLM Model": chain.config.llm_model,
    "Embedding Model": chain.config.embedding_model,
    "Temperature": chain.config.temperature,
    "Query Expansion": "Enabled" if chain.config.use_query_expansion else "Disabled",
    "Reranking": "Enabled" if chain.config.use_reranker else "Disabled",
    "Initial K": chain.config.initial_k,
    "Final K": chain.config.final_k,
    "BM25 Weight": f"{chain.config.bm25_weight:.1f}",
    "Semantic Weight": f"{chain.config.semantic_weight:.1f}",
    "Chunk Size": chain.config.chunk_size,
    "Chunk Overlap": chain.config.chunk_overlap,
}

col1, col2 = st.columns(2)
items = list(config_data.items())
mid = len(items) // 2

with col1:
    for key, value in items[:mid]:
        st.text(f"{key}: {value}")

with col2:
    for key, value in items[mid:]:
        st.text(f"{key}: {value}")

# Local Model Setup Guide
st.markdown("---")
with st.expander("Setting Up Local Models (Ollama)"):
    st.markdown("""
    ### Running Models Locally with Ollama

    Ollama allows you to run LLMs locally without any API costs.

    **Installation:**
    ```bash
    # macOS
    brew install ollama

    # Linux
    curl -fsSL https://ollama.ai/install.sh | sh

    # Windows
    # Download from https://ollama.ai/download
    ```

    **Start Ollama:**
    ```bash
    ollama serve
    ```

    **Pull Models:**
    ```bash
    # Recommended models for RAG
    ollama pull llama3.1:8b          # Great balance of speed/quality
    ollama pull mistral:7b           # Fast and efficient
    ollama pull nomic-embed-text     # For embeddings
    ollama pull mxbai-embed-large    # Better quality embeddings
    ```

    **Benefits of Local Models:**
    - No API costs
    - Data stays on your machine
    - Works offline
    - No rate limits

    **Recommended Local Setup:**
    - LLM: `llama3.1:8b` or `mistral:7b`
    - Embeddings: `nomic-embed-text` or `mxbai-embed-large`
    """)

# API Key Setup Guide
with st.expander("Setting Up API Keys"):
    st.markdown("""
    ### Configuring API Keys

    Add the following to your `.env` file:

    ```bash
    # OpenAI (required for GPT-4o, GPT-4o-mini)
    OPENAI_API_KEY=sk-...

    # Anthropic (required for Claude models)
    ANTHROPIC_API_KEY=sk-ant-...

    # Google (required for Gemini models)
    GOOGLE_API_KEY=...

    # Cohere (required for Command R models)
    COHERE_API_KEY=...
    ```

    **Getting API Keys:**
    - OpenAI: https://platform.openai.com/api-keys
    - Anthropic: https://console.anthropic.com/
    - Google: https://makersuite.google.com/app/apikey
    - Cohere: https://dashboard.cohere.com/api-keys
    """)
