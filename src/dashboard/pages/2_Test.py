"""
Enhanced chat interface with source citations and confidence scores.
"""

import html
import streamlit as st
from src.dashboard.llm import LlmChain, RAGConfig
import os


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS."""
    if text is None:
        return ""
    return html.escape(str(text))

st.set_page_config(
    page_title="Test Bot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat display
st.markdown("""
<style>
.source-card {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 3px solid #0066cc;
}
.confidence-high { color: #28a745; }
.confidence-medium { color: #ffc107; }
.confidence-low { color: #dc3545; }
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 15px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM chain with config
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = LlmChain()

# Initialize sources storage
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

st.title("🧪 Test Bot")
st.markdown("""
Test the RAG system with your course materials. The bot uses **hybrid search** (BM25 + semantic),
**cross-encoder reranking**, and provides **source citations** with confidence scores.
""")

# Sidebar with context information and settings
with st.sidebar:
    st.markdown("### RAG Configuration")

    # Show current config
    config = st.session_state.llm_chain.config
    st.info(f"""
    **Model:** {config.llm_model}
    **Embeddings:** {config.embedding_model}
    **Chunk Size:** {config.chunk_size}
    **Top-K:** {config.final_k}
    """)

    st.markdown("### Context Status")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    try:
        files = []
        for root, dirs, filenames in os.walk(data_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in filenames:
                if f.endswith(('.txt', '.pdf', '.csv', '.md')):
                    files.append(os.path.relpath(os.path.join(root, f), data_dir))
    except Exception:
        files = []

    if files:
        st.success(f"✅ {len(files)} files loaded")
        with st.expander("View files"):
            for file in files[:10]:  # Show first 10
                st.write(f"📄 {file}")
            if len(files) > 10:
                st.write(f"... and {len(files) - 10} more")
    else:
        st.warning("""
            ⚠️ No files uploaded yet!

            Please go to the Upload page and upload your course materials first.
        """)

    st.divider()

    # Actions
    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding vector index..."):
            success = st.session_state.llm_chain.rebuild_index()
            if success:
                st.success("Index rebuilt successfully!")
            else:
                st.error("Failed to rebuild index")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# Main content area with two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Chat")

    # Chat container
    chat_container = st.container()
    with chat_container:
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show sources for assistant messages if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for src in message["sources"]:
                            # Escape user-controlled content to prevent XSS
                            safe_source = escape_html(src['source'])
                            safe_page = escape_html(str(src['page'])) if src.get('page') else ""
                            safe_content = escape_html(src['content'][:150])
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{safe_source}</strong>
                                {f" (Page {safe_page})" if safe_page else ""}
                                <br><small>Relevance: {src['relevance']:.1%}</small>
                                <br><small style="color: #666;">{safe_content}...</small>
                            </div>
                            """, unsafe_allow_html=True)

        # Accept user input
        if prompt := st.chat_input("Ask me anything about your course..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and analyzing..."):
                    try:
                        # Get structured response with sources
                        response = st.session_state.llm_chain.get_structured_response(prompt)

                        # Display answer
                        st.markdown(response.answer)

                        # Store sources for sidebar display
                        sources_data = [
                            {
                                'source': src.source,
                                'page': src.page,
                                'relevance': src.relevance_score,
                                'content': src.content
                            }
                            for src in response.sources
                        ]
                        st.session_state.last_sources = sources_data

                        # Show sources in expander
                        if sources_data:
                            with st.expander("📚 View Sources"):
                                for src in sources_data:
                                    # Escape user-controlled content to prevent XSS
                                    safe_source = escape_html(src['source'])
                                    safe_page = escape_html(str(src['page'])) if src.get('page') else ""
                                    safe_content = escape_html(src['content'][:150])
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>{safe_source}</strong>
                                        {f" (Page {safe_page})" if safe_page else ""}
                                        <br><small>Relevance: {src['relevance']:.1%}</small>
                                        <br><small style="color: #666;">{safe_content}...</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Add to history with sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": sources_data,
                            "confidence": response.confidence
                        })

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        if "openai" in str(e).lower():
                            st.warning("Please check if your OpenAI API key is properly set in the .env file")

with col2:
    st.markdown("### Response Metrics")

    # Show metrics from last response
    if st.session_state.messages:
        last_assistant = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_assistant = msg
                break

        if last_assistant:
            confidence = last_assistant.get("confidence", 0)
            sources = last_assistant.get("sources", [])

            # Confidence indicator
            if confidence >= 0.7:
                conf_class = "confidence-high"
                conf_label = "High"
            elif confidence >= 0.4:
                conf_class = "confidence-medium"
                conf_label = "Medium"
            else:
                conf_class = "confidence-low"
                conf_label = "Low"

            st.markdown(f"""
            <div class="metric-box">
                <h3>Confidence</h3>
                <h1 class="{conf_class}">{confidence:.0%}</h1>
                <p>{conf_label} Confidence</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Source stats
            st.metric("Sources Retrieved", len(sources))

            if sources:
                avg_relevance = sum(s['relevance'] for s in sources) / len(sources)
                st.metric("Avg. Relevance", f"{avg_relevance:.1%}")

                # Top source
                if sources:
                    top_source = max(sources, key=lambda x: x['relevance'])
                    st.markdown("**Top Source:**")
                    st.info(f"📄 {top_source['source']}")

    else:
        st.info("Ask a question to see response metrics")

    # Quick actions
    st.markdown("### Quick Actions")

    example_questions = [
        "What topics are covered in this course?",
        "What are the grading policies?",
        "When are the assignments due?",
        "What are the prerequisites?",
    ]

    for q in example_questions:
        if st.button(f"💬 {q[:40]}...", key=f"example_{hash(q)}"):
            # Simulate asking the question
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
