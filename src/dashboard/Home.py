"""
TAlker - Production-Grade RAG Teaching Assistant
Main application entry point.
"""

import yaml
import os
import streamlit as st
import streamlit_authenticator as stauth
from src.piazza_bot.profile import Profile

st.set_page_config(
    page_title="TAlker - RAG Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "posts_df" not in st.session_state:
    st.session_state["posts_df"] = None


def get_new_posts():
    """Fetch new posts from Piazza."""
    try:
        profile = Profile()
        posts = profile.get_posts(time_limit=300)
        st.session_state["posts_df"] = posts
        return posts
    except Exception as e:
        st.warning(f"Could not fetch posts: {str(e)}")
        return None


@st.cache_data
def load_posts():
    """Load and display posts in the session state."""
    if st.session_state["posts_df"] is not None:
        st.table(st.session_state["posts_df"])


def main(authenticator, name):
    """Main application logic."""
    with st.sidebar:
        st.write(f"Welcome, *{name}*")
        authenticator.logout("Logout")

    st.title("🎓 TAlker - RAG Teaching Assistant")
    st.markdown("""
    A production-grade RAG system for automating teaching assistant tasks.
    Upload course materials and get AI-powered answers with source citations.
    """)

    # Feature cards
    st.markdown("### Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 📤 Upload
        Import course materials:
        - PDF documents
        - Text files
        - Markdown files
        - CSV data
        """)

    with col2:
        st.markdown("""
        #### 🧪 Test
        Chat with your RAG system:
        - Hybrid search (BM25 + semantic)
        - Cross-encoder reranking
        - Source citations
        - Confidence scores
        """)

    with col3:
        st.markdown("""
        #### 📊 Analysis
        Course analytics:
        - Post activity trends
        - User engagement
        - Content analysis
        - Response patterns
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📈 Evaluation
        RAGAS metrics:
        - Faithfulness
        - Answer relevancy
        - Context precision/recall
        - Automated reports
        """)

    with col2:
        st.markdown("""
        #### ⚙️ Settings
        Configure providers:
        - OpenAI, Anthropic, Google, Cohere
        - Ollama for local/offline models
        - Multiple embedding options
        - RAG parameters
        """)

    # Quick start guide
    st.markdown("---")
    st.markdown("### Quick Start")
    st.markdown("""
    1. **Upload** your course materials (syllabus, lectures, assignments)
    2. **Test** the bot by asking questions about your course
    3. **Evaluate** the quality using RAGAS metrics
    4. **Configure** providers in Settings for your needs
    """)

    # Status indicators
    st.markdown("---")
    st.markdown("### System Status")

    col1, col2, col3 = st.columns(3)

    # Check for data files
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    file_count = 0
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            file_count += len([f for f in files if f.endswith(('.txt', '.pdf', '.csv', '.md'))])

    with col1:
        if file_count > 0:
            st.success(f"📚 {file_count} documents loaded")
        else:
            st.warning("📚 No documents uploaded")

    with col2:
        # Check for Piazza config
        piazza_email = os.getenv("PIAZZA_EMAIL")
        if piazza_email:
            st.success("🔗 Piazza configured")
        else:
            st.info("🔗 Piazza not configured")

    with col3:
        # Check for API keys
        try:
            from src.dashboard.providers import validate_api_keys
            api_status = validate_api_keys()
            available = sum(1 for v in api_status.values() if v)
            st.success(f"🤖 {available} providers available")
        except Exception:
            st.info("🤖 Provider status unknown")


def run():
    """Run the Streamlit application."""
    # Get the absolute path to config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "..", "config.yaml")

    # Check for environment variable to explicitly allow no-auth mode
    allow_no_auth = os.getenv("TALKER_ALLOW_NO_AUTH", "false").lower() == "true"

    if not os.path.exists(config_path):
        if allow_no_auth:
            st.info("No config.yaml found. Running without authentication.")
            main_no_auth()
        else:
            st.error("Authentication configuration not found (config.yaml missing).")
            st.warning("Set TALKER_ALLOW_NO_AUTH=true environment variable to run without authentication.")
            st.stop()
        return

    try:
        with open(config_path, encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)

        # Validate required config sections
        required_sections = ["credentials", "cookie"]
        for section in required_sections:
            if section not in config:
                st.error(f"Invalid config.yaml: missing '{section}' section.")
                st.stop()
                return

        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config.get("preauthorized", {})
        )

        # Place login in sidebar
        with st.sidebar:
            name, authentication_status, username = authenticator.login("Login")

        if authentication_status:
            main(authenticator, name)
        elif authentication_status is False:
            st.error("Username/password is incorrect")
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            show_demo_info()

    except yaml.YAMLError as e:
        st.error(f"Configuration file has invalid YAML syntax: {str(e)}")
        st.stop()
    except KeyError as e:
        st.error(f"Configuration file missing required key: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.stop()


def main_no_auth():
    """Run main app without authentication."""
    st.title("🎓 TAlker - RAG Teaching Assistant")
    st.markdown("""
    A production-grade RAG system for automating teaching assistant tasks.
    Upload course materials and get AI-powered answers with source citations.
    """)

    # Feature cards
    st.markdown("### Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 📤 Upload
        Import course materials (PDF, TXT, MD, CSV)
        """)

    with col2:
        st.markdown("""
        #### 🧪 Test
        Chat with hybrid search and reranking
        """)

    with col3:
        st.markdown("""
        #### ⚙️ Settings
        Configure LLM and embedding providers
        """)

    st.markdown("---")
    st.info("👈 Select a page from the sidebar to get started.")


def show_demo_info():
    """Show demo information for unauthenticated users."""
    st.markdown("---")
    st.markdown("### About TAlker")
    st.markdown("""
    TAlker is a production-grade RAG (Retrieval-Augmented Generation) system
    designed to help teaching assistants automate responses to student questions.

    **Key Features:**
    - Multi-provider LLM support (OpenAI, Anthropic, Google, Cohere, Ollama)
    - Hybrid search with BM25 and semantic retrieval
    - Cross-encoder reranking for better relevance
    - RAGAS evaluation framework
    - Works offline with local models
    """)


if __name__ == "__main__":
    run()
