import streamlit as st
from src.dashboard.llm import LlmChain
import os

st.set_page_config(
    page_title="Test Bot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM chain
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = LlmChain()

st.title("🧪 Test Bot")
st.markdown("""
    Test the bot's responses in a chat-like interface. The bot uses the context from your uploaded files
    to provide accurate and relevant answers about your course.
""")

# Sidebar with context information
with st.sidebar:
    st.markdown("### Context Status")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    if files:
        st.success("✅ Files loaded:")
        for file in files:
            st.write(f"📄 {file}")
    else:
        st.warning("""
            ⚠️ No files uploaded yet!
            
            Please go to the Upload page and upload your course materials first:
            - Course syllabus
            - Assignment descriptions
            - Lecture slides
            - Any other relevant course documents
            
            This helps me provide accurate answers about your course.
        """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat container for better styling
chat_container = st.container()
with chat_container:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything about your course..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_chain.get_response(prompt)
                    
                    # If no context is available, provide a more helpful message
                    if "Please upload some course materials" in response:
                        st.warning("""
                            I don't have access to your course materials yet. To get accurate answers:
                            1. Go to the Upload page (📤)
                            2. Upload your course documents (syllabus, assignments, etc.)
                            3. Come back here and ask your question again
                        """)
                    else:
                        st.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if "openai" in str(e).lower():
                        st.warning("Please check if your OpenAI API key is properly set in the .env file")
