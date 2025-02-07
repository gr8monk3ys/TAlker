import yaml
import os
import streamlit as st
import streamlit_authenticator as stauth
from src.piazza_bot.profile import Profile

st.set_page_config(
    page_title="TAlker Main page",
    page_icon="👋",
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
        st.write(f"Welcome *{name}*")
        authenticator.logout("Logout")
    
    st.title("Piazza TA Bot 👋")
    st.markdown(
        """
        Welcome to the Piazza TA Bot! This application helps teaching assistants
        manage and respond to Piazza posts efficiently.
        ### Features:
        - **Upload**: Import your Piazza data and configure the bot
        - **Test**: Try out the bot's responses before deployment
        - **Analysis**: View insights about student engagement and post patterns
        To get started, select a feature from the sidebar.
        """
    )

    # Features section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📤 Upload")
        st.write("Import your Piazza data and configure the bot settings.")
    with col2:
        st.subheader("🧪 Test")
        st.write("Test the bot's responses in a safe environment.")
    with col3:
        st.subheader("📊 Analysis")
        st.write("Get insights into student engagement and post patterns.")

    # Initialize posts
    get_new_posts()

def run():
    """Run the Streamlit application."""
    try:
        # Get the absolute path to config.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "config.yaml")
        
        if not os.path.exists(config_path):
            st.error("Configuration file not found. Please create a config.yaml file in the project root directory. "
                    "You can copy config.example.yaml and update it with your credentials.")
            return

        with open(config_path, encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)

        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config["preauthorized"]
        )

        # Place login in sidebar
        with st.sidebar:
            name, authentication_status, username = authenticator.login("Login")

        if authentication_status:
            if username == "gr8monk3ys":
                main(authenticator, name)
        elif authentication_status is False:
            st.error("Username/password is incorrect")
        elif authentication_status is None:
            st.warning("Please enter your username and password")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:")
        st.write(f"Current directory: {os.path.abspath(os.path.dirname(__file__))}")
        st.write(f"Config path: {config_path}")

if __name__ == "__main__":
    run()
