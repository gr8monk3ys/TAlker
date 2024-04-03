import yaml, os, sys
from llm import *
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
from textblob import TextBlob
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

file_directory = os.path.abspath('../piazza_bot/')
if file_directory not in sys.path:
    sys.path.append(file_directory)

from profile import *

posts_df = pd.read_csv('../data/posts.csv')

st.set_page_config(
    page_title="TAlker Dashboard",
    page_icon="🦙",
    layout="wide",
)

def list_files(directory):
    """Lists files in the specified directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def add_file_to_directory(uploaded_file, directory='../data/'):
    """Saves the uploaded file to the specified directory."""
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def delete_file_from_directory(filename, directory='../data/'):
    """Deletes the specified file from the directory and updates the session state."""
    try:
        os.remove(os.path.join(directory, filename))
        return True
    except FileNotFoundError:
        return False

def initialize_files_list(directory='../data/'):
    """Initializes the files list in session state with files from the directory."""
    if 'files' not in st.session_state:
        st.session_state['files'] = list_files(directory)

def display_uploaded_files(directory='../data/'):
    """Displays the list of uploaded files with an option to delete."""
    if 'files' in st.session_state:
        st.write("Uploaded Files:")
        to_delete = []
        for filename in st.session_state['files']:
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.write(filename)
            with col2:
                if st.button("X", key=filename):
                    # If deletion is successful, mark this filename for removal from session state
                    if delete_file_from_directory(filename, directory):
                        to_delete.append(filename)
                    else:
                        st.error(f"Failed to delete {filename}")
        # Update session state by removing deleted files
        for filename in to_delete:
            st.session_state['files'].remove(filename)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def get_new_posts():
    profile = Profile()
    st.sidebar.button('Get New Posts', on_click=profile.main())

@st.cache_data
def load_posts():
    st.table(posts_df)

def prof_ui(authenticator):
    llm_instance = Llm_chain()

    with st.sidebar:
        st.title("TAlker")
        if 'OPENAI_API_KEY' in st.secrets:
            st.success('API key already provided!', icon='✅')
            openai_api = st.secrets['OPENAI_API_KEY']
        else:
            openai_api = st.text_input('Enter OpenAI API token:', type='password')
            if not (openai_api.startswith('sk-') and len(openai_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['OPENAI_API_KEY'] = openai_api
        
        choice = st.radio("Navigation", ["Upload", "Data Source", "Analysis"])
        st.write("This dashboard is multi-use for simplifying everyday information of the TA bot knowledge base")
        clear_chat_history()
        authenticator.logout('Logout', 'main')
    
    if choice == "Upload":
        col1, col2 = st.columns(2)
        with col1:
            initialize_files_list()
            uploaded_file = st.file_uploader("Drag or Upload PDFs or Slides here", type=['pdf', 'png', 'jpg', 'jpeg', 'mp4', 'mp3'])
            submit_button = st.button('Submit')

            if submit_button:
                if uploaded_file is not None:
                    # Save the uploaded file to the ../data/ directory and update the session state
                    add_file_to_directory(uploaded_file)
                    st.session_state['files'].append(uploaded_file.name)
                    st.success("File submitted!")
                else:
                    st.error("Please upload a file before submitting.")

            display_uploaded_files()

        with col2:
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"role": "TA assistant", "content": "How may I assist you today?"}]

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if prompt := st.chat_input(disabled=not openai_api):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                # Generate a new response if last message is not from assistant
            if st.session_state.messages[-1]["role"] != "Teacher assistant":
                with st.chat_message("Teacher assistant"):
                    with st.spinner("Thinking..."):
                        response = llm_instance.generate_response(str(prompt), 'DSCI 560')
                        placeholder = st.empty()
                        full_response = ''
                        for item in response:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)
                        message = {"role": "Teacher assistant", "content": full_response}
                st.session_state.messages.append(message)

    if choice == "Data Source":
        st.title("The data that already existed in database")
        load_posts()
    if choice == "Analysis":
        col1, col2 = st.columns(2)
        with col1:
            posts_df['date'] = pd.to_datetime(posts_df['timestamp'])  # Assuming 'created' is a column with timestamps
            posts_over_time = posts_df.resample('W', on='date').size()  # Weekly aggregation
            fig = px.line(posts_over_time, title='Posts Over Time', color_discrete_sequence=['indianred'])
            st.plotly_chart(fig)

            user_posts_counts = posts_df['username'].value_counts()  # Assuming 'username' denotes the post author
            fig = px.histogram(x=user_posts_counts.values, nbins=20, title='User Participation', color_discrete_sequence=['indianred'])
            st.plotly_chart(fig)

        with col2:
            posts_df['post_length'] = posts_df['content'].apply(len)
            fig = px.histogram(posts_df, x='post_length', title='Distribution of Post Lengths',
                            labels={'post_length': 'Post Length (characters)'},
                            nbins=50,  # Adjust the number of bins for better granularity
                            color_discrete_sequence=['indianred'])  # Color can be adjusted
            st.plotly_chart(fig)

            posts_df['sentiment'] = posts_df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
            fig = px.histogram(posts_df, x='sentiment', nbins=20, title='Sentiment Distribution of Posts', color_discrete_sequence=['indianred'])
            st.plotly_chart(fig)
            
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login(fields={'Login':'Login', 'Form name':'Login'})
    if authentication_status:
        if username == 'gr8monk3ys':
            st.write(f'Welcome *{name}*')
            prof_ui(authenticator)
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
