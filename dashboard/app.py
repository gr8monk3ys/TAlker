import pandas as pd
import streamlit as st
from pathlib import Path
# from llm import *

def check_and_open_file(file_path):
    # Convert to a Path object for easier handling
    file_path = Path(file_path)
    
    # Check if the file exists
    if not file_path.is_file():
        print(f"File does not exist: {file_path}")
        return None
    
    # Attempt to open the file, handling possible permission errors
    try:
        return open(file_path, "rb")
    except PermissionError:
        print(f"Permission denied for file: {file_path}")
    except Exception as e:
        print(f"An error occurred while opening file {file_path}: {e}")

def calculate_column_widths(dataframe):
            # Calculate the maximum width for each column (based on character count)
            widths = []
            for column in dataframe.columns:
                max_len = max(
                    dataframe[column].astype(str).apply(len).max(),  # max length in column
                    len(str(column)),
                )  # length of column name\\header
                widths.append(max_len)
            return widths

def load_css(file_name):
    with open(file_name, "r") as f:
        return f.read()

def local_css(file_name):
    st.markdown(f'<style>{load_css(file_name)}</style>', unsafe_allow_html=True)

def add_file_to_list(uploaded_file):
    # Create a files list in session state if it doesn't exist
    if 'files' not in st.session_state:
        st.session_state['files'] = []

    # Add file information to the list
    st.session_state['files'].append({
        "filename": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size
    })
    
if __name__ == '__main__':
    local_css('style.css')
    with st.sidebar:
        st.title("TAlker")
        choice = st.radio("Navigation", ["Upload", "Data Source", "Analysis"])
        st.info("This dashboard is multi-use for simplifying everyday information of the TA bot knowledge base")
    
    if choice == "Upload":
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Drag or Upload PDFs or Slides here", type=['pdf', 'png', 'jpg', 'jpeg', 'mp4', 'mp3'])
            submit_button = st.button('Submit')

            # Handle file submission
            if submit_button:
                if uploaded_file is not None:
                    add_file_to_list(uploaded_file)
                    st.success("File submitted!")
                else:
                    st.error("Please upload a file before submitting.")

            # Display the list of uploaded files
            if 'files' in st.session_state:
                st.write("Uploaded Files:")
                for file_info in st.session_state['files']:
                    st.write(f"{file_info['filename']}")

        with col2:
            # Optionally, display a text area for additional interactions
            message = st.text_area("Test message")
            message_submit = st.button('Test')
            if message_submit:
                st.write("Generating best practice message...")
                result = llm_chain.generate_response(message)
                st.info(result)

    if choice == "Data Source":
        st.title("The data that already existed in database")
        data = '../data/customers-100.csv'
        df = pd.read_csv(data)
        st.table(df)

    if choice == "Analysis":
        st.title("Look at figures and statistics")
        st.write("The most frequently asked questions")
        st.write("The most active users:")
        st.write("The questions that can't be answered by the bot:")

