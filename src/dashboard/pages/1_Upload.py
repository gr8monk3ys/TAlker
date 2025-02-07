import os
import streamlit as st
from datetime import datetime
import PyPDF2
from io import BytesIO

# Set page config
st.set_page_config(page_title="Upload", page_icon="📤", layout="wide", initial_sidebar_state="expanded")

# Initialize directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def initialize_files_list():
    """Initialize the list of files in session state."""
    if "files" not in st.session_state:
        st.session_state["files"] = list_files(DATA_DIR)

@st.cache_data
def list_files(directory):
    """List all files in the directory with their details."""
    files = []
    for f in os.listdir(directory):
        file_path = os.path.join(directory, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            files.append({
                "name": f,
                "size": f"{size / 1024:.2f} KB",
                "modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
                "type": os.path.splitext(f)[1].lower()
            })
    return files

def process_pdf(uploaded_file):
    """Extract text from PDF and save as TXT."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Save as txt file
        txt_filename = os.path.splitext(uploaded_file.name)[0] + ".txt"
        txt_path = os.path.join(DATA_DIR, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def save_file(uploaded_file):
    """Save an uploaded file."""
    if uploaded_file is not None:
        try:
            # Handle PDFs
            if uploaded_file.type == "application/pdf":
                if process_pdf(uploaded_file):
                    st.success(f"PDF {uploaded_file.name} processed and saved as text!")
                else:
                    return
            
            # Save original file
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} saved successfully!")
            
            # Update files list
            st.session_state["files"] = list_files(DATA_DIR)
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")

def delete_file(filename):
    """Delete a file."""
    file_path = os.path.join(DATA_DIR, filename)
    try:
        os.remove(file_path)
        # Also remove corresponding txt file if it exists
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(txt_path):
            os.remove(txt_path)
        st.success(f"File {filename} deleted successfully!")
        # Update files list
        st.session_state["files"] = list_files(DATA_DIR)
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")

# Initialize files list
initialize_files_list()

# Main UI
st.title("📤 Upload Course Materials")
st.write(
    """
    Upload your course materials here to provide context for the bot's responses.
    Supported file types:
    - PDF files (will be converted to text)
    - Text files (.txt)
    - CSV files (for structured data)
    """
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose files to upload",
    type=["txt", "pdf", "csv"],
    help="Upload your course materials here. PDFs will be automatically converted to text for better processing.",
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.write("### File Details")
    file_details = {
        "Filename": uploaded_file.name,
        "File type": uploaded_file.type,
        "File size": f"{uploaded_file.size / 1024:.2f} KB"
    }
    for key, value in file_details.items():
        st.write(f"- {key}: {value}")
    
    if st.button("Save File", type="primary"):
        save_file(uploaded_file)

# Display existing files
if st.session_state["files"]:
    st.write("### Uploaded Files")
    
    # Create a table of files
    files_data = []
    for file in st.session_state["files"]:
        col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
        with col1:
            st.write(f"📄 {file['name']}")
        with col2:
            st.write(file['size'])
        with col3:
            st.write(file['modified'])
        with col4:
            if st.button("🗑️ Delete", key=file['name']):
                delete_file(file['name'])

    # Show total context size
    total_size = sum([float(f['size'].replace(' KB', '')) for f in st.session_state["files"]])
    st.info(f"📚 Total context size: {total_size:.2f} KB")
else:
    st.warning("No files uploaded yet. Upload some course materials to get started!")
