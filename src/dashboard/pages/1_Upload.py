import os
import re
import streamlit as st
from datetime import datetime
import PyPDF2
from io import BytesIO

# Set page config
st.set_page_config(page_title="Upload", page_icon="📤", layout="wide", initial_sidebar_state="expanded")

# Initialize directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Security constants
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.csv', '.md'}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other security issues."""
    # Get only the basename (remove any directory components)
    filename = os.path.basename(filename)
    # Remove any null bytes
    filename = filename.replace('\x00', '')
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename

def validate_file_path(filepath: str) -> bool:
    """Validate that the file path is within DATA_DIR to prevent path traversal."""
    # Resolve to absolute path
    abs_filepath = os.path.abspath(filepath)
    abs_data_dir = os.path.abspath(DATA_DIR)
    # Check if the file path starts with the data directory
    return abs_filepath.startswith(abs_data_dir + os.sep) or abs_filepath == abs_data_dir

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

        # Sanitize filename and save as txt file
        safe_name = sanitize_filename(uploaded_file.name)
        txt_filename = os.path.splitext(safe_name)[0] + ".txt"
        txt_path = os.path.join(DATA_DIR, txt_filename)

        # Validate path is within DATA_DIR
        if not validate_file_path(txt_path):
            st.error("Invalid file path detected.")
            return False

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def save_file(uploaded_file):
    """Save an uploaded file with security validations."""
    if uploaded_file is not None:
        try:
            # Sanitize filename
            safe_name = sanitize_filename(uploaded_file.name)
            if not safe_name:
                st.error("Invalid filename.")
                return

            # Validate file extension
            _, ext = os.path.splitext(safe_name)
            if ext.lower() not in ALLOWED_EXTENSIONS:
                st.error(f"File type '{ext}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
                return

            # Validate file size
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
                return

            # Construct and validate file path
            file_path = os.path.join(DATA_DIR, safe_name)
            if not validate_file_path(file_path):
                st.error("Invalid file path detected.")
                return

            # Handle PDFs
            if uploaded_file.type == "application/pdf":
                if process_pdf(uploaded_file):
                    st.success(f"PDF {safe_name} processed and saved as text!")
                else:
                    return

            # Save original file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {safe_name} saved successfully!")

            # Update files list and clear cache
            list_files.clear()
            st.session_state["files"] = list_files(DATA_DIR)
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")

def delete_file(filename):
    """Delete a file with security validations."""
    # Sanitize filename
    safe_name = sanitize_filename(filename)
    if not safe_name:
        st.error("Invalid filename.")
        return

    file_path = os.path.join(DATA_DIR, safe_name)

    # Validate path is within DATA_DIR
    if not validate_file_path(file_path):
        st.error("Invalid file path detected.")
        return

    try:
        if os.path.exists(file_path):
            os.remove(file_path)

        # Also remove corresponding txt file if it exists
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        if validate_file_path(txt_path) and os.path.exists(txt_path):
            os.remove(txt_path)

        st.success(f"File {safe_name} deleted successfully!")
        # Update files list and clear cache
        list_files.clear()
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
    - Markdown files (.md)
    - CSV files (for structured data)
    """
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose files to upload",
    type=["txt", "pdf", "csv", "md"],
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
