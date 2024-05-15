import fitz
import logging

logging.basicConfig(filename='logs/processing.log', level=logging.INFO)

def extract_text_from_pages(file_path, num_pages):
    """Extracts text from the first 'num_pages' of a given PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            pages_to_read = min(num_pages, len(doc))  # Ensure we don't exceed the actual number of pages
            for page in doc[:pages_to_read]:
                text += page.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
    return text
