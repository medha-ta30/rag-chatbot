import os
from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF file.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        list[dict]: A list of dictionaries representing pages:
            [{"filename": "...", "page_number": 1, "text": "..."}]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no pages or is empty.
        Exception: For general pypdf reading errors (e.g., corrupted file).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    filename = os.path.basename(file_path)
    pages_data = []

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise Exception(f"Failed to read PDF file '{filename}': {str(e)}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"The PDF file '{filename}' contains no pages.")

    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        try:
            text = page.extract_text()
        except Exception as e:
            text = ""
        
        # Ensure we default to an empty string if pypdf returns None or empty
        if text is None:
            text = ""
            
        pages_data.append({
            "filename": filename,
            "page_number": page_number,
            "text": text.strip()
        })

    return pages_data
