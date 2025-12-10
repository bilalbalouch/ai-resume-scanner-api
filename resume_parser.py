import pdfplumber
import docx2txt
import re

def extract_text(file_path):
    """Extract text from PDF or DOCX resume"""
    text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file_path.lower().endswith(".docx"):
        text = docx2txt.process(file_path)
    else:
        return ""
    return text.strip()


def extract_email(text):
    """Extract email address from text"""
    pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_phone(text):
    """Extract phone number from text"""
    pattern = r"(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}"
    match = re.search(pattern, text)
    return match.group(0) if match else ""
