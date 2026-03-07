import re
import PyPDF2
from contextlib import nullcontext
import io

def clean_document(text):
    """
    Cleans document text by removing extra whitespaces, 
    Jupyter Notebook artifacts, and common noise.
    """
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        # Filtering code-like artifacts if any
        if (line.startswith("{") or line.startswith("}") or 
            "nbformat" in line or "cell_type" in line):
            continue
        clean_lines.append(line)
    return " ".join(clean_lines)

def split_into_sentences(text):
    """
    Splits text into valid sentences based on regex.
    """
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter very short or fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    return sentences

def extract_text_from_pdf(file_input):
    """
    Extracts text from a binary file-like object or PDF path.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_input)
        for p in reader.pages:
            extr = p.extract_text()
            if extr:
                text += extr
    except Exception as e:
        print(f"Extraction error: {e}")
    return text
