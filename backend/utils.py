import re
import PyPDF2
from contextlib import nullcontext

def clean_document(text):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if (line.startswith("{") or line.startswith("}") or line.startswith("\"") or 
            "nbformat" in line or "cell_type" in line or "execution_count" in line):
            continue
        clean_lines.append(line)
    return " ".join(clean_lines)

def split_into_sentences(text):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def extract_text_from_pdf(file_input):
    text = ""
    if isinstance(file_input, str):
        context_manager = open(file_input, "rb")
    else:
        context_manager = nullcontext(file_input)

    with context_manager as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            if p.extract_text():
                text += p.extract_text()
    return text
