import os
import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, BertModel
import io
from bert_score import score

from model import SpectralKrylovTransformerBlock
from utils import clean_document, split_into_sentences, extract_text_from_pdf

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
krylov_model = SpectralKrylovTransformerBlock(vocab_size=tokenizer.vocab_size, d_model=128).to(device)
krylov_model.eval()

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    cand: str
    ref: str

def get_krylov_summary(text):
    cleaned = clean_document(text)
    sentences = split_into_sentences(cleaned)
    if not sentences: return "No valid sentences found.", 0
    
    start_time = time.time()
    embs = []
    with torch.no_grad():
        for s in sentences:
            enc = tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            out = krylov_model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            embs.append(out[:, 0, :].squeeze(0).cpu())
    
    if not embs: return "Could not generate embeddings.", 0
    
    embs = torch.stack(embs)
    scores = torch.norm(embs, dim=1)
    
    top_k = min(3, len(sentences))
    idx = torch.topk(scores, top_k).indices.tolist()
    summary = " ".join([sentences[i] for i in sorted(idx)])
    duration = time.time() - start_time
    return summary, duration

def get_bert_summary(text):
    cleaned = clean_document(text)
    sentences = split_into_sentences(cleaned)
    if not sentences: return "No valid sentences found.", 0
    
    start_time = time.time()
    embs = []
    with torch.no_grad():
        for s in sentences:
            enc = tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            out = bert_model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            embs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu())
    
    if not embs: return "Could not generate embeddings.", 0
    
    embs = torch.stack(embs)
    scores = torch.norm(embs, dim=1)
    
    top_k = min(3, len(sentences))
    idx = torch.topk(scores, top_k).indices.tolist()
    summary = " ".join([sentences[i] for i in sorted(idx)])
    duration = time.time() - start_time
    return summary, duration

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        content = await file.read()
        pdf_file = io.BytesIO(content)
        text = extract_text_from_pdf(pdf_file)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bert")
async def bert_api(req: TextRequest):
    try:
        summary, duration = get_bert_summary(req.text)
        return {"summary": summary, "time": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/krylov")
async def krylov_api(req: TextRequest):
    try:
        summary, duration = get_krylov_summary(req.text)
        return {"summary": summary, "time": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_api(req: CompareRequest):
    try:
        P, R, F10 = score([req.cand], [req.ref], lang="en", verbose=False)
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F10[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
