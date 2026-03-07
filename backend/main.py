import os
import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, BertModel
import io
import logging
from bert_score import score

from model import SpectralKrylovTransformerBlock
from utils import clean_document, split_into_sentences, extract_text_from_pdf

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Krylov-BERT Full-Stack API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on Device: {device}")

# Model Loading Registry (Cached at Start)
tokenizer = None
krylov_model = None
bert_model = None

@app.on_event("startup")
async def load_models():
    """Initializes models and tokenizers globally at start."""
    global tokenizer, krylov_model, bert_model
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        krylov_model = SpectralKrylovTransformerBlock(vocab_size=tokenizer.vocab_size, d_model=128).to(device)
        krylov_model.eval()

        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()
        logger.info("All heavy models loaded onto device successfully.")
    except Exception as e:
        logger.error(f"Critical error loading models: {str(e)}")

class TextSubmit(BaseModel):
    text: str

class PairSubmit(BaseModel):
    cand: str
    ref: str

def compute_summary(text, mode="krylov"):
    """Generic summarization logic using model choice."""
    cleaned = clean_document(text)
    sentences = split_into_sentences(cleaned)
    if not sentences: return "No valid content.", 0
    
    start = time.time()
    embs = []
    active_m = krylov_model if mode == "krylov" else bert_model
    
    with torch.no_grad():
        for s in sentences:
            inputs = tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
            if mode == "krylov":
                out = krylov_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                embs.append(out[:, 0, :].squeeze(0).cpu())
            else:
                out = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                embs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu())
    
    if not embs: return "Failed to process text.", 0
    
    embs = torch.stack(embs)
    scores = torch.norm(embs, dim=1)
    
    # Pick top 3 sentences for summary
    k = min(3, len(sentences))
    idx = torch.topk(scores, k).indices.tolist()
    summary = " ".join([sentences[i] for i in sorted(idx)])
    duration = time.time() - start
    return summary, duration

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/extract")
async def extract_api(file: UploadFile = File(...)):
    """Receives PDF/TXT and extracts cleansed text."""
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Use PDF or TXT only.")
    try:
        content = await file.read()
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(io.BytesIO(content))
        else:
            text = content.decode("utf-8", errors="ignore")
        return {"text": text, "count": len(text)}
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/krylov")
async def run_krylov(req: TextSubmit):
    """Processes text with Spectral Krylov Block Attention."""
    summ, dt = compute_summary(req.text, mode="krylov")
    return {"summary": summ, "time": dt}

@app.post("/bert")
async def run_bert(req: TextSubmit):
    """Benchmarks against standard BERT baseline."""
    summ, dt = compute_summary(req.text, mode="baseline")
    return {"summary": summ, "time": dt}

@app.post("/compare")
async def compare_api(req: PairSubmit):
    """Computes BERTScore similarity metric."""
    try:
        P, R, F1 = score([req.cand], [req.ref], lang="en", verbose=False)
        return {"precision": float(P[0]), "recall": float(R[0]), "f1": float(F1[0])}
    except Exception as e:
        logger.error(f"Metric failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
