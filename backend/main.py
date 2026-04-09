import os
import time
import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, BertModel
from bert_score import score

from model import SpectralKrylovTransformerBlock
from utils import clean_document, split_into_sentences, extract_text_from_pdf

# ✅ App init

app = FastAPI(title="KryloBERT API 🚀")

# ✅ CORS (allow frontend)

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# ✅ Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load models ONCE (important)

try:
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

```
krylov_model = SpectralKrylovTransformerBlock(
    vocab_size=tokenizer.vocab_size,
    d_model=128
).to(device)
krylov_model.eval()

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

print("✅ Models loaded successfully")
```

except Exception as e:
print("❌ Model loading failed:", e)
tokenizer = None
krylov_model = None
bert_model = None

# ✅ Request schemas

class TextRequest(BaseModel):
text: str

class CompareRequest(BaseModel):
cand: str
ref: str

# ✅ Health check

@app.get("/")
def health():
return {"status": "Backend running 🚀"}

# =========================

# 🔹 CORE FUNCTIONS

# =========================

def get_krylov_summary(text):
if not tokenizer or not krylov_model:
return "Model not loaded", 0

```
cleaned = clean_document(text)
sentences = split_into_sentences(cleaned)

if not sentences:
    return "No valid sentences found.", 0

start_time = time.time()
embs = []

try:
    with torch.no_grad():
        for s in sentences:
            enc = tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=32
            )
            out = krylov_model(enc["input_ids"].to(device))
            embs.append(out.mean(dim=1).squeeze(0).cpu())

    if not embs:
        return "Embedding failed", 0

    embs = torch.stack(embs)
    doc_emb = embs.mean(dim=0, keepdim=True)
    scores = F.cosine_similarity(embs, doc_emb)

    top_k = min(3, len(sentences))
    idx = torch.topk(scores, top_k).indices.tolist()

    summary = " ".join([sentences[i] for i in sorted(idx)])
    return summary, round(time.time() - start_time, 3)

except Exception as e:
    return f"Error: {str(e)}", 0
```

def get_bert_summary(text):
if not tokenizer or not bert_model:
return "Model not loaded", 0

```
cleaned = clean_document(text)
sentences = split_into_sentences(cleaned)

if not sentences:
    return "No valid sentences found.", 0

start_time = time.time()
embs = []

try:
    with torch.no_grad():
        for s in sentences:
            enc = tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )

            out = bert_model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            )

            embs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu())

    embs = torch.stack(embs)
    doc_emb = embs.mean(dim=0, keepdim=True)
    scores = F.cosine_similarity(embs, doc_emb)

    top_k = min(3, len(sentences))
    idx = torch.topk(scores, top_k).indices.tolist()

    summary = " ".join([sentences[i] for i in sorted(idx)])
    return summary, round(time.time() - start_time, 3)

except Exception as e:
    return f"Error: {str(e)}", 0
```

# =========================

# 🔹 API ROUTES

# =========================

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
if not file.filename.endswith(".pdf"):
raise HTTPException(status_code=400, detail="Only PDF files allowed")

```
try:
    content = await file.read()
    text = extract_text_from_pdf(io.BytesIO(content))
    return {"text": text}

except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.post("/bert")
async def bert_api(req: TextRequest):
summary, duration = get_bert_summary(req.text)
return {"summary": summary, "time": duration}

@app.post("/krylov")
async def krylov_api(req: TextRequest):
summary, duration = get_krylov_summary(req.text)
return {"summary": summary, "time": duration}

@app.post("/compare")
async def compare_api(req: CompareRequest):
try:
P, R, F1 = score([req.cand], [req.ref], lang="en", verbose=False)
return {
"precision": float(P[0]),
"recall": float(R[0]),
"f1": float(F1[0])
}
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))

# =========================

# 🔹 LOCAL RUN

# =========================

if **name** == "**main**":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
