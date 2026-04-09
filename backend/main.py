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

app = FastAPI(title="KryloBERT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    krylov_model = SpectralKrylovTransformerBlock(
        vocab_size=tokenizer.vocab_size,
        d_model=128
    ).to(device)
    krylov_model.eval()

    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()

except Exception as e:
    print("Model loading failed:", e)
    tokenizer = None
    krylov_model = None
    bert_model = None


class TextRequest(BaseModel):
    text: str


class CompareRequest(BaseModel):
    cand: str
    ref: str


@app.get("/")
def health():
    return {"status": "Backend running"}


def get_bert_summary(text):
    if not tokenizer or not bert_model:
        return "Model not loaded", 0

    sentences = split_into_sentences(clean_document(text))
    if not sentences:
        return "No valid sentences", 0

    start = time.time()
    embs = []

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

    idx = torch.topk(scores, min(3, len(sentences))).indices.tolist()
    summary = " ".join([sentences[i] for i in sorted(idx)])

    return summary, round(time.time() - start, 3)


@app.post("/bert")
async def bert_api(req: TextRequest):
    summary, duration = get_bert_summary(req.text)
    return {"summary": summary, "time": duration}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    content = await file.read()
    text = extract_text_from_pdf(io.BytesIO(content))
    return {"text": text}


@app.post("/compare")
async def compare_api(req: CompareRequest):
    P, R, F1 = score([req.cand], [req.ref], lang="en", verbose=False)
    return {
        "precision": float(P[0]),
        "recall": float(R[0]),
        "f1": float(F1[0])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
