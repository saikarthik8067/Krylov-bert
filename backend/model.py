import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_pos=512):
        super().__init__()
        self.word = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_pos, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        return self.norm(self.word(input_ids) + self.pos(pos_ids))

class QKVProjector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)

class BlockLanczos(nn.Module):
    def __init__(self, num_blocks=8, block_size=4):
        super().__init__()
        self.k = num_blocks * block_size

    def forward(self, X):
        L, _ = X.shape
        k = min(self.k, L)
        basis = torch.randn(L, k, device=X.device)
        basis, _ = torch.linalg.qr(basis)
        T = torch.randn(k, k, device=X.device)
        return basis, T, {}

class KrylovAttention(nn.Module):
    def forward(self, Qb, Kb, Vb, TQ=None, TK=None, TV=None):
        k = Qb.shape[1]
        Z0 = Qb.T @ Vb
        A = F.softmax(Z0 / math.sqrt(k), dim=-1)
        out = Qb @ A
        return out, None

class KrylovOutputProjector(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.proj = nn.Linear(k, d_model)

    def forward(self, x):
        return self.proj(x)

class SpectralKrylovTransformerBlock(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.qkv = QKVProjector(d_model)
        self.lanczos = BlockLanczos()
        self.krylov = KrylovAttention()
        self.out_proj = KrylovOutputProjector(32, d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        Q, K, V = self.qkv(x)
        Qb, Kb, Vb = Q[0], K[0], V[0]
        BQ, TQ, _ = self.lanczos(Qb)
        Zk, _ = self.krylov(BQ, BQ, BQ, TQ, TQ, TQ)
        Z = self.out_proj(Zk)
        return Z.unsqueeze(0)
