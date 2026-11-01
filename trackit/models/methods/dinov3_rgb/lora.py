import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankProjection(nn.Module):
    def __init__(self, dim, rank, alpha, dropout):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Expected rank >= 1, alpha > 0 and 0 <= dropout < 1')
        self.A = nn.Parameter(torch.empty(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x):
        return F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scaling


class QVAdapter(nn.Module):
    def __init__(self, dim, rank, alpha, dropout):
        super().__init__()
        self.q = LowRankProjection(dim, rank, alpha, dropout)
        self.v = LowRankProjection(dim, rank, alpha, dropout)

    def forward(self, x):
        q, v = self.q(x), self.v(x)
        return torch.cat((q, torch.zeros_like(q), v), dim=-1)


class TrackingBlock(nn.Module):
    """Frozen official block plus explicit, stateless Q/V adapter routing.

    The official block's QKV projection (including its masked K bias), RoPE,
    attention output projection, LayerScale, MLP and norms are retained.
    """
    def __init__(self, base, adapter_name, rank, alpha, dropout):
        super().__init__()
        if base.sample_drop_ratio != 0:
            raise ValueError('TrackingBlock expects the official zero-drop-path backbone')
        self.base = base.requires_grad_(False)
        dim = base.attn.qkv.in_features
        self.adapters = nn.ModuleDict({adapter_name: QVAdapter(dim, rank, alpha, dropout)})

    def forward(self, x, rope, adapter_name):
        base = self.base
        normalized = base.norm1(x)
        qkv = base.attn.qkv(normalized) + self.adapters[adapter_name](normalized)
        attention = base.attn.compute_attention(qkv=qkv, rope=rope)
        attention = base.attn.proj_drop(base.attn.proj(attention))
        x = x + base.ls1(attention)
        return x + base.ls2(base.mlp(base.norm2(x)))
