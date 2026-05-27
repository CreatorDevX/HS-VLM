import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import RMSNorm


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gelu(self.w1(x)))


class VisualPrefixEncoder(nn.Module):
    def __init__(self, d_model: int, d_clip: int, n_queries: int = 16):
        super().__init__()
        self.n_queries = n_queries

        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.clip_proj = nn.Linear(d_clip, d_model, bias=False)

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.cross_norm = RMSNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.self_norm = RMSNorm(d_model)

        self.ffn = FeedForward(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        B = clip_embed.shape[0]
        q = self.query_tokens.expand(B, -1, -1)
        k = self.clip_proj(clip_embed).unsqueeze(1)

        q = q + self.cross_norm(self.cross_attn(q, k, k)[0])
        q = q + self.self_norm(self.self_attn(q, q, q)[0])
        q = q + self.ffn_norm(self.ffn(q))

        return q
