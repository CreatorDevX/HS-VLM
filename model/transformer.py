import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.moe import MoELayer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps
        )
        x_normed = x.float() / rms
        return (x_normed * self.weight).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(2)
        if T > self.cos.size(0):
            self._set_cos_sin_cache(T)
        cos = self.cos[:T].view(1, 1, T, self.d_head)
        sin = self.sin[:T].view(1, 1, T, self.d_head)
        q_embed = (q.float() * cos) + (rotate_half(q).float() * sin)
        k_embed = (k.float() * cos) + (rotate_half(k).float() * sin)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv, seqlen, d_head = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv, n_rep, seqlen, d_head)
        .reshape(bs, n_kv * n_rep, seqlen, d_head)
    )


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.rope = RoPE(d_head, max_seq_len, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff_expert: int,
        n_experts: int,
        top_k: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        capacity_factor: float = 1.25,
        z_loss_coeff: float = 1e-4,
        load_balance_coeff: float = 1e-2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.attention = Attention(
            d_model, n_heads, n_kv_heads, d_head,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff_expert,
            n_experts=n_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            z_loss_coeff=z_loss_coeff,
            load_balance_coeff=load_balance_coeff,
        )
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.norm1(x))
        moe_out, aux_loss, expert_util, router_logits = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss, expert_util, router_logits
