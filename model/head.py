import torch
import torch.nn as nn


class MultiTokenLMHead(nn.Module):
    def __init__(self, d_model: int, d_embed: int, n_pred_tokens: int = 4):
        super().__init__()
        self.n_pred_tokens = n_pred_tokens

        if n_pred_tokens == 1:
            self.proj = nn.Linear(d_model, d_embed, bias=False)
        else:
            self.shared = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.GELU(),
            )
            self.proj = nn.Linear(d_model, n_pred_tokens * d_embed, bias=False)

    def forward(
        self, h: torch.Tensor, embed_weight: torch.Tensor
    ) -> torch.Tensor:
        if self.n_pred_tokens == 1:
            x = self.proj(h)
            logits = torch.matmul(x.float(), embed_weight.T.float())
            return logits.to(h.dtype)

        x = self.shared(h)
        x = self.proj(x)
        x = x.view(*x.shape[:-1], self.n_pred_tokens, self.d_embed)
        logits = torch.matmul(x.float(), embed_weight.T.float())
        return logits.to(h.dtype)
