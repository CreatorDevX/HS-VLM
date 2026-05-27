import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        d_clip: int = 512,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_clip)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_clip, d_clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)
