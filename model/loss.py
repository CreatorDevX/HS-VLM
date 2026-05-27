import torch
import torch.nn.functional as F


def multi_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    if logits.dim() == 3:
        B, T, vocab = logits.shape
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            targets[:, 1:].reshape(-1),
        )

    B, T, n_pred, vocab = logits.shape
    loss = 0.0
    for k in range(n_pred):
        shift_logits = logits[:, : T - k - 1, k]
        shift_targets = targets[:, k + 1 :]
        loss = loss + weights[k] * F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_targets.reshape(-1),
        )
    return loss
