import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int,
        capacity_factor: float = 1.25,
        z_loss_coeff: float = 1e-4,
        load_balance_coeff: float = 1e-2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.z_loss_coeff = z_loss_coeff
        self.load_balance_coeff = load_balance_coeff

        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Stacked expert weights: one big batched matmul instead of 32 small ones
        self.w1 = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)
        self.w3 = nn.Parameter(torch.randn(n_experts, d_model, d_ff) * 0.02)

    def _top_k_routing(self, router_logits: torch.Tensor):
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        return router_probs, top_k_probs, top_k_indices

    def _compute_aux_losses(
        self, router_logits: torch.Tensor, router_probs: torch.Tensor,
        top_k_indices: torch.Tensor
    ):
        num_tokens = router_logits.shape[0]

        tokens_per_expert = top_k_indices.flatten()
        f_i = torch.zeros(self.n_experts, device=router_logits.device)
        f_i.scatter_add_(
            0, tokens_per_expert,
            torch.ones_like(tokens_per_expert, dtype=torch.float)
        )
        f_i = f_i / (num_tokens * self.top_k)

        p_i = router_probs.mean(dim=0)

        load_balance_loss = self.n_experts * torch.sum(f_i * p_i)

        z_loss = torch.mean(router_logits.float() ** 2)

        return load_balance_loss, z_loss

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        tokens = x.view(-1, D)
        num_tokens = tokens.shape[0]

        router_logits = self.router(tokens)
        router_probs, top_k_probs, top_k_indices = self._top_k_routing(router_logits)

        load_balance_loss, z_loss = self._compute_aux_losses(
            router_logits, router_probs, top_k_indices
        )

        capacity = math.ceil(num_tokens * self.top_k * self.capacity_factor / self.n_experts)

        final_output = torch.zeros_like(tokens)
        expert_util = torch.zeros(self.n_experts, device=x.device, dtype=torch.long)

        # Flatten routing assignments
        flat_experts = top_k_indices.flatten()  # (num_tokens * top_k,)
        flat_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        flat_route = torch.arange(self.top_k, device=x.device).repeat(num_tokens)

        # Sort by expert for contiguous groups
        order = torch.argsort(flat_experts)
        flat_experits = flat_experts[order]
        flat_tokens = flat_tokens[order]
        flat_route = flat_route[order]

        unique_experts, counts = torch.unique_consecutive(flat_experits, return_counts=True)
        ends = counts.cumsum(dim=0)
        starts = ends - counts

        # Build padded per-expert input buffers
        expert_buf = torch.zeros(self.n_experts, capacity, D, device=x.device, dtype=x.dtype)
        weight_buf = torch.zeros(self.n_experts, capacity, 1, device=x.device, dtype=x.dtype)
        token_map = torch.full((self.n_experts, capacity), -1, device=x.device, dtype=torch.long)

        for expert_idx, start, count in zip(unique_experts.tolist(), starts.tolist(), counts.tolist()):
            end = start + count
            tok_ids = flat_tokens[start:end]
            rt_ids = flat_route[start:end]

            if count > capacity:
                probs = top_k_probs[tok_ids, rt_ids]
                keep = probs.argsort(descending=True)[:capacity]
                tok_ids = tok_ids[keep]
                rt_ids = rt_ids[keep]
                count = capacity

            expert_buf[expert_idx, :count] = tokens[tok_ids]
            weight_buf[expert_idx, :count, 0] = top_k_probs[tok_ids, rt_ids]
            token_map[expert_idx, :count] = tok_ids
            expert_util[expert_idx] = count

        # Batched SwiGLU: all experts computed in one go
        h1 = torch.bmm(expert_buf, self.w1.transpose(-1, -2))  # (n, cap, d_ff)
        h2 = torch.bmm(expert_buf, self.w2.transpose(-1, -2))
        expert_out = torch.bmm(F.silu(h1) * h2, self.w3.transpose(-1, -2))  # (n, cap, d_model)
        expert_out = expert_out * weight_buf  # (n, cap, d_model)

        # Scatter back
        mask = token_map != -1
        final_output[token_map[mask]] += expert_out[mask]

        aux_loss = (
            self.load_balance_coeff * load_balance_loss
            + self.z_loss_coeff * z_loss
        )

        return (
            final_output.view(B, T, D),
            aux_loss,
            expert_util,
            router_logits.detach(),
        )
