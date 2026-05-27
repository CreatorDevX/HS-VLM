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

        # Stacked expert weights: single batched matmul per SwiGLU projection
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

        # Build flat routing assignments and sort by expert (all on GPU, no .tolist())
        flat_experts = top_k_indices.flatten()
        flat_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        flat_route = torch.arange(self.top_k, device=x.device).repeat(num_tokens)
        probs_flat = top_k_probs[flat_tokens, flat_route]

        # Sort by expert index for contiguous groups
        order = torch.argsort(flat_experts)
        flat_experts = flat_experts[order]
        flat_tokens = flat_tokens[order]
        flat_route = flat_route[order]
        probs_flat = probs_flat[order]

        N = flat_experts.size(0)
        unique_experts, counts = torch.unique_consecutive(flat_experts, return_counts=True)
        ends = counts.cumsum(dim=0)
        starts = ends - counts

        # Map each expert to its group start offset
        expert_to_start = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        expert_to_start[unique_experts] = starts

        # Compute rank of each assignment within its expert group
        ranks = torch.arange(N, device=x.device) - expert_to_start[flat_experts]

        # Drop tokens that exceed capacity
        keep = ranks < capacity
        flat_experts = flat_experts[keep]
        flat_tokens = flat_tokens[keep]
        flat_route = flat_route[keep]
        probs_flat = probs_flat[keep]
        ranks = ranks[keep]

        # Build padded expert buffers via single GPU scatter (no Python loop)
        expert_buf = tokens.new_zeros(self.n_experts, capacity, D)
        weight_buf = tokens.new_zeros(self.n_experts, capacity, 1)
        token_map = torch.full((self.n_experts, capacity), -1, device=x.device, dtype=torch.long)

        expert_buf[flat_experts, ranks] = tokens[flat_tokens]
        weight_buf[flat_experts, ranks, 0] = probs_flat
        token_map[flat_experts, ranks] = flat_tokens

        # Compute expert util counts
        expert_util = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        expert_util.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=torch.long))

        # Batched SwiGLU: all experts computed as a single grouped matmul
        h1 = torch.bmm(expert_buf, self.w1.transpose(-1, -2))
        h2 = torch.bmm(expert_buf, self.w2.transpose(-1, -2))
        expert_out = torch.bmm(F.silu(h1) * h2, self.w3.transpose(-1, -2))
        expert_out = expert_out * weight_buf

        # Scatter expert outputs back to token positions
        final_output = torch.zeros_like(tokens)
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
