import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


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
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

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

        # Flatten all routing assignments: (num_tokens * top_k,) each
        flat_experts = top_k_indices.flatten()
        flat_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        flat_route = torch.arange(self.top_k, device=x.device).repeat(num_tokens)

        # Sort by expert index so tokens for the same expert are contiguous
        order = torch.argsort(flat_experts)
        flat_experts = flat_experts[order]
        flat_tokens = flat_tokens[order]
        flat_route = flat_route[order]

        # Find group boundaries for each expert that has tokens
        unique_experts, counts = torch.unique_consecutive(flat_experts, return_counts=True)
        ends = counts.cumsum(dim=0)
        starts = ends - counts

        for expert_idx, start, count in zip(unique_experts.tolist(), starts.tolist(), counts.tolist()):
            end = start + count
            token_ids = flat_tokens[start:end]
            route_ids = flat_route[start:end]

            if count > capacity:
                probs = top_k_probs[token_ids, route_ids]
                keep = probs.argsort(descending=True)[:capacity]
                token_ids = token_ids[keep]
                route_ids = route_ids[keep]
                count = capacity

            expert_input = tokens[token_ids]
            expert_output = self.experts[expert_idx](expert_input)
            weights = top_k_probs[token_ids, route_ids].unsqueeze(-1)
            final_output[token_ids] += expert_output * weights
            expert_util[expert_idx] = count

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
