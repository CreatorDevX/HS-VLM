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

        for expert_idx in range(self.n_experts):
            mask = top_k_indices == expert_idx
            token_indices, route_idx = torch.where(mask)

            if len(token_indices) > capacity:
                probs_for_expert = top_k_probs[token_indices, route_idx]
                _, sorted_idx = torch.sort(probs_for_expert, descending=True)
                token_indices = token_indices[sorted_idx[:capacity]]
                route_idx = route_idx[sorted_idx[:capacity]]

            if len(token_indices) > 0:
                expert_input = tokens[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weights = top_k_probs[token_indices, route_idx].unsqueeze(-1)
                final_output[token_indices] += expert_output * weights
                expert_util[expert_idx] = len(token_indices)

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
