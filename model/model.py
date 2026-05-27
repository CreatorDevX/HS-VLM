from typing import Dict, Optional

import torch
import torch.nn as nn

from model.transformer import TransformerBlock, RMSNorm
from model.head import MultiTokenLMHead


class MoETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(
            config.vocab_size, config.d_embed
        )
        self.embed_proj = nn.Linear(config.d_embed, config.d_model, bias=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff_expert=config.d_ff_expert,
                n_experts=config.n_experts,
                top_k=config.top_k,
                max_seq_len=config.max_seq_len + config.n_pred_tokens,
                rope_base=config.rope_base,
                capacity_factor=config.capacity_factor,
                z_loss_coeff=config.z_loss_coeff,
                load_balance_coeff=config.load_balance_coeff,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = MultiTokenLMHead(
            d_model=config.d_model,
            d_embed=config.d_embed,
            n_pred_tokens=config.n_pred_tokens,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_prefix: Optional[torch.Tensor] = None,
        return_layer: Optional[int] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h = self.token_embed(input_ids)
        h = self.embed_proj(h)

        n_prefix = 0
        if image_prefix is not None:
            n_prefix = image_prefix.shape[1]
            h = torch.cat([image_prefix, h], dim=1)

        total_aux_loss = 0.0
        expert_utils = []
        router_logits_list = []
        hidden_states = None
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, aux_loss, expert_util, router_logits, new_kv = block(h, past_kv=past_kv, use_cache=use_cache)
            total_aux_loss = total_aux_loss + aux_loss
            expert_utils.append(expert_util)
            router_logits_list.append(router_logits)
            if use_cache:
                new_past_key_values.append(new_kv)
            if return_layer is not None and i == return_layer:
                if n_prefix > 0:
                    hidden_states = h[:, n_prefix:]
                else:
                    hidden_states = h

        h = self.final_norm(h)

        h_text = h[:, n_prefix:] if n_prefix > 0 else h
        logits = self.lm_head(h_text, self.token_embed.weight)

        result = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "expert_utils": torch.stack(expert_utils),
            "router_logits": router_logits_list,
        }
        if use_cache:
            result["past_key_values"] = new_past_key_values
        if hidden_states is not None:
            result["hidden_states"] = hidden_states
        return result
