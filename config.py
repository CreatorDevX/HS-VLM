from dataclasses import dataclass, field
from typing import Optional


def compute_ntk_base(base: float, extension_factor: float, d_head: int) -> float:
    return base * (extension_factor ** (d_head / (d_head - 2)))


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 384
    d_embed: int = 64
    d_head: int = 64
    n_heads: int = 6
    n_kv_heads: int = 3
    n_layers: int = 8
    d_ff_expert: int = 148
    n_experts: int = 32
    top_k: int = 2
    n_pred_tokens: int = 4
    capacity_factor: float = 1.25
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    z_loss_coeff: float = 1e-3
    load_balance_coeff: float = 1e-2

    @property
    def _q_dim(self) -> int:
        return self.n_heads * self.d_head

    @property
    def _kv_dim(self) -> int:
        return self.n_kv_heads * self.d_head

    @property
    def _attn_params(self) -> int:
        q_proj = self.d_model * self._q_dim
        k_proj = self.d_model * self._kv_dim
        v_proj = self.d_model * self._kv_dim
        out_proj = self._q_dim * self.d_model
        return q_proj + k_proj + v_proj + out_proj

    @property
    def total_params_estimate(self) -> float:
        embed = self.vocab_size * self.d_embed + self.d_embed * self.d_model
        attn = self._attn_params
        norms = 2 * self.d_model
        router = self.d_model * self.n_experts
        dense_per = attn + norms + router
        expert = 3 * self.d_model * self.d_ff_expert
        moe_per = self.n_experts * expert
        head_shared = self.d_model * self.d_model
        head_out = self.d_model * self.n_pred_tokens * self.d_embed
        total = embed + self.n_layers * (dense_per + moe_per) + head_shared + head_out
        return total / 1e6

    @property
    def activated_params_estimate(self) -> float:
        embed = self.vocab_size * self.d_embed + self.d_embed * self.d_model
        attn = self._attn_params
        norms = 2 * self.d_model
        router = self.d_model * self.n_experts
        dense_per = attn + norms + router
        expert = 3 * self.d_model * self.d_ff_expert
        moe_activated_per = self.top_k * expert
        head_shared = self.d_model * self.d_model
        head_out = self.d_model * self.n_pred_tokens * self.d_embed
        total = embed + self.n_layers * (dense_per + moe_activated_per) + head_shared + head_out
        return total / 1e6

    @property
    def sparsity(self) -> float:
        total = self.total_params_estimate
        active = self.activated_params_estimate
        return (1 - active / total) * 100


@dataclass
class TrainingConfig:
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    tokenizer_path: str = "tokenizer.json"
    seq_len: int = 2048
    total_tokens: int = 1_000_000_000

    # Optimization
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.965
    beta2: float = 0.99
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # MoE aux losses
    z_loss_coeff: float = 1e-3
    load_balance_coeff: float = 1e-2

    # Multi-token prediction
    pred_token_weights: tuple = (1.0, 1.0, 1.0, 1.0)

    # Batch
    micro_batch_size: int = 8
    grad_accum_steps: int = 32

    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    inference_interval: int = 100
    save_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5
    use_wandb: bool = True
    wandb_project: str = "hypersparsity-lm"
    wandb_run_name: str = "moe-50m-82sparse"

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    @property
    def max_steps(self) -> int:
        return int(self.total_tokens / self.tokens_per_step)

    warmup_steps: int = 0  # computed as max(1, max_steps // 20) if not set



