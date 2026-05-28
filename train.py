import os
import math
import time
import argparse
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from sophia import SophiaG
    SOPHIA_AVAILABLE = True
except ImportError:
    SOPHIA_AVAILABLE = False

try:
    from bitsandbytes.optim import AdamW8bit
    ADAMW_8BIT = True
    AdamW_fallback = AdamW8bit
except ImportError:
    from torch.optim import AdamW
    ADAMW_8BIT = False
    AdamW_fallback = AdamW

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import ModelConfig, TrainingConfig, compute_ntk_base
from model.model import MoETransformer
from model.loss import multi_token_cross_entropy
from data.dataset import create_dataloader, load_or_train_tokenizer


INFERENCE_PROMPTS = [
    "The future of artificial intelligence lies in",
    "In the beginning, the universe was nothing but",
    "The old man sat by the window and watched",
]


def setup_ddp():
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        return world_size, rank, True
    return 1, 0, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
):
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    lr: float,
):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    pred_token_weights: tuple,
    amp_dtype: torch.dtype,
    num_batches: int = 10,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches_run = 0
    for batch in val_loader:
        if num_batches_run >= num_batches:
            break
        input_ids = batch["input_ids"].cuda()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(input_ids)
            logits = outputs["logits"]
            ce_loss = multi_token_cross_entropy(
                logits, input_ids, weights=pred_token_weights
            )
        total_loss = total_loss + ce_loss.item()
        num_batches_run += 1
    model.train()
    avg_loss = total_loss / max(1, num_batches_run)
    return {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss)}


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    amp_dtype: torch.dtype = torch.float16,
):
    input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0).cuda()

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs["past_key_values"]
    logits = outputs["logits"]
    if logits.dim() == 4:
        logits = logits[0, -1, 0]
    else:
        logits = logits[0, -1]

    for _ in range(max_new_tokens):
        logits = logits / temperature

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs["past_key_values"]
        logits = outputs["logits"]
        if logits.dim() == 4:
            logits = logits[0, -1, 0]
        else:
            logits = logits[0, -1]

    return tokenizer.decode(input_ids[0].tolist())


def generate_text_table(
    model: nn.Module, tokenizer, prompts, amp_dtype, max_new_tokens=64,
) -> list:
    results = []
    for prompt in prompts:
        output = generate(
            model, prompt, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.8, top_k=50, amp_dtype=amp_dtype,
        )
        results.append({"prompt": prompt, "output": output})
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    save_dir: str,
    keep_last: int = 5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step}.pt")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    old_ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("step_") and f.endswith(".pt")]
    )
    while len(old_ckpts) > keep_last:
        os.remove(os.path.join(save_dir, old_ckpts.pop(0)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HypersparsityLM — MoE pretraining")

    # === Mode ===
    p.add_argument("--phase2", action="store_true", help="Phase 2: extend context to 16K")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume / initialise Phase 2 from")

    # === Model ===
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--d-embed", type=int, default=64)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-kv-heads", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff-expert", type=int, default=148)
    p.add_argument("--n-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--n-pred-tokens", type=int, default=4)
    p.add_argument("--no-mtp", action="store_true", help="Disable multi-token prediction (n_pred_tokens=1)")
    p.add_argument("--capacity-factor", type=float, default=1.25)
    p.add_argument("--z-loss-coeff", type=float, default=1e-3)
    p.add_argument("--load-balance-coeff", type=float, default=1e-2)
    p.add_argument("--norm-eps", type=float, default=1e-6)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--rope-base", type=float, default=10000.0)

    # === Data ===
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--tokenizer-path", type=str, default="tokenizer.json")

    # === Training ===
    p.add_argument("--total-tokens", type=float, default=1e9, help="Total training tokens")
    p.add_argument("--optimizer", type=str, default="sophia", choices=["adamw", "sophia"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=None, help="Warmup steps (default: 5%% of total)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.965)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--rho", type=float, default=0.03, help="Sophia clipping threshold")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    # === Batch ===
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=32)

    # === Logging ===
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--inference-interval", type=int, default=100)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--keep-last", type=int, default=5)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", action="store_false", dest="compile")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb-project", type=str, default="hypersparsity-lm")
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--wandb-key", type=str, default=None, help="W&B API key (avoids interactive login)")

    return p.parse_args()


def build_configs(args: argparse.Namespace):
    total_steps = max(1, int(args.total_tokens / (args.micro_batch_size * args.max_seq_len * args.grad_accum_steps)))

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_embed=args.d_embed,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        d_ff_expert=args.d_ff_expert,
        n_experts=args.n_experts,
        top_k=args.top_k,
        n_pred_tokens=1 if args.no_mtp else args.n_pred_tokens,
        capacity_factor=args.capacity_factor,
        z_loss_coeff=args.z_loss_coeff,
        load_balance_coeff=args.load_balance_coeff,
        norm_eps=args.norm_eps,
        max_seq_len=args.max_seq_len,
        rope_base=args.rope_base,
    )

    run_name = args.wandb_run or f"moe-50m-{model_cfg.sparsity:.0f}sparse"

    if args.phase2:
        ext = model_cfg.max_seq_len // 2048
        model_cfg.rope_base = compute_ntk_base(10000.0, ext, model_cfg.d_head)
        run_name += f"-16k"

    warmup_steps = args.warmup if args.warmup is not None else max(1, total_steps // 20)

    train_cfg = TrainingConfig(
        dataset_name=args.dataset,
        tokenizer_path=args.tokenizer_path,
        seq_len=model_cfg.max_seq_len,
        total_tokens=int(args.total_tokens),
        learning_rate=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        inference_interval=args.inference_interval,
        save_dir=args.save_dir,
        keep_last_n_checkpoints=args.keep_last,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
    )
    train_cfg.warmup_steps = warmup_steps

    return model_cfg, train_cfg


def train(args: argparse.Namespace, rank: int = 0, world_size: int = 1, ddp_enabled: bool = False):
    model_cfg, train_cfg = build_configs(args)

    total_steps = train_cfg.max_steps
    tokens_per_step = train_cfg.tokens_per_step
    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        amp_dtype = torch.float32
    else:
        amp_dtype = torch.float16
    main = is_main_process(rank)

    # === Print summary ===
    if main:
        phase = "PHASE 2 — 16K context" if args.phase2 else "PHASE 1 — 2K context"
        flops = 6 * model_cfg.activated_params_estimate * 1e6 * total_steps * tokens_per_step
        print(f"{'='*60}")
        print(f"  {phase}")
        print(f"{'='*60}")
        print(f"Model:     {model_cfg.total_params_estimate:.2f}M ({model_cfg.activated_params_estimate:.2f}M activated)")
        print(f"Sparsity:  {model_cfg.sparsity:.1f}%")
        print(f"Context:   {model_cfg.max_seq_len}  (RoPE base={model_cfg.rope_base:.0f})")
        print(f"Tokens:    {train_cfg.total_tokens/1e9:.1f}B  |  Steps: {total_steps} ({tokens_per_step:,}/step)")
        print(f"Data:      {train_cfg.dataset_name}")
        print(f"LR:        {train_cfg.learning_rate} -> {train_cfg.min_lr}  warmup={train_cfg.warmup_steps}")
        print(f"Optimizer: {args.optimizer}")
        print(f"Compile:   {'torch.compile' if args.compile else 'disabled'}  |  {args.dtype.upper()} + GradScaler")
        print(f"GPUs:      {world_size} {'DDP' if ddp_enabled else 'single'}")
        print(f"Micro-batch: {train_cfg.micro_batch_size}  |  Grad accum: {train_cfg.grad_accum_steps}")
        print(f"Compute:   ~{flops/1e15:.1f} PFLOPs  |  ~{flops/1e15/0.15:.0f}s @ 150 TFLOPs")
        print(f"{'='*60}")

    # === Wandb ===
    if main and train_cfg.use_wandb and WANDB_AVAILABLE:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.wandb_run_name,
            config={"phase": "2-16k" if args.phase2 else "1-2k", **vars(args)},
        )

    # === Model ===
    raw_model = MoETransformer(model_cfg).cuda()

    if args.phase2:
        ckpt_path = args.checkpoint or "checkpoints/step_1907.pt"
        state_dict = torch.load(ckpt_path, map_location="cuda")["model"]
        missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
        if main:
            if missing:
                print(f"  Missing keys (expected for Phase 2): {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print(f"Loaded checkpoint: {ckpt_path}")
            print(f"RoPE base extended to {model_cfg.rope_base:.0f}")

    if args.compile:
        raw_model = torch.compile(raw_model, mode="default")

    if main:
        print(f"Params: {sum(p.numel() for p in raw_model.parameters()):,}")
        if not args.compile:
            print("torch.compile disabled")

    model = DDP(raw_model, device_ids=[rank]) if ddp_enabled else raw_model

    # === Optimiser & LR schedule ===
    if args.optimizer == "sophia" and SOPHIA_AVAILABLE:
        optimizer = SophiaG(
            model.parameters(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
            rho=args.rho,
            weight_decay=train_cfg.weight_decay,
        )
        if is_main_process(rank):
            print(f"Optimizer: SophiaG (rho={args.rho})")
    else:
        param_groups = get_weight_decay_param_groups(model, train_cfg.weight_decay, train_cfg.learning_rate)
        opt_name = "AdamW 8-bit" if ADAMW_8BIT else "AdamW (32-bit)"
        if args.optimizer == "sophia" and not SOPHIA_AVAILABLE and is_main_process(rank):
            print(f"SophiaG not available, falling back to {opt_name}")
        optimizer = AdamW_fallback(param_groups, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=train_cfg.warmup_steps, total_steps=total_steps, min_lr=train_cfg.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=2.**8, growth_factor=1.5)

    # === Data ===
    train_loader = create_dataloader(
        tokenizer_path=train_cfg.tokenizer_path, seq_len=train_cfg.seq_len,
        batch_size=train_cfg.micro_batch_size, split="train",
        dataset_name=train_cfg.dataset_name,
    )
    val_loader = None
    if main:
        val_loader = create_dataloader(
            tokenizer_path=train_cfg.tokenizer_path, seq_len=train_cfg.seq_len,
            batch_size=train_cfg.micro_batch_size, split="validation",
            dataset_name=train_cfg.dataset_name,
        )

    tokenizer = load_or_train_tokenizer(
        tokenizer_path=train_cfg.tokenizer_path,
    )

    # === Training loop ===
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_ce = 0.0
    accum_aux = 0.0
    start_time = time.time()
    data_iter = iter(train_loader)
    best_val_loss = float("inf")

    for step in range(1, total_steps + 1):
        torch.compiler.cudagraph_mark_step_begin()
        micro_loss = 0.0
        micro_ce = 0.0
        micro_aux = 0.0
        micro_expert_util = 0.0

        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].cuda()
            no_sync = ddp_enabled and micro_step < train_cfg.grad_accum_steps - 1
            ctx = model.no_sync() if no_sync else nullcontext()

            with ctx, torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids)
                logits = outputs["logits"]
                ce_loss = multi_token_cross_entropy(logits, input_ids, weights=train_cfg.pred_token_weights)
                aux_loss = outputs["aux_loss"]
                loss = (ce_loss + aux_loss) / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            micro_loss += loss.item()
            micro_ce += (ce_loss / train_cfg.grad_accum_steps).item()
            micro_aux += (aux_loss / train_cfg.grad_accum_steps).item()
            micro_expert_util += outputs["expert_utils"].float().mean().item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        accum_loss += micro_loss
        accum_ce += micro_ce
        accum_aux += micro_aux

        # --- Logging ---
        if step % train_cfg.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = train_cfg.tokens_per_step * train_cfg.log_interval / elapsed
            current_lr = scheduler.get_last_lr()[0]
            avg_util = micro_expert_util / train_cfg.grad_accum_steps

            log_data = {
                "loss": accum_loss / train_cfg.log_interval,
                "ce_loss": accum_ce / train_cfg.log_interval,
                "aux_loss": accum_aux / train_cfg.log_interval,
                "lr": current_lr,
                "tokens_per_sec": tok_per_sec,
                "avg_expert_util": avg_util,
                "grad_norm": grad_norm,
            }

            if main:
                if train_cfg.use_wandb and WANDB_AVAILABLE:
                    wandb.log(log_data, step=step)
                print(
                    f"Step {step:>6d}/{total_steps} | "
                    f"loss: {log_data['loss']:.4f} | "
                    f"ce: {log_data['ce_loss']:.4f} | "
                    f"aux: {log_data['aux_loss']:.6f} | "
                    f"lr: {current_lr:.2e} | "
                    f"tok/s: {tok_per_sec:.0f} | "
                    f"grad: {grad_norm:.2f}"
                )

            accum_loss = 0.0
            accum_ce = 0.0
            accum_aux = 0.0
            start_time = time.time()

        # --- Evaluation ---
        if step % train_cfg.eval_interval == 0 and main and val_loader is not None:
            metrics = evaluate(model, val_loader, train_cfg.pred_token_weights, amp_dtype, num_batches=10)
            if train_cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics, step=step)
            print(f"  Eval step {step}: val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_ppl']:.2f}")
            model.train()
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(
                    model.module if ddp_enabled else model, optimizer, scheduler,
                    step, train_cfg.save_dir, train_cfg.keep_last_n_checkpoints, scaler,
                )

        # --- Inference ---
        if step % train_cfg.inference_interval == 0 and main:
            generations = generate_text_table(
                model.module if ddp_enabled else model, tokenizer,
                INFERENCE_PROMPTS, amp_dtype, max_new_tokens=64,
            )
            print(f"\n  --- Generation at step {step} ---")
            for g in generations:
                print(f"    prompt: {g['prompt']}")
                print(f"    output: {g['output'][:120]}...")
                print()
            if train_cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "generations": wandb.Table(
                        columns=["step", "prompt", "output"],
                        data=[[step, g["prompt"], g["output"]] for g in generations],
                    )
                }, step=step)

        # --- Checkpoint ---
        if step % train_cfg.save_interval == 0 and main:
            save_checkpoint(
                model.module if ddp_enabled else model, optimizer, scheduler,
                step, train_cfg.save_dir, train_cfg.keep_last_n_checkpoints, scaler,
            )
            print(f"  Saved checkpoint at step {step}")

    # === Final checkpoint & inference ===
    if main:
        save_checkpoint(
            model.module if ddp_enabled else model, optimizer, scheduler,
            total_steps, train_cfg.save_dir, train_cfg.keep_last_n_checkpoints, scaler,
        )
        print(f"Saved final checkpoint at step {total_steps}")

        generations = generate_text_table(
            model.module if ddp_enabled else model, tokenizer,
            INFERENCE_PROMPTS, amp_dtype, max_new_tokens=128,
        )
        print("\n" + "=" * 60)
        print("FINAL INFERENCE")
        print("=" * 60)
        for g in generations:
            print(f"\nPrompt: {g['prompt']}")
            print(f"Output: {g['output']}")
            print("-" * 60)

        if train_cfg.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "generations": wandb.Table(
                    columns=["step", "prompt", "output"],
                    data=[[total_steps, g["prompt"], g["output"]] for g in generations],
                )
            }, step=total_steps)
            wandb.finish()

    print(f"Training complete (rank {rank})")


if __name__ == "__main__":
    args = parse_args()
    world_size, rank, ddp_enabled = setup_ddp()
    try:
        train(args, rank=rank, world_size=world_size, ddp_enabled=ddp_enabled)
    finally:
        cleanup_ddp()
