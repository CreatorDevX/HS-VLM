import os
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPModel, CLIPImageProcessor

from config import ModelConfig
from model.model import MoETransformer
from model.visual_prefix import VisualPrefixEncoder
from model.projector import Projector
from data.dataset import load_or_train_tokenizer
from data.clip_dataset import create_clip_dataloader


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


def multi_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HypersparsityLM — Phase 3b Visual Prefix Training")

    p.add_argument("--clip-model", type=str, default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    p.add_argument("--lm-checkpoint", type=str, required=True, help="Phase 1/2 LM checkpoint")
    p.add_argument("--projector-checkpoint", type=str, default=None, help="Phase 3a projector checkpoint (optional)")

    p.add_argument("--n-queries", type=int, default=16, help="Number of visual prefix tokens")

    p.add_argument("--dataset", type=str, default="lambdalabs/pokemon-blip-captions")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--image-key", type=str, default="image")
    p.add_argument("--text-key", type=str, default="text")

    p.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.02)

    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--gen-interval", type=int, default=200)
    p.add_argument("--save-dir", type=str, default="checkpoints")

    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb-project", type=str, default="hypersparsity-lm")
    p.add_argument("--wandb-run", type=str, default=None)

    return p.parse_args()


@torch.no_grad()
def generate_caption(
    lm: MoETransformer,
    prefix_encoder: VisualPrefixEncoder,
    clip_model: CLIPModel,
    clip_processor: CLIPImageProcessor,
    image,
    tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cuda"),
):
    lm.eval()
    prefix_encoder.eval()

    image_tensor = clip_processor(image, return_tensors="pt").pixel_values.to(device)
    clip_embed = F.normalize(clip_model.get_image_features(pixel_values=image_tensor), dim=-1)
    prefix = prefix_encoder(clip_embed)

    input_ids = torch.tensor([[tokenizer.token_to_id("<bos>")]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = lm(input_ids, image_prefix=prefix)
        logits = outputs["logits"][0, -1, 0]
        logits = logits / temperature

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    lm.train()
    prefix_encoder.train()
    return tokenizer.decode(input_ids[0].tolist())


def train(args: argparse.Namespace, rank: int = 0, world_size: int = 1, ddp_enabled: bool = False):
    main = is_main_process(rank)
    device = torch.device(rank)

    if main:
        print("=" * 60)
        print("  PHASE 3b — Visual Prefix for Generative VLM")
        print("=" * 60)
        print(f"CLIP:        {args.clip_model}")
        print(f"Queries:     {args.n_queries}")
        print(f"Dataset:     {args.dataset} [{args.split}]")
        print(f"Steps:       {args.steps}  |  Batch: {args.batch_size * args.grad_accum * world_size}")
        print(f"LR:          {args.lr}  warmup={args.warmup}")
        print(f"GPUs:        {world_size} {'DDP' if ddp_enabled else 'single'}")
        print(f"LM frozen:   yes  |  Prefix encoder trainable: yes")
        print("=" * 60)

    # === CLIP (frozen) ===
    if main:
        print("Loading CLIP...")
    clip = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip.requires_grad_(False)
    clip.eval()
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # === LM (frozen) ===
    if main:
        print("Loading LM checkpoint...")
    state = torch.load(args.lm_checkpoint, map_location=device, weights_only=True)
    model_cfg = ModelConfig()
    lm = MoETransformer(model_cfg).to(device)
    lm.load_state_dict(state["model"], strict=False)
    lm.requires_grad_(False)
    lm.eval()

    # === Projector (frozen, optional) ===
    projector = None
    if args.projector_checkpoint and os.path.exists(args.projector_checkpoint):
        p_state = torch.load(args.projector_checkpoint, map_location=device, weights_only=True)
        projector = Projector(d_model=model_cfg.d_model, d_clip=clip.config.projection_dim).to(device)
        projector.load_state_dict(p_state["projector"])
        projector.requires_grad_(False)
        projector.eval()
        if main:
            print(f"Loaded projector from {args.projector_checkpoint}")

    # === Visual Prefix Encoder (trainable) ===
    prefix_encoder = VisualPrefixEncoder(
        d_model=model_cfg.d_model,
        d_clip=clip.config.projection_dim,
        n_queries=args.n_queries,
    ).to(device)

    vlm_model = prefix_encoder
    if ddp_enabled:
        vlm_model = DDP(prefix_encoder, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        prefix_encoder.parameters(), lr=args.lr, weight_decay=args.wd
    )

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, (s + 1) / max(1, args.warmup)),
    )

    # === Tokenizer & Data ===
    tokenizer = load_or_train_tokenizer(
        tokenizer_path=args.tokenizer_path,
    )

    loader = create_clip_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        image_key=args.image_key,
        text_key=args.text_key,
        batch_size=args.batch_size,
        clip_image_processor=image_processor,
        lm_tokenizer=tokenizer,
        seq_len=args.seq_len,
    )
    data_iter = iter(loader)

    # === W&B ===
    if main and args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"vlm-{args.n_queries}q",
            config=vars(args),
        )

    # === Training ===
    prefix_encoder.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for step in range(1, args.steps + 1):
        micro_loss = 0.0

        for micro_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                clip_embed = clip.get_image_features(pixel_values=images)
                clip_embed = F.normalize(clip_embed, dim=-1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                prefix = prefix_encoder(clip_embed)
                outputs = lm(input_ids, image_prefix=prefix)
                logits = outputs["logits"]
                loss = multi_token_cross_entropy(logits, input_ids)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            micro_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(prefix_encoder.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        warmup_scheduler.step()
        optimizer.zero_grad()

        accum_loss += micro_loss

        # === Logging ===
        if step % 10 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = accum_loss / max(1, step % 10 or 10)

            if main:
                msg = (
                    f"Step {step:>4d}/{args.steps} | "
                    f"loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )
                print(msg)
                if args.wandb:
                    import wandb
                    wandb.log({"loss": avg_loss, "lr": current_lr}, step=step)

            accum_loss = 0.0
            start_time = time.time()

        # === Generation samples ===
        if step % args.gen_interval == 0 and main:
            sample_batch = next(iter(loader))
            sample_image = sample_batch["image"][0]
            sample_caption = sample_batch["caption"][0]
            generated = generate_caption(
                lm, prefix_encoder, clip, image_processor,
                sample_image, tokenizer, max_new_tokens=48, device=device,
            )
            print(f"\n  --- Generation at step {step} ---")
            print(f"    reference: {sample_caption}")
            print(f"    generated: {generated}")
            if args.wandb:
                import wandb
                wandb.log({
                    "generation_ref": sample_caption,
                    "generation_out": generated,
                }, step=step)

        # === Checkpoint ===
        if step % args.save_interval == 0 and main:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"vlm_prefix_step{step}.pt")
            torch.save({
                "prefix_encoder": prefix_encoder.state_dict(),
                "args": args,
            }, path)
            print(f"  Saved {path}")

    # === Final ===
    if main:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, "vlm_prefix_final.pt")
        torch.save({
            "prefix_encoder": prefix_encoder.state_dict(),
            "args": args,
        }, path)
        print(f"Saved {path}")
        if args.wandb:
            import wandb
            wandb.finish()

    print(f"Visual prefix training complete (rank {rank})")


if __name__ == "__main__":
    args = parse_args()
    world_size, rank, ddp_enabled = setup_ddp()
    try:
        train(args, rank=rank, world_size=world_size, ddp_enabled=ddp_enabled)
    finally:
        cleanup_ddp()
