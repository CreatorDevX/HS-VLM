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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import ModelConfig
from model.model import MoETransformer
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


def contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    B = image_embeds.size(0)
    logits = logit_scale * text_embeds @ image_embeds.T
    labels = torch.arange(B, device=logits.device)
    loss_i2t = F.cross_entropy(logits.T, labels)
    loss_t2i = F.cross_entropy(logits, labels)
    return (loss_i2t + loss_t2i) / 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HypersparsityLM — Phase 3 CLIP alignment")

    p.add_argument("--clip-model", type=str, default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")

    p.add_argument("--checkpoint", type=str, required=True, help="Phase 1/2 LM checkpoint")
    p.add_argument("--layer", type=int, default=4, help="LM layer to extract hidden states from (0-7, default 4)")

    p.add_argument("--dataset", type=str, default="lambdalabs/pokemon-blip-captions")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--image-key", type=str, default="image")
    p.add_argument("--text-key", type=str, default="text")

    p.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=77)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--logit-scale-init", type=float, default=1.0)

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=200)

    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb-project", type=str, default="hypersparsity-lm")
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--wandb-key", type=str, default=None, help="W&B API key (avoids interactive login)")

    return p.parse_args()


def train(args: argparse.Namespace, rank: int = 0, world_size: int = 1, ddp_enabled: bool = False):
    main = is_main_process(rank)
    device = torch.device(rank)

    if main:
        print("=" * 60)
        print("  PHASE 3 — CLIP Alignment")
        print("=" * 60)
        print(f"CLIP:        {args.clip_model}")
        print(f"LM layer:    {args.layer}  (early phase connector)")
        print(f"Dataset:     {args.dataset} [{args.split}]")
        print(f"Steps:       {args.steps}  |  Batch: {args.batch_size * args.grad_accum * world_size}")
        print(f"LR:          {args.lr}  warmup={args.warmup}")
        print(f"GPUs:        {world_size} {'DDP' if ddp_enabled else 'single'}")
        print("=" * 60)

    # === CLIP model (frozen) ===
    if main:
        print("Loading CLIP...")
    clip = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip.requires_grad_(False)
    clip.eval()

    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # === LM (frozen) ===
    if main:
        print("Loading LM checkpoint...")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = ModelConfig()
    lm = MoETransformer(model_cfg).to(device)
    lm.load_state_dict(state["model"], strict=False)
    lm.requires_grad_(False)
    lm.eval()

    # === Projector (trainable) ===
    projector = Projector(d_model=model_cfg.d_model, d_clip=clip.config.projection_dim).to(device)
    logit_scale = nn.Parameter(torch.tensor(args.logit_scale_init)).to(device)

    model = projector
    if ddp_enabled:
        model = DDP(projector, device_ids=[rank], find_unused_parameters=False)

    trainable_params = list(projector.parameters()) + [logit_scale]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

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
    if main and args.wandb and WANDB_AVAILABLE:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"align-layer{args.layer}",
            config=vars(args),
        )

    # === Training ===
    projector.train()
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    accum_loss = 0.0
    start_time = time.time()

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
                image_embeds = clip.get_image_features(pixel_values=images)
                image_embeds = F.normalize(image_embeds, dim=-1)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lm_out = lm(input_ids, return_layer=args.layer)
                    h = lm_out["hidden_states"]

            if h is None:
                raise RuntimeError(
                    f"return_layer={args.layer} out of range (LM has {len(lm.blocks)} layers)"
                )

            with torch.amp.autocast("cuda", dtype=torch.float16):
                text_embeds = projector(h)

            loss = contrastive_loss(image_embeds, text_embeds, logit_scale)
            loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            micro_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        warmup_scheduler.step()
        optimizer.zero_grad()

        accum_loss += micro_loss

        if step % 10 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            scale_val = logit_scale.item()
            avg_loss = accum_loss / max(1, step % 10 or 10)

            if main:
                msg = (
                    f"Step {step:>4d}/{args.steps} | "
                    f"loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"scale: {scale_val:.3f} | "
                    f"{elapsed:.1f}s"
                )
                print(msg)
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({"loss": avg_loss, "lr": current_lr, "logit_scale": scale_val, "step": step})

            accum_loss = 0.0
            start_time = time.time()

        if step % args.save_interval == 0 and main:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/projector_step{step}.pt"
            torch.save({"projector": projector.state_dict(), "logit_scale": logit_scale, "args": args}, path)
            print(f"  Saved {path}")

    # === Final save ===
    if main:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/projector_final.pt"
        torch.save({"projector": projector.state_dict(), "logit_scale": logit_scale, "args": args}, path)
        print(f"Saved {path}")
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    print(f"Alignment complete (rank {rank})")


if __name__ == "__main__":
    args = parse_args()
    world_size, rank, ddp_enabled = setup_ddp()
    try:
        train(args, rank=rank, world_size=world_size, ddp_enabled=ddp_enabled)
    finally:
        cleanup_ddp()
