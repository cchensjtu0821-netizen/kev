"""Fine-tune a dense CameraAgent checkpoint for Kev image decisions."""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from kev.vision_data import ImageDecisionDataset

from .data import CameraImageDataset
from .model import CameraDecisionModel, load_camera


def datasets(args):
    train = ImageDecisionDataset(data=args.train_data, meta=args.train_meta)
    train_images = {train[i]["image"] for i in range(len(train))}
    if not (args.val_data or args.val_meta):
        return train, None
    valid = ImageDecisionDataset(data=args.val_data, meta=args.val_meta)
    train_files = {source[1].resolve() for source, _, _ in train.sources}
    valid_files = {source[1].resolve() for source, _, _ in valid.sources}
    if train_files & valid_files:
        raise ValueError("train and validation share an annotation file")
    if any(count != len(offsets) for _, offsets, count in valid.sources):
        raise ValueError("validation repeat_time must be 1")
    valid_images = {valid[i]["image"] for i in range(len(valid))}
    overlap = train_images & valid_images
    if overlap:
        raise ValueError(f"train and validation share {len(overlap)} image paths, for example {next(iter(overlap))}")
    return train, valid


def first_item(items):
    return items[0]


def evaluate(model, loader, *, device):
    model.eval()
    correct = count = 0
    brier = 0.0
    with torch.inference_mode():
        for item in loader:
            pixels = item["pixels"].to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                       non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model([item["record"]], pixels)[0]
            for question, z in zip(item["record"]["questions"], logits):
                p = z.float().softmax(-1)
                label = question["label"]
                correct += int(p.argmax().item() == label)
                brier += float(((p - F.one_hot(torch.tensor(label, device=device), p.numel())) ** 2).sum())
                count += 1
    return {"questions": count, "accuracy": correct / count, "brier": brier / count}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    train = parser.add_mutually_exclusive_group(required=True)
    train.add_argument("--train-data")
    train.add_argument("--train-meta")
    valid = parser.add_mutually_exclusive_group()
    valid.add_argument("--val-data")
    valid.add_argument("--val-meta")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--model", help="local CameraAgent checkpoint directory")
    parser.add_argument("--out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8, help="images per DataLoader batch per GPU")
    parser.add_argument("--accum", type=int, default=2, help="DataLoader batches per optimizer step")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--freeze-mlp", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.accum, args.max_length, args.image_size, args.head_dim) < 1:
        parser.error("epochs, batch, accum, max-length, image-size and head-dim must be positive")
    if args.lr <= 0 or args.head_lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("invalid optimizer settings")
    if args.max_steps < 0 or args.save_every_steps < 0 or args.num_workers < 0:
        parser.error("max-steps, save-every-steps and num-workers must be nonnegative")
    return args


def train_batch(wrapped, items, *, device, group_images, sync_at_end):
    """Forward the image batch together and average each image's questions."""
    pixels = torch.cat([item["pixels"] for item in items]).to(
        device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        non_blocking=True)
    records = [item["record"] for item in items]
    context = nullcontext() if sync_at_end or not isinstance(wrapped, DistributedDataParallel) else wrapped.no_sync()
    with context, torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
        logits = wrapped(records, pixels)
        losses = []
        for record, image_logits in zip(records, logits):
            question_losses = [F.cross_entropy(z.float().unsqueeze(0),
                                               torch.tensor([question["label"]], device=device))
                               for question, z in zip(record["questions"], image_logits)]
            losses.append(torch.stack(question_losses).mean())
        loss = torch.stack(losses).sum() / group_images
    loss.backward()
    return float(loss.detach()) * group_images


def main(argv=None):
    args = parse_args(argv)
    train, valid = datasets(args)
    if args.validate_data:
        print(json.dumps({"train_images": len(train), "validation_images": len(valid) if valid is not None else 0}))
        return
    if not args.model or not args.out:
        raise ValueError("training needs --model and --out")
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"CameraAgent checkpoint directory is unavailable: {args.model}")
    if Path(args.out).exists():
        raise FileExistsError(f"output already exists: {args.out}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CameraAgent training needs a CUDA GPU with bf16 support")
    from transformers import AutoTokenizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    camera = load_camera(args.model)
    config = camera.config
    if getattr(config, "model_type", None) != "camera_agent":
        raise ValueError("--model must point to a camera_agent checkpoint")
    if getattr(config, "template", None) != "internvl2_5" or getattr(config, "ps_version", None) != "v2":
        raise ValueError("this adapter expects CameraAgent internvl2_5 with ps_version v2")
    if args.image_size != getattr(config, "force_image_size", None):
        raise ValueError(f"image-size must match checkpoint force_image_size={config.force_image_size}")
    camera.config.use_cache = False
    camera.language_model.config.use_cache = False
    output_embeddings = camera.language_model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.requires_grad_(False)
    if hasattr(camera.language_model, "gradient_checkpointing_enable"):
        camera.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    for flag, name in ((args.freeze_backbone, "vision_model"), (args.freeze_llm, "language_model"),
                       (args.freeze_mlp, "mlp1")):
        if flag:
            if not hasattr(camera, name):
                raise ValueError(f"--freeze-{name} requested but checkpoint has no {name}")
            getattr(camera, name).requires_grad_(False)
    model = CameraDecisionModel(camera, tokenizer, head_dim=args.head_dim,
                                max_length=args.max_length).to(device)
    backbone = [p for p in model.camera.parameters() if p.requires_grad]
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW(({"params": backbone, "lr": args.lr},
                                   {"params": head, "lr": args.head_lr}),
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    train_images = CameraImageDataset(train, args.image_size)
    sampler = DistributedSampler(train_images, num_replicas=world, rank=local_rank, shuffle=True,
                                 seed=args.seed) if world > 1 else None
    loader = DataLoader(train_images, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                        collate_fn=list, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    valid_loader = None
    if valid is not None and local_rank == 0:
        valid_loader = DataLoader(CameraImageDataset(valid, args.image_size), batch_size=1,
                                  collate_fn=first_item, num_workers=args.num_workers,
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    total_steps = args.epochs * math.ceil(len(loader) / args.accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    step = 0
    started = last_step = time.monotonic()
    step_loss = step_images = 0
    if local_rank == 0:
        print(json.dumps({"event": "train_start", "images": len(train_images), "epochs": args.epochs,
                          "progress": f"0/{total_steps}", "gpus": world, "batch_per_gpu": args.batch,
                          "accum": args.accum,
                          "attention_backend": camera.language_model.config._attn_implementation}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        seen = 0
        epoch_images = len(sampler) if sampler is not None else len(train_images)
        for index, items in enumerate(loader):
            group_images = min(args.batch * args.accum,
                               epoch_images - (index // args.accum) * args.batch * args.accum)
            sync = (index + 1) % args.accum == 0 or index + 1 == len(loader)
            batch_loss = train_batch(wrapped, items, device=device,
                                     group_images=group_images, sync_at_end=sync)
            train_loss += batch_loss
            step_loss += batch_loss
            step_images += len(items)
            seen += len(items)
            if sync:
                backbone_lr, head_lr = (group["lr"] for group in optimizer.param_groups)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stats = torch.tensor((step_loss, step_images), device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(stats)
                if local_rank == 0:
                    now = time.monotonic()
                    elapsed = now - started
                    print(json.dumps({"event": "train_progress", "epoch": epoch + 1,
                                      "progress": f"{step}/{total_steps}",
                                      "progress_pct": round(100 * step / total_steps, 1),
                                      "loss": round((stats[0] / stats[1]).item(), 6),
                                      "images_per_second": round(stats[1].item() / max(now - last_step, 1e-9), 2),
                                      "elapsed": str(timedelta(seconds=round(elapsed))),
                                      "eta": str(timedelta(seconds=round(elapsed / step * (total_steps - step)))),
                                      "backbone_lr": backbone_lr, "head_lr": head_lr}), flush=True)
                    last_step = now
                step_loss = step_images = 0
                if args.save_every_steps and step % args.save_every_steps == 0:
                    if world > 1:
                        dist.barrier()
                    if local_rank == 0:
                        model.save(Path(args.out) / f"checkpoint-{step:07d}",
                                   source=args.model, image_size=args.image_size)
                    if world > 1:
                        dist.barrier()
                if step >= total_steps:
                    break
        if world > 1:
            stats = torch.tensor((train_loss, seen), device=device)
            dist.all_reduce(stats)
            train_loss, seen = stats.tolist()
            dist.barrier()
        if local_rank == 0:
            result = {"epoch": epoch + 1, "step": step, "train_loss": train_loss / seen}
            if valid is not None:
                result["validation"] = evaluate(model, valid_loader, device=device)
            print(json.dumps(result), flush=True)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            with (out / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
        if world > 1:
            dist.barrier()
        if step >= total_steps:
            break
    if local_rank == 0:
        model.save(Path(args.out) / "final", source=args.model, image_size=args.image_size)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
