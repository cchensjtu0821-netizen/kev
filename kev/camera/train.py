"""Fine-tune a dense CameraAgent checkpoint for Kev image decisions."""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from kev.vision_data import ImageDecisionDataset
from kev.suite import read_json, read_jsonl, write_jsonl

from .data import CameraImageDataset
from .model import CameraDecisionModel, checkpoint_config, load, load_camera
from .resume import (ResumableBatchSampler, check_resume, data_signature, find_resume,
                     read_state, restore_rng, save_checkpoint)


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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="raw CameraAgent checkpoint; initialize a new pointer head")
    source.add_argument("--checkpoint", help="Camera Kev checkpoint; keep the trained backbone and pointer head")
    source.add_argument("--resume", help="complete training checkpoint directory, or latest under --out")
    parser.add_argument("--out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8, help="images per DataLoader batch per GPU")
    parser.add_argument("--accum", type=int, default=2, help="DataLoader batches per optimizer step")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, help="default: checkpoint setting, or 2560 for a raw model")
    parser.add_argument("--image-size", type=int, help="default: checkpoint setting, or 384 for a raw model")
    parser.add_argument("--head-dim", type=int, help="default: checkpoint setting, or 256 for a raw model")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0, help="save and exit after this step, keeping the full LR schedule")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--freeze-mlp", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if min(value for value in (args.epochs, args.batch, args.accum, args.max_length,
                              args.image_size, args.head_dim) if value is not None) < 1:
        parser.error("epochs, batch, accum, max-length, image-size and head-dim must be positive")
    if args.lr <= 0 or args.head_lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("invalid optimizer settings")
    if min(args.max_steps, args.save_every_steps, args.stop_after, args.num_workers) < 0:
        parser.error("max-steps, save-every-steps, stop-after and num-workers must be nonnegative")
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
    if not (args.model or args.checkpoint or args.resume) or not args.out:
        raise ValueError("training needs --model, --checkpoint or --resume, and --out")
    resume_path = find_resume(args.resume, args.out) if args.resume else None
    state = read_state(resume_path) if resume_path else None
    checkpoint = resume_path or args.checkpoint
    source, metadata = checkpoint_config(checkpoint) if checkpoint else (Path(args.model), {})
    for key, default in (("image_size", 384), ("head_dim", 256), ("max_length", 2560)):
        requested = getattr(args, key)
        if requested is None:
            setattr(args, key, metadata.get(key, default))
        elif checkpoint and key in ("image_size", "head_dim") and requested != metadata[key]:
            raise ValueError(f"--{key.replace('_', '-')} must match checkpoint {key}={metadata[key]}")
    if not source.is_dir():
        raise FileNotFoundError(f"CameraAgent checkpoint directory is unavailable: {source}")
    if not checkpoint and ((source / "camera_decision.json").exists() or
                           read_json(source / "config.json").get("kev")):
        raise ValueError("--model is for a raw CameraAgent; use --checkpoint to preserve the trained pointer head")
    if Path(args.out).exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.out}")
    if resume_path and Path(args.out).exists():
        latest = find_resume("latest", args.out)
        if latest.resolve() != resume_path.resolve():
            raise ValueError("resume an older checkpoint into a new --out directory to preserve later checkpoints")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CameraAgent training needs a CUDA GPU with bf16 support")
    from transformers import AutoTokenizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if checkpoint:
        model, _ = load(checkpoint, device=device)
        model.max_length = args.max_length
        camera = model.camera
    else:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)
        camera = load_camera(source)
        model = CameraDecisionModel(camera, tokenizer, head_dim=args.head_dim,
                                    max_length=args.max_length).to(device)
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
    backbone = [p for p in model.camera.parameters() if p.requires_grad]
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW(({"params": backbone, "lr": args.lr},
                                   {"params": head, "lr": args.head_lr}),
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    train_images = CameraImageDataset(train, args.image_size)
    sampler = DistributedSampler(train_images, num_replicas=world, rank=local_rank, shuffle=True,
                                 seed=args.seed)
    batches = ResumableBatchSampler(sampler, args.batch, drop_last=False)
    loader = DataLoader(train_images, batch_sampler=batches,
                        collate_fn=list, num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed),
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    valid_loader = None
    if valid is not None and local_rank == 0:
        valid_loader = DataLoader(CameraImageDataset(valid, args.image_size), batch_size=1,
                                  collate_fn=first_item, num_workers=args.num_workers,
                                  generator=torch.Generator().manual_seed(args.seed),
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    batches_per_epoch = len(loader)
    total_steps = args.epochs * math.ceil(batches_per_epoch / args.accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    signature = data_signature(train, valid)
    step = start_epoch = next_batch = 0
    if state:
        check_resume(state, args, world=world, signature=signature, total_steps=total_steps)
        step, start_epoch, next_batch = state["step"], state["epoch"], state["next_batch"]
        if state["complete"]:
            if local_rank == 0:
                print(json.dumps({"event": "already_complete", "step": step, "total_steps": total_steps}), flush=True)
            if world > 1:
                dist.destroy_process_group()
            return
        if args.stop_after and args.stop_after <= step:
            raise ValueError("--stop-after must exceed the resumed step")
        optimizer.load_state_dict(state.pop("optimizer"))
        scheduler.load_state_dict(state.pop("scheduler"))
        if local_rank == 0:
            metrics = Path(args.out) / "training_metrics.jsonl"
            if metrics.exists():
                write_jsonl(metrics, [row for row in read_jsonl(metrics) if row["epoch"] <= start_epoch])
    initial_step = step
    started = last_step = time.monotonic()
    step_loss = step_images = 0
    if local_rank == 0:
        print(json.dumps({"event": "train_start", "images": len(train_images), "epochs": args.epochs,
                          "progress": f"{step}/{total_steps}", "resume": str(resume_path) if resume_path else None,
                          "gpus": world, "batch_per_gpu": args.batch,
                          "accum": args.accum,
                          "attention_backend": camera.language_model.config._attn_implementation}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        batches.start = next_batch if epoch == start_epoch else 0
        model.train()
        train_loss = state["ranks"][local_rank]["train_loss"] if state and epoch == start_epoch else 0.0
        seen = state["ranks"][local_rank]["seen"] if state and epoch == start_epoch else 0
        epoch_images = len(sampler)
        # Image decoding is deterministic; a dedicated loader generator keeps worker seeds off the model RNG stream.
        if state and epoch == start_epoch:
            restore_rng(state["ranks"][local_rank]["rng"])
        next_index = batches.start
        for index, items in enumerate(loader if step < total_steps else (), start=batches.start):
            next_index = index + 1
            group_images = min(args.batch * args.accum,
                               epoch_images - (index // args.accum) * args.batch * args.accum)
            sync = (index + 1) % args.accum == 0 or index + 1 == batches_per_epoch
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
                                      "eta": str(timedelta(seconds=round(elapsed / (step - initial_step) * (total_steps - step)))),
                                      "backbone_lr": backbone_lr, "head_lr": head_lr}), flush=True)
                    last_step = now
                step_loss = step_images = 0
                stopping = args.stop_after and step >= args.stop_after
                if stopping or (args.save_every_steps and step % args.save_every_steps == 0):
                    save_checkpoint(Path(args.out) / f"checkpoint-{step:07d}", model, optimizer, scheduler,
                                    args=args, source=source, world=world, rank=local_rank,
                                    signature=signature, total_steps=total_steps, step=step, epoch=epoch,
                                    next_batch=index + 1, train_loss=train_loss, seen=seen)
                if stopping:
                    if local_rank == 0:
                        print(json.dumps({"event": "stopped", "step": step}), flush=True)
                    if world > 1:
                        dist.destroy_process_group()
                    return
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
    save_checkpoint(Path(args.out) / "final", model, optimizer, scheduler,
                    args=args, source=source, world=world, rank=local_rank,
                    signature=signature, total_steps=total_steps, step=step, epoch=epoch,
                    next_batch=next_index, train_loss=train_loss, seen=seen, complete=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
