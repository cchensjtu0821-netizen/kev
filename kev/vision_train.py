"""Train a ViT bridge on labelled image decisions."""

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .device import DEVICES, DEVICE_HELP, select
from .model import SERVE_MAX_STATE_8K, training_context
from .vision import VisionDecisionModel
from .vision_data import ImageDecisionDataset


def score(model, tok, dataset):
    from PIL import Image

    model.eval()
    correct = total = 0
    brier = 0.0
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            enc = model.encode(tok, item["record"])
            with Image.open(item["image"]) as image:
                probs = model.probs(enc, model.pixels(image))
            for question, p in zip(item["record"]["questions"], probs):
                label = question["label"]
                correct += int(p.argmax().item() == label)
                brier += float(((p - F.one_hot(torch.tensor(label), p.numel())) ** 2).sum())
                total += 1
    return {"questions": total, "accuracy": correct / total, "brier": brier / total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    train = parser.add_mutually_exclusive_group(required=True)
    train.add_argument("--train-data", help="labelled JSONL with image paths relative to its directory")
    train.add_argument("--train-meta", help="CameraModel-style dataset collection JSON")
    parser.add_argument("--val-data", help="separate labelled JSONL for validation")
    parser.add_argument("--val-meta", help="separate CameraModel-style collection JSON")
    parser.add_argument("--validate-data", action="store_true", help="check records and image paths without loading weights")
    parser.add_argument("--text-checkpoint", help="released Kev checkpoint or local checkpoint")
    parser.add_argument("--vision-model", default="facebook/dinov2-base", help="Hugging Face ViT encoder")
    parser.add_argument("--vision-revision", default=None)
    parser.add_argument("--visual-tokens", type=int, default=16)
    parser.add_argument("--max-state", type=int, default=SERVE_MAX_STATE_8K, help="text state plus image tokens; reject longer records")
    parser.add_argument("--train-vision", action="store_true", help="fine-tune the ViT; default freezes it")
    parser.add_argument("--train-text", action="store_true", help="train the source LoRA and pointer head (adapter checkpoints only)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    parser.add_argument("--out", help="new directory for the visual bridge")
    args = parser.parse_args(argv)
    if bool(args.val_data) == bool(args.val_meta):
        parser.error("pass exactly one of --val-data or --val-meta")
    train_data = ImageDecisionDataset(data=args.train_data, meta=args.train_meta)
    val_data = ImageDecisionDataset(data=args.val_data, meta=args.val_meta)
    train_files = {info[1].resolve() for info, _, _ in train_data.sources}
    val_files = {info[1].resolve() for info, _, _ in val_data.sources}
    if train_files & val_files:
        parser.error("train and validation collections share an annotation file")
    if any(count != len(offsets) for _, offsets, count in val_data.sources):
        parser.error("validation repeat_time must be 1")
    if args.validate_data:
        for dataset in (train_data, val_data):
            for index in range(len(dataset)):
                dataset[index]
        print(json.dumps({"train": len(train_data), "validation": len(val_data)}, indent=2))
        return
    if not args.text_checkpoint or not args.out:
        parser.error("training needs --text-checkpoint and --out")
    if args.epochs < 1 or args.accum < 1 or args.lr <= 0 or args.visual_tokens < 1 or args.max_steps < 0:
        parser.error("epochs, accum, lr and visual-tokens must be positive; max-steps must be nonnegative")
    context = training_context(args.max_state)
    if args.visual_tokens >= context["max_state"]:
        parser.error("visual-tokens must be smaller than max-state")
    out = Path(args.out)
    if out.exists():
        parser.error(f"output already exists: {out}")
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from .checkpoint import Checkpoint, LoadOptions

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    checkpoint = Checkpoint(args.text_checkpoint)
    if args.train_text and checkpoint.full:
        parser.error("--train-text needs an adapter text checkpoint; full-weight 27B supports the projector stage")
    device = select(args.device)
    tok, decision = checkpoint.load(device, LoadOptions(merge=False))
    decision.requires_grad_(False)
    decision.head.temperature = 1.0
    if args.train_text:
        adapter_params = [p for name, p in decision.lm.named_parameters()
                          if "lora_" in name or "trainable_tokens" in name or "modules_to_save" in name]
        if not adapter_params:
            raise ValueError("checkpoint has no trainable adapter tensors")
        for p in adapter_params:
            p.requires_grad_(True)
        decision.head.requires_grad_(True)
    vision = AutoModel.from_pretrained(args.vision_model, revision=args.vision_revision)
    processor = AutoImageProcessor.from_pretrained(args.vision_model, revision=args.vision_revision, use_fast=False)
    model = VisionDecisionModel(decision, vision, processor, visual_tokens=args.visual_tokens,
                                train_vision=args.train_vision)
    model.max_state = args.max_state
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    planned_steps = args.epochs * math.ceil(len(loader) / args.accum)
    if args.max_steps:
        planned_steps = min(planned_steps, args.max_steps)
    metrics = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        if not args.train_text:
            decision.eval()
        loss_sum = 0.0
        for batch_index, item in enumerate(loader):
            enc = model.encode(tok, item["record"])
            with Image.open(item["image"]) as image:
                pixels = model.pixels(image)
            bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
            with autocast:
                logits = model(enc, pixels)
                loss = torch.stack([F.cross_entropy(z.unsqueeze(0), torch.tensor([q["label"]], device=z.device))
                                    for z, q in zip(logits, item["record"]["questions"])]).mean()
            remaining = len(loader) - (batch_index // args.accum) * args.accum
            (loss / min(args.accum, remaining)).backward()
            loss_sum += loss.item()
            if (batch_index + 1) % args.accum == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step >= planned_steps:
                    break
        result = {"epoch": epoch + 1, "steps": step, "train_loss": loss_sum / (batch_index + 1),
                  "validation": score(model, tok, val_data)}
        metrics.append(result)
        print(json.dumps(result))
        if step >= planned_steps:
            break
    model.save_bridge(out, text_checkpoint=args.text_checkpoint, vision_name=args.vision_model,
                      vision_revision=args.vision_revision, train_text=args.train_text)
    (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
