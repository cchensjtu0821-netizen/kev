"""Fit one multi-label decision threshold per question on held-out images."""

import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kev.device import select
from kev.suite import digest, read_json, write_json
from kev.vision_data import ImageDecisionDataset

from .data import CameraImageDataset
from .model import checkpoint_config, load
from .questions import materialize
from .train import first_item, predictions


def micro_f1(tp, predicted, positives):
    return 2 * tp / (predicted + positives) if predicted + positives else 1.0


def fit_threshold(probabilities, targets):
    """Exact micro-F1 sweep with tied scores kept together; selection uses p > threshold."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(targets).ravel()
    if not len(p) or p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("calibration needs matching, finite probabilities in [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("calibration targets must be binary")
    positives = int(y.sum())
    selected = p > 0.5
    best = (micro_f1(int(y[selected].sum()), int(selected.sum()), positives), 0.0, 0.5)
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    cumulative = np.cumsum(y)
    # At an observed probability t, all scores equal to t are excluded.
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    for index in starts:
        threshold = float(p[index])
        tp = int(cumulative[index - 1]) if index else 0
        candidate = (micro_f1(tp, int(index), positives), -abs(threshold - 0.5), threshold)
        best = max(best, candidate)
    if p[-1] > 0:
        best = max(best, (micro_f1(positives, len(p), positives), -0.5, 0.0))
    return {"threshold": best[2], "calibration_micro_f1": best[0], "positive_labels": positives}


def checkpoint_identity(path):
    path = Path(path)
    base, metadata = checkpoint_config(path)
    files = {"config": base / "config.json", "head": path / metadata.get("head_weights", "pointer.pt")}
    adapter = path / "llm_adapter" / "adapter_model.safetensors"
    if adapter.is_file():
        files["adapter"] = adapter
    return {name: digest(file) for name, file in files.items()}


def load_thresholds(path, checkpoint):
    artifact = read_json(path)
    if artifact.get("format") != 1 or artifact.get("comparison") != ">":
        raise ValueError("unsupported threshold calibration format")
    if artifact["checkpoint_identity"] != checkpoint_identity(checkpoint):
        raise ValueError("threshold calibration belongs to a different checkpoint")
    return artifact["questions"]


def question_threshold(question, thresholds, default=0.5):
    value = thresholds.get(question["qid"])
    if value is not None:
        if value["keys"] != question["keys"] or value["instructions"] != question["instr"]:
            raise ValueError(f"calibration schema changed for {question['qid']}")
        default = value["threshold"]
    if not math.isfinite(default) or not 0 <= default <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    return default


def calibrate(predicted):
    collected = {}
    for question, probabilities in predicted:
        if question.get("qtype") != "multilabel":
            continue
        qid = question["qid"]
        schema = {"keys": question["keys"], "instructions": question["instr"]}
        entry = collected.setdefault(qid, {**schema, "probabilities": [], "targets": []})
        if any(entry[key] != value for key, value in schema.items()):
            raise ValueError(f"calibration schema varies between images for {qid}")
        entry["probabilities"].append(probabilities.tolist())
        entry["targets"].append(question["label"])
    if not collected:
        raise ValueError("validation has no multi-label questions to calibrate")
    return {qid: {"keys": entry["keys"], "instructions": entry["instructions"],
                  "images": len(entry["targets"]), **fit_threshold(entry["probabilities"], entry["targets"])}
            for qid, entry in collected.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    valid = parser.add_mutually_exclusive_group(required=True)
    valid.add_argument("--val-data")
    valid.add_argument("--val-meta")
    parser.add_argument("--out", required=True, help="new calibration JSON file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    _, config = checkpoint_config(args.checkpoint)
    records = ImageDecisionDataset(data=args.val_data, meta=args.val_meta,
                                  materializer=partial(materialize, layout=config.get("question_layout", "original")))
    if any(count != len(offsets) for _, offsets, count in records.sources):
        raise ValueError("calibration repeat_time must be 1")
    # Reuse the image-disjoint check when this machine still has the training annotations.
    training = config.get("training", {}).get("args", {})
    training_path = training.get("train_data") or training.get("train_meta")
    if training_path and Path(training_path).is_file():
        train = ImageDecisionDataset(data=training.get("train_data"), meta=training.get("train_meta"),
                                     materializer=materialize)
        if {train[i]["image"] for i in range(len(train))} & {records[i]["image"] for i in range(len(records))}:
            raise ValueError("calibration and training share image paths")
    device = torch.device(select(args.device))
    model, _ = load(args.checkpoint, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    loader = DataLoader(CameraImageDataset(records, config["image_size"]), batch_size=1,
                        collate_fn=first_item, num_workers=args.num_workers)
    questions = calibrate(predictions(model, loader, device=device))
    artifact = {"format": 1, "comparison": ">", "objective": "micro_f1",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_identity": checkpoint_identity(args.checkpoint),
                "validation": [{"annotation": str(source[1].resolve()), "sha256": digest(source[1])}
                               for source, _, _ in records.sources], "questions": questions}
    write_json(args.out, artifact, atomic=True)
    print(f"saved thresholds for {len(questions)} multi-label questions to {args.out}")


if __name__ == "__main__":
    main()
