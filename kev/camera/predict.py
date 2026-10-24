"""Score one image request with a trained CameraAgent pointer checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from kev.device import select

from .calibrate import load_thresholds, question_threshold
from .data import load_image_tensor
from .model import load
from .questions import to_record


def answers(record, metadata, probabilities, *, thresholds=None, threshold=0.5):
    result = {}
    for question, meta, p in zip(record["questions"], metadata, probabilities):
        if meta["type"] == "multilabel":
            cutoff = question_threshold(question, thresholds or {}, threshold)
            result[meta["id"]] = {"probabilities": dict(zip(meta["keys"], p)), "threshold": cutoff,
                                   "selected": [key for key, value in zip(meta["keys"], p) if value > cutoff]}
        elif meta["type"] == "grouped_choice":
            groups = {}
            for group in meta["groups"]:
                values = p[group["start"]:group["stop"]]
                groups[group["name"]] = {"choice": group["keys"][max(range(len(values)), key=values.__getitem__)],
                                          "probabilities": dict(zip(group["keys"], values))}
            result[meta["id"]] = groups
        else:
            result[meta["id"]] = dict(zip(meta["keys"], p))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="training output's final or checkpoint directory")
    parser.add_argument("--request", required=True, help="JSON with image, state and questions")
    parser.add_argument("--calibration", help="threshold JSON from kev.camera.calibrate")
    parser.add_argument("--threshold", type=float, default=0.5, help="fallback multi-label threshold")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    image_path = Path(request.pop("image"))
    if not image_path.is_absolute():
        image_path = request_path.resolve().parent / image_path
    device = torch.device(select(args.device))
    thresholds = load_thresholds(args.calibration, args.checkpoint) if args.calibration else {}
    model, config = load(args.checkpoint, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    record, metadata = to_record(request, layout=config.get("question_layout", "original"))
    pixels = load_image_tensor(image_path, config["image_size"]).to(
        device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        probs = [p.tolist() for p in model.probabilities(record, pixels)]
    result = answers(record, metadata, probs, thresholds=thresholds, threshold=args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
