"""Score one image request with a trained CameraAgent pointer checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from kev.api import SystemOneRequest, to_record

from .model import image_tensor, load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="training output's final or checkpoint directory")
    parser.add_argument("--request", required=True, help="JSON with image, state and questions")
    parser.add_argument("--questions-per-pass", type=int, default=4)
    args = parser.parse_args(argv)
    if args.questions_per_pass < 1:
        parser.error("questions-per-pass must be positive")
    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    image_path = Path(request.pop("image"))
    if not image_path.is_absolute():
        image_path = request_path.resolve().parent / image_path
    record, metadata = to_record(SystemOneRequest.model_validate(request))
    model, config = load(args.checkpoint)
    with Image.open(image_path) as image:
        pixels = image_tensor(image, config["image_size"]).to(device="cuda", dtype=torch.bfloat16)
    probs = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(record["questions"]), args.questions_per_pass):
            logits = model(record, pixels, question_start=start,
                           question_end=start + args.questions_per_pass)
            probs.extend([z.float().softmax(-1).tolist() for z in logits])
    result = {item["id"]: dict(zip(item["keys"], p)) for item, p in zip(metadata, probs)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
