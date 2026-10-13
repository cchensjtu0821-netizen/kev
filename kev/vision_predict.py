"""Score one image and a Kev request through a trained visual bridge."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, help="directory written by kev.vision_train")
    parser.add_argument("--request", required=True, help="JSON with image, state and Kev questions")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    from PIL import Image

    from .api import SystemOneRequest, to_record
    from .vision import load_vision_model

    request_path = Path(args.request)
    source = json.loads(request_path.read_text(encoding="utf-8"))
    image_path = (request_path.resolve().parent / source.pop("image")).resolve()
    record, metadata = to_record(SystemOneRequest.model_validate(source))
    tok, model = load_vision_model(args.bridge, args.device)
    enc = model.encode(tok, record)
    with Image.open(image_path) as image:
        probabilities = model.probs(enc, model.pixels(image))
    print(json.dumps({m["id"]: dict(zip(m["keys"], p.tolist())) for m, p in zip(metadata, probabilities)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
