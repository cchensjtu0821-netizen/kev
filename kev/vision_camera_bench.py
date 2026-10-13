"""Score a visual Kev bridge directly on CameraModel's MMBench/AesBench TSVs."""

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from .camera_bench import _file_sha256, _tsv_rows, question_from_tsv


DATASETS = {"mmbench_cn": "mmbench-cn", "mmbench_en": "mmbench-en", "aesbench_val": "aesbench-val"}


def request_from_tsv(row, dataset):
    question, _ = question_from_tsv(row, dataset)
    return {"state": "", "questions": {"answer": question}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, flag in DATASETS.items():
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = {name: getattr(args, name) for name in DATASETS if getattr(args, name)}
    if not sources:
        parser.error("give at least one benchmark TSV")
    from PIL import Image

    from .data import materialize
    from .vision import load_vision_model

    tok, model = load_vision_model(args.bridge, args.device)
    rows = []
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for dataset, path in sources.items():
        seen = set()
        for row in _tsv_rows(path):
            index = str(row["index"])
            if not index.strip():
                raise ValueError(f"{path}: empty index")
            if index in seen:
                raise ValueError(f"{path}: duplicate index {index}")
            seen.add(index)
            record = materialize(request_from_tsv(row, dataset))
            blob = base64.b64decode(row["image"], validate=True)
            with Image.open(BytesIO(blob)) as image:
                probabilities = model.probs(model.encode(tok, record), model.pixels(image))[0].tolist()
            keys = record["questions"][0]["keys"]
            prediction = keys[max(range(len(probabilities)), key=probabilities.__getitem__)]
            answer = record["questions"][0]["keys"][record["questions"][0]["label"]]
            correct = int(prediction == answer)
            category = (row.get("category") or "").strip() if dataset == "aesbench_val" else None
            groups = [dataset] + ([f"{dataset}/{category}"] if category else [])
            for group in groups:
                counts[group]["correct"] += correct
                counts[group]["total"] += 1
            rows.append({"dataset": dataset, "index": index, "category": category,
                         "image_sha256": hashlib.sha256(blob).hexdigest(), "answer": answer,
                         "prediction": prediction, "probabilities": dict(zip(keys, probabilities))})
    report = {"evaluation": "direct_multimodal", "bridge": str(args.bridge),
              "inputs": {name: {"path": str(path), "sha256": _file_sha256(path)} for name, path in sources.items()},
              "metrics": {name: {**numbers, "accuracy": numbers["correct"] / numbers["total"]}
                          for name, numbers in counts.items()}}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
