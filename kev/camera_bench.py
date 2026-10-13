"""CameraModel MMBench/AesBench text-proxy evaluation for Kev.

Kev checkpoints only score text. An independently produced, image-hash-bound
description is required for every image; the result is a two-stage metric, not
a direct multimodal benchmark score.
"""

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


DATASETS = {
    "mmbench_cn": ("A", "B", "C", "D", "E"),
    "mmbench_en": ("A", "B", "C", "D", "E"),
    "aesbench_val": ("A", "B", "C", "D"),
}


def _tsv_rows(path):
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"index", "image", "question", "answer", "A", "B"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing TSV columns {sorted(required - set(reader.fieldnames or []))}")
        yield from reader


def _image_sha256(image):
    if not isinstance(image, str):
        raise ValueError("missing base64 image")
    try:
        blob = base64.b64decode(image, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image") from exc
    if not blob:
        raise ValueError("empty image")
    return hashlib.sha256(blob).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence(path):
    """Index image descriptions by dataset/index, rejecting ambiguous entries."""
    evidence = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item.get("dataset"), str(item.get("index")))
            if (key[0] not in DATASETS or item.get("index") is None
                    or not isinstance(item.get("image_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", item["image_sha256"]) is None
                    or not isinstance(item.get("description"), str) or not item["description"].strip()):
                raise ValueError(f"{path}:{line_no}: require dataset, index, image_sha256, nonempty description")
            if key in evidence:
                raise ValueError(f"{path}:{line_no}: duplicate evidence for {key}")
            evidence[key] = item
    if not evidence:
        raise ValueError(f"{path}: no image descriptions")
    return evidence


def question_from_tsv(row, dataset):
    options = {letter: row[letter].strip() for letter in DATASETS[dataset]
               if row.get(letter) and row[letter].strip()}
    if len(options) < 2:
        raise ValueError(f"fewer than two choices for {dataset}/{row.get('index')}")
    answer = (row.get("answer") or "").strip()
    if answer not in options:
        raise ValueError(f"invalid answer {answer!r} for {dataset}/{row.get('index')}")
    question = (row.get("question") or "").strip()
    if dataset.startswith("mmbench") and row.get("hint") and row["hint"].strip():
        question = row["hint"].strip() + "\n" + question
    if not question:
        raise ValueError(f"empty question for {dataset}/{row.get('index')}")
    category = (row.get("category") or "").strip() if dataset == "aesbench_val" else None
    if dataset == "aesbench_val" and category not in {"AesA", "AesE", "AesP"}:
        raise ValueError(f"unknown AesBench category {category!r} for {row.get('index')}")
    return {"type": "choice", "instructions": question, "criteria": options,
            "label": answer, "src": dataset}, category


def build_requests(sources, evidence):
    """Return Kev requests and a row map, with no image bytes in either output."""
    requests, rows, seen = [], [], set()
    for dataset, path in sources.items():
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        for row in _tsv_rows(path):
            index = str(row["index"])
            if not index.strip():
                raise ValueError(f"{path}: empty index")
            key = (dataset, index)
            if key in seen:
                raise ValueError(f"{path}: duplicate index {index}")
            seen.add(key)
            if key not in evidence:
                raise ValueError(f"{path}: missing image description for {dataset}/{index}")
            image_hash = _image_sha256(row["image"])
            item = evidence[key]
            if item["image_sha256"] != image_hash:
                raise ValueError(f"{path}: image hash mismatch for {dataset}/{index}")
            question, category = question_from_tsv(row, dataset)
            requests.append({
                "state": {"image_description": item["description"].strip()},
                "questions": {"answer": question},
            })
            rows.append({"dataset": dataset, "index": index, "category": category,
                         "image_sha256": image_hash})
    extra = set(evidence) - seen
    if extra:
        raise ValueError(f"image descriptions have {len(extra)} unused entries, first: {sorted(extra)[0]}")
    if not requests:
        raise ValueError("no benchmark rows")
    return requests, rows


def summarize(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("prediction count differs from request count")
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for row, pred in zip(rows, predictions):
        group = row["dataset"]
        groups[group]["total"] += 1
        groups[group]["correct"] += int(pred["correct"])
        if row["category"]:
            sub = f"{group}/{row['category']}"
            groups[sub]["total"] += 1
            groups[sub]["correct"] += int(pred["correct"])
    return {group: {**counts, "accuracy": counts["correct"] / counts["total"]}
            for group, counts in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in DATASETS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="JSONL of dataset, index, image_sha256, description; descriptions must be generated independently of the answers")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--run", help="local Kev checkpoint or Hub model id")
    model.add_argument("--remote", help="System One endpoint URL")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    if not args.prepare_only and not (args.run or args.remote):
        parser.error("scoring requires --run or --remote")
    sources = {name: getattr(args, name) for name in DATASETS if getattr(args, name)}
    if not sources:
        parser.error("give at least one benchmark TSV")
    evidence = load_evidence(args.evidence)
    requests, row_map = build_requests(sources, evidence)
    args.out.mkdir(parents=True, exist_ok=False)
    request_file = args.out / "requests.jsonl"
    with request_file.open("w", encoding="utf-8") as output:
        for request in requests:
            output.write(json.dumps(request, ensure_ascii=False) + "\n")
    (args.out / "row_map.json").write_text(json.dumps(row_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"evaluation": "text_proxy_two_stage", "direct_multimodal": False,
                "evidence_sha256": _file_sha256(args.evidence),
                "inputs": {name: {"path": str(path), "sha256": _file_sha256(path)}
                           for name, path in sources.items()}, "records": len(requests),
                "model": args.run or args.remote, "prepared_only": args.prepare_only}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(requests)} text-proxy requests in {args.out}")
        return

    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.data import load_records
    from kev.device import select
    from kev.predictors import LocalPredictor, RemotePredictor
    from kev.suite import CONTEXT

    records = load_records(request_file, source="camera_bench")
    if args.remote:
        predictor = RemotePredictor(args.remote, "kev-latest", os.environ.get("KEV_REMOTE_API_KEY", "local"))
    else:
        options = LoadOptions.from_env()
        device = select(args.device, dtype=options.dtype) if options.dtype is not None else select(args.device)
        predictor = LocalPredictor(args.run, device, options, context=CONTEXT)
    report, scored = evaluate_records(records, predictor, args.out / "kev", skip_overlong=True)
    if report["coverage"]["rejected_records"]:
        raise RuntimeError("some questions exceeded Kev context; inspect kev/rejected.json before reporting accuracy")
    predictions = [{"correct": int(row["keys"][max(range(len(row["p"])), key=row["p"].__getitem__)] == row["keys"][row["label"]])}
                   for row in scored]
    result = {"evaluation": "text_proxy_two_stage", "direct_multimodal": False,
              "metrics": summarize(row_map, predictions), "coverage": report["coverage"]}
    (args.out / "camera_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
