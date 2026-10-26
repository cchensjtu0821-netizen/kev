#!/usr/bin/env python3
"""Convert moshi_seka benchmark files into merged Camera question formats.

Run from the Kev repository root:
    python -m benchmark.convert_moshi_seka_to_kev --data-dir /path/to/data --out-dir /path/to/converted

Each source produces .modes.jsonl (nine questions) and .all.jsonl (two questions).
Questions use the same merger as training. Only annotated tasks have ground truth;
_meta.eval_labels names the question/group answers that may be scored.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile

from kev.camera.labels import MODES, NONE, PALETTE_GROUPS, prepare_record


KEYWORDS = tuple(name for names in PALETTE_GROUPS.values() for name in names)
MODE_SUFFIXES = ("普通", "典型")
MODE_FILES = (
    "doc_test_gt.json",
    "fireworks_test_gt_update0616.json",
    "test_data_6k_260316_processed_CBG_rules_updated_version4.json",
    "gaoxiangsu_test_gt.json",
    "adjust0626.json",
)
SEKA_FILES = (
    "new_seka_test_20260713.jsonl",
    "20260713_seka_test.json",
    "new_seka_test_huahai.jsonl",
    "new_seka_test_camp.jsonl",
    "new_seka_test_bar.jsonl",
)
ALL_FILES = MODE_FILES + SEKA_FILES
IMAGE_DIR = "/srv/workspace/Kirin_AI_Workspace/AIC_I/c00932551/CameraBaseModel/benchmark_lora/moshi_seka"
SEKA_DATA_BASE = "/srv/workspace/Kirin_AI_Workspace/AIC_I/l00883213/camera_agent/hisi_dinov3_llm022/domain_data/seka_moshi_0530/data"
GAOXIANGSU_TEST_DIR = "/srv/workspace/Kirin_AI_Workspace/AIC_I/l00883213/camera_agent/hisi_dinov3_llm022/domain_test/moshi_seka_test/test_data/260415_gaoxiangsu_test"


def remap_image_path(image_path, *, image_dir, seka_data_base=SEKA_DATA_BASE,
                     gaoxiangsu_test_dir=GAOXIANGSU_TEST_DIR):
    """Keep the original benchmark's prefix mapping and resolve relative images."""
    if not image_path:
        return None
    mappings = (
        ("/mnt/15_data_yh/15_data/vlm/seka_wcy/train_data/260415_gaoxiangsu_test", gaoxiangsu_test_dir),
        ("/mnt/15_data_yh/15_data/vlm/seka_wcy/train_data/0519", Path(image_dir) / "0519"),
        ("/mnt/15_data_yh/15_data/vlm/seka_wcy/train_data/20260713sekatest", Path(image_dir) / "20260713sekatest"),
        ("/mnt/15_data_yh/15_data/vlm/seka_wcy/train_data", seka_data_base),
        ("/mnt/15_data_yh/15_data/vlm/result/20260719", Path(image_dir) / "20260719"),
        ("/夜景/l00939411/q00876272/data", seka_data_base),
        ("/home/ma-user/work", "/srv/workspace/Kirin_AI_Workspace"),
    )
    for old, new in mappings:
        if image_path == old or image_path.startswith(old + "/"):
            image_path = str(new).rstrip("/") + image_path[len(old):]
            break
    path = Path(image_path).expanduser()
    return (path if path.is_absolute() else Path(image_dir) / path).resolve()


def load_json_or_jsonl(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    return value if isinstance(value, list) else [value]


def mode_labels(answer):
    labels = set()
    for token in answer.split():
        if token not in MODES:
            for suffix in MODE_SUFFIXES:
                if token.endswith(suffix):
                    token = token[:-len(suffix)]
        if token in MODES:
            labels.add(token)
    return [name for name in MODES if name in labels]


def seka_labels(answer):
    tokens = {token.strip() for token in answer.replace(";", "").split("-")}
    return [name for name in KEYWORDS if name in tokens]


def question_template():
    """Unlabelled 24-question template, including the weather Choice used in training."""
    questions = {
        f"mode_{name}": {"type": "noul", "instructions": f"这幅画面的模式是否为「{name}」？", "src": "lora_seka"}
        for name in MODES
    }
    for group, names in PALETTE_GROUPS.items():
        criteria = dict.fromkeys(names)
        if group != "色调":
            criteria[NONE] = "以上都不是"
        questions[f"seka_{group}"] = {"type": "choice", "instructions": f"这幅画面的{group}标签是哪一项？",
                                     "criteria": criteria, "src": "lora_seka"}
    return questions


def convert_item(item, *, mode_file, image, source, formats=("modes", "all")):
    answer = item.get("answer", "")
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    palette_labels = {}
    if not mode_file:
        present = set(seka_labels(answer))
        for group, names in PALETTE_GROUPS.items():
            selected = [name for name in names if name in present]
            if len(selected) > 1:
                raise ValueError(f"seka_{group} has overlapping labels: {selected}")
            if selected:
                palette_labels[group] = selected[0]
            elif group != "色调":
                palette_labels[group] = NONE
    base = {"id": item.get("original_id", item.get("id")), "image": str(image),
            "state": "", "answer": answer}
    converted = {}
    for layout in formats:
        row = prepare_record({**base, "questions": question_template()}, layout)
        labels = {}
        if mode_file:
            labels["modes"] = mode_labels(answer)
            row["questions"]["modes"]["label"] = labels["modes"]
        elif layout == "all":
            labels["palette"] = dict(palette_labels)
            if len(palette_labels) == len(PALETTE_GROUPS):
                row["questions"]["palette"]["label"] = dict(palette_labels)
        else:
            for group, label in palette_labels.items():
                key = f"seka_{group}"
                labels[key] = label
                row["questions"][key]["label"] = label
        row["_meta"] = {"source": source, "eval_task": "mode" if mode_file else "seka",
                        "question_layout": layout, "eval_labels": labels}
        converted[layout] = row
    return converted


def convert_source(source, *, args):
    items = load_json_or_jsonl(args.data_dir / source)
    targets = {layout: args.out_dir / f"{source}.{layout}.jsonl" for layout in args.formats}
    temporary = {}
    counts = {"items": len(items), "rows": 0, "empty_items": 0, "missing_image": 0, "unlabelled_tone": 0}
    try:
        with ExitStack() as stack:
            streams = {}
            for layout in args.formats:
                stream = stack.enter_context(tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=args.out_dir,
                    prefix=f".{targets[layout].name}.", suffix=".tmp", delete=False))
                streams[layout] = stream
                temporary[layout] = Path(stream.name)
            for index, item in enumerate(items, 1):
                if not item:
                    counts["empty_items"] += 1
                    continue
                try:
                    image = remap_image_path(item.get("image", ""), image_dir=args.image_dir,
                                             seka_data_base=args.seka_data_base,
                                             gaoxiangsu_test_dir=args.gaoxiangsu_test_dir)
                    if image is None or not image.is_file():
                        if args.missing_images == "error":
                            raise FileNotFoundError(f"missing image: {image}")
                        counts["missing_image"] += 1
                        continue
                    rows = convert_item(item, mode_file=source in MODE_FILES, image=image,
                                        source=source, formats=args.formats)
                except (ValueError, TypeError, AttributeError, FileNotFoundError) as error:
                    raise ValueError(f"{source}: item {index}: {error}") from error
                if source in SEKA_FILES and not set(seka_labels(item.get("answer") or "")) & set(PALETTE_GROUPS["色调"]):
                    counts["unlabelled_tone"] += 1
                for layout, row in rows.items():
                    streams[layout].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                counts["rows"] += 1
        for layout, path in temporary.items():
            os.replace(path, targets[layout])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return {**counts, "eval_task": "mode" if source in MODE_FILES else "seka",
            "outputs": {layout: str(path) for layout, path in targets.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "converted")
    parser.add_argument("--image-dir", type=Path, default=Path(os.environ.get("MOSHI_SEKA_IMAGE_DIR", IMAGE_DIR)))
    parser.add_argument("--seka-data-base", type=Path, default=Path(SEKA_DATA_BASE))
    parser.add_argument("--gaoxiangsu-test-dir", type=Path, default=Path(GAOXIANGSU_TEST_DIR))
    parser.add_argument("--formats", nargs="+", choices=("modes", "all"), default=["modes", "all"])
    parser.add_argument("--sources", nargs="+", choices=ALL_FILES, default=list(ALL_FILES))
    parser.add_argument("--missing-images", choices=("skip", "error"), default="skip")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.formats = list(dict.fromkeys(args.formats))
    args.sources = list(dict.fromkeys(args.sources))
    for source in args.sources:
        if not (args.data_dir / source).is_file():
            parser.error(f"missing benchmark file: {args.data_dir / source}")
    outputs = [args.out_dir / f"{source}.{layout}.jsonl" for source in args.sources for layout in args.formats]
    stats_path = args.out_dir / "conversion_stats.json"
    if not args.overwrite and any(path.exists() for path in [*outputs, stats_path]):
        parser.error("output exists; pass --overwrite to replace it")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for source in args.sources:
        stats[source] = convert_source(source, args=args)
        print(f"{source}: {stats[source]['rows']} rows per format, {stats[source]['missing_image']} missing images")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"total rows per format: {sum(value['rows'] for value in stats.values())}")
    return stats


if __name__ == "__main__":
    main()
