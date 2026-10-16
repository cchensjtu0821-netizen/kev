"""Convert 16 mode and 23 palette Noul questions into 24 image questions.

Usage: python scripts/convert_vision_labels.py input.jsonl output.jsonl
"""

import argparse
import json
import os
import tempfile
from pathlib import Path


MODES = (
    "雨景", "剪影", "全景", "夜景", "舞台", "原生", "球类", "设施",
    "车流", "月亮", "动物", "博物馆", "小烟花", "大烟花", "仙女棒", "文档",
)
PALETTE_GROUPS = {
    "天空": ("蓝色系天空", "冷灰系天空", "暖色系天空", "冷暖混合天空"),
    "植物": ("绿色植物", "红黄橙植物"),
    "天气": ("阴雨",),
    "室内外": ("室内", "室外", "窗边"),
    "时间": ("白天", "夕阳", "蓝调", "夜晚"),
    "建筑": ("历史建筑", "其他建筑"),
    "色调": ("暖调", "冷调", "中性色调", "混合色调"),
    "其他": ("美食", "花卉", "古装人物"),
}
NONE = "无"


def convert_record(row, *, native_conflict="error"):
    if native_conflict not in {"error", "drop-native", "drop-others"}:
        raise ValueError(f"invalid native conflict policy: {native_conflict}")
    questions = row.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("questions must be an object")
    food_key = "seka_美食"
    alias_key = "seka_餐饮美食"
    if food_key in questions and alias_key in questions:
        raise ValueError("both seka_美食 and seka_餐饮美食 exist; resolve their labels first")
    if food_key not in questions and alias_key not in questions:
        raise ValueError("missing seka_美食 or seka_餐饮美食")

    expected = {f"mode_{name}" for name in MODES}
    expected.update(f"seka_{name}" for names in PALETTE_GROUPS.values() for name in names)
    if alias_key in questions:
        expected.remove(food_key)
        expected.add(alias_key)
    missing, extra = expected - questions.keys(), questions.keys() - expected
    if missing or extra:
        raise ValueError(f"expected 39 questions; missing={sorted(missing)}, extra={sorted(extra)}")
    for key, question in questions.items():
        if not isinstance(question, dict) or question.get("type") != "noul" or type(question.get("label")) is not bool:
            raise ValueError(f"{key}: expected a Noul question with a boolean label")

    output = {key: value for key, value in row.items() if key != "questions"}
    converted = {f"mode_{name}": dict(questions[f"mode_{name}"]) for name in MODES}
    active_modes = [name for name in MODES if converted[f"mode_{name}"]["label"]]
    if "原生" in active_modes and len(active_modes) > 1:
        if native_conflict == "error":
            raise ValueError(f"mode_原生 conflicts with {active_modes}; inspect the image or select --native-conflict")
        if native_conflict == "drop-native":
            converted["mode_原生"]["label"] = False
        else:
            for name in active_modes:
                if name != "原生":
                    converted[f"mode_{name}"]["label"] = False
    active_modes = [name for name in MODES if converted[f"mode_{name}"]["label"]]
    for exclusive in ("动物", "博物馆", "文档"):
        if exclusive in active_modes and len(active_modes) > 1:
            raise ValueError(f"mode_{exclusive} conflicts with {active_modes}")

    for group, names in PALETTE_GROUPS.items():
        keys = [alias_key if name == "美食" and alias_key in questions else f"seka_{name}" for name in names]
        selected = [name for name, key in zip(names, keys) if questions[key]["label"]]
        if len(selected) > 1:
            raise ValueError(f"seka_{group} has conflicting labels: {selected}")
        if group == "色调" and not selected:
            raise ValueError("seka_色调 requires exactly one label")
        sources = {questions[key].get("src") for key in keys}
        if len(sources) > 1:
            raise ValueError(f"seka_{group} has inconsistent src values: {sources}")
        src = next(iter(sources))
        if group == "天气":
            question = dict(questions[keys[0]])
            converted["seka_阴雨"] = question
            continue
        criteria = {name: None for name in names}
        if group != "色调":
            criteria[NONE] = "以上都不是"
        choice = {
            "type": "choice",
            "instructions": f"这幅画面的{group}标签是哪一项？",
            "criteria": criteria,
            "label": selected[0] if selected else NONE,
        }
        if src is not None:
            choice["src"] = src
        converted[f"seka_{group}"] = choice

    assert len(converted) == 24
    output["questions"] = converted
    return output


def convert_file(source, destination, *, native_conflict="error", overwrite=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output must be different files")
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("put output beside input so relative image paths still point to the same files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    count = 0
    try:
        with source.open(encoding="utf-8") as stream, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as output:
            temp_name = output.name
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row must be an object")
                    converted = convert_record(row, native_conflict=native_conflict)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
        if not count:
            raise ValueError(f"{source}: no records")
        os.replace(temp_name, destination)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source JSONL containing 39 Noul questions per image")
    parser.add_argument("output", type=Path, help="converted JSONL; must differ from input")
    parser.add_argument("--native-conflict", choices=("error", "drop-native", "drop-others"), default="error",
                        help="how to handle original/native co-occurrence; default stops for manual review")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output after conversion succeeds")
    args = parser.parse_args()
    try:
        count = convert_file(args.input, args.output, native_conflict=args.native_conflict, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"converted {count} images to {args.output} (24 questions each)")


if __name__ == "__main__":
    main()
