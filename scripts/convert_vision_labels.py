"""Convert 16 mode and 23 palette Noul questions into 24 image questions.

Usage: python -m scripts.convert_vision_labels input.jsonl output.jsonl [--merge modes|all]
"""

import argparse
import json
import os
import tempfile
from pathlib import Path


from kev.camera.labels import MODES, PALETTE_GROUPS, convert_record, prepare_record


def convert_file(source, destination, *, native_conflict="error", overwrite=False, merge="original"):
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
                    if len(row.get("questions", {})) == 39 or merge == "original":
                        row = convert_record(row, native_conflict=native_conflict)
                    converted = prepare_record(row, merge)
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
    parser.add_argument("input", type=Path, help="source JSONL: 39 Noul questions, or 24 questions with --merge")
    parser.add_argument("output", type=Path, help="converted JSONL; must differ from input")
    parser.add_argument("--native-conflict", choices=("error", "drop-native", "drop-others"), default="error",
                        help="how to handle original/native co-occurrence; default stops for manual review")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output after conversion succeeds")
    parser.add_argument("--merge", choices=("original", "modes", "all"), default="original",
                        help="original: 24 questions; modes: 9; all: 2 with grouped palette choices")
    args = parser.parse_args()
    try:
        count = convert_file(args.input, args.output, native_conflict=args.native_conflict,
                             overwrite=args.overwrite, merge=args.merge)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"converted {count} images to {args.output} (layout={args.merge})")


if __name__ == "__main__":
    main()
