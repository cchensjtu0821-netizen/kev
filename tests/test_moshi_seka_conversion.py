"""Merged benchmark prompts must match training and never invent missing labels."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark.convert_moshi_seka_to_kev import (
    ALL_FILES, MODE_FILES, SEKA_FILES, convert_item, load_json_or_jsonl,
    main, mode_labels, question_template, remap_image_path, seka_labels,
)
from kev.camera.labels import prepare_record
from kev.camera.questions import to_record


def rows(answer, *, mode_file):
    return convert_item({"original_id": 45, "id": 3, "answer": answer}, mode_file=mode_file,
                        image="/images/45.jpg", source="source.json")


def test_mode_labels_preserve_multi_selection_and_drop_unknown_tokens():
    assert mode_labels("夜景普通 大烟花典型 夜景 无效标签") == ["夜景", "大烟花"]
    assert seka_labels("暖色系天空-室外-夜晚-中性色调-无效标签;") == ["暖色系天空", "室外", "夜晚", "中性色调"]


def test_mode_benchmark_keeps_palette_unlabelled_in_both_formats():
    converted = rows("剪影普通 不在列表", mode_file=True)
    for layout, count in (("modes", 9), ("all", 2)):
        row = converted[layout]
        assert len(row["questions"]) == count
        assert row["id"] == 45 and row["answer"] == "剪影普通 不在列表"
        assert row["questions"]["modes"]["label"] == ["剪影"]
        assert row["_meta"]["eval_labels"] == {"modes": ["剪影"]}
        assert all("label" not in q for key, q in row["questions"].items() if key != "modes")
        record, _ = to_record(row)
        expected, _ = to_record(prepare_record({"state": "", "questions": question_template()}, layout))
        for actual, trained in zip(record["questions"], expected["questions"]):
            assert (actual["instr"], actual["options"]) == (trained["instr"], trained["options"])


def test_seka_benchmark_keeps_modes_unlabelled_and_scores_all_known_groups():
    answer = "暖色系天空-室外-夜晚-中性色调"
    converted = rows(answer, mode_file=False)
    labels = {"天空": "暖色系天空", "植物": "无", "天气": "无", "室内外": "室外",
              "时间": "夜晚", "建筑": "无", "色调": "中性色调", "其他": "无"}
    for row in converted.values():
        assert "label" not in row["questions"]["modes"]
        assert row["answer"] == answer
        assert len(to_record(row)[0]["questions"]) == len(row["questions"])
    assert converted["all"]["questions"]["palette"]["label"] == labels
    assert converted["all"]["_meta"]["eval_labels"] == {"palette": labels}
    assert converted["modes"]["_meta"]["eval_labels"] == {f"seka_{g}": y for g, y in labels.items()}
    assert list(converted["all"]["questions"]["palette"]["criteria"]["天气"]) == ["阴雨", "无"]


def test_missing_tone_does_not_become_null_or_an_invented_tone():
    converted = rows("室外-夜晚", mode_file=False)
    assert "label" not in converted["modes"]["questions"]["seka_色调"]
    assert "seka_色调" not in converted["modes"]["_meta"]["eval_labels"]
    palette = converted["all"]["questions"]["palette"]
    assert "label" not in palette
    assert len(palette["criteria"]["色调"]) == 4
    assert "色调" not in converted["all"]["_meta"]["eval_labels"]["palette"]
    assert len(converted["all"]["_meta"]["eval_labels"]["palette"]) == 7
    to_record(converted["all"])


def test_conflicting_palette_labels_are_rejected():
    with pytest.raises(ValueError, match="色调.*overlapping labels"):
        rows("暖调-冷调", mode_file=False)


def test_path_remapping_has_prefix_boundaries_and_absolute_resolution(tmp_path):
    root = "/mnt/15_data_yh/15_data/vlm/seka_wcy/train_data"
    assert remap_image_path(root + "/0519/a.jpg", image_dir=tmp_path) == tmp_path / "0519/a.jpg"
    assert remap_image_path("relative.jpg", image_dir=tmp_path) == tmp_path / "relative.jpg"
    assert remap_image_path(root + "/05190/a.jpg", image_dir=tmp_path,
                            seka_data_base=tmp_path / "seka") == tmp_path / "seka/05190/a.jpg"
    assert remap_image_path(root + "/260415_gaoxiangsu_test/a.jpg", image_dir=tmp_path,
                            gaoxiangsu_test_dir=tmp_path / "high") == tmp_path / "high/a.jpg"
    assert remap_image_path("", image_dir=tmp_path) is None


@pytest.mark.parametrize("contents,count", [("[]", 0), ('[{"id": 1}]', 1),
                                          ('{"id":1}\n{"id":2}\n', 2), ('{\n "id": 1\n}', 1)])
def test_json_and_jsonl_loading(tmp_path, contents, count):
    path = tmp_path / "source.jsonl"
    path.write_text(contents, encoding="utf-8")
    assert len(load_json_or_jsonl(path)) == count


def test_cli_converts_all_ten_sources_and_counts_missing_images(tmp_path):
    data, images, out = tmp_path / "data", tmp_path / "images", tmp_path / "out"
    data.mkdir()
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"synthetic image path")
    for source in ALL_FILES:
        answer = "剪影典型" if source in MODE_FILES else "室外-中性色调"
        values = [{"id": 1, "image": "photo.jpg", "answer": answer},
                  {"id": 2, "image": "missing.jpg", "answer": answer}]
        (data / source).write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    args = ["--data-dir", str(data), "--out-dir", str(out), "--image-dir", str(images)]
    stats = main(args)
    assert len(stats) == 10
    for source in ALL_FILES:
        assert stats[source]["rows"] == 1 and stats[source]["missing_image"] == 1
        for layout, count in (("modes", 9), ("all", 2)):
            row = json.loads((out / f"{source}.{layout}.jsonl").read_text(encoding="utf-8"))
            assert row["image"] == str(images / "photo.jpg")
            assert len(row["questions"]) == count
            assert row["_meta"]["eval_task"] == ("mode" if source in MODE_FILES else "seka")
    with pytest.raises(SystemExit):
        main(args)
    main([*args, "--overwrite", "--formats", "all"])


def test_bad_source_does_not_replace_existing_outputs(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"synthetic")
    source = SEKA_FILES[0]
    contents = json.dumps([
        {"image": str(image), "answer": "中性色调"},
        {"image": str(image), "answer": "冷调-暖调"}])
    (tmp_path / source).write_text(contents, encoding="utf-8")
    output = tmp_path / f"{source}.all.jsonl"
    output.write_text("previous output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="item 2.*色调"):
        main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path), "--sources", source,
              "--formats", "all", "--overwrite"])
    assert output.read_text(encoding="utf-8") == "previous output\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_image_can_stop_conversion_without_publishing_rows(tmp_path):
    source = MODE_FILES[0]
    (tmp_path / source).write_text('[{"image": "missing.jpg", "answer": "剪影"}]', encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="item 1.*missing image"):
        main(["--data-dir", str(tmp_path), "--out-dir", str(out), "--image-dir", str(tmp_path),
              "--sources", source, "--missing-images", "error"])
    assert not list(out.glob("*.jsonl"))
    assert not list(out.glob(".*.tmp"))


def test_converter_help_works_without_third_party_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-m", "benchmark.convert_moshi_seka_to_kev", "--help"],
                            cwd=root, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--formats" in result.stdout
