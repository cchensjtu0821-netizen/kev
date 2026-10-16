"""The palette converter must preserve multi-label modes and reject bad source labels."""

import json

import pytest

from kev.data import materialize
from scripts.convert_vision_labels import MODES, PALETTE_GROUPS, convert_file, convert_record


def record():
    questions = {f"mode_{name}": {"type": "noul", "instructions": name, "label": name == "原生", "src": "lora_seka"}
                 for name in MODES}
    questions.update({f"seka_{name}": {"type": "noul", "instructions": name, "label": name == "中性色调", "src": "lora_seka"}
                      for names in PALETTE_GROUPS.values() for name in names})
    return {"id": 73, "image": "images/73.jpg", "state": "", "questions": questions}


def test_converts_all_groups_and_keeps_composable_modes():
    source = record()
    source["questions"]["mode_原生"]["label"] = False
    source["questions"]["mode_夜景"]["label"] = True
    source["questions"]["mode_大烟花"]["label"] = True
    source["questions"]["seka_夜晚"]["label"] = True
    source["questions"]["seka_室外"]["label"] = True
    source["questions"]["seka_中性色调"]["label"] = False
    source["questions"]["seka_冷调"]["label"] = True
    output = convert_record(source)
    assert len(output["questions"]) == 24
    assert (output["id"], output["image"], output["state"]) == (73, "images/73.jpg", "")
    assert output["questions"]["mode_夜景"]["label"] is True
    assert output["questions"]["mode_大烟花"]["label"] is True
    assert output["questions"]["seka_时间"]["label"] == "夜晚"
    assert output["questions"]["seka_室内外"]["label"] == "室外"
    assert output["questions"]["seka_天空"]["label"] == "无"
    assert "无" not in output["questions"]["seka_色调"]["criteria"]
    assert output["questions"]["seka_色调"]["label"] == "冷调"
    assert output["questions"]["seka_阴雨"]["type"] == "noul"
    assert len(materialize(output)["questions"]) == 24


def test_food_alias_requires_consistent_source_labels():
    source = record()
    source["questions"]["seka_餐饮美食"] = source["questions"].pop("seka_美食")
    source["questions"]["seka_餐饮美食"]["label"] = True
    assert convert_record(source)["questions"]["seka_其他"]["label"] == "美食"
    source["questions"]["seka_花卉"]["label"] = True
    with pytest.raises(ValueError, match="conflicting labels"):
        convert_record(source)


def test_native_conflict_needs_explicit_decision():
    source = record()
    source["questions"]["mode_小烟花"]["label"] = True
    with pytest.raises(ValueError, match="mode_原生 conflicts"):
        convert_record(source)
    assert convert_record(source, native_conflict="drop-native")["questions"]["mode_原生"]["label"] is False
    assert convert_record(source, native_conflict="drop-others")["questions"]["mode_小烟花"]["label"] is False


def test_file_conversion_is_atomic_on_bad_row(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "converted.jsonl"
    bad = record()
    bad["questions"]["seka_室内"]["label"] = True
    bad["questions"]["seka_室外"]["label"] = True
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (record(), bad)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source.jsonl:2: seka_室内外 has conflicting labels"):
        convert_file(source, destination)
    assert not destination.exists()
    source.write_text(json.dumps(record(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert convert_file(source, destination) == 1
    assert len(json.loads(destination.read_text(encoding="utf-8"))["questions"]) == 24
    with pytest.raises(ValueError, match="relative image paths"):
        convert_file(source, tmp_path / "other" / "converted.jsonl")
