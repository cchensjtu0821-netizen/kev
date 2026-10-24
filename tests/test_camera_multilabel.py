"""Merged Camera questions: label semantics, task weights, calibration and CLI flows."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kev.camera.calibrate import calibrate, fit_threshold, question_threshold
from kev.camera.labels import MODES, PALETTE_GROUPS, convert_record, prepare_record
from kev.camera.model import CameraDecisionModel, encode_record, pack_record
from kev.camera.objectives import image_loss, loss_terms, question_probabilities
from kev.camera.predict import answers
from kev.camera.questions import materialize, to_record
from kev.camera.train import datasets, evaluate, parse_args, train_batch
from kev.suite import read_jsonl, write_jsonl
from scripts.convert_vision_labels import convert_file
from test_camera_decision import TinyCamera, TinyTokenizer, cpu_training_control_flow, tiny_checkpoints


def palette_record():
    questions = {f"mode_{name}": {"type": "noul", "instructions": name,
                                    "label": name in {"夜景", "大烟花"}, "src": "camera"}
                 for name in MODES}
    questions.update({f"seka_{name}": {"type": "noul", "instructions": name,
                                       "label": name in {"冷调", "夜晚"}, "src": "camera"}
                      for names in PALETTE_GROUPS.values() for name in names})
    return {"image": "image.jpg", "state": "", "questions": questions}


def small_record():
    return {"state": "", "questions": {
        "modes": {"type": "multilabel", "criteria": {"night": None, "rain": None, "fireworks": None},
                  "label": ["night", "fireworks"]},
        "palette": {"type": "grouped_choice", "criteria": {
            "tone": {"warm": None, "cold": None}, "time": {"day": None, "night": None}},
            "label": {"tone": "cold", "time": "night"}}}}


@pytest.mark.parametrize("layout,count", [("original", 24), ("modes", 9), ("all", 2)])
def test_merging_preserves_mode_and_palette_targets(layout, count):
    row = convert_record(palette_record())
    untouched = copy.deepcopy(row)
    merged = prepare_record(row, layout)
    assert row == untouched
    assert len(merged["questions"]) == count
    rec = materialize(merged)
    assert sum(len(loss_terms(q, torch.zeros(len(q["options"])))) for q in rec["questions"]) == 24
    if layout != "original":
        assert merged["questions"]["modes"]["label"] == ["夜景", "大烟花"]
        assert prepare_record(merged, layout) == merged
    if layout == "all":
        assert merged["questions"]["palette"]["label"]["色调"] == "冷调"
        assert merged["questions"]["palette"]["label"]["天气"] == "无"
        assert merged["questions"]["palette"]["label"]["时间"] == "夜晚"
        assert prepare_record(palette_record(), layout) == merged


def test_raw_unlabelled_request_uses_the_same_merged_prompt_as_training():
    source = palette_record()
    training = materialize(source, layout="all")
    for q in source["questions"].values():
        del q["label"]
    inference, _ = to_record(source, layout="all")
    for a, b in zip(training["questions"], inference["questions"]):
        assert (a["instr"], a["options"], a["keys"]) == (b["instr"], b["options"], b["keys"])


@pytest.mark.parametrize("label", ["阴雨", "无"])
def test_weather_choice_merges_with_its_original_options_and_label(label):
    source = convert_record(palette_record())
    del source["questions"]["seka_阴雨"]
    source["questions"]["seka_天气"] = {
        "type": "choice", "instructions": "这幅画面的天气标签是哪一项？",
        "criteria": {"阴雨": None, "无": "以上都不是"}, "label": label, "src": "camera"}
    modes = prepare_record(source, "modes")
    assert len(modes["questions"]) == 9
    assert modes["questions"]["seka_天气"] == source["questions"]["seka_天气"]
    merged = prepare_record(source, "all")
    assert len(merged["questions"]) == 2
    palette = merged["questions"]["palette"]
    assert palette["criteria"]["天气"] == source["questions"]["seka_天气"]["criteria"]
    assert palette["label"]["天气"] == label
    training = materialize(source, layout="all")
    for question in source["questions"].values():
        del question["label"]
    inference, _ = to_record(source, layout="all")
    for a, b in zip(training["questions"], inference["questions"]):
        assert (a["instr"], a["options"], a["keys"]) == (b["instr"], b["options"], b["keys"])


def test_converter_can_merge_existing_24_or_39_question_files(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "merged.jsonl"
    write_jsonl(source, [palette_record(), convert_record(palette_record())])
    assert convert_file(source, output, merge="all") == 2
    assert all(len(row["questions"]) == 2 for row in read_jsonl(output))


def test_multi_positive_loss_and_group_softmax_have_original_task_weights():
    rec = materialize(small_record())
    modes = torch.zeros(3, requires_grad=True)
    palette = torch.zeros(4, requires_grad=True)
    loss = image_loss(rec, [modes, palette])
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    torch.testing.assert_close(modes.grad, torch.tensor([-0.1, 0.1, -0.1]))
    torch.testing.assert_close(palette.grad, torch.tensor([0.1, -0.1, 0.1, -0.1]))
    assert question_probabilities(rec["questions"][0], modes).sum() == 1.5
    p = question_probabilities(rec["questions"][1], palette)
    assert p[:2].sum() == 1 and p[2:].sum() == 1
    for labels in ([], ["night"], ["night", "rain", "fireworks"]):
        row = small_record()
        row["questions"]["modes"]["label"] = labels
        q = materialize(row)["questions"][0]
        assert loss_terms(q, torch.zeros(3)).mean().item() == pytest.approx(np.log(2))


@pytest.mark.parametrize("change", [
    lambda q: q["modes"].update(label=["missing"]),
    lambda q: q["modes"].update(label=["night", "night"]),
    lambda q: q["modes"].update(label="night"),
    lambda q: q["palette"].update(label={"tone": "cold"}),
    lambda q: q["palette"]["label"].update(tone="unknown"),
    lambda q: q["palette"]["criteria"].update(time={}),
])
def test_bad_merged_labels_are_rejected(change):
    row = small_record()
    change(row["questions"])
    with pytest.raises(ValueError):
        materialize(row)


def test_mixed_layout_batch_trains_and_reports_multiple_positive_labels():
    torch.manual_seed(3)
    records = [materialize(small_record()), materialize({"state": "", "questions": {
        "rain": {"type": "noul", "label": False}}})]
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    items = [{"record": row, "pixels": torch.tensor([float(i + 1)])} for i, row in enumerate(records)]
    value = train_batch(model, items, device=torch.device("cpu"), group_images=2, sync_at_end=True)
    assert np.isfinite(value)
    assert model.head.q.weight.grad.abs().sum() > 0
    assert model.camera.projector.weight.grad.abs().sum() > 0
    report = evaluate(model, items, device=torch.device("cpu"))
    assert report["questions"] == 6
    assert report["multilabel"]["questions"] == 1
    assert 0 <= report["multilabel"]["micro_f1"] <= 1


def test_merged_packed_input_is_shorter_with_synthetic_tokenizer():
    tok = TinyTokenizer()
    lengths = []
    for layout in ("original", "modes", "all"):
        rec = materialize(convert_record(palette_record()), layout=layout)
        lengths.append(len(pack_record(encode_record(tok, rec, 64, 3000))["ids"]))
    assert lengths[0] > lengths[1] > lengths[2]


def test_exact_threshold_sweep_matches_brute_force_with_ties_and_empty_sets():
    rng = np.random.default_rng(19)
    for _ in range(100):
        p = rng.integers(0, 11, (7, 3)) / 10
        y = rng.integers(0, 2, p.shape)
        fitted = fit_threshold(p, y)
        def score(threshold):
            selected = p > threshold
            denominator = int(selected.sum() + y.sum())
            return 2 * (selected * y).sum() / denominator if denominator else 1.0
        assert fitted["calibration_micro_f1"] == max(score(t) for t in [0, 0.5, 1, *p.ravel()])
        assert score(fitted["threshold"]) == fitted["calibration_micro_f1"]
    for y in (np.zeros(3), np.ones(3)):
        fitted = fit_threshold([0.2, 0.2, 0.8], y)
        assert np.isfinite(fitted["threshold"])
    with pytest.raises(ValueError):
        fit_threshold([float("nan")], [1])


def test_prediction_uses_calibrated_threshold_and_preserves_palette_groups():
    source = small_record()
    rec = materialize(source)
    _, meta = to_record(source)
    p = [torch.tensor([0.4, 0.3, 0.8]), torch.tensor([0.1, 0.9, 0.2, 0.8])]
    fitted = calibrate(zip(rec["questions"], p))
    result = answers(rec, meta, [v.tolist() for v in p], thresholds=fitted)
    assert result["modes"]["selected"] == ["night", "fireworks"]
    assert result["palette"]["tone"]["choice"] == "cold"
    assert result["palette"]["time"]["choice"] == "night"
    changed = copy.deepcopy(rec["questions"][0])
    changed["keys"].reverse()
    with pytest.raises(ValueError, match="schema changed"):
        question_threshold(changed, fitted)


def test_bfloat_logits_use_float_probabilities_for_threshold_parity():
    q = materialize(small_record())["questions"][0]
    z = torch.tensor([0.8, -0.3, 0.7], dtype=torch.bfloat16)
    torch.testing.assert_close(question_probabilities(q, z), z.float().sigmoid(), atol=0, rtol=0)


@pytest.mark.parametrize("layout,count", [("modes", 9), ("all", 2)])
def test_dataset_preflight_merges_before_training(tmp_path, layout, count):
    (tmp_path / "image.jpg").write_bytes(b"image")
    data = tmp_path / "train.jsonl"
    write_jsonl(data, [convert_record(palette_record())])
    args = parse_args(["--train-data", str(data), "--question-layout", layout, "--validate-data"])
    train, _ = datasets(args)
    assert len(train[0]["record"]["questions"]) == count


@pytest.mark.parametrize("script,layout,rank", [("train_camera_kev_lora_modes.sh", "modes", 128),
                                               ("train_camera_kev_lora_modes_seka.sh", "all", 32)])
def test_merged_lora_shell_launchers_forward_layout_and_resume(tmp_path, script, layout, rank):
    launcher = tmp_path / "torchrun"
    body = f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    launcher.write_text(body, encoding="utf-8")
    launcher.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "TORCHRUN": str(launcher), "RESUME": "latest",
           "TRAIN_DATA": "/data/train.jsonl", "OUTPUT_DIR": str(tmp_path / "out")}
    result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "shell/lora" / script)],
                            env=env, capture_output=True, text=True, check=True)
    argv = json.loads(result.stdout)
    assert argv[argv.index("--question-layout") + 1] == layout
    assert argv[argv.index("--lora-r") + 1] == str(rank)
    assert argv[argv.index("--resume") + 1] == "latest"


def test_merged_training_resume_preserves_weights(tmp_path, cpu_training_control_flow):
    from kev.camera import model as camera_model, train as camera_train
    raw, args = cpu_training_control_flow
    data = Path(args[args.index("--train-data") + 1])
    write_jsonl(data, [{**row, "questions": small_record()["questions"]} for row in read_jsonl(data)])
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    camera_train.main([*args, "--model", str(raw), "--out", str(baseline)])
    camera_train.main([*args, "--model", str(raw), "--out", str(resumed), "--stop-after", "1"])
    camera_train.main([*args, "--resume", "latest", "--out", str(resumed)])
    left, _ = camera_model.load(baseline / "final", device="cpu", dtype=torch.float32)
    right, _ = camera_model.load(resumed / "final", device="cpu", dtype=torch.float32)
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)


def test_calibration_and_prediction_cli_roundtrip(tmp_path, cpu_training_control_flow, capsys):
    from kev.camera import calibrate as calibration, predict, train as camera_train
    raw, args = cpu_training_control_flow
    training = Path(args[args.index("--train-data") + 1])
    validation = Path(args[args.index("--val-data") + 1])
    for path in (training, validation):
        write_jsonl(path, [{**row, "questions": small_record()["questions"]} for row in read_jsonl(path)])
    out = tmp_path / "trained"
    camera_train.main([*args, "--model", str(raw), "--out", str(out)])
    calibration_file = tmp_path / "thresholds.json"
    calibration.main(["--checkpoint", str(out / "final"), "--val-data", str(validation),
                      "--out", str(calibration_file), "--device", "cpu", "--num-workers", "0"])
    thresholds = calibration.load_thresholds(calibration_file, out / "final")
    assert thresholds["modes"]["images"] == 1
    request = read_jsonl(validation)[0]
    for q in request["questions"].values():
        del q["label"]
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    capsys.readouterr()
    predict.main(["--checkpoint", str(out / "final"), "--request", str(request_file),
                  "--calibration", str(calibration_file), "--device", "cpu"])
    result = json.loads(capsys.readouterr().out)
    assert result["modes"]["threshold"] == thresholds["modes"]["threshold"]
    assert set(result["palette"]) == {"tone", "time"}
    with pytest.raises(ValueError, match="share image paths"):
        calibration.main(["--checkpoint", str(out / "final"), "--val-data", str(training),
                          "--out", str(tmp_path / "bad.json"), "--device", "cpu", "--num-workers", "0"])
    with torch.no_grad():
        model, _ = predict.load(out / "final", device="cpu", dtype=torch.float32)
        model.head.q.weight.add_(1)
        torch.save(model.head.state_dict(), out / "final" / "pointer.pt")
    with pytest.raises(ValueError, match="different checkpoint"):
        calibration.load_thresholds(calibration_file, out / "final")
