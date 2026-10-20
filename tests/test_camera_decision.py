"""CameraAgent adapter contracts without private model weights."""

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from kev.camera import model as camera_model
from kev.camera import train as camera_train
from kev.camera.data import CameraImageDataset
from kev.camera.model import CameraDecisionModel, encode_record, load_camera, pack_record
from kev.camera.resume import (ResumableBatchSampler, check_resume, data_signature,
                               find_resume, read_state, restore_rng, rng_state, save_checkpoint)
from kev.camera.train import datasets, evaluate, first_item, parse_args, train_batch
from kev.suite import write_json, write_jsonl


class TinyTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[2 + ord(character) % 126 for character in text])

    def convert_tokens_to_ids(self, token):
        return 1 if token == "<IMG_CONTEXT>" else 0

    def convert_ids_to_tokens(self, token):
        return "<IMG_CONTEXT>" if token == 1 else "<unk>"


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_shape = None

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict, position_ids=None):
        self.calls += 1
        self.last_shape = inputs_embeds.shape
        if attention_mask.ndim == 4:
            allowed = (attention_mask[:, 0] == 0).to(inputs_embeds.dtype)
            hidden = allowed @ inputs_embeds
        else:
            hidden = inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.embeddings = nn.Embedding(128, 4)
        self.config = SimpleNamespace(hidden_size=4, _attn_implementation="eager")
        self.model.config = self.config

    def get_input_embeddings(self):
        return self.embeddings


class TinyCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguage()
        self.projector = nn.Linear(1, 4)
        self.config = SimpleNamespace(llm_config=self.language_model.config)

    def extract_feature(self, pixels):
        return self.projector(pixels.reshape(-1, 1, 1)).expand(-1, 3, self.projector.out_features)


def record():
    return {"state": "photo", "questions": [
        {"instr": "Is there a cat?", "options": ["no", "yes"], "label": 1},
        {"instr": "Which colour?", "options": ["red", "blue", "none"], "label": 0},
    ]}


@pytest.mark.parametrize("as_dict", [False, True])
def test_camera_loader_configures_nested_attention_before_model_construction(monkeypatch, as_dict):
    from transformers import AutoConfig, AutoModelForCausalLM

    text_config = {"_attn_implementation": "flash_attention_2", "attn_implementation": "flash_attention_2"}
    if not as_dict:
        text_config = SimpleNamespace(**text_config)
    config = SimpleNamespace(llm_config=text_config)
    camera = TinyCamera()

    def load_config(path, **kwargs):
        assert path == "camera-checkpoint"
        assert kwargs == {"trust_remote_code": True}
        return config

    def load_model(path, **kwargs):
        assert path == "camera-checkpoint"
        assert kwargs["config"] is config
        values = text_config if as_dict else vars(text_config)
        assert values["_attn_implementation"] == values["attn_implementation"] == "eager"
        assert kwargs["attn_implementation"] == "eager"
        assert kwargs["use_flash_attn"] is False
        assert kwargs["dtype"] == torch.bfloat16
        return camera

    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    assert load_camera("camera-checkpoint") is camera


@pytest.mark.parametrize("target", ["language", "backbone", "attention"])
def test_camera_rejects_nested_flash_attention_before_forward(target):
    camera = TinyCamera()
    camera.config._attn_implementation = "eager"
    if target == "language":
        module = camera.language_model
    elif target == "backbone":
        module = camera.language_model.model
    else:
        camera.language_model.model.self_attn = nn.Module()
        module = camera.language_model.model.self_attn
    module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    with pytest.raises(ValueError, match="4D question-branch mask requires eager"):
        CameraDecisionModel(camera, TinyTokenizer())


def test_qwen3_eager_packed_questions_match_rows_with_checkpointing():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(8)
    config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, bos_token_id=1, eos_token_id=0, pad_token_id=0,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    camera = TinyCamera()
    camera.language_model = Qwen3ForCausalLM(config)
    camera.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    camera.config.llm_config = camera.language_model.config
    camera.projector = nn.Linear(1, 16)
    model = CameraDecisionModel(camera, TinyTokenizer(), head_dim=3, max_length=500).train()
    rows = [record(), {"state": "longer state", "questions": record()["questions"][:1]}]
    pixels = torch.tensor([1.0, 2.0])
    batched = model(rows, pixels)
    for index, row in enumerate(rows):
        direct = model(row, pixels[index:index + 1])
        for actual, expected in zip(batched[index], direct):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    sum(z.square().sum() for group in batched for z in group).backward()
    assert torch.isfinite(camera.projector.weight.grad).all()
    assert camera.projector.weight.grad.abs().sum() > 0


def test_camera_pointer_keeps_questions_isolated_and_trains_visual_path():
    torch.manual_seed(3)
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    first = model(record(), torch.tensor([1.0]))
    changed = record()
    changed["questions"][1]["instr"] = "Entirely different question"
    second = model(changed, torch.tensor([1.0]))
    assert torch.equal(first[0], second[0])
    assert first[1].shape == (3,)
    assert not torch.equal(first[1], second[1])
    assert not torch.equal(first[0], model(record(), torch.tensor([2.0]))[0])
    first[0].sum().backward()
    assert model.camera.projector.weight.grad.abs().sum() > 0
    assert model.head.q.weight.grad.abs().sum() > 0


def test_train_batch_averages_images_with_unequal_question_counts_and_partial_group():
    torch.manual_seed(4)
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    reference = copy.deepcopy(model)
    records = [record(), {"state": "second", "questions": record()["questions"][:1]}, record()]
    items = [{"record": row, "pixels": torch.tensor([float(i + 1)])}
             for i, row in enumerate(records)]
    first_loss = train_batch(model, items[:2], device=torch.device("cpu"),
                             group_images=3, sync_at_end=False)
    last_loss = train_batch(model, items[2:], device=torch.device("cpu"),
                            group_images=3, sync_at_end=True)
    expected = []
    for item in items:
        logits = reference(item["record"], item["pixels"])
        losses = [F.cross_entropy(z.unsqueeze(0), torch.tensor([q["label"]]))
                  for z, q in zip(logits, item["record"]["questions"])]
        expected.append(torch.stack(losses).mean())
    torch.stack(expected).mean().backward()
    assert first_loss + last_loss == pytest.approx(sum(float(loss.detach()) for loss in expected), abs=1e-6)
    for trained, direct in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(trained.grad, direct.grad, atol=1e-6, rtol=1e-6)


def test_batched_camera_forward_matches_individual_rows():
    torch.manual_seed(5)
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    records = [record(), {"state": "a longer state", "questions": record()["questions"][:1]}]
    pixels = torch.tensor([1.0, 2.0])
    batched = model(records, pixels, question_start=0, question_end=2)
    assert [len(group) for group in batched] == [2, 1]
    for index, row in enumerate(records):
        direct = model(row, pixels[index:index + 1])
        for actual, expected in zip(batched[index], direct):
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    changed = copy.deepcopy(records)
    changed[0]["questions"][1]["instr"] = "Different second task"
    assert torch.allclose(model(changed, pixels)[0][0], batched[0][0], atol=1e-6, rtol=1e-6)
    encoded = encode_record(model.tokenizer, records[0], image_tokens=3, max_length=500)
    packed = pack_record(encoded)
    prefix = encoded[0]["prefix_length"]
    assert len(packed["ids"]) == sum(len(row["ids"]) for row in encoded) - prefix
    assert packed["segments"].count(0) == prefix
    assert packed["positions"][prefix] == packed["positions"][packed["segments"].index(2)]


def test_identical_question_templates_need_no_batch_padding():
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    rows = [record(), record()]
    encoded = encode_record(model.tokenizer, rows[0], image_tokens=3, max_length=500)
    expected_length = len(pack_record(encoded)["ids"])
    assert len(model(rows, torch.tensor([1.0, 2.0]))) == 2
    assert model.camera.language_model.model.last_shape == (2, expected_length, 4)


def test_validation_and_prediction_run_all_questions_in_one_packed_forward():
    model = CameraDecisionModel(TinyCamera(), TinyTokenizer(), head_dim=3, max_length=500)
    item = {"record": record(), "pixels": torch.tensor([1.0])}
    result = evaluate(model, [item], device=torch.device("cpu"))
    assert result["questions"] == 2
    assert model.camera.language_model.model.calls == 1
    assert len(model.probabilities(item["record"], item["pixels"])) == 2
    assert model.camera.language_model.model.calls == 2


def test_encoding_counts_actual_visual_tokens_and_rejects_overflow():
    tok = TinyTokenizer()
    rows = encode_record(tok, record(), image_tokens=3, max_length=500)
    assert len(rows) == 2
    assert rows[0]["ids"][rows[0]["image_start"]:rows[0]["image_start"] + 3] == [1, 1, 1]
    assert len(rows[0]["option_indices"]) == 2
    with pytest.raises(ValueError, match="over max_length"):
        encode_record(tok, record(), image_tokens=3, max_length=20)


def test_dataset_preflight_accepts_noul_and_choice_and_catches_image_overlap(tmp_path):
    train_image = tmp_path / "train.jpg"
    train_image.write_bytes(b"image")
    val_image = tmp_path / "val.jpg"
    val_image.write_bytes(b"image")
    row = {"image": "train.jpg", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False},
        "tone": {"type": "choice", "instructions": "Which tone?",
                 "criteria": {"warm": None, "cold": None}, "label": "warm"}}}
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "val.jsonl"
    training.write_text(json.dumps(row) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({**row, "image": "val.jpg"}) + "\n", encoding="utf-8")
    args = parse_args(["--train-data", str(training), "--val-data", str(validation), "--validate-data"])
    assert (args.batch, args.accum) == (8, 2)
    train, val = datasets(args)
    assert [q["label"] for q in train[0]["record"]["questions"]] == [0, 0]
    assert len(val) == 1
    validation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="share 1 image paths"):
        datasets(args)


def test_dataset_preflight_allows_no_validation(tmp_path):
    (tmp_path / "train.jpg").write_bytes(b"image")
    training = tmp_path / "train.jsonl"
    write_jsonl(training, [{"image": "train.jpg", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}])
    args = parse_args(["--train-data", str(training), "--validate-data"])
    train, valid = datasets(args)
    assert len(train) == 1
    assert valid is None
    write_jsonl(training, [{"image": "missing.jpg", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}])
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        datasets(args)


def small_worker_pixel_limit(worker_id):
    Image.MAX_IMAGE_PIXELS = 16


@pytest.mark.parametrize("num_workers", [0, 2])
def test_image_decode_runs_with_cameramodel_pixel_limit(tmp_path, monkeypatch, num_workers):
    Image.new("RGB", (12, 18), (255, 0, 0)).save(tmp_path / "train.png")
    training = tmp_path / "train.jsonl"
    write_jsonl(training, [{"image": "train.png", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}])
    args = parse_args(["--train-data", str(training), "--validate-data"])
    records, _ = datasets(args)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    loader = DataLoader(CameraImageDataset(records, 32), batch_size=1, num_workers=num_workers,
                        collate_fn=first_item, worker_init_fn=small_worker_pixel_limit)
    item = next(iter(loader))
    assert item["pixels"].shape == (1, 3, 32, 32)
    assert item["pixels"][0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)
    assert item["record"]["questions"][0]["label"] == 0


@pytest.fixture
def tiny_checkpoints(monkeypatch):
    """Exercise checkpoint I/O with synthetic tensors, without private CameraAgent code or weights."""
    from transformers import AutoTokenizer

    class StochasticBackbone(TinyBackbone):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.2)

        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.last_hidden_state = self.dropout(output.last_hidden_state)
            return output

    def make_camera():
        camera = TinyCamera()
        camera.language_model.model = StochasticBackbone()
        camera.language_model.model.config = camera.language_model.config
        camera.config.model_type = "camera_agent"
        camera.config.template = "internvl2_5"
        camera.config.ps_version = "v2"
        camera.config.force_image_size = 8
        camera.language_model.get_output_embeddings = lambda: None
        return camera

    def save_camera(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "config.json", {
            "model_type": "camera_agent", "llm_config": {"hidden_size": 4},
            "force_image_size": 8, "template": "internvl2_5", "ps_version": "v2",
        })
        torch.save(self.state_dict(), path / "tiny.pt")

    def load_camera(path, **kwargs):
        camera = make_camera()
        camera.load_state_dict(torch.load(Path(path) / "tiny.pt", weights_only=True))
        return camera

    def save_tokenizer(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(TinyCamera, "save_pretrained", save_camera, raising=False)
    monkeypatch.setattr(TinyTokenizer, "save_pretrained", save_tokenizer, raising=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(camera_model, "load_camera", load_camera)
    monkeypatch.setattr(camera_train, "load_camera", load_camera)
    return make_camera


def test_flat_config_and_legacy_checkpoint_preserve_backbone_head_and_predictions(tmp_path, tiny_checkpoints):
    model = CameraDecisionModel(tiny_checkpoints(), TinyTokenizer(), head_dim=3, max_length=500).eval()
    source = tmp_path / "source"
    source.mkdir()
    (source / "modeling_camera_agent.py").write_text("pass\n", encoding="utf-8")
    flat = tmp_path / "flat"
    model.save(flat, source=source, image_size=8)
    config = json.loads((flat / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "camera_agent"
    assert config["llm_config"]["hidden_size"] == 4
    assert config["kev"]["head_dim"] == 3
    assert config["kev"]["image_size"] == 8
    assert (flat / "modeling_camera_agent.py").is_file()
    assert not (flat / "camera").exists()
    assert not (flat / "camera_decision.json").exists()
    loaded, _ = camera_model.load(flat, device="cpu", dtype=torch.float32)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip(model(record(), torch.tensor([1.])),
                                                loaded(record(), torch.tensor([1.]))))

    legacy = tmp_path / "legacy"
    model.camera.save_pretrained(legacy / "camera")
    model.tokenizer.save_pretrained(legacy / "camera")
    torch.save(model.head.state_dict(), legacy / "pointer.pt")
    write_json(legacy / "camera_decision.json", {
        "source": str(source), "format": 1, "head_dim": 3, "image_size": 8, "max_length": 500,
    })
    loaded, _ = camera_model.load(legacy, device="cpu", dtype=torch.float32)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="no training_state.pt"):
        find_resume(str(legacy), tmp_path)


@pytest.fixture
def cpu_training_control_flow(tmp_path, monkeypatch, tiny_checkpoints):
    """Run the trainer's control flow on a synthetic CPU model; this does not verify CameraAgent/CUDA."""
    class CpuTorch:
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                               set_device=lambda rank: None)

        def device(self, *args):
            return torch.device("cpu")

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(camera_train, "torch", CpuTorch())
    real_loader = camera_train.DataLoader

    def cpu_loader(*args, **kwargs):
        kwargs["pin_memory"] = False
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(camera_train, "DataLoader", cpu_loader)
    # Adapt the synthetic visual projector to real image tensor shapes.
    def extract_feature(self, pixels):
        value = pixels.flatten(1).mean(-1).reshape(-1, 1, 1)
        return self.projector(value).expand(-1, 3, 4)

    monkeypatch.setattr(TinyCamera, "extract_feature", extract_feature)
    raw = tmp_path / "raw"
    tiny_checkpoints().save_pretrained(raw)
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "val.jsonl"
    rows = []
    for index in range(6):
        Image.new("RGB", (8, 8), (index * 30, 100, 200)).save(tmp_path / f"{index}.png")
        rows.append({"image": f"{index}.png", "state": "", "questions": {
            "cat": {"type": "noul", "instructions": "Cat?", "label": index % 2 == 0}}})
    training.write_text("".join(json.dumps(row) + "\n" for row in rows[:5]), encoding="utf-8")
    validation.write_text(json.dumps(rows[5]) + "\n", encoding="utf-8")
    args = ["--train-data", str(training), "--val-data", str(validation),
            "--epochs", "2", "--batch", "2", "--accum", "2", "--image-size", "8",
            "--head-dim", "3", "--max-length", "500", "--num-workers", "0"]
    return raw, args


def assert_tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tensor_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stop_step,workers", [(1, 0), (2, 0), (4, 0), (1, 2)])
def test_resume_control_flow_matches_uninterrupted_with_dropout_and_partial_batches(
        tmp_path, cpu_training_control_flow, stop_step, workers):
    raw, args = cpu_training_control_flow
    args = [*args, "--num-workers", str(workers)]
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    camera_train.main([*args, "--model", str(raw), "--out", str(baseline)])
    camera_train.main([*args, "--model", str(raw), "--out", str(resumed), "--stop-after", str(stop_step)])
    # A failed checkpoint write is hidden and must never become the latest resume point.
    (resumed / ".checkpoint-9999999-interrupted").mkdir()
    checkpoint = find_resume("latest", resumed)
    assert checkpoint.name == f"checkpoint-{stop_step:07d}"
    camera_train.main([*args, "--resume", "latest", "--out", str(resumed)])
    expected, _ = camera_model.load(baseline / "final", device="cpu")
    actual, _ = camera_model.load(resumed / "final", device="cpu")
    assert_tensor_tree_equal(actual.state_dict(), expected.state_dict())
    expected_state, actual_state = read_state(baseline / "final"), read_state(resumed / "final")
    assert expected_state["step"] == actual_state["step"] == 4
    assert_tensor_tree_equal(expected_state["optimizer"], actual_state["optimizer"])
    assert_tensor_tree_equal(expected_state["scheduler"], actual_state["scheduler"])
    assert (baseline / "training_metrics.jsonl").read_bytes() == (resumed / "training_metrics.jsonl").read_bytes()
    camera_train.main([*args, "--resume", "latest", "--out", str(resumed)])


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_initializes_a_new_run_with_the_saved_head(tmp_path, cpu_training_control_flow, monkeypatch, legacy):
    raw, args = cpu_training_control_flow
    original = tmp_path / "original"
    camera_train.main([*args, "--model", str(raw), "--out", str(original)])
    saved, _ = camera_model.load(original / "final", device="cpu")
    checkpoint = original / "final"
    if legacy:
        checkpoint = tmp_path / "legacy"
        saved.camera.save_pretrained(checkpoint / "camera")
        saved.tokenizer.save_pretrained(checkpoint / "camera")
        torch.save(saved.head.state_dict(), checkpoint / "pointer.pt")
        write_json(checkpoint / "camera_decision.json", {
            "source": str(raw), "format": 1, "head_dim": 3, "image_size": 8, "max_length": 500,
        })
    real_batch = camera_train.train_batch
    seen = []

    def check_first_batch(model, *args, **kwargs):
        if not seen:
            assert_tensor_tree_equal(model.state_dict(), saved.state_dict())
        seen.append(True)
        return real_batch(model, *args, **kwargs)

    monkeypatch.setattr(camera_train, "train_batch", check_first_batch)
    # No shape arguments: the new run must inherit the checkpoint's non-default settings.
    args = [arg for index, arg in enumerate(args)
            if arg not in ("--head-dim", "--image-size", "--max-length")
            and (index == 0 or args[index - 1] not in ("--head-dim", "--image-size", "--max-length"))]
    camera_train.main([*args, "--checkpoint", str(checkpoint), "--out", str(tmp_path / "new"),
                       "--stop-after", "1"])
    assert seen
    state = read_state(tmp_path / "new" / "checkpoint-0000001")
    assert state["step"] == 1
    assert (state["args"]["head_dim"], state["args"]["image_size"], state["args"]["max_length"]) == (3, 8, 500)
    with pytest.raises(ValueError, match="raw CameraAgent"):
        camera_train.main([*args, "--model", str(original / "final"), "--out", str(tmp_path / "wrong")])


def test_rng_restore_includes_python_numpy_and_torch():
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(5))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(5), expected[2])


def test_failed_checkpoint_write_cannot_be_selected_as_latest(tmp_path):
    class BrokenModel:
        def save(self, path, **kwargs):
            (path / "pointer.pt").write_bytes(b"incomplete")
            raise OSError("disk full")

    target = tmp_path / "checkpoint-0000001"
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(target, BrokenModel(), None, None, args=SimpleNamespace(image_size=8),
                        source=tmp_path, world=1, rank=0, signature=[], total_steps=2,
                        step=1, epoch=0, next_batch=1, train_loss=1.0, seen=1)
    assert not target.exists()
    with pytest.raises(FileNotFoundError, match="no complete"):
        find_resume("latest", tmp_path)


def test_resume_rejects_changed_training_and_data(tmp_path, cpu_training_control_flow):
    raw, args = cpu_training_control_flow
    out = tmp_path / "run"
    camera_train.main([*args, "--model", str(raw), "--out", str(out), "--stop-after", "1"])
    state = read_state(find_resume("latest", out))
    parsed = SimpleNamespace(**state["args"])
    signature = data_signature(*datasets(parsed))
    check_resume(state, parsed, world=1, signature=signature, total_steps=4)
    for field, value in (("lr", 0.001), ("accum", 1), ("epochs", 3), ("freeze_llm", True)):
        changed = copy.copy(parsed)
        setattr(changed, field, value)
        with pytest.raises(ValueError, match="settings changed"):
            check_resume(state, changed, world=1, signature=signature, total_steps=4)
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        check_resume(state, parsed, world=2, signature=signature, total_steps=4)
    Path(parsed.train_data).write_text(Path(parsed.train_data).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotations"):
        check_resume(state, parsed, world=1, signature=data_signature(*datasets(parsed)), total_steps=4)


def test_resume_sampler_skips_batches_before_decoding_images():
    class Records:
        def __getitem__(self, index):
            assert index >= 4
            return index

    sampler = ResumableBatchSampler(range(7), batch_size=2, drop_last=False)
    sampler.start = 2
    assert len(sampler) == 2
    assert [batch.tolist() for batch in DataLoader(Records(), batch_sampler=sampler)] == [[4, 5], [6]]


@pytest.mark.parametrize("script,mode", [
    ("train_camera_kev.sh", "base"), ("train_camera_kev_from_ckpt.sh", "checkpoint"),
    ("train_camera_kev.sh", "resume"), ("train_camera_kev_from_ckpt.sh", "resume"),
])
def test_camera_shell_entrypoints_select_one_source(tmp_path, script, mode):
    launcher = tmp_path / "torchrun"
    launcher.write_bytes(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n".encode("utf-8"))
    launcher.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    env = {**os.environ, "TORCHRUN": str(launcher), "TRAIN_DATA": "/data/train.jsonl", "TRAIN_META": "",
           "CAMERA_CODE_DIR": "", "OUTPUT_DIR": str(tmp_path / "out"), "RESUME": "",
           "MODEL_NAME_OR_PATH": str(model), "CAMERA_CHECKPOINT": str(model),
           "IMAGE_SIZE": "", "MAX_LENGTH": "", "HEAD_DIM": ""}
    if mode == "resume":
        env["RESUME"] = "latest"
        env["MODEL_NAME_OR_PATH"] = "/missing/original/base"
        env["CAMERA_CHECKPOINT"] = ""
    result = subprocess.run(["bash", f"shell/{script}"], env=env, capture_output=True, text=True, check=True)
    argv = json.loads(result.stdout)
    flag = {"base": "--model", "checkpoint": "--checkpoint", "resume": "--resume"}[mode]
    assert argv[argv.index(flag) + 1] == ("latest" if mode == "resume" else str(model))
    assert sum(arg in ("--model", "--checkpoint", "--resume") for arg in argv) == 1
    assert "--head-dim" not in argv


@pytest.mark.parametrize("script,rank,gpus,batch", [
    ("train_camera_kev_lora.sh", 64, 8, 16),
    ("train_camera_kev_lora_goutu.sh", 64, 8, 16),
    ("train_camera_kev_lora_seka.sh", 128, 4, 8),
])
def test_camera_lora_shell_defaults_and_resume(tmp_path, script, rank, gpus, batch):
    launcher = tmp_path / "torchrun"
    launcher.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    launcher.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    env = {"PATH": os.environ["PATH"], "TORCHRUN": str(launcher), "MODEL_NAME_OR_PATH": str(model),
           "META_PATH": "/data/kev-collection.json", "OUTPUT_DIR": str(tmp_path / "out")}
    for resume in (False, True):
        if resume:
            env.update(RESUME="latest", MODEL_NAME_OR_PATH="/missing/base")
        result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "shell/lora" / script),
                                 "--max-steps", "1"], cwd=tmp_path, env=env,
                                capture_output=True, text=True, check=True)
        argv = json.loads(result.stdout)
        assert argv[argv.index("--lora-r") + 1] == str(rank)
        assert f"--nproc-per-node={gpus}" in argv
        assert argv[argv.index("--batch") + 1] == str(batch)
        assert argv[argv.index("--accum") + 1] == "1"
        assert argv[argv.index("--train-meta") + 1] == "/data/kev-collection.json"
        assert argv[argv.index("--lr") + 1] == "1e-4"
        assert argv[-2:] == ["--max-steps", "1"]
        flag = "--resume" if resume else "--model"
        assert argv[argv.index(flag) + 1] == ("latest" if resume else str(model))
        assert sum(arg in ("--model", "--checkpoint", "--resume") for arg in argv) == 1


def test_camera_lora_roundtrip_preserves_frozen_base_and_updated_adapter(tmp_path, tiny_checkpoints, monkeypatch):
    """Synthetic Qwen tensors exercise PEFT gradients and the standalone save format."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kev.camera import lora

    def make_camera():
        camera = tiny_checkpoints()
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, bos_token_id=1, eos_token_id=0, pad_token_id=0)
        config._attn_implementation = "eager"
        camera.language_model = Qwen3ForCausalLM(config)
        camera.config.llm_config = config
        camera.projector = nn.Linear(1, 16)
        return camera

    def save_camera(self, path, *, state_dict=None, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "config.json", {"model_type": "camera_agent", "llm_config": {"hidden_size": 16}})
        torch.save(self.state_dict() if state_dict is None else state_dict, path / "tiny.pt")

    def load_camera(path, **kwargs):
        camera = make_camera()
        camera.load_state_dict(torch.load(Path(path) / "tiny.pt", weights_only=True), strict=True)
        return camera

    monkeypatch.setattr(TinyCamera, "save_pretrained", save_camera)
    monkeypatch.setattr(camera_model, "load_camera", load_camera)
    model = CameraDecisionModel(make_camera(), TinyTokenizer(), head_dim=3, max_length=500)
    base = {name: value.clone() for name, value in model.camera.state_dict().items()}
    lora.attach(model, 2)
    model.camera.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert any("lora_" in name for name in trainable)
    assert all("lora_" in name or name.startswith("head.") for name in trainable)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model.train()
    logits = model([record()], torch.tensor([1.0]))[0]
    sum(F.cross_entropy(z.unsqueeze(0), torch.tensor([q["label"]]))
        for q, z in zip(record()["questions"], logits)).backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0
               for name, value in model.named_parameters() if "lora_B" in name)
    assert all(value.grad is None for name, value in model.camera.named_parameters() if "lora_" not in name)
    optimizer.step()
    model.eval()
    expected = model.probabilities(record(), torch.tensor([1.0]))
    path = tmp_path / "checkpoint"
    model.save(path, source=tmp_path / "absent-original-base", image_size=8)
    saved_base = torch.load(path / "tiny.pt", weights_only=True)
    assert_tensor_tree_equal(saved_base, base)
    loaded, metadata = camera_model.load(path, device="cpu", dtype=torch.float32)
    assert metadata["lora_rank"] == 2
    assert loaded.lora_rank == 2
    actual = loaded.probabilities(record(), torch.tensor([1.0]))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("options", [["--lora-r", "-1"], ["--lora-r", "2", "--freeze-llm"]])
def test_camera_lora_rejects_invalid_trainability(options):
    with pytest.raises(SystemExit):
        parse_args(["--train-data", "/data/train.jsonl", *options])
