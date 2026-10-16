"""CameraAgent adapter contracts without private model weights."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kev.camera.model import CameraDecisionModel, encode_record
from kev.camera.train import datasets, parse_args


class TinyTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[2 + ord(character) % 126 for character in text])

    def convert_tokens_to_ids(self, token):
        return 1 if token == "<IMG_CONTEXT>" else 0

    def convert_ids_to_tokens(self, token):
        return "<IMG_CONTEXT>" if token == 1 else "<unk>"


class TinyBackbone(nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.embeddings = nn.Embedding(128, 4)

    def get_input_embeddings(self):
        return self.embeddings


class TinyCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguage()
        self.projector = nn.Linear(1, 4)
        self.config = SimpleNamespace(llm_config=SimpleNamespace(hidden_size=4))

    def extract_feature(self, pixels):
        return self.projector(pixels.reshape(1, 1, 1)).expand(1, 3, 4)


def record():
    return {"state": "photo", "questions": [
        {"instr": "Is there a cat?", "options": ["no", "yes"], "label": 1},
        {"instr": "Which colour?", "options": ["red", "blue", "none"], "label": 0},
    ]}


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
    train, val = datasets(args)
    assert [q["label"] for q in train[0]["record"]["questions"]] == [0, 0]
    assert len(val) == 1
    validation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="share 1 image paths"):
        datasets(args)


def test_dataset_preflight_allows_no_validation(tmp_path):
    (tmp_path / "train.jpg").write_bytes(b"image")
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"image": "train.jpg", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}) + "\n",
        encoding="utf-8")
    args = parse_args(["--train-data", str(training), "--validate-data"])
    train, valid = datasets(args)
    assert len(train) == 1
    assert valid is None
    training.write_text(json.dumps({"image": "missing.jpg", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}) + "\n",
        encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        datasets(args)
