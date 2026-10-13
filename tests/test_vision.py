"""Local contracts for image rows and CameraModel-style data collections."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kev.model import PointerHead
from kev.vision import VisionDecisionModel
from kev.vision_data import ImageDecisionDataset
from kev.vision_camera_bench import request_from_tsv


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, position_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3, num_register_tokens=0)

    def forward(self, *, pixel_values):
        value = pixel_values.reshape(1, 1)
        patches = torch.stack((value, value + 1, torch.ones_like(value)), dim=-1).expand(1, 4, 3)
        return SimpleNamespace(last_hidden_state=torch.cat((torch.zeros(1, 1, 3), patches), dim=1))


def encoding():
    return {"ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "seg": [0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "pos": list(range(6)) + list(range(2, 6)),
            "decide_idx": [5, 9], "opt_idx": [[3, 4], [7, 8]]}


def test_image_bridge_keeps_questions_isolated_and_trains_projector():
    torch.manual_seed(4)
    decision = SimpleNamespace(lm=TinyText(), head=PointerHead(4, 3), device="cpu", option_isolation=False)
    model = VisionDecisionModel(decision, TinyVision(), None, visual_tokens=2)
    first = model(encoding(), torch.tensor([1.0]))
    changed = encoding()
    changed["ids"][-4:] = [11, 12, 13, 14]
    second = model(changed, torch.tensor([1.0]))
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
    changed_image = model(encoding(), torch.tensor([2.0]))
    assert not torch.equal(first[0], changed_image[0])
    first[0].sum().backward()
    assert model.projector[1].weight.grad is not None
    assert model.projector[1].weight.grad.abs().sum() > 0


def test_collection_uses_lazy_rows_and_virtual_repeats(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"sample")
    row = {"image": "a.jpg", "state": "", "questions": {
        "kind": {"type": "choice", "instructions": "What?", "criteria": {"a": "A", "b": "B"}, "label": "a"}}}
    annotation = tmp_path / "train.jsonl"
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"photos": {"root": str(root), "annotation": str(annotation), "repeat_time": 2}}), encoding="utf-8")
    dataset = ImageDecisionDataset(meta=meta)
    assert len(dataset) == 2
    assert dataset[0]["record"] == dataset[1]["record"]
    assert dataset[0]["record"]["questions"][0]["label"] == 0


def test_collection_rejects_missing_label_and_escaped_path(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"sample")
    path = tmp_path / "rows.jsonl"
    row = {"image": "a.jpg", "questions": {
        "kind": {"type": "choice", "instructions": "What?", "criteria": {"a": "A", "b": "B"}}}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlabelled"):
        ImageDecisionDataset(data=path)[0]
    row["questions"]["kind"]["label"] = "a"
    row["image"] = "../a.jpg"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes root"):
        ImageDecisionDataset(data=path)[0]


def test_camera_tsv_request_keeps_hint_and_answer():
    row = {"index": "7", "image": "unused", "question": "Which one?", "hint": "Look left.",
           "answer": "B", "A": "dog", "B": "cat", "C": "", "D": "", "E": ""}
    request = request_from_tsv(row, "mmbench_en")
    assert request["questions"]["answer"] == {
        "type": "choice", "instructions": "Look left.\nWhich one?",
        "criteria": {"A": "dog", "B": "cat"}, "label": "B", "src": "mmbench_en"}
