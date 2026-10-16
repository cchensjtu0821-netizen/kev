"""CameraAgent adapter contracts without private model weights."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from kev.camera.data import CameraImageDataset
from kev.camera.model import CameraDecisionModel, encode_record, load_camera, pack_record
from kev.camera.train import datasets, evaluate, first_item, parse_args, train_batch


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


def test_image_decode_runs_through_dataloader_workers(tmp_path):
    Image.new("RGB", (12, 18), (255, 0, 0)).save(tmp_path / "train.png")
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"image": "train.png", "state": "", "questions": {
        "rain": {"type": "noul", "instructions": "Is it raining?", "label": False}}}) + "\n",
        encoding="utf-8")
    args = parse_args(["--train-data", str(training), "--validate-data"])
    records, _ = datasets(args)
    loader = DataLoader(CameraImageDataset(records, 32), batch_size=1, num_workers=2,
                        collate_fn=first_item)
    item = next(iter(loader))
    assert item["pixels"].shape == (1, 3, 32, 32)
    assert item["pixels"][0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)
    assert item["record"]["questions"][0]["label"] == 0
