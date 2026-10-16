"""Read CameraAgent image features with Kev's option pointer head."""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.model import PointerHead, user_tokens


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def image_tensor(image, size=384):
    """CameraModel's fixed-size, single-tile ImageNet normalization path."""
    from PIL import Image
    import numpy as np

    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
    mean = torch.tensor(IMAGE_MEAN)[:, None, None]
    std = torch.tensor(IMAGE_STD)[:, None, None]
    return ((pixels - mean) / std).unsqueeze(0)


def _tokens(tok, text):
    return user_tokens(tok, text)


def encode_record(tok, record, image_tokens, max_length):
    """Tokenize one image and independent question rows without adding vocabulary entries."""
    if image_tokens < 1:
        raise ValueError("CameraAgent returned no image tokens")
    image_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    if image_id is None or tok.convert_ids_to_tokens(image_id) != "<IMG_CONTEXT>":
        raise ValueError("tokenizer needs the CameraAgent <IMG_CONTEXT> token")
    prefix = _tokens(tok, "State:\n" + record["state"] + "\nImage: <img>")
    suffix = _tokens(tok, "</img>\n")
    prefix += [image_id] * image_tokens + suffix
    image_start = len(prefix) - image_tokens - len(suffix)
    rows = []
    for question in record["questions"]:
        branch = _tokens(tok, "Question: " + question["instr"] + "\n")
        ends = []
        for option in question["options"]:
            option_ids = _tokens(tok, "Option: " + option + "\n")
            if not option_ids:
                raise ValueError("an option encoded to zero tokens")
            branch.extend(option_ids)
            ends.append(len(branch) - 1)
        branch.extend(_tokens(tok, "Answer:"))
        if len(prefix) + len(branch) > max_length:
            raise ValueError(f"image question has {len(prefix) + len(branch)} tokens, over max_length={max_length}")
        rows.append({"ids": prefix + branch, "image_start": image_start,
                     "option_indices": [len(prefix) + end for end in ends],
                     "decide_index": len(prefix) + len(branch) - 1})
    return rows


class CameraDecisionModel(nn.Module):
    def __init__(self, camera, tokenizer, *, head_dim=256, max_length=2560):
        super().__init__()
        self.camera = camera
        self.tokenizer = tokenizer
        self.max_length = max_length
        for attr in ("extract_feature", "language_model"):
            if not hasattr(camera, attr):
                raise TypeError(f"CameraAgent checkpoint needs {attr}; inspect its modeling_camera_agent.py")
        language = camera.language_model
        if not hasattr(language, "model") or not hasattr(language, "get_input_embeddings"):
            raise TypeError("CameraAgent language_model must expose model and get_input_embeddings")
        llm_config = camera.config.llm_config
        hidden_size = llm_config["hidden_size"] if isinstance(llm_config, dict) else llm_config.hidden_size
        self.head = PointerHead(hidden_size, head_dim)

    def _features(self, pixels):
        visual = self.camera.extract_feature(pixels)
        if isinstance(visual, (tuple, list)):
            visual = visual[0]
        if visual.ndim != 3 or visual.shape[0] != 1 or visual.shape[2] != self.head.q.in_features:
            raise ValueError(f"expected one image with projected hidden size {self.head.q.in_features}, got {tuple(visual.shape)}")
        return visual

    def forward(self, record, pixels, *, question_start=0, question_end=None):
        visual = self._features(pixels)
        rows = encode_record(self.tokenizer, record, visual.shape[1], self.max_length)
        embedding = self.camera.language_model.get_input_embeddings()
        outputs = []
        for row in rows[question_start:question_end]:
            ids = torch.tensor(row["ids"], device=pixels.device).unsqueeze(0)
            inputs = embedding(ids)
            start = row["image_start"]
            inputs = torch.cat((inputs[:, :start], visual.to(inputs.dtype),
                                inputs[:, start + visual.shape[1]:]), dim=1)
            hidden = self.camera.language_model.model(
                inputs_embeds=inputs, attention_mask=torch.ones_like(ids),
                use_cache=False, return_dict=True,
            ).last_hidden_state[0].float()
            opts = torch.tensor(row["option_indices"], device=pixels.device)
            outputs.append(self.head(hidden[row["decide_index"]], hidden[opts]))
        return outputs

    @torch.no_grad()
    def probabilities(self, record, pixels):
        self.eval()
        return [F.softmax(logits, dim=-1).cpu() for logits in self(record, pixels)]

    def save(self, path, *, source, image_size):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.camera.save_pretrained(path / "camera", safe_serialization=True)
        self.tokenizer.save_pretrained(path / "camera")
        if Path(source).is_dir():
            for code in Path(source).glob("*.py"):
                shutil.copy2(code, path / "camera" / code.name)
        torch.save(self.head.state_dict(), path / "pointer.pt")
        (path / "camera_decision.json").write_text(json.dumps({
            "source": str(source), "image_size": image_size, "head_dim": self.head.q.out_features,
            "max_length": self.max_length, "format": 1,
        }, indent=2) + "\n", encoding="utf-8")


def load(path, *, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(path)
    metadata = json.loads((path / "camera_decision.json").read_text(encoding="utf-8"))
    base = path / "camera"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
    camera = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, dtype=dtype, attn_implementation="eager").to(device)
    model = CameraDecisionModel(camera, tokenizer, head_dim=metadata["head_dim"],
                                max_length=metadata["max_length"]).to(device)
    model.head.load_state_dict(torch.load(path / "pointer.pt", map_location="cpu", weights_only=True))
    return model.eval(), metadata
