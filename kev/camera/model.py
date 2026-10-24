"""Read CameraAgent image features with Kev's option pointer head."""

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.model import PointerHead, branch_mask_batch, user_tokens
from kev.suite import read_json, write_json

from .objectives import question_probabilities


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def load_camera(path, *, dtype=torch.bfloat16):
    """Configure the nested Qwen model before CameraAgent constructs it."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_config = config.llm_config
    if isinstance(text_config, dict):
        text_config.update(_attn_implementation="eager", attn_implementation="eager")
    else:
        text_config._attn_implementation = "eager"
        text_config.attn_implementation = "eager"
    return AutoModelForCausalLM.from_pretrained(
        path, config=config, trust_remote_code=True, dtype=dtype,
        attn_implementation="eager", use_flash_attn=False,
    )


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
                     "prefix_length": len(prefix),
                     "option_indices": [len(prefix) + end for end in ends],
                     "decide_index": len(prefix) + len(branch) - 1})
    return rows


def pack_record(rows, question_start=0, question_end=None):
    """Place the shared image prefix once and keep every question in its own causal branch."""
    prefix_length = rows[0]["prefix_length"]
    ids = rows[0]["ids"][:prefix_length]
    segments = [0] * prefix_length
    positions = list(range(prefix_length))
    readouts = []
    for question_index, row in enumerate(rows[question_start:question_end], start=1):
        branch = row["ids"][prefix_length:]
        base = len(ids) - prefix_length
        ids.extend(branch)
        segments.extend([question_index] * len(branch))
        positions.extend(range(prefix_length, prefix_length + len(branch)))
        readouts.append((base + row["decide_index"],
                         [base + index for index in row["option_indices"]]))
    return {"ids": ids, "segments": segments, "positions": positions,
            "image_start": rows[0]["image_start"], "readouts": readouts}


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
        for name, module in language.named_modules():
            if name not in ("", "model") and not name.endswith(".self_attn"):
                continue
            implementation = getattr(module.config, "_attn_implementation", None)
            if implementation != "eager":
                raise ValueError(f"CameraAgent {name or 'language_model'} uses attention backend "
                                 f"{implementation!r}; the 4D question-branch mask requires eager. "
                                 "Load with load_camera() to configure llm_config and use_flash_attn=False.")
        llm_config = camera.config.llm_config
        hidden_size = llm_config["hidden_size"] if isinstance(llm_config, dict) else llm_config.hidden_size
        self.head = PointerHead(hidden_size, head_dim)
        self.lora_rank = 0
        self.question_layout = "original"

    def _features(self, pixels):
        visual = self.camera.extract_feature(pixels)
        if isinstance(visual, (tuple, list)):
            visual = visual[0]
        if visual.ndim != 3 or visual.shape[0] != 1 or visual.shape[2] != self.head.q.in_features:
            raise ValueError(f"expected one image with projected hidden size {self.head.q.in_features}, got {tuple(visual.shape)}")
        return visual

    def forward(self, record, pixels, *, question_start=0, question_end=None):
        if isinstance(record, list):
            return self.forward_batch(record, pixels, question_start=question_start,
                                      question_end=question_end)
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

    def forward_batch(self, records, pixels, *, question_start=0, question_end=None):
        visual = self.camera.extract_feature(pixels)
        if isinstance(visual, (tuple, list)):
            visual = visual[0]
        if visual.ndim != 3 or visual.shape[0] != len(records) or visual.shape[2] != self.head.q.in_features:
            raise ValueError(f"expected {len(records)} images with projected hidden size "
                             f"{self.head.q.in_features}, got {tuple(visual.shape)}")
        packed = [pack_record(encode_record(self.tokenizer, record, visual.shape[1], self.max_length),
                              question_start, question_end) for record in records]
        if not any(item["readouts"] for item in packed):
            return [[] for _ in records]
        embedding = self.camera.language_model.get_input_embeddings()
        max_length = max(len(item["ids"]) for item in packed)
        same_length = all(len(item["ids"]) == max_length for item in packed)
        padded = []
        positions = torch.zeros((len(packed), max_length), device=pixels.device, dtype=torch.long)
        for index, item in enumerate(packed):
            ids = torch.tensor(item["ids"], device=pixels.device)
            embedded = embedding(ids)
            start = item["image_start"]
            value = torch.cat((embedded[:start], visual[index].to(embedded.dtype),
                               embedded[start + visual.shape[1]:]), dim=0)
            padded.append(value if same_length else F.pad(value, (0, 0, 0, max_length - len(value))))
            positions[index, :len(ids)] = torch.tensor(item["positions"], device=pixels.device)
        inputs = torch.stack(padded)
        same_segments = all(item["segments"] == packed[0]["segments"] for item in packed)
        mask = branch_mask_batch([packed[0]["segments"]] if same_segments else
                                 [item["segments"] for item in packed], pixels.device,
                                 dtype=inputs.dtype, length=max_length)
        if same_segments:
            mask = mask.expand(len(packed), -1, -1, -1)
        hidden = self.camera.language_model.model(
            inputs_embeds=inputs, position_ids=positions, attention_mask=mask,
            use_cache=False, return_dict=True,
        ).last_hidden_state.float()
        return [[self.head(hidden[index, decide], hidden[index, torch.tensor(options, device=pixels.device)])
                 for decide, options in item["readouts"]] for index, item in enumerate(packed)]

    @torch.no_grad()
    def probabilities(self, record, pixels):
        self.eval()
        return [question_probabilities(q, z).cpu() for q, z in zip(record["questions"], self([record], pixels)[0])]

    def save(self, path, *, source, image_size):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.lora_rank:
            from . import lora

            lora.save(self, path)
        else:
            self.camera.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        if Path(source).is_dir():
            for code in Path(source).glob("*.py"):
                if code.resolve() != (path / code.name).resolve():
                    shutil.copy2(code, path / code.name)
        torch.save(self.head.state_dict(), path / "pointer.pt")
        config_path = path / "config.json"
        config = read_json(config_path)
        config["kev"] = {
            "source": str(source), "image_size": image_size, "head_dim": self.head.q.out_features,
            "max_length": self.max_length, "format": 2, "head_type": "pointer",
            "head_weights": "pointer.pt", "image_mean": IMAGE_MEAN, "image_std": IMAGE_STD,
            "image_tiles": 1, "question_attention": "block_causal",
        }
        if self.lora_rank:
            config["kev"]["lora_rank"] = self.lora_rank
        if self.question_layout != "original":
            config["kev"]["question_layout"] = self.question_layout
        write_json(config_path, config)


def checkpoint_config(path):
    """Return the backbone directory and Kev settings for either checkpoint format."""
    path = Path(path)
    if (path / "camera_decision.json").is_file():
        metadata = read_json(path / "camera_decision.json")
        if metadata["format"] != 1:
            raise ValueError(f"unsupported Camera Kev checkpoint format: {metadata['format']}")
        return path / "camera", metadata
    config = read_json(path / "config.json")
    metadata = config.get("kev")
    if not metadata or metadata.get("format") != 2:
        raise ValueError(f"{path} is not a Camera Kev checkpoint; use --model for a raw CameraAgent")
    return path, metadata


def load(path, *, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoTokenizer

    path = Path(path)
    base, metadata = checkpoint_config(path)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
    camera = load_camera(base, dtype=dtype).to(device)
    model = CameraDecisionModel(camera, tokenizer, head_dim=metadata["head_dim"],
                                max_length=metadata["max_length"]).to(device)
    if metadata.get("lora_rank"):
        from . import lora

        lora.attach(model, metadata["lora_rank"], adapter=path / "llm_adapter")
    model.head.load_state_dict(torch.load(path / metadata.get("head_weights", "pointer.pt"),
                                         map_location="cpu", weights_only=True))
    model.question_layout = metadata.get("question_layout", "original")
    return model.eval(), metadata
