"""Image tokens in Kev's shared state, with its existing pointer readout."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, SERVE_MAX_STATE_8K, rows_of, training_context


class VisionDecisionModel(nn.Module):
    """Run a ViT once, then put projected patch tokens after Kev's state marker.

    Every question gets an independent causal row, including on hybrid Qwen
    backbones. The image is inside the common state and never leaks between
    question branches. The visual bridge is separate from text checkpoints.
    """

    def __init__(self, decision, vision, processor, *, visual_tokens=16, train_vision=False):
        super().__init__()
        if visual_tokens < 1:
            raise ValueError("visual_tokens must be positive")
        if decision.option_isolation:
            raise ValueError("option isolation is not supported by the vision row path")
        self.decision = decision
        self.vision = vision
        self.processor = processor
        self.visual_tokens = visual_tokens
        self.train_vision = train_vision
        self.max_state = SERVE_MAX_STATE_8K
        vision_dim = vision.config.hidden_size
        text_dim = decision.lm.config.hidden_size
        self.projector = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, text_dim),
                                       nn.GELU(), nn.Linear(text_dim, text_dim))
        self.vision.requires_grad_(train_vision)
        self.vision.to(decision.device)
        self.projector.to(decision.device)

    def pixels(self, image):
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.decision.device)

    def encode(self, tok, record, *, max_state=None):
        context = training_context(max_state or self.max_state)
        return self.decision.encode(tok, record, max_state=context["max_state"] - self.visual_tokens,
                                    max_branch=context["max_branch"] - self.visual_tokens, strict=True)

    def image_tokens(self, pixel_values):
        if not self.train_vision:
            self.vision.eval()
        with torch.set_grad_enabled(self.train_vision and self.training):
            hidden = self.vision(pixel_values=pixel_values).last_hidden_state
        if hidden.shape[1] > 1:
            reserved = 1 + int(getattr(self.vision.config, "num_register_tokens", 0) or 0)
            hidden = hidden[:, reserved:]  # retain the patch grid
        if hidden.shape[1] == 0:
            raise ValueError("vision encoder returned no patch tokens")
        hidden = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.visual_tokens).transpose(1, 2)
        return self.projector(hidden.to(self.projector[1].weight.dtype))

    def forward(self, enc, pixel_values):
        """Return one logits tensor per question, retaining projector gradients."""
        state_ids, _, branches = rows_of(enc)
        if len(state_ids) + self.visual_tokens > SERVE_MAX_STATE:
            raise ValueError("state plus image tokens exceeds the serving context")
        image = self.image_tokens(pixel_values)
        if image.shape[0] != 1:
            raise ValueError("one image per record is supported")
        embed = self.decision.lm.get_input_embeddings()
        dtype = embed.weight.dtype
        prefix_ids = torch.tensor(state_ids, device=self.decision.device)
        prefix = embed(prefix_ids).unsqueeze(0)
        prefix = torch.cat((prefix[:, :1], image.to(dtype), prefix[:, 1:]), dim=1)
        results = []
        for branch in branches:
            if prefix.shape[1] + len(branch["ids"]) > SERVE_MAX_BRANCH:
                raise ValueError("question row plus image tokens exceeds the serving context")
            branch_ids = torch.tensor(branch["ids"], device=self.decision.device)
            row = torch.cat((prefix, embed(branch_ids).unsqueeze(0)), dim=1)
            positions = torch.arange(row.shape[1], device=row.device).unsqueeze(0)
            hidden = self.decision.lm(inputs_embeds=row, position_ids=positions,
                                      attention_mask=torch.ones_like(positions)).last_hidden_state[0].float()
            offset = prefix.shape[1]
            opts = torch.tensor(branch["opts"], device=row.device) + offset
            results.append(self.decision.head(hidden[offset + branch["decide"]], hidden[opts]))
        return results

    @torch.no_grad()
    def probs(self, enc, pixel_values):
        return [F.softmax(logits, dim=-1).cpu() for logits in self.forward(enc, pixel_values)]

    def save_bridge(self, out, *, text_checkpoint, vision_name, vision_revision=None, train_text=False):
        """Save the trained visual parts and enough provenance to reload them."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"projector": self.projector.state_dict(),
                    "vision": self.vision.state_dict() if self.train_vision else None}, out / "vision.pt")
        if train_text:
            self.decision.lm.save_pretrained(out / "adapter")
            torch.save(self.decision.head.state_dict(), out / "pointer.pt")
        if Path(text_checkpoint).is_dir():
            text_checkpoint = str(Path(text_checkpoint).resolve())
        if Path(vision_name).is_dir():
            vision_name = str(Path(vision_name).resolve())
        config = json.dumps({
            "text_checkpoint": text_checkpoint, "vision_model": vision_name,
            "vision_revision": vision_revision, "visual_tokens": self.visual_tokens,
            "train_vision": self.train_vision, "train_text": train_text,
            "max_state": self.max_state,
        }, indent=2) + "\n"
        (out / "vision_config.json").write_text(config, encoding="utf-8")


def load_vision_model(path, device="cpu"):
    """Load a visual bridge. Text checkpoints keep their original format."""
    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModel

    from .checkpoint import Checkpoint, LoadOptions

    path = Path(path)
    config = json.loads((path / "vision_config.json").read_text(encoding="utf-8"))
    checkpoint = Checkpoint(config["text_checkpoint"])
    if config["train_text"]:
        if checkpoint.full:
            raise ValueError("train_text bridge requires an adapter text checkpoint")
        from .model import DecisionModel, load_tokenizer
        meta = checkpoint.meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        decision = DecisionModel(meta.base, tok, device, revision=meta.base_revision, head_dim=meta.head_dim,
                                 option_isolation=meta.option_isolation,
                                 dtype=torch.bfloat16 if meta.weights_dtype == "bf16" else torch.float32)
        decision.lm = PeftModel.from_pretrained(decision.lm, str(path / "adapter")).to(device)
        decision.head.load_state_dict(torch.load(path / "pointer.pt", map_location="cpu", weights_only=True))
        decision.head.temperature = 1.0
    else:
        tok, decision = checkpoint.load(device, LoadOptions(merge=False))
    decision.head.temperature = 1.0
    revision = config.get("vision_revision")
    vision = AutoModel.from_pretrained(config["vision_model"], revision=revision)
    processor = AutoImageProcessor.from_pretrained(config["vision_model"], revision=revision, use_fast=False)
    model = VisionDecisionModel(decision, vision, processor, visual_tokens=config["visual_tokens"],
                                train_vision=config["train_vision"])
    model.max_state = config.get("max_state", SERVE_MAX_STATE_8K)
    saved = torch.load(path / "vision.pt", map_location="cpu", weights_only=True)
    model.projector.load_state_dict(saved["projector"])
    if saved["vision"] is not None:
        model.vision.load_state_dict(saved["vision"])
    return tok, model.eval()
