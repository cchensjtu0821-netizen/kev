"""Attention LoRA for CameraAgent's dense Qwen decoder."""

from pathlib import Path

import torch.nn as nn


def attach(model, rank, *, adapter=None):
    """Freeze CameraAgent and train decoder adapters plus the Kev pointer head."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if rank <= 0:
        raise ValueError("Camera LoRA rank must be positive")
    decoder = model.camera.language_model.model
    if isinstance(decoder, PeftModel):
        raise ValueError("Camera decoder already has a LoRA adapter")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found = {name.rsplit(".", 1)[-1] for name, layer in decoder.named_modules()
             if isinstance(layer, nn.Linear)}
    if missing := set(targets) - found:
        raise ValueError(f"Camera decoder is missing attention projections: {sorted(missing)}")
    model.camera.requires_grad_(False)
    if adapter is None:
        config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                            bias="none", task_type="FEATURE_EXTRACTION", target_modules=targets)
        decoder = get_peft_model(decoder, config)
    else:
        decoder = PeftModel.from_pretrained(decoder, adapter, is_trainable=True)
        if decoder.peft_config["default"].r != rank:
            raise ValueError("Camera LoRA rank does not match the saved adapter")
    model.camera.language_model.model = decoder
    model.lora_rank = rank


def save(model, path):
    """Save a standalone base and a separate adapter without merging or rounding it."""
    decoder = model.camera.language_model.model
    decoder.save_pretrained(Path(path) / "llm_adapter", safe_serialization=True)
    # The outer CameraAgent serializer must receive the original decoder's keys.
    # These tensors share storage with the model; no second backbone is allocated.
    prefix = "language_model.model."
    wrapped = prefix + "base_model.model."
    state = {}
    for name, value in model.camera.state_dict().items():
        if name.startswith(wrapped):
            if ".lora_" in name:
                continue
            name = prefix + name[len(wrapped):].replace(".base_layer.", ".")
        state[name] = value
    model.camera.save_pretrained(path, state_dict=state, safe_serialization=True)
