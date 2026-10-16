# CameraAgent 0.4B image decisions

This path loads the dense `CameraAgentModel` checkpoint, uses its DINOv3 encoder and image projector, and trains a Kev pointer head over Qwen3 hidden states. It accepts both the 39-question `noul` records and the 24-question `noul` plus `choice` records. The two formats can be mixed in the same training JSONL or collection. The loss averages over questions within each image, then over images in an optimizer step.

The adapter uses one 384-pixel tile per image, ImageNet normalization, `ps_version=v2`, and the checkpoint's `extract_feature()` method. The single-tile choice follows the launcher's `dynamic_image_size=False` and `use_thumbnail=False` settings; confirm the normalization against that machine's CameraModel `build_transform` before the full run. It reads the number of projected visual tokens from the model at runtime. Training, validation and prediction place the image and state prefix once per image, followed by all its question branches in one packed forward pass. A block-causal mask lets each branch read the shared prefix and its own earlier tokens, but no other branch. Prefix activations and their keys and values are computed once while training gradients from every question flow through them. A batch whose records share the same question template and empty state has equal packed lengths, so it stacks without padding and shares one branch mask. Mixed templates or states still use padding. It does not use the older ViT bridge in `kev.vision`.

## Data

Use the existing [image decision format](vision-training.md#data). A mixed record can look like this:

```json
{"image":"images/001.jpg","state":"","questions":{"mode_夜景":{"type":"noul","instructions":"这张图是夜景吗？","label":true},"seka_色调":{"type":"choice","instructions":"这幅画面的色调是哪一项？","criteria":{"暖调":null,"冷调":null,"中性色调":null,"混合色调":null},"label":"冷调"}}}
```

For a collection, set `TRAIN_META` to a JSON file with CameraModel's `root`, `annotation`, and `repeat_time` fields. For a single JSONL file, set `TRAIN_DATA`. Set either `VAL_META` or `VAL_DATA` only if you have a separate image validation set; leave both empty to skip validation. Image paths in JSONL are relative to the JSONL directory; image paths in a collection are relative to its `root`. Validation collections must have `repeat_time=1`.

The preflight checks every label and image path. When validation data is supplied, it also rejects a train/validation image path overlap. Split by original image or scene before converting 39 questions into 24; the path check cannot detect duplicate image bytes under different filenames. Kev's existing frozen training and validation suites contain text decisions, not image paths, so this command does not use them as an image validation set.

The dataset keeps JSONL byte offsets and image paths in memory, not decoded images. Startup scans the annotations once to check labels and paths. During training, PyTorch DataLoader workers read individual JSONL rows and open, resize, and normalize their images as needed; `DATALOADER_NUM_WORKERS` defaults to 4 per GPU process. The workers prefetch a small queue of samples. The same worker path handles validation images when a validation set is provided.

```bash
python -m kev.camera.train --train-data /data/train.jsonl --validate-data
```

## Train

Run this on the machine with the checkpoint directory and its `modeling_camera_agent.py`, `configuration_camera_agent.py`, tokenizer files, and weights. Hugging Face's local `auto_map` loader reads those Python modules from the model directory. `CAMERA_CODE_DIR` is optional: set it only if those modules import other packages from a separate CameraModel source tree. Adding a directory to `PYTHONPATH` does not replace missing `modeling_camera_agent.py` or `configuration_camera_agent.py` in the checkpoint directory. The custom code and its dependencies must work in the selected Python environment. `trust_remote_code=True` executes that checkpoint's Python code. The checkpoint supplied in the request is not mounted in this repository's environment, so its runtime API still needs a one-step check there.

```bash
TRAIN_DATA=/data/train.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-04b \
  GPUS=8 bash shell/train_camera_kev.sh --max-steps 1

TRAIN_DATA=/data/train.jsonl VAL_DATA=/data/val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-04b-full \
  GPUS=8 bash shell/train_camera_kev.sh

TRAIN_DATA=/data/train.jsonl FREEZE_BACKBONE=true FREEZE_MLP=true \
  MAX_STEPS=1 bash shell/train_camera_kev.sh
```

The shell defaults to the supplied 0.4B checkpoint path, the CameraModel conda environment, eight GPUs and bf16. Model loading explicitly selects eager attention for compatibility with that environment's Transformers version, including when reloading for prediction. It uses AdamW with the source launcher's 0.9/0.95 betas, `1e-8` epsilon, `0.01` weight decay, 3% warmup, cosine decay, gradient checkpointing, and 2560-token row admission. Backbone learning rate is `1e-5`; the newly initialized pointer head uses `1e-4`. As in CameraModel's launcher, `PER_DEVICE_BATCH_SIZE` and `GRADIENT_ACC` are separate settings. Their defaults are 8 and 2, so each optimizer step covers up to 16 images per GPU, or 128 across eight GPUs. The forward pass processes eight packed image records together. Every task of an image shares the same packed pass, so there is no `QUESTIONS_PER_PASS` setting. Set `GPUS=1` for a single GPU. `CAMERA_ENV_BIN`, `CAMERA_CODE_DIR`, `TORCHRUN`, `MODEL_NAME_OR_PATH`, `IMAGE_SIZE`, `MAX_LENGTH`, `HEAD_DIM`, `SAVE_STEPS`, `MAX_STEPS`, `SEED`, and learning rate variables can override the defaults. `FREEZE_BACKBONE`, `FREEZE_LLM`, and `FREEZE_MLP` accept `true`/`false` or `1`/`0` and default to `false`. The corresponding `--freeze-*` flags and other trainer options can also be passed after the shell script name.

Rank 0 prints a JSON start record and one progress record after every optimizer step. Each record shows the current step and total steps, progress percentage, mean loss for that step across GPUs, processed images per second, elapsed time, estimated remaining time and both learning rates. The existing per-epoch metrics continue to be written to `training_metrics.jsonl`.

This trainer uses PyTorch distributed data parallel instead of the source script's DeepSpeed ZeRO-2 and packed generative loss. It loads the checkpoint's language model and its LM head, but does not run the LM head or generate text. The LM head is frozen. A new Kev pointer head starts from random weights, reads option-boundary hidden states, and is trained together with the selected CameraAgent weights. The save includes the unchanged LM head for CameraAgent checkpoint compatibility, plus `final/pointer.pt` for the trained decision head. It also writes `final/camera_decision.json` and per-epoch training metrics; validation metrics appear only when validation data was supplied. Intermediate checkpoints go under `checkpoint-<step>/` when `SAVE_STEPS` is reached. An existing output directory is refused.

## Predict

Create a JSON request with `image`, `state`, and unlabelled `questions`, then run:

```bash
python -m kev.camera.predict \
  --checkpoint /data/runs/camera-kev-04b-full/final \
  --request /data/request.json
```

The result maps every question ID to its option probabilities. For a `noul` question the keys are `false` and `true`. For `choice` they are its `criteria` keys. Fit a temperature on held-out images before using the probabilities as calibrated confidence.
