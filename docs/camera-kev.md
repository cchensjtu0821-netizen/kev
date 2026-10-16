# CameraAgent 0.4B image decisions

This path loads the dense `CameraAgentModel` checkpoint, uses its DINOv3 encoder and image projector, and trains a Kev pointer head over Qwen3 hidden states. It accepts both the 39-question `noul` records and the 24-question `noul` plus `choice` records. The two formats can be mixed in the same training JSONL or collection. The loss averages over questions within each image, then over images in an optimizer step.

The adapter uses one 384-pixel tile per image, ImageNet normalization, `ps_version=v2`, and the checkpoint's `extract_feature()` method. The single-tile choice follows the launcher's `dynamic_image_size=False` and `use_thumbnail=False` settings; confirm the normalization against that machine's CameraModel `build_transform` before the full run. It reads the number of projected visual tokens from the model at runtime. Each question runs in an independent causal row; the image encoder is recomputed for each `--questions-per-pass` chunk to limit GPU memory. The current path handles one image per record. It does not use the older ViT bridge in `kev.vision`.

## Data

Use the existing [image decision format](vision-training.md#data). A mixed record can look like this:

```json
{"image":"images/001.jpg","state":"","questions":{"mode_夜景":{"type":"noul","instructions":"这张图是夜景吗？","label":true},"seka_色调":{"type":"choice","instructions":"这幅画面的色调是哪一项？","criteria":{"暖调":null,"冷调":null,"中性色调":null,"混合色调":null},"label":"冷调"}}}
```

For a collection, set `TRAIN_META` and `VAL_META` to JSON files with CameraModel's `root`, `annotation`, and `repeat_time` fields. For single JSONL files, set `TRAIN_DATA` and `VAL_DATA`. Image paths in JSONL are relative to the JSONL directory; image paths in a collection are relative to its `root`. Validation collections must have `repeat_time=1`.

The preflight checks every label and image path and rejects a train/validation image path overlap. Split by original image or scene before converting 39 questions into 24; the path check cannot detect duplicate image bytes under different filenames. Keep the validation images separate from training.

```bash
python -m kev.camera.train --train-data /data/train.jsonl --val-data /data/val.jsonl --validate-data
```

## Train

Run this on the machine with the checkpoint directory and its `modeling_camera_agent.py`, `configuration_camera_agent.py`, tokenizer files, and weights. The custom code and its dependencies must work in the selected CameraModel Python environment. `trust_remote_code=True` executes that checkpoint's Python code. The checkpoint supplied in the request is not mounted in this repository's environment, so its runtime API still needs a one-step check there.

```bash
TRAIN_DATA=/data/train.jsonl VAL_DATA=/data/val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-04b \
  GPUS=8 bash shell/train_camera_kev.sh --max-steps 1

TRAIN_DATA=/data/train.jsonl VAL_DATA=/data/val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-04b-full \
  GPUS=8 bash shell/train_camera_kev.sh
```

The shell defaults to the supplied 0.4B checkpoint path, the CameraModel conda environment, eight GPUs, bf16 and FlashAttention 2. It uses AdamW with the source launcher's 0.9/0.95 betas, `1e-8` epsilon, `0.01` weight decay, 3% warmup, cosine decay, gradient checkpointing, and 2560-token rows. Backbone learning rate is `1e-5`; the newly initialized pointer head uses `1e-4`. `GRADIENT_ACC=8` means eight images per optimizer step per GPU. `QUESTIONS_PER_PASS=4` limits simultaneous question graphs. Set `GPUS=1` for a single GPU. `CAMERA_ENV_BIN`, `CAMERA_CODE_DIR`, `TORCHRUN`, `MODEL_NAME_OR_PATH`, `IMAGE_SIZE`, `MAX_LENGTH`, `SAVE_STEPS`, and learning rate variables can override the defaults. `--freeze-backbone`, `--freeze-llm`, and `--freeze-mlp` are available for ablations.

This trainer uses PyTorch distributed data parallel instead of the source script's DeepSpeed ZeRO-2 and packed generative loss. It trains the dense model and pointer head on image-level decision loss. It saves `final/camera/`, `final/pointer.pt`, `final/camera_decision.json`, and per-epoch validation metrics. Intermediate checkpoints go under `checkpoint-<step>/` when `SAVE_STEPS` is reached. An existing output directory is refused.

## Predict

Create a JSON request with `image`, `state`, and unlabelled `questions`, then run:

```bash
python -m kev.camera.predict \
  --checkpoint /data/runs/camera-kev-04b-full/final \
  --request /data/request.json
```

The result maps every question ID to its option probabilities. For a `noul` question the keys are `false` and `true`. For `choice` they are its `criteria` keys. Fit a temperature on held-out images before using the probabilities as calibrated confidence.
