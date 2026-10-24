# CameraAgent 0.4B image decisions

This path loads the dense `CameraAgentModel` checkpoint, uses its DINOv3 encoder and image projector, and trains a Kev pointer head over Qwen3 hidden states. It accepts the original 39-question `noul` records, 24-question `noul` plus `choice` records, and merged `multilabel` / `grouped_choice` records. Formats can be mixed in the same training JSONL or collection. The loss averages over label decisions and choice groups within each image, then over images in an optimizer step.

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

The shell defaults to the supplied 0.4B checkpoint path, the CameraModel conda environment, eight GPUs and bf16. Model loading explicitly selects eager attention for compatibility with that environment's Transformers version, including when reloading for prediction. It uses AdamW with the source launcher's 0.9/0.95 betas, `1e-8` epsilon, `0.01` weight decay, 3% warmup, cosine decay, gradient checkpointing, and 2560-token row admission. Backbone learning rate is `1e-5`; the newly initialized pointer head uses `1e-4`. As in CameraModel's launcher, `PER_DEVICE_BATCH_SIZE` and `GRADIENT_ACC` are separate settings. Their defaults are 8 and 2, so each optimizer step covers up to 16 images per GPU, or 128 across eight GPUs. The forward pass processes eight packed image records together. Every task of an image shares the same packed pass, so there is no `QUESTIONS_PER_PASS` setting. Set `GPUS=1` for a single GPU. `CAMERA_ENV_BIN`, `CAMERA_CODE_DIR`, `TORCHRUN`, `MODEL_NAME_OR_PATH`, `IMAGE_SIZE`, `MAX_LENGTH`, `HEAD_DIM`, `SAVE_STEPS`, `MAX_STEPS`, `STOP_AFTER`, `RESUME`, `SEED`, and learning rate variables can override the defaults. `FREEZE_BACKBONE`, `FREEZE_LLM`, and `FREEZE_MLP` accept `true`/`false` or `1`/`0` and default to `false`. The corresponding `--freeze-*` flags and other trainer options can also be passed after the shell script name.

Rank 0 prints a JSON start record and one progress record after every optimizer step. The `progress` field shows the current and total steps together, such as `12/300`. Each progress record also includes the percentage, mean loss for that step across GPUs, processed images per second, elapsed time, estimated remaining time and both learning rates. Like CameraModel, `elapsed` and `eta` use readable durations such as `0:03:20` and `2:15:00` (hours:minutes:seconds), with days included for runs over 24 hours. The existing per-epoch metrics continue to be written to `training_metrics.jsonl`.

Both training and prediction configure the nested `llm_config` for eager attention and pass `use_flash_attn=False` to the CameraAgent constructor. Setting only the outer model's `attn_implementation` can leave Qwen3 on FlashAttention, whose padding-mask handling cannot consume the four-dimensional branch mask. The adapter checks the language model and each attention layer before the first forward pass; `train_start` reports the effective `attention_backend`.

Training and prediction share an image reader that sets `PIL.Image.MAX_IMAGE_PIXELS = None` before opening an image, matching CameraModel's policy for local datasets. The setting runs inside each DataLoader worker as well, so high-resolution source images can be decoded and resized to the configured training size.

This trainer uses PyTorch distributed data parallel instead of the source script's DeepSpeed ZeRO-2 and packed generative loss. It loads the checkpoint's language model and its LM head, but does not run the LM head or generate text. The LM head is frozen. `train_camera_kev.sh` starts from a raw CameraAgent with a new random Kev pointer head. The head reads option-boundary hidden states and trains together with the selected CameraAgent weights. Passing a Kev checkpoint through `--model` is refused so that its trained decision head cannot be silently discarded.

## LoRA from the dense 0.4B model

The launchers under `shell/lora/` follow CameraModel's LoRA settings while using the dense 0.4B checkpoint and Kev's decision loss. They freeze all CameraAgent base weights, including the vision encoder, projector, embeddings and LM head. They train only Qwen attention `q_proj`, `k_proj`, `v_proj`, `o_proj` adapters and the Kev pointer head. LoRA alpha is twice the rank and dropout is 0.05. No vocabulary entries are added.

```bash
MODEL_NAME_OR_PATH=/data/models/camera-agent-04b \
  TRAIN_META=/data/kev/goutu.json \
  OUTPUT_DIR=/data/runs/kev-lora-goutu \
  bash shell/lora/train_camera_kev_lora_goutu.sh

MODEL_NAME_OR_PATH=/data/models/camera-agent-04b \
  TRAIN_META=/data/kev/seka.json \
  OUTPUT_DIR=/data/runs/kev-lora-seka \
  bash shell/lora/train_camera_kev_lora_seka.sh

# Generic launcher, with one GPU and a one-step integration check.
CUDA_VISIBLE_DEVICES=0 GPUS=1 PER_DEVICE_BATCH_SIZE=1 \
  TRAIN_DATA=/data/train.jsonl MODEL_NAME_OR_PATH=/data/models/camera-agent-04b \
  OUTPUT_DIR=/data/runs/kev-lora-check MAX_STEPS=1 \
  bash shell/lora/train_camera_kev_lora.sh
```

Goutu defaults to eight GPUs, batch 16 per GPU and rank 64. Seka defaults to GPUs 4-7, batch 8 per GPU and rank 128. Both use accumulation 1, three epochs, adapter and head learning rates of `1e-4`, and save every 2,000 steps. Override these with the same environment variables as the full-weight launcher plus `LORA_R`. `META_PATH` aliases `TRAIN_META`, and `NUM_TRAIN_EPOCHS` aliases `EPOCHS`. The collection's annotations must contain Kev image decision records as described above; CameraModel's generative conversation annotations need conversion first. Set validation data explicitly if needed.

Python also accepts `--lora-r 64`. A positive rank freezes the entire CameraAgent even without `--freeze-*` flags. Omit `FREEZE_LLM` and `--freeze-llm`, which would also freeze the adapters. The launcher uses Kev's existing DDP, bf16 and eager block-causal attention path. It does not use CameraModel's MoE objectives, DeepSpeed or generative packing plan. Batch defaults follow the reference scripts and still need a GPU memory check with the actual question count.

LoRA checkpoints include the original unmerged CameraAgent weights, `llm_adapter/` and `pointer.pt`; `config.json` records `kev.lora_rank`. They load without the original base directory through the same prediction and checkpoint commands. The base and adapter remain separate so saving does not round a merged update into bf16. `--checkpoint` inherits the saved rank; `--resume` also restores the optimizer and RNG. For the LoRA shell launcher, keep `LORA_R` and all training settings the same when setting `RESUME=latest`. The full-weight checkpoint launcher can continue a LoRA checkpoint without specifying `LORA_R`. `peft` must be installed in the CameraModel Python environment.

These scripts and Python modules have only been checked statically in this remote workspace. Run the one-step check on the GPU machine with the dense checkpoint and training data before starting a full run.

## Merge modes and palette questions

The mode launcher combines 16 Noul mode questions into one `multilabel` question. A 24-question record becomes nine questions. The modes-and-palette launcher also combines the eight palette dimensions into one `grouped_choice` question, leaving two questions per image. Both inherit the Seka launcher's rank 128, GPUs 4-7, batch 8 per GPU and accumulation 1. Override those settings as needed.

```bash
# Mode merging only; 39- and 24-question annotations are merged during loading.
MODEL_NAME_OR_PATH=/data/models/camera-agent-04b \
  TRAIN_DATA=/data/train.jsonl VAL_DATA=/data/val.jsonl \
  OUTPUT_DIR=/data/runs/kev-lora-modes \
  bash shell/lora/train_camera_kev_lora_modes.sh

# One mode question and one palette question.
MODEL_NAME_OR_PATH=/data/models/camera-agent-04b \
  TRAIN_META=/data/kev/train.json VAL_META=/data/kev/val.json \
  OUTPUT_DIR=/data/runs/kev-lora-modes-seka \
  bash shell/lora/train_camera_kev_lora_modes_seka.sh

# One-step check before a full run. Use a separate output directory.
CUDA_VISIBLE_DEVICES=0 GPUS=1 PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=1 \
  MODEL_NAME_OR_PATH=/data/models/camera-agent-04b TRAIN_DATA=/data/train.jsonl \
  OUTPUT_DIR=/data/runs/kev-lora-merged-check \
  bash shell/lora/train_camera_kev_lora_modes_seka.sh
```

Any existing launcher accepts `QUESTION_LAYOUT=modes` or `QUESTION_LAYOUT=all`; Python accepts `--question-layout`. The default `original` keeps the sample's supplied layout, including already merged samples. Checkpoints save the load-time layout so prediction and calibration apply the same transformation. Python inherits it when continuing from a checkpoint; keep it unchanged when using `RESUME=latest` with a shell launcher.

To write merged annotations instead of merging during loading, keep the output beside its input so relative image paths stay valid:

```bash
python -m scripts.convert_vision_labels /data/train.jsonl /data/train-modes.jsonl --merge modes
python -m scripts.convert_vision_labels /data/train.jsonl /data/train-merged.jsonl --merge all
python -m kev.camera.train --train-data /data/train.jsonl --question-layout all --validate-data
```

The converter accepts 39- or 24-question input when `--merge` is supplied. It retains the existing conflicting-label checks on the 39-question source. Without `--merge`, it retains the original 39-to-24 conversion. Neither operation changes image splits. Mode names and palette labels come from the same conversion rules as before.

A small example of the new formats follows. Production samples can list all 16 modes and eight palette dimensions.

```json
{
  "image": "images/001.jpg",
  "state": "",
  "questions": {
    "modes": {
      "type": "multilabel",
      "instructions": "选择这张图适用的全部模式。",
      "criteria": {"夜景": null, "大烟花": null, "雨景": null},
      "label": ["夜景", "大烟花"]
    },
    "palette": {
      "type": "grouped_choice",
      "instructions": "选择各维度的色卡标签，每个维度选一项。",
      "criteria": {
        "色调": {"暖调": null, "冷调": null, "中性色调": null, "混合色调": null},
        "时间": {"白天": null, "夜晚": null, "无": null}
      },
      "label": {"色调": "冷调", "时间": "夜晚"}
    }
  }
}
```

`multilabel` uses one independent logit per option and binary cross-entropy with logits. Its label is a list of selected names; an empty list is valid. Every unselected option is a negative. The loss averages over all option labels, rather than dividing by the number of positive labels. `grouped_choice` uses one packed branch with a separate softmax and cross-entropy for each dimension. Weather has the two options `无` and `阴雨`. Other groups retain their explicit `无` option where the original converter provided it. A custom palette with genuinely independent labels can also use `multilabel`; that changes its label constraints.

Merging preserves the original 24-task weights: each mode contributes one BCE term, and each palette dimension contributes one CE term. The image loss is their sum divided by 24, including when the input has only two questions. Validation reports the same task-level accuracy and Brier, plus multi-label micro-F1 and exact match at threshold 0.5. These Camera formats do not extend the text-only System One API or the older vision bridge.

## Calibrate multi-label thresholds

After training, fit thresholds on held-out images. The command selects one threshold per multi-label question to maximize micro-F1 on those images, keeping tied scores together. Selection uses `probability > threshold`. This fits a decision threshold; it does not calibrate probabilities or estimate performance on a fresh test set.

```bash
python -m kev.camera.calibrate \
  --checkpoint /data/runs/kev-lora-modes-seka/final \
  --val-data /data/val.jsonl \
  --out /data/runs/kev-lora-modes-seka/thresholds.json

python -m kev.camera.predict \
  --checkpoint /data/runs/kev-lora-modes-seka/final \
  --request /data/request.json \
  --calibration /data/runs/kev-lora-modes-seka/thresholds.json
```

Calibration also accepts `--val-meta`; repeat counts must be one. Use images disjoint from training, split by original image or scene. When the checkpoint's training annotation paths are still available, calibration checks image-path overlap. If data was moved, keep that split yourself. The artifact records annotation hashes, the checkpoint config/head/adapter hashes, option order and instruction text. Prediction rejects an artifact from a different checkpoint or a changed question schema. Refit it after further training. `--out` must name a new file.

For prediction, omit `label` fields. Raw known Camera questions are merged according to the checkpoint's layout; already merged questions can be supplied directly. Multi-label results contain `probabilities`, `threshold` and `selected`. Grouped palette results contain one `choice` and probability distribution per dimension. Without a calibration file, multi-label selection uses 0.5; `--threshold` changes the fallback. Both commands accept `--device cpu` for environments where the checkpoint's custom code supports CPU.

## Checkpoint format

Each new `checkpoint-0002500/` and `final/` directory contains a complete model and training state:

```text
config.json                       # CameraAgent config, with Kev settings under "kev"
model*.safetensors                # CameraAgent weights, including its frozen LM head
tokenizer* / other tokenizer files
modeling_*.py / configuration_*.py # custom CameraAgent code copied from the source
pointer.pt                        # trained Kev decision head
training_state.pt                 # optimizer, scheduler, per-rank RNG and training position
```

The root `config.json` retains CameraAgent's full configuration, including its vision and language model configurations. Its `kev` field adds the head type and dimension, head weight filename, image size and normalization, token limit, branch attention rule, source path, and training arguments and position. The checkpoint carries its own backbone and tokenizer; loading it does not require the original 0.4B directory. Its custom Python code still needs its runtime dependencies and any separately imported CameraModel packages.

The earlier layout kept CameraAgent under `camera/` and used `camera_decision.json` plus `pointer.pt` for the Kev wrapper. This reused CameraAgent's native serialization without changing its config. The new layout puts CameraAgent files at the root and records the wrapper in the same config. `pointer.pt` remains a separate weight file because the CameraAgent serializer saves its own parameters. Use `kev.camera.model.load` or the training and prediction commands to load the complete Kev model; a plain CameraAgent `AutoModelForCausalLM` load only loads the CameraAgent backbone.

Both layouts remain readable for prediction and for starting a new training run. Old checkpoints have no optimizer or progress state and cannot provide an exact interrupted-run resume. All new saves use the root-config layout.

## Train from a saved Kev checkpoint

Use the new entrypoint to retain both the CameraAgent weights and the trained pointer head:

```bash
CAMERA_CHECKPOINT=/data/runs/camera-kev-04b-full/final \
  TRAIN_DATA=/data/new-train.jsonl VAL_DATA=/data/new-val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-next \
  LEARNING_RATE=2e-6 HEAD_LR=1e-5 EPOCHS=2 \
  GPUS=8 bash shell/train_camera_kev_from_ckpt.sh
```

This is a new run: its optimizer, learning-rate schedule and step count start fresh. Image size, head dimension and token limit default to the checkpoint's settings. Explicit image size and head dimension must match it; `MAX_LENGTH` can change the row admission limit. Both old and new Kev checkpoint layouts work. Set a new `OUTPUT_DIR`; existing directories are refused for new runs.

The equivalent Python entrypoint is `python -m kev.camera.train --checkpoint /path/to/ckpt --train-data /data/train.jsonl --out /data/new-run` with the usual training options. Both shell scripts use the same trainer and common launcher settings.

## Resume an interrupted run

Repeat the original command with the same data, GPU count and training settings, and add `RESUME=latest`. For example, with the original script's default recipe:

```bash
TRAIN_DATA=/data/train.jsonl VAL_DATA=/data/val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-04b-full RESUME=latest \
  GPUS=8 bash shell/train_camera_kev.sh
```

The second script supports the same operation. For the new-run example above:

```bash
TRAIN_DATA=/data/new-train.jsonl VAL_DATA=/data/new-val.jsonl \
  OUTPUT_DIR=/data/runs/camera-kev-next RESUME=latest \
  LEARNING_RATE=2e-6 HEAD_LR=1e-5 EPOCHS=2 \
  GPUS=8 bash shell/train_camera_kev_from_ckpt.sh
```

Neither original base path nor `CAMERA_CHECKPOINT` is needed when `RESUME` is set. `latest` selects `final/` if present, otherwise the most recent complete `checkpoint-<step>/`. To select a checkpoint explicitly, set `RESUME=/data/runs/camera-kev-next/checkpoint-0002500`. Resume an older checkpoint into a new output directory if its run already contains later checkpoints. Python accepts `--resume latest` or `--resume /path/to/ckpt`.

Each save occurs after a complete optimizer step, including gradient accumulation. It saves AdamW and scheduler state, global step, epoch and next batch, partial epoch loss counters, and Python, NumPy, CPU Torch and local CUDA RNG state for every rank. The epoch's deterministic sampler reconstructs the same data order and skips completed batches before decoding their images. Annotation hashes, dataset roots and repeat counts are checked along with the GPU count and training arguments. Changes to epochs, batch size, accumulation, seed, freeze flags or learning-rate settings are refused; use `CAMERA_CHECKPOINT` for a new run with different settings. Keep image contents, dependencies and CUDA hardware consistent when comparing a resumed run with an uninterrupted run. Worker count and save frequency may change.

Checkpoint writes go to a hidden staging directory first. Only after weights, config and training state are written does the directory acquire its final name; incomplete writes are never selected by `latest`. No checkpoints are deleted. Resume starts from the last completed save, so work since that save is repeated. `SAVE_STEPS` defaults to 2500; lower it for more frequent recovery points. A run that has already saved `final/` reports `already_complete` when resumed. Starting another training phase from it uses `CAMERA_CHECKPOINT`.

To exercise recovery on the GPU machine, run with `STOP_AFTER=10`, then repeat the same command with `RESUME=latest` and without `STOP_AFTER`. This saves step 10 and exits while retaining the full intended schedule. `MAX_STEPS=10` instead defines a ten-step training schedule and completes that run. Per-epoch metrics live in `OUTPUT_DIR/training_metrics.jsonl`; validation metrics are written only when validation data is supplied. When resuming the same run, metrics after the restored position are discarded and recomputed. Only resume trusted checkpoints, since `training_state.pt` includes Python and NumPy state loaded through PyTorch's pickle format.

Repository tests use synthetic CPU modules to check checkpoint serialization and resume control flow. They do not verify private CameraAgent weights, inference or multi-GPU training; those checks must run on the machine holding the model and data.

## Predict

Create a JSON request with `image`, `state`, and unlabelled `questions`, then run:

```bash
python -m kev.camera.predict \
  --checkpoint /data/runs/camera-kev-04b-full/final \
  --request /data/request.json
```

The result maps every question ID to its option probabilities. For a `noul` question the keys are `false` and `true`. For `choice` they are its `criteria` keys. Fit a temperature on held-out images before using the probabilities as calibrated confidence.
