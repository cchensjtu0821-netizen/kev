# Image decision training

Kev's published checkpoints came from these stages. Their text API and scores do not include image input. The [model cards](model-cards/) record the full recipes and measured results.

| Checkpoint | Base and training route |
|---|---|
| [Kev-0.8B](model-cards/kev-0.8b.md) | Qwen3.5-0.8B-Base. Two epochs of `decision-v7` with rank-16 LoRA, then a dates/unknowable delta and one documents, hard-task and developer-tooling delta. |
| [Kev-4B](model-cards/kev-4b.md) | Qwen3.5-4B-Base. The same two-epoch LoRA base recipe, then dates/unknowable, documents and skill deltas. |
| [Kev-9B](model-cards/kev-9b.md) | Qwen3.5-9B-Base. The same LoRA base recipe, dates/unknowable, then a documents and skills delta selected in round 27. |
| [Kev-27B](model-cards/kev-27b.md) | Post-trained Qwen3.8-27B. One full-weight epoch on the 145,840-record `sft-v2-r22` corpus, then an 85:15 weight blend with the earlier LoRA Kev-27B. |

This experimental path adds a ViT and a two-layer MLP projector. It follows CameraModel's ViT, projector and language-model layout and reads its `root` / `annotation` / `repeat_time` data collection format. Kev's pointer head still scores supplied options. It does not generate captions. ViT patch tokens enter the common state after Kev's state marker; each question then runs as an isolated causal row.

For the existing 0.4B dense `CameraAgentModel` checkpoint, use the separate [CameraAgent pointer path](camera-kev.md). It loads that model's own vision encoder, projector and Qwen3 weights and accepts both `noul` and mixed `noul`/`choice` image records.

CameraModel's checked-in launcher points to image and annotation files on a separate machine. They are not in this checkout. Its general `conversations` rows have free-form answers, which do not provide reliable labels for Kev's option probabilities. Add labelled `questions` to the rows you train on. Keep MMBench dev and AesBench val for evaluation, not training. The existing [camera benchmark guide](camera-bench.md) evaluates released text checkpoints through independent captions; it measures a different, two-stage system.

## Data

One JSONL row has one local image and one or more labelled Kev questions:

```json
{"image":"images/example.jpg","state":"A customer took this photo of a damaged package.","questions":{"condition":{"type":"choice","instructions":"What is visible?","criteria":{"damaged":"The box is torn","intact":"The box is intact"},"label":"damaged","src":"package_photos"}}}
```

`state` may be text or Kev's structured JSON state. `image` is relative to the JSONL directory for `--train-data` and to a collection item's `root` for `--train-meta`. The current path accepts one image per row. A CameraModel-style collection can list multiple such JSONL files:

```json
{
  "packages": {
    "root": "/data/package_images",
    "annotation": "/data/package_decisions_train.jsonl",
    "repeat_time": 1.0
  }
}
```

Use a separate validation file or collection. Split by original image or scene before generating variants, so related photos cannot cross the split. The loader indexes byte offsets and reads rows on demand, including virtual repeats. It validates paths and refuses unlabelled questions. `--validate-data` checks all rows and image paths without loading model weights.

## Train

Install Kev's vision extra. The ViT name can point to a local Hugging Face encoder. Pin both model revisions for a reproducible run. Start with the 0.8B checkpoint and a frozen ViT and text model; this trains only the bridge. Use `--train-text` on an adapter checkpoint to fine-tune its LoRA and pointer head too. `--train-vision` fine-tunes all ViT weights and uses more memory. The 27B full-weight checkpoint supports the bridge stage, while full-weight text fine-tuning is outside this path.

```bash
uv sync --extra vision
uv run python -m kev.vision_train --train-data /data/train.jsonl --val-data /data/val.jsonl --validate-data
uv run python -m kev.vision_train \
  --train-data /data/train.jsonl --val-data /data/val.jsonl \
  --text-checkpoint jaredpalmer/kev-0.8b@v1.0 \
  --vision-model facebook/dinov2-base --visual-tokens 16 \
  --epochs 2 --accum 8 --out runs/kev-vision-08b
```

The same command accepts `--train-meta` and `--val-meta` instead of the JSONL paths. `--max-state` is the combined text and image-token budget during training, default 8,192; overlong rows fail rather than truncate. The output contains `vision.pt`, `vision_config.json`, and `training_metrics.json`. With `--train-text`, it also contains the trained adapter and pointer head. The original text checkpoint is separate and is never modified. Validation reports accuracy and Brier score. Fit a temperature on held-out image data before treating the visual model's confidence as calibrated.

## Predict

Write a request JSON with an `image` path, `state` and ordinary Kev `questions` without labels:

```json
{"image":"images/example.jpg","state":"","questions":{"condition":{"type":"choice","instructions":"What is visible?","criteria":{"damaged":"Torn box","intact":"Intact box"}}}}
```

```bash
uv run python -m kev.vision_predict --bridge runs/kev-vision-08b --request /data/request.json --device cuda
```

This returns a probability for every option. No visual checkpoint has been trained or benchmarked in this repository yet. Run the command on a machine with the image files and GPU, then compare it against text-only and image-caption baselines on a held-out image set.

CameraModel's local MMBench and AesBench TSVs can be read directly after training:

```bash
uv run python -m kev.vision_camera_bench --bridge runs/kev-vision-08b \
  --mmbench-cn /data/mmbench/mmbench_dev_cn_20231003.tsv \
  --mmbench-en /data/mmbench/mmbench_dev_en_20231003.tsv \
  --aesbench-val /data/AesBench/AesBench_VAL.tsv \
  --device cuda --out runs/kev-vision-camera-eval
```

This command decodes each TSV image and reports direct image-model accuracy and per-row option probabilities. It writes image hashes, not image bytes. Treat those TSVs as evaluation data; do not add them to a training collection.
