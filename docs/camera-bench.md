# CameraModel MMBench and AesBench with Kev

Kev's released checkpoints are **text decision models**. `kev.model.DecisionModel`
loads a text backbone and takes tokenized state, question and choices. It does not
load a vision tower or accept `pixel_values`. CameraModel's evaluator decodes an
image from every MMBench/AesBench TSV row and calls `model.chat` with those pixels.
Its MoE task policy also assigns MMBench to `reasoning` and AesBench to
`aesthetic`; that policy belongs to the CameraModel checkpoint, not Kev.

Consequently, **there is no direct Kev MMBench/AesBench score**. This bridge
evaluates an explicitly named *two-stage text proxy*: another visual model first
describes each image, then Kev answers the multiple-choice question from that
description. Its accuracy must be reported as `vision model + Kev`, alongside
the vision model/checkpoint and prompt used to produce descriptions. Do not
compare it as if it were a direct CameraModel VLM score. AesBench in particular
may need aesthetic evidence that a generic object caption omits.

## Inputs

Use the same TSVs as `CameraModel/tools/benchmark_eval/eval_combined.py`:

- `data/mmbench/mmbench_dev_cn_20231003.tsv` -> `--mmbench-cn`
- `data/mmbench/mmbench_dev_en_20231003.tsv` -> `--mmbench-en`
- `data/AesBench/AesBench_VAL.tsv` -> `--aesbench-val`

Create a JSONL file of independent image descriptions, one per TSV row:

```json
{"dataset":"mmbench_en","index":"42","image_sha256":"64 lowercase hex digits","description":"A person holds a red umbrella on a street."}
```

`dataset` must be `mmbench_cn`, `mmbench_en`, or `aesbench_val`. Hash the
**decoded image bytes** from the TSV's `image` column with SHA-256. Produce the
description from the image alone, without the ground-truth answer, answer
letter, or evaluation feedback. Keep the vision model ID, revision, and prompt
with the evidence file so the two-stage result is reproducible. The converter
checks every image hash and refuses missing, duplicate, or unused descriptions.

## Prepare and score

```bash
python -m kev.camera_bench \
  --mmbench-cn /path/to/mmbench_dev_cn_20231003.tsv \
  --mmbench-en /path/to/mmbench_dev_en_20231003.tsv \
  --aesbench-val /path/to/AesBench_VAL.tsv \
  --evidence /path/to/image_descriptions.jsonl \
  --out /path/to/kev_camera_run \
  --run jaredpalmer/kev-4b
```

Use `--prepare-only` without `--run` to check the TSVs and evidence on a
machine without weights. It writes `requests.jsonl` in Kev's existing custom
benchmark format, `row_map.json`, and `manifest.json` with input hashes. A
scored run additionally writes `kev/report.json`, `kev/rows.json`, and
`camera_metrics.json` with CN, EN, AesBench overall, and AesA/AesE/AesP
accuracy. An output directory must be new for each run. Overlong requests cause
the command to fail before reporting CameraModel-style accuracy.

No benchmark TSVs, images, or model weights are included in this repository.
