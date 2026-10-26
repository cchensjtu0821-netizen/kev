# Camera benchmark conversion

`convert_moshi_seka_to_kev.py` converts the ten moshi_seka benchmark files into the two
merged Camera formats. It uses `kev.camera.labels.prepare_record`, the same merger as
training. It needs Python and this checkout, with no model weights, GPU or third-party
Python packages for conversion.

Run from the Kev repository root:

```bash
python -m benchmark.convert_moshi_seka_to_kev \
  --data-dir /srv/workspace/Kirin_AI_Workspace/AIC_I/c00932551/CameraBaseModel/benchmark_lora/moshi_seka/data \
  --out-dir benchmark/converted
```

The default image root is the original `benchmark_lora/moshi_seka` directory. Override
it with `--image-dir` or `MOSHI_SEKA_IMAGE_DIR` if the benchmark images moved. The old
absolute path remapping is retained; `--seka-data-base` and `--gaoxiangsu-test-dir`
override the other two image roots. Relative image paths resolve against `--image-dir`.
Every output image path is absolute.

Each source produces two files:

| Suffix | Training layout | Input questions |
| --- | --- | --- |
| `.modes.jsonl` | `QUESTION_LAYOUT=modes` | One `modes` multilabel question and eight palette Choice questions |
| `.all.jsonl` | `QUESTION_LAYOUT=all` | One `modes` multilabel question and one `palette` grouped-choice question |

Use `--formats all` to produce only the two-question format. `--sources doc_test_gt.json`
selects one source instead of all ten. `--overwrite` replaces existing outputs. Missing
images are skipped and counted in `conversion_stats.json`; `--missing-images error`
stops instead. Contradictory labels within a palette group stop conversion with the
source filename and item number. A source's output files are replaced only after all
its rows convert successfully.

The original `answer` string and `original_id` (falling back to `id`) are retained.
Mode qualifiers `普通` and `典型` are stripped; out-of-list mode and palette tokens are
dropped from the structured targets, matching the supplied benchmark converter.

## Partial benchmark labels

These are evaluation records, not training records. Mode sources annotate only modes;
Seka sources annotate only palette labels. Both formats retain the complete input
question template. An unannotated task has no question-level `label`, rather than a
fabricated negative or a null Choice label.

`_meta.eval_labels` is the authoritative map of answers to score. For a mode source:

```json
{"eval_task": "mode", "question_layout": "all", "eval_labels": {"modes": ["剪影"]}}
```

For a Seka source:

```json
{
  "eval_task": "seka",
  "question_layout": "all",
  "eval_labels": {
    "palette": {
      "天空": "暖色系天空", "植物": "无", "天气": "无", "室内外": "室外",
      "时间": "夜晚", "建筑": "无", "色调": "中性色调", "其他": "无"
    }
  }
}
```

If a Seka answer has no tone, `色调` is absent from the palette target map and must not
be scored. The tone question and its four options still appear in the input. If all
eight palette labels are known, the grouped question also carries the ordinary
training-style `label` dictionary; otherwise its partial targets exist only under
`_meta.eval_labels`. In the nine-question format, each known palette question has its
own `label` and entry under `_meta.eval_labels`.

Build inference inputs with `kev.camera.questions.to_record(row, labelled=False)` or
the Camera predictor. Keep the source row separately for scoring only the targets in
`_meta.eval_labels`. The labelled training dataset reader intentionally rejects rows
with unlabelled questions. Do not feed mode-only palette predictions or Seka-only mode
predictions into accuracy/F1 as though their answers were known. Conversion does not
run model evaluation or fit thresholds; fit mode thresholds on separate calibration
images before reading these benchmark sources.
