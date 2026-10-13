"""Small fixture tests for the CameraModel TSV to Kev bridge (no model weights)."""

import base64
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kev.camera_bench import build_requests, load_evidence, main, summarize


def fixture(tmp_path, image=b"fake-image", image_hash=None):
    tsv = tmp_path / "mmbench.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "image", "question", "hint", "answer", "A", "B"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"index": "42", "image": base64.b64encode(image).decode(), "question": "Which object?",
                         "hint": "Look at the center.", "answer": "B", "A": "dog", "B": "cat"})
    evidence = tmp_path / "evidence.jsonl"
    payload = json.dumps({"dataset": "mmbench_en", "index": "42",
                          "image_sha256": image_hash or hashlib.sha256(image).hexdigest(),
                          "description": "A cat on a chair."}) + "\n"
    evidence.write_text(payload, encoding="utf-8")
    return tsv, evidence


class CameraBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_bridge_preserves_labels_and_only_sends_text(self):
        tsv, evidence_file = fixture(self.root)
        requests, row_map = build_requests({"mmbench_en": tsv}, load_evidence(evidence_file))
        self.assertEqual(requests[0]["state"], {"image_description": "A cat on a chair."})
        question = requests[0]["questions"]["answer"]
        self.assertEqual(question["instructions"], "Look at the center.\nWhich object?")
        self.assertEqual(question["criteria"], {"A": "dog", "B": "cat"})
        self.assertEqual(question["label"], "B")
        self.assertNotIn("fake-image", json.dumps(requests))
        self.assertEqual(row_map[0]["image_sha256"], hashlib.sha256(b"fake-image").hexdigest())
        self.assertEqual(summarize(row_map, [{"correct": 1}])["mmbench_en"],
                         {"correct": 1, "total": 1, "accuracy": 1.0})

    def test_bridge_rejects_wrong_image_description(self):
        tsv, evidence_file = fixture(self.root, image_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "image hash mismatch"):
            build_requests({"mmbench_en": tsv}, load_evidence(evidence_file))

    def test_bridge_rejects_missing_description(self):
        tsv, evidence_file = fixture(self.root)
        with self.assertRaisesRegex(ValueError, "missing image description"):
            build_requests({"mmbench_cn": tsv}, load_evidence(evidence_file))

    def test_bridge_rejects_duplicate_evidence(self):
        _, evidence_file = fixture(self.root)
        evidence_file.write_text(evidence_file.read_text(encoding="utf-8") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate evidence"):
            load_evidence(evidence_file)

    def test_prepare_only_writes_reproducible_manifest(self):
        tsv, evidence_file = fixture(self.root)
        out = self.root / "prepared"
        main(["--mmbench-en", str(tsv), "--evidence", str(evidence_file),
              "--out", str(out), "--prepare-only"])
        requests = [json.loads(line) for line in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(manifest["evaluation"], "text_proxy_two_stage")
        self.assertFalse(manifest["direct_multimodal"])
        self.assertEqual(manifest["records"], 1)
        self.assertEqual(manifest["evidence_sha256"], hashlib.sha256(evidence_file.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
