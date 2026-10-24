"""Image decision records and CameraModel-style dataset collections."""

import json
from pathlib import Path

from .data import materialize


class ImageDecisionDataset:
    """Index JSONL byte offsets, then read and materialize one record at a time.

    A collection uses CameraModel's ``root``, ``annotation`` and ``repeat_time``
    fields. Its rows need Kev's labelled ``questions``; a free-form caption is
    not a decision label and cannot be converted without annotation.
    """

    def __init__(self, *, data=None, meta=None, materializer=materialize):
        if bool(data) == bool(meta):
            raise ValueError("pass exactly one of data or meta")
        self.sources = []
        self.materializer = materializer
        if data:
            self._add("data", Path(data), Path(data).resolve().parent, 1.0)
        else:
            collection = json.loads(Path(meta).read_text(encoding="utf-8"))
            if not isinstance(collection, dict) or not collection:
                raise ValueError("meta must contain a nonempty dataset collection")
            for name, item in collection.items():
                annotation = Path(item["annotation"])
                if not annotation.is_absolute():
                    annotation = Path(meta).resolve().parent / annotation
                root = Path(item["root"])
                if not root.is_absolute():
                    root = Path(meta).resolve().parent / root
                self._add(name, annotation, root, float(item.get("repeat_time", 1)))
        if not self.sources:
            raise ValueError("no records")
        self.index = [(source_index, i) for source_index, (_, _, count) in enumerate(self.sources)
                      for i in range(count)]

    def _add(self, name, annotation, root, repeat):
        if repeat <= 0:
            raise ValueError(f"{name}: repeat_time must be positive")
        offsets = []
        with annotation.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        if not offsets:
            raise ValueError(f"{annotation}: empty annotation")
        count = int(len(offsets) * repeat)
        if count == 0:
            raise ValueError(f"{name}: repeat_time selects zero records")
        self.sources.append(((name, annotation, root), offsets, count))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        source_index, virtual_index = self.index[index]
        (name, annotation, root), offsets, _ = self.sources[source_index]
        with annotation.open("rb") as stream:
            stream.seek(offsets[virtual_index % len(offsets)])
            row = json.loads(stream.readline())
        if not isinstance(row.get("questions"), dict) or not row["questions"]:
            raise ValueError(f"{annotation}: row {virtual_index} needs labelled Kev questions")
        images = row.get("image", row.get("images"))
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"{annotation}: row {virtual_index} needs exactly one image path")
        image = (root / images[0]).resolve()
        if not image.is_relative_to(root.resolve()):
            raise ValueError(f"{annotation}: image escapes root: {images[0]}")
        if not image.is_file():
            raise FileNotFoundError(image)
        for question in row["questions"].values():
            if "label" not in question:
                raise ValueError(f"{annotation}: row {virtual_index} has an unlabelled question")
        request = {"state": row.get("state", ""), "questions": {
            key: {**question, "src": question.get("src", name)}
            for key, question in row["questions"].items()}}
        record = self.materializer(request)
        return {"image": image, "record": record, "source": name}
