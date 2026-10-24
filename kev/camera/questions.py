"""Camera question formats, including multi-label and grouped-choice readouts."""

from kev.api import MAX_OPTIONS, SystemOneRequest, option_text, render, to_record as api_record
from kev.data import materialize as api_materialize

from .labels import prepare_record


def _criteria(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_OPTIONS:
        raise ValueError(f"criteria must contain 1..{MAX_OPTIONS} named options")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError("option names must be nonempty strings")
    return value


def to_record(request, *, labelled=False, layout="original"):
    request = prepare_record(request, layout)
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    if all(q.get("type") not in {"multilabel", "grouped_choice"} for q in questions.values()):
        record, metadata = api_record(SystemOneRequest.model_validate(request))
        if labelled:
            record = api_materialize({**request, "questions": {
                key: {**q, "src": q.get("src", "camera")} for key, q in questions.items()}})
        return record, metadata
    output, metadata = [], []
    for qid, source in questions.items():
        kind = source.get("type")
        if labelled and "label" not in source:
            raise ValueError(f"{qid} needs a label")
        if kind not in {"multilabel", "grouped_choice"}:
            single = {"state": request.get("state", ""), "questions": {qid: source}}
            record, meta = api_record(SystemOneRequest.model_validate(single))
            if labelled:
                single["questions"][qid] = {**source, "src": source.get("src", "camera")}
                record = api_materialize(single)
            output.extend(record["questions"])
            metadata.extend(meta)
            continue
        criteria = _criteria(source.get("criteria"))
        question = {"qid": qid, "qtype": kind, "instr": render(source.get("instructions")),
                    "src": source.get("src", "camera")}
        if kind == "multilabel":
            keys = list(criteria)
            question["options"] = [option_text(key, value) for key, value in criteria.items()]
            label = source.get("label", []) if labelled else []
            if not isinstance(label, list) or any(not isinstance(key, str) for key in label):
                raise ValueError(f"{qid}: multi-label label must be a list of option names")
            if len(set(label)) != len(label) or set(label) - criteria.keys():
                raise ValueError(f"{qid}: duplicate or unknown multi-label options")
            question["label"] = [int(key in label) for key in keys]
        else:
            label = source.get("label", {}) if labelled else {}
            if not isinstance(label, dict) or (labelled and set(label) != set(criteria)):
                raise ValueError(f"{qid}: grouped label must name every group exactly once")
            keys, options, groups, labels = [], [], [], []
            for name, values in criteria.items():
                values = _criteria(values)
                start = len(keys)
                keys.extend(f"{name}/{key}" for key in values)
                options.extend(f"{name}/{option_text(key, value)}" for key, value in values.items())
                if labelled and label[name] not in values:
                    raise ValueError(f"{qid}/{name}: unknown label {label[name]!r}")
                labels.append(list(values).index(label[name]) if labelled else 0)
                groups.append({"name": name, "keys": list(values), "start": start, "stop": len(keys)})
            if len(keys) > MAX_OPTIONS or len(set(keys)) != len(keys):
                raise ValueError(f"{qid}: grouped options must have unique names and total at most {MAX_OPTIONS}")
            question.update(options=options, groups=groups, label=labels)
        question["keys"] = keys
        output.append(question)
        metadata.append({"id": qid, "type": kind, "keys": keys,
                         **({"groups": question["groups"]} if kind == "grouped_choice" else {})})
    return {"state": render(request.get("state", "")), "questions": output}, metadata


def materialize(request, *, layout="original"):
    request = prepare_record(request, layout)
    if all(q.get("type") not in {"multilabel", "grouped_choice"} for q in request["questions"].values()):
        return api_materialize({**request, "questions": {
            key: {**q, "src": q.get("src", "camera")} for key, q in request["questions"].items()}})
    return to_record(request, labelled=True)[0]
