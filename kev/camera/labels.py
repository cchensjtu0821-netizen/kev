"""Camera mode and palette label conversion and question merging."""


MODES = (
    "雨景", "剪影", "全景", "夜景", "舞台", "原生", "球类", "设施",
    "车流", "月亮", "动物", "博物馆", "小烟花", "大烟花", "仙女棒", "文档",
)
PALETTE_GROUPS = {
    "天空": ("蓝色系天空", "冷灰系天空", "暖色系天空", "冷暖混合天空"),
    "植物": ("绿色植物", "红黄橙植物"),
    "天气": ("阴雨",),
    "室内外": ("室内", "室外", "窗边"),
    "时间": ("白天", "夕阳", "蓝调", "夜晚"),
    "建筑": ("历史建筑", "其他建筑"),
    "色调": ("暖调", "冷调", "中性色调", "混合色调"),
    "其他": ("美食", "花卉", "古装人物"),
}
NONE = "无"


def convert_record(row, *, native_conflict="error", labelled=True):
    if native_conflict not in {"error", "drop-native", "drop-others"}:
        raise ValueError(f"invalid native conflict policy: {native_conflict}")
    questions = row.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("questions must be an object")
    food_key = "seka_美食"
    alias_key = "seka_餐饮美食"
    if food_key in questions and alias_key in questions:
        raise ValueError("both seka_美食 and seka_餐饮美食 exist; resolve their labels first")
    if food_key not in questions and alias_key not in questions:
        raise ValueError("missing seka_美食 or seka_餐饮美食")

    expected = {f"mode_{name}" for name in MODES}
    expected.update(f"seka_{name}" for names in PALETTE_GROUPS.values() for name in names)
    if alias_key in questions:
        expected.remove(food_key)
        expected.add(alias_key)
    missing, extra = expected - questions.keys(), questions.keys() - expected
    if missing or extra:
        raise ValueError(f"expected 39 questions; missing={sorted(missing)}, extra={sorted(extra)}")
    for key, question in questions.items():
        if (not isinstance(question, dict) or question.get("type") != "noul" or
                (labelled and type(question.get("label")) is not bool)):
            raise ValueError(f"{key}: expected a Noul question with a boolean label")

    output = {key: value for key, value in row.items() if key != "questions"}
    converted = {f"mode_{name}": dict(questions[f"mode_{name}"]) for name in MODES}
    active_modes = [name for name in MODES if labelled and converted[f"mode_{name}"]["label"]]
    if "原生" in active_modes and len(active_modes) > 1:
        if native_conflict == "error":
            raise ValueError(f"mode_原生 conflicts with {active_modes}; inspect the image or select --native-conflict")
        if native_conflict == "drop-native":
            converted["mode_原生"]["label"] = False
        else:
            for name in active_modes:
                if name != "原生":
                    converted[f"mode_{name}"]["label"] = False
    active_modes = [name for name in MODES if labelled and converted[f"mode_{name}"]["label"]]
    for exclusive in ("动物", "博物馆", "文档"):
        if exclusive in active_modes and len(active_modes) > 1:
            raise ValueError(f"mode_{exclusive} conflicts with {active_modes}")

    for group, names in PALETTE_GROUPS.items():
        keys = [alias_key if name == "美食" and alias_key in questions else f"seka_{name}" for name in names]
        selected = [name for name, key in zip(names, keys) if labelled and questions[key]["label"]]
        if len(selected) > 1:
            raise ValueError(f"seka_{group} has conflicting labels: {selected}")
        if labelled and group == "色调" and not selected:
            raise ValueError("seka_色调 requires exactly one label")
        sources = {questions[key].get("src") for key in keys}
        if len(sources) > 1:
            raise ValueError(f"seka_{group} has inconsistent src values: {sources}")
        src = next(iter(sources))
        if group == "天气":
            question = dict(questions[keys[0]])
            converted["seka_阴雨"] = question
            continue
        criteria = {name: None for name in names}
        if group != "色调":
            criteria[NONE] = "以上都不是"
        choice = {
            "type": "choice",
            "instructions": f"这幅画面的{group}标签是哪一项？",
            "criteria": criteria,
        }
        if labelled:
            choice["label"] = selected[0] if selected else NONE
        if src is not None:
            choice["src"] = src
        converted[f"seka_{group}"] = choice

    assert len(converted) == 24
    output["questions"] = converted
    return output


def _source(questions):
    sources = {q.get("src") for q in questions}
    if len(sources) != 1:
        raise ValueError("merged questions must have the same src")
    source = next(iter(sources))
    return {"src": source} if source is not None else {}


def prepare_record(row, layout="original"):
    """Merge known Camera tasks; premerged records pass through unchanged."""
    if layout not in {"original", "modes", "all"}:
        raise ValueError(f"unknown question layout: {layout}")
    if layout == "original":
        return row
    questions = row["questions"]
    if len(questions) == 39 and all(q.get("type") == "noul" for q in questions.values()):
        labelled = any("label" in q for q in questions.values())
        row = convert_record(row, labelled=labelled)
        questions = row["questions"]
    questions = dict(questions)
    mode_keys = [f"mode_{name}" for name in MODES]
    if any(key in questions for key in mode_keys):
        if "modes" in questions:
            raise ValueError("record has both merged and separate mode questions")
        if not all(key in questions and questions[key].get("type") == "noul" for key in mode_keys):
            raise ValueError("mode merging needs all 16 Noul questions")
        originals = [questions.pop(key) for key in mode_keys]
        modes = {"type": "multilabel", "instructions": "选择这张图适用的全部模式。",
                 "criteria": dict.fromkeys(MODES), **_source(originals)}
        if any("label" in q for q in originals):
            if not all(type(q.get("label")) is bool for q in originals):
                raise ValueError("mode labels must all be booleans")
            modes["label"] = [name for name, q in zip(MODES, originals) if q["label"]]
        questions = {"modes": modes, **questions}
    if layout == "all":
        if "seka_天气" in questions and "seka_阴雨" in questions:
            raise ValueError("record has both seka_天气 and seka_阴雨")
        weather_key = "seka_天气" if "seka_天气" in questions else "seka_阴雨"
        palette_keys = [weather_key if group == "天气" else f"seka_{group}" for group in PALETTE_GROUPS]
        if any(key in questions for key in palette_keys):
            if "palette" in questions:
                raise ValueError("record has both merged and separate palette questions")
            if not all(key in questions for key in palette_keys):
                raise ValueError("palette merging needs all eight palette dimensions")
            originals = [questions.pop(key) for key in palette_keys]
            palette = {"type": "grouped_choice", "instructions": "选择各维度的色卡标签，每个维度选一项。",
                       "criteria": {}, **_source(originals)}
            labelled = any("label" in q for q in originals)
            if labelled:
                if not all("label" in q for q in originals):
                    raise ValueError("palette dimensions must all have labels")
                palette["label"] = {}
            for group, q in zip(PALETTE_GROUPS, originals):
                binary_weather = group == "天气" and q.get("type") == "noul"
                expected = "noul" if binary_weather else "choice"
                if q.get("type") != expected:
                    raise ValueError(f"{group} must have type {expected}")
                palette["criteria"][group] = {NONE: None, "阴雨": None} if binary_weather else q["criteria"]
                if labelled:
                    label = q["label"]
                    if binary_weather:
                        if type(label) is not bool:
                            raise ValueError("weather label must be a boolean")
                        label = "阴雨" if label else NONE
                    palette["label"][group] = label
            questions["palette"] = palette
    return {**row, "questions": questions}
