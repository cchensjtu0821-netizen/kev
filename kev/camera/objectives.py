"""Loss and probabilities share the same per-label or per-group interpretation."""

import torch
import torch.nn.functional as F


def loss_terms(question, logits):
    logits = logits.float()
    label = question["label"]
    kind = question.get("qtype")
    if kind == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, logits.new_tensor(label), reduction="none")
    if kind == "grouped_choice":
        return torch.stack([F.cross_entropy(logits[group["start"]:group["stop"]].unsqueeze(0),
                                             torch.tensor([target], device=logits.device))
                            for group, target in zip(question["groups"], label)])
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=logits.device)).unsqueeze(0)


def image_loss(record, logits):
    return torch.cat([loss_terms(q, z) for q, z in zip(record["questions"], logits)]).mean()


def question_probabilities(question, logits):
    kind = question.get("qtype")
    if kind == "multilabel":
        return logits.float().sigmoid()
    if kind == "grouped_choice":
        return torch.cat([logits[group["start"]:group["stop"]].float().softmax(-1) for group in question["groups"]])
    return logits.softmax(-1)


def task_scores(question, probabilities):
    """Return accuracy and Brier contributions, one per original task."""
    if question.get("qtype") == "multilabel":
        targets = probabilities.new_tensor(question["label"])
        return ((probabilities > 0.5) == targets).tolist(), (2 * (probabilities - targets).square()).tolist()
    if question.get("qtype") == "grouped_choice":
        pairs = [(probabilities[g["start"]:g["stop"]], target)
                 for g, target in zip(question["groups"], question["label"])]
    else:
        pairs = [(probabilities, question["label"])]
    correct, brier = [], []
    for p, label in pairs:
        correct.append(int(p.argmax().item() == label))
        brier.append(float((p - F.one_hot(torch.tensor(label, device=p.device), p.numel())).square().sum()))
    return correct, brier
