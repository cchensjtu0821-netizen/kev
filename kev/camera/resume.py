"""Save complete Camera Kev training steps and restore their per-rank state."""

import random
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler

from kev.suite import digest, read_json, write_json


class ResumableBatchSampler(BatchSampler):
    start = 0

    def __iter__(self):
        return islice(super().__iter__(), self.start, None)

    def __len__(self):
        return max(0, super().__len__() - self.start)


def data_signature(*datasets):
    result = []
    for dataset in datasets:
        sources = []
        if dataset is not None:
            for (name, annotation, root), _, count in dataset.sources:
                sources.append({"name": name, "annotation": str(annotation.resolve()),
                                "root": str(root.resolve()), "count": count,
                                "sha256": digest(annotation)})
        result.append(sources)
    return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def find_resume(value, out):
    if value != "latest":
        path = Path(value)
        if not (path / "training_state.pt").is_file():
            raise ValueError(f"{path} has no training_state.pt; use --checkpoint to start a new run from weights")
        return path
    out = Path(out)
    final = out / "final"
    if (final / "training_state.pt").is_file():
        return final
    checkpoints = sorted(path for path in out.glob("checkpoint-*")
                         if (path / "training_state.pt").is_file())
    if not checkpoints:
        raise FileNotFoundError(f"no complete training checkpoint in {out}")
    return checkpoints[-1]


def read_state(path):
    # Resume files contain Python/NumPy RNG state. Only load trusted local checkpoints.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["format"] != 1:
        raise ValueError("unsupported Camera Kev training state format")
    return state


def check_resume(state, args, *, world, signature, total_steps):
    if state["world"] != world:
        raise ValueError(f"resume needs WORLD_SIZE={state['world']}, got {world}")
    ignored = {"model", "checkpoint", "resume", "out", "num_workers", "save_every_steps", "stop_after", "validate_data"}
    changed = [key for key, value in state["args"].items()
               if key not in ignored and getattr(args, key) != value]
    if changed:
        raise ValueError(f"resume settings changed: {', '.join(changed)}; use --checkpoint for a new training run")
    if state["data"] != signature:
        raise ValueError("resume dataset annotations, roots or repeat counts changed")
    if state["total_steps"] != total_steps:
        raise ValueError("resume total_steps changed")


def save_checkpoint(path, model, optimizer, scheduler, *, args, source, world, rank,
                    signature, total_steps, step, epoch, next_batch, train_loss, seen, complete=False):
    local = {"rng": rng_state(), "train_loss": train_loss, "seen": seen}
    ranks = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, ranks, dst=0)
    else:
        ranks = [local]
    if rank == 0:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}; resume into a new --out directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        model.save(staging, source=source, image_size=args.image_size)
        state = {"format": 1, "args": vars(args), "world": world, "data": signature,
                 "total_steps": total_steps, "step": step, "epoch": epoch, "next_batch": next_batch,
                 "complete": complete,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "ranks": ranks}
        torch.save(state, staging / "training_state.pt")
        config_path = staging / "config.json"
        config = read_json(config_path)
        config["kev"]["training"] = {"args": vars(args), "world_size": world, "step": step,
                                      "epoch": epoch, "next_batch": next_batch, "total_steps": total_steps,
                                      "state_file": "training_state.pt"}
        write_json(config_path, config)
        staging.rename(path)
    if world > 1:
        dist.barrier()
