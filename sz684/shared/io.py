"""Shared manifests, hashing, device selection, and masked token pooling."""

import hashlib
import json
import os
import torch
from shared.paths import PROJECT_ROOT as ROOT

# Preserve the project-local caches used before separating the baseline CLI.
# Set these before callers import transformers/Hugging Face or Matplotlib.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def device_for(requested):
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable; run outside sandbox or explicitly choose CPU"
        )
    return requested


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def load_manifest(run):
    return json.loads((run / "manifest.json").read_text())


def pooled(hidden, mask):
    # Cast before reduction to avoid overflow if an explicit lower-precision experiment is added.
    weights = mask.to(dtype=torch.float32).unsqueeze(-1)
    return (hidden.float() * weights).sum(1) / weights.sum(1)
