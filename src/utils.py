"""Small generic utilities shared by notebooks."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create output directories, including missing parents."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and PyTorch when it is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_torch_device():
    """Select CUDA when available, otherwise CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text())


def save_json(value: Any, path: Path) -> None:
    """Write a JSON-serializable value with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

