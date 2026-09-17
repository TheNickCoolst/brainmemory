"""Automatic accelerator selection: CUDA, then MPS, then CPU."""

from __future__ import annotations

import torch


def select_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device | None = None) -> str:
    d = device if device is not None else select_device()
    if d.type == "cuda":
        idx = d.index or 0
        try:
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        except Exception:
            return f"cuda:{idx}"
    return d.type
