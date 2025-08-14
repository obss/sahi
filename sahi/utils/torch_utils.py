from __future__ import annotations
import torch

def select_device(device: str | int | None = None) -> str:
    if device is None or str(device).lower() == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return str(device)

def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()