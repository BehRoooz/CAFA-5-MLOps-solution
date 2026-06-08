"""Shared utilities: reproducibility, device selection, logging."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return torch device per ``CAFA_DEVICE`` (auto | cuda | cpu). Default: auto (CUDA if available)."""
    preference = os.environ.get("CAFA_DEVICE", "auto").strip().lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CAFA_DEVICE=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name() -> str:
    """String name of the resolved torch device (e.g. ``cuda`` or ``cpu``)."""
    return str(get_device())


def get_device_info() -> dict[str, str | bool]:
    """Device metadata for service health endpoints."""
    device = get_device()
    info: dict[str, str | bool] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cafa_device": os.environ.get("CAFA_DEVICE", "auto"),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def setup_logger(
    name: str = "cafa5",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes to console and optionally to a file.

    Args:
        name: Logger name.
        log_dir: If provided, a file handler is added writing to ``log_dir/train.log``.
        level: Logging level.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
