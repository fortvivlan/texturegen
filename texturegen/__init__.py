"""Utilities for preparing texture data and training an SDXL LoRA."""

from .config import TrainingConfig
from .data import prepare_huggingface_dataset, prepare_local_dataset

__all__ = [
    "TrainingConfig",
    "prepare_huggingface_dataset",
    "prepare_local_dataset",
]

__version__ = "0.1.0"

