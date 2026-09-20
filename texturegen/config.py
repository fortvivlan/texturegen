from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    """Serializable settings for an SDXL texture LoRA run."""

    dataset_dir: str
    output_dir: str = "outputs/texture-lora"
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    resolution: int = 1024
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 3000
    learning_rate: float = 1e-4
    rank: int = 16
    seed: int = 42
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    enable_xformers: bool = False
    caption_dropout: float = 0.05
    random_flip: bool = True
    random_roll: bool = True
    circular_padding: bool = True
    dataloader_num_workers: int = 2
    checkpointing_steps: int = 500
    resume_from_checkpoint: str | None = None
    report_to: str = "tensorboard"

    def __post_init__(self) -> None:
        self.dataset_dir = str(self.dataset_dir)
        if not self.dataset_dir.strip():
            raise ValueError("dataset_dir must point to a prepared dataset")
        if self.resolution <= 0 or self.resolution % 8:
            raise ValueError("resolution must be positive and divisible by 8")
        if self.rank <= 0 or self.max_train_steps <= 0:
            raise ValueError("rank and max_train_steps must be positive")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        values: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown training settings: {', '.join(sorted(unknown))}")
        return cls(**values)
