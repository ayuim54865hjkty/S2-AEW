from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    image_size: int
    num_channels: int
    num_classes: int


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    checkpoint: str | None = None


@dataclass(frozen=True)
class WatermarkConfig:
    key: str
    eps: float = 0.3
    lr: float = 3e-3
    steps: int = 200
    lam: float = 1e-3
    spectral_weight: float = 2.2
    keep_weight: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    watermark: WatermarkConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ExperimentConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        watermark=WatermarkConfig(**raw["watermark"]),
    )
