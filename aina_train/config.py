from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str = "llama"
    vocab_size: int | None = None
    sequence_length: int = 2048
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int | None = None
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_learning_rate: float = 3e-5


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    project_name: str
    stage: str
    dataset_dir: str
    output_dir: str
    model: ModelConfig
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    tokenizer_path: str | None = None
    tokenizer_fallback: str | None = None
    init_checkpoint: str | None = None
    init_s3_output: str | None = None
    s3_dataset: str | None = None
    s3_output: str | None = None
    seed: int = 42
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 100
    log_interval: int = 10
    num_workers: int = 0
    dtype: str = "auto"
    device: str = "auto"
    compile: bool = False
    resume: bool = True
    sft_max_length: int | None = None
    sft_assistant_only_loss: bool = True
    report_path: str | None = None

    @property
    def resolved_report_path(self) -> Path:
        if self.report_path:
            return Path(self.report_path)
        return Path(self.output_dir) / "training_report.json"

    def with_overrides(
        self,
        *,
        output_dir: str | None = None,
        dataset_dir: str | None = None,
        s3_dataset: str | None = None,
        max_steps: int | None = None,
        batch_size: int | None = None,
        grad_accum_steps: int | None = None,
    ) -> "TrainConfig":
        return dataclasses.replace(
            self,
            output_dir=output_dir if output_dir is not None else self.output_dir,
            dataset_dir=dataset_dir if dataset_dir is not None else self.dataset_dir,
            s3_dataset=s3_dataset if s3_dataset is not None else self.s3_dataset,
            max_steps=max_steps if max_steps is not None else self.max_steps,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            grad_accum_steps=grad_accum_steps if grad_accum_steps is not None else self.grad_accum_steps,
        )


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["model"] = ModelConfig(**raw["model"])
    raw["optimizer"] = OptimizerConfig(**raw.get("optimizer", {}))
    return TrainConfig(**raw)


def config_hash(config: TrainConfig) -> str:
    payload = dataclasses.asdict(config)
    for operational_key in ["init_s3_output", "s3_output", "resume", "report_path"]:
        payload.pop(operational_key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_dirs(config: TrainConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.resolved_report_path.parent.mkdir(parents=True, exist_ok=True)


def dataclass_to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value
