from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .tokenizer import TokenizerLike, encode_messages_assistant_labels


def load_dataset_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset metadata: {path}")
    return json.loads(path.read_text())


def _np_dtype(dtype: str) -> np.dtype:
    if dtype == "uint16":
        return np.dtype(np.uint16)
    if dtype == "uint32":
        return np.dtype(np.uint32)
    raise ValueError(f"unsupported token dtype: {dtype}")


class PackedBinaryDataset:
    def __init__(self, dataset_dir: str | Path, split: str, *, seed: int) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata = load_dataset_metadata(self.dataset_dir)
        self.sequence_length = int(self.metadata["tokens_per_sample"])
        self.dtype = _np_dtype(self.metadata["dtype"])
        self.rng = random.Random(seed)
        self.shards: list[tuple[np.memmap, int]] = []
        for shard in self.metadata.get("shards", []):
            if shard.get("split") != split:
                continue
            path = self.dataset_dir / shard["path"]
            if not path.exists():
                raise FileNotFoundError(f"missing shard: {path}")
            token_count = int(shard["tokens"])
            self.shards.append((np.memmap(path, dtype=self.dtype, mode="r"), token_count))
        if not self.shards:
            raise ValueError(f"no {split} binary shards found in {self.dataset_dir}")

    def get_batch(self, batch_size: int, device):
        torch = require_torch()
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for _ in range(batch_size):
            shard, token_count = self.rng.choice(self.shards)
            sequence_count = max(1, token_count // self.sequence_length)
            sequence_index = self.rng.randrange(sequence_count)
            start = sequence_index * self.sequence_length
            block = np.asarray(shard[start : start + self.sequence_length], dtype=np.int64)
            if len(block) < self.sequence_length:
                block = np.pad(block, (0, self.sequence_length - len(block)))
            ids = torch.from_numpy(block.astype(np.int64, copy=True))
            xs.append(ids)
            ys.append(ids.clone())
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


class SftJsonlDataset:
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        tokenizer: TokenizerLike,
        max_length: int,
        assistant_only_loss: bool,
        seed: int,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata = load_dataset_metadata(self.dataset_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_only_loss = assistant_only_loss
        self.rng = random.Random(seed)
        self.records = list(self._iter_records(split))
        if not self.records:
            raise ValueError(f"no {split} SFT records found in {self.dataset_dir}")

    def _iter_records(self, split: str) -> Iterable[dict[str, Any]]:
        for shard in self.metadata.get("shards", []):
            if shard.get("split") != split:
                continue
            path = self.dataset_dir / shard["path"]
            if not path.exists():
                raise FileNotFoundError(f"missing shard: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def get_batch(self, batch_size: int, device):
        torch = require_torch()
        encoded: list[tuple[list[int], list[int]]] = []
        max_len = 0
        for _ in range(batch_size):
            ids: list[int] = []
            labels: list[int] = []
            for _attempt in range(20):
                record = self.rng.choice(self.records)
                ids, labels = encode_messages_assistant_labels(
                    self.tokenizer,
                    record["messages"],
                    max_length=self.max_length,
                    assistant_only_loss=self.assistant_only_loss,
                )
                if len(ids) >= 2 and any(label != -100 for label in labels[1:]):
                    break
            if len(ids) < 2 or not any(label != -100 for label in labels[1:]):
                eos_id = int(getattr(self.tokenizer, "eos_token_id", 0) or 0)
                ids = [eos_id, eos_id]
                labels = [-100, eos_id]
            encoded.append((ids, labels))
            max_len = max(max_len, len(ids))

        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        xs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        ys = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for row, (ids, labels) in enumerate(encoded):
            xs[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            ys[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        return xs.to(device), ys.to(device)


def build_datasets(config, tokenizer: TokenizerLike | None, device):
    del device
    if config.stage == "pretrain":
        return (
            PackedBinaryDataset(config.dataset_dir, "train", seed=config.seed),
            PackedBinaryDataset(config.dataset_dir, "val", seed=config.seed + 1),
        )
    if config.stage == "sft":
        if tokenizer is None:
            raise ValueError("SFT requires a tokenizer")
        max_length = config.sft_max_length or config.model.sequence_length
        return (
            SftJsonlDataset(
                config.dataset_dir,
                "train",
                tokenizer=tokenizer,
                max_length=max_length,
                assistant_only_loss=config.sft_assistant_only_loss,
                seed=config.seed,
            ),
            SftJsonlDataset(
                config.dataset_dir,
                "val",
                tokenizer=tokenizer,
                max_length=max_length,
                assistant_only_loss=config.sft_assistant_only_loss,
                seed=config.seed + 1,
            ),
        )
    raise ValueError(f"unsupported stage: {config.stage}")


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for dataset batching/training") from exc
    return torch
