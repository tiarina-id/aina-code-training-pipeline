from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from aina_train.data import PackedBinaryDataset, SftJsonlDataset
from aina_train.tokenizer import ByteTokenizer


def write_pretrain_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train = np.arange(64, dtype=np.uint16)
    val = np.arange(64, 128, dtype=np.uint16)
    train.tofile(root / "train-00000.bin")
    val.tofile(root / "val-00000.bin")
    metadata = {
        "vocab_size": 256,
        "dtype": "uint16",
        "sequence_length": 16,
        "tokens_per_sample": 16,
        "shards": [
            {"split": "train", "path": "train-00000.bin", "tokens": 64, "sequences": 4},
            {"split": "val", "path": "val-00000.bin", "tokens": 64, "sequences": 4},
        ],
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


def write_sft_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "messages": [
            {"role": "system", "content": "You are Aina."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi"},
        ],
        "source": "unit",
    }
    for split in ["train", "val"]:
        (root / f"{split}-00000.jsonl").write_text(json.dumps(row) + "\n")
    metadata = {
        "output_mode": "sft_jsonl",
        "artifact_format": "jsonl_messages",
        "shards": [
            {"split": "train", "path": "train-00000.jsonl", "samples": 1, "tokens": 20},
            {"split": "val", "path": "val-00000.jsonl", "samples": 1, "tokens": 20},
        ],
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


@unittest.skipIf(torch is None, "PyTorch is not installed locally")
class DataTests(unittest.TestCase):
    def test_packed_binary_batch_is_shifted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_pretrain_dataset(root)
            dataset = PackedBinaryDataset(root, "train", seed=1)
            x, y = dataset.get_batch(2, torch.device("cpu"))
            self.assertEqual(tuple(x.shape), (2, 16))
            self.assertEqual(tuple(y.shape), (2, 16))
            self.assertTrue(torch.equal(x, y))

    def test_sft_masks_non_assistant_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_sft_dataset(root)
            dataset = SftJsonlDataset(
                root,
                "train",
                tokenizer=ByteTokenizer(),
                max_length=64,
                assistant_only_loss=True,
                seed=1,
            )
            _, y = dataset.get_batch(1, torch.device("cpu"))
            self.assertGreater(int((y != -100).sum().item()), 0)
            self.assertGreater(int((y == -100).sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
