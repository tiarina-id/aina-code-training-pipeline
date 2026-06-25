from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aina_train.upload_s3 import (
    dataset_ready,
    expected_dataset_files,
    get_ready_json,
    parse_s3_uri,
    should_skip_dataset_key,
)


class S3DatasetTests(unittest.TestCase):
    def test_dataset_ready_checks_metadata_and_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = {
                "dtype": "uint16",
                "shards": [
                    {"split": "train", "path": "train-00000.bin", "tokens": 8},
                    {"split": "val", "path": "val-00000.bin", "tokens": 8},
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            (root / "manifest.json").write_text("{}")
            (root / "train-00000.bin").write_bytes(b"1234")
            self.assertFalse(dataset_ready(root))
            (root / "val-00000.bin").write_bytes(b"5678")
            self.assertTrue(dataset_ready(root))

    def test_expected_dataset_files_for_sft(self):
        metadata = {
            "output_mode": "sft_jsonl",
            "shards": [{"path": "train-00000.jsonl"}, {"path": "val-00000.jsonl"}],
        }
        self.assertEqual(
            expected_dataset_files(metadata),
            ["metadata.json", "train-00000.jsonl", "val-00000.jsonl"],
        )

    def test_parse_s3_uri_normalizes_prefix(self):
        self.assertEqual(parse_s3_uri("s3://bucket/path/to/data"), ("bucket", "path/to/data/"))

    def test_dataset_sync_skips_checkpoint_prefix(self):
        self.assertTrue(should_skip_dataset_key("checkpoint/READY.json"))
        self.assertTrue(should_skip_dataset_key("metadata.partial.json"))
        self.assertFalse(should_skip_dataset_key("train-00000.bin"))

    def test_ready_json_returns_none_when_missing(self):
        class MissingClient:
            def get_object(self, Bucket, Key):
                del Bucket, Key
                raise FakeClientError("NoSuchKey")

        self.assertIsNone(get_ready_json(MissingClient(), "bucket", "checkpoint/READY.json", FakeClientError))


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


if __name__ == "__main__":
    unittest.main()
