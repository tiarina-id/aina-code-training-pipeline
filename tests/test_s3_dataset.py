from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aina_train.upload_s3 import (
    dataset_ready,
    expected_dataset_files,
    get_ready_json,
    parse_s3_uri,
    should_skip_dataset_key,
    should_upload_final_output,
    upload_outputs,
    upload_training_checkpoint,
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

    def test_checkpoint_upload_excludes_numbered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint-latest.pt").write_bytes(b"latest")
            (root / "checkpoint-step-00001000.pt").write_bytes(b"numbered")
            (root / "training_report.json").write_text("{}")

            client = FakeS3Client()
            with fake_boto3(client):
                uploaded = upload_training_checkpoint(root, "s3://bucket/out/", step=1000)

        self.assertEqual(
            [item[1] for item in client.uploads],
            ["out/checkpoint-latest.pt", "out/training_report.json"],
        )
        self.assertEqual(len(client.puts), 1)
        ready = json.loads(client.puts[0][1].decode("utf-8"))
        self.assertEqual(ready["latest_checkpoint"], "checkpoint-latest.pt")
        self.assertNotIn("numbered_checkpoint", ready)
        self.assertEqual(len(uploaded), 3)

    def test_final_upload_excludes_numbered_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint-latest.pt").write_bytes(b"latest")
            (root / "checkpoint-step-00001000.pt").write_bytes(b"numbered")
            (root / "training_report.json").write_text("{}")
            (root / "final_hf").mkdir()
            (root / "final_hf" / "model.safetensors").write_bytes(b"model")

            client = FakeS3Client()
            with fake_boto3(client):
                upload_outputs(root, "s3://bucket/out/")

        self.assertEqual(
            sorted(item[1] for item in client.uploads),
            [
                "out/checkpoint-latest.pt",
                "out/final_hf/model.safetensors",
                "out/training_report.json",
            ],
        )
        self.assertEqual(len(client.puts), 1)

    def test_should_upload_final_output(self):
        self.assertTrue(should_upload_final_output("checkpoint-latest.pt"))
        self.assertTrue(should_upload_final_output("training_report.json"))
        self.assertTrue(should_upload_final_output("final_hf/model.safetensors"))
        self.assertFalse(should_upload_final_output("checkpoint-step-00001000.pt"))
        self.assertFalse(should_upload_final_output("checkpoint/READY.json"))


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.puts: list[tuple[str, bytes]] = []

    def upload_file(self, local_file, bucket, key):
        del bucket
        self.uploads.append((Path(local_file).name, key))

    def put_object(self, *, Bucket, Key, Body, ContentType):
        del Bucket, ContentType
        self.puts.append((Key, Body))


def fake_boto3(client):
    boto3_module = types.SimpleNamespace(client=lambda service: client)
    return patch.dict(sys.modules, {"boto3": boto3_module})


if __name__ == "__main__":
    unittest.main()
