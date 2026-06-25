from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from aina_train.cli import build_parser
from aina_train.config import load_config


class ConfigCliTests(unittest.TestCase):
    def test_load_config_and_cli_overrides_without_torch(self):
        raw = {
            "project_name": "unit",
            "stage": "pretrain",
            "dataset_dir": "/tmp/dataset",
            "s3_dataset": "s3://bucket/datasets/unit/",
            "output_dir": "/tmp/out",
            "batch_size": 2,
            "max_steps": 10,
            "model": {
                "name": "tiny",
                "vocab_size": 256,
                "sequence_length": 16,
                "hidden_size": 16,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(raw))
            config = load_config(path).with_overrides(
                max_steps=3,
                batch_size=1,
                s3_dataset="s3://other/unit/",
            )
        self.assertEqual(config.max_steps, 3)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.model.name, "tiny")
        self.assertEqual(config.s3_dataset, "s3://other/unit/")

    def test_parser_help_does_not_import_torch(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                "config.yaml",
                "--s3-dataset",
                "s3://bucket/data/",
                "--skip-upload",
                "--no-resume",
            ]
        )
        self.assertEqual(args.config, "config.yaml")
        self.assertEqual(args.s3_dataset, "s3://bucket/data/")
        self.assertFalse(args.resume)
        self.assertTrue(args.skip_upload)


if __name__ == "__main__":
    unittest.main()
