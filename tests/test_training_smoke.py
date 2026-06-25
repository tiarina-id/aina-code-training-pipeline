from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

from aina_train.config import ModelConfig, OptimizerConfig, TrainConfig

from test_data import write_pretrain_dataset, write_sft_dataset


def tiny_model() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=256,
        sequence_length=16,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        dropout=0.0,
    )


def tiny_optimizer() -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        min_learning_rate=1e-4,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed locally")
class TrainingSmokeTests(unittest.TestCase):
    def test_pretrain_cpu_smoke_and_resume_checkpoint(self):
        from aina_train.trainer import run_training

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            output_dir = root / "out"
            write_pretrain_dataset(dataset_dir)
            config = TrainConfig(
                project_name="unit-pretrain",
                stage="pretrain",
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                model=tiny_model(),
                optimizer=tiny_optimizer(),
                batch_size=2,
                max_steps=2,
                eval_interval=1,
                eval_batches=1,
                checkpoint_interval=1,
                log_interval=1,
                device="cpu",
            )
            report = run_training(config, resume=False, skip_upload=True)
            self.assertTrue((output_dir / "checkpoint-latest.pt").exists())
            self.assertTrue((output_dir / "final_hf" / "config.json").exists())
            from transformers import AutoConfig, AutoModelForCausalLM

            hf_config = AutoConfig.from_pretrained(output_dir / "final_hf")
            self.assertEqual(hf_config.model_type, "llama")
            AutoModelForCausalLM.from_pretrained(output_dir / "final_hf")
            self.assertTrue(report["completed"])
            resumed = run_training(config.with_overrides(max_steps=3), resume=True, skip_upload=True)
            self.assertTrue(resumed["completed"])

    def test_sft_cpu_smoke(self):
        from aina_train.trainer import run_training

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "sft"
            output_dir = root / "out"
            write_sft_dataset(dataset_dir)
            config = TrainConfig(
                project_name="unit-sft",
                stage="sft",
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                tokenizer_fallback="byte",
                model=tiny_model(),
                optimizer=tiny_optimizer(),
                batch_size=1,
                max_steps=2,
                eval_interval=1,
                eval_batches=1,
                checkpoint_interval=1,
                log_interval=1,
                device="cpu",
                sft_max_length=32,
            )
            report = run_training(config, resume=False, skip_upload=True)
            self.assertTrue((output_dir / "training_report.json").exists())
            self.assertTrue((output_dir / "final_hf" / "config.json").exists())
            self.assertFalse(report["history"][-1]["val_loss"] != report["history"][-1]["val_loss"])
            self.assertEqual(report["stage"], "sft")

    def test_checkpoint_load_skips_unsupported_rng_state(self):
        from aina_train.checkpoint import restore_rng_state

        restore_rng_state({"rng_state": ("legacy", "state"), "cuda_rng_state": ("legacy",)})


if __name__ == "__main__":
    unittest.main()
