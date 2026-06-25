from __future__ import annotations

import argparse

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Aina Code models.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset input directory.")
    parser.add_argument("--s3-dataset", default=None, help="Override S3 dataset input prefix.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override micro batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from checkpoint.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh run.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload outputs to S3.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config).with_overrides(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        s3_dataset=args.s3_dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    try:
        from .trainer import run_training
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required to run training. Install it manually on the server, "
                "or use `python -m pip install -e .[server]` if that matches your CUDA setup."
            ) from exc
        raise
    run_training(config, resume=args.resume, skip_upload=args.skip_upload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
