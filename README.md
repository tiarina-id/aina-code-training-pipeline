# Aina Code Training Pipeline

Pipeline training untuk output dari `preproc-pipeline`.

Flow sama untuk lokal dan server: lokal hanya memakai config `3m` di CPU untuk smoke test, sedangkan server memakai config `50m` atau `500m` di single RTX 6000/H100.

## Setup

Fish local:

```fish
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -e .
```

Bash server:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is intentionally not installed by this repo for local setup. Install the correct PyTorch build manually on the server for its CUDA/driver stack before running training.

The current trainer uses Hugging Face LLaMA-compatible models and exports `final_hf/`. Checkpoints produced by the older custom GPT backend are not compatible; use `--no-resume` or a clean `output_dir` after migrating.

## Local CPU

Pretrain 3M:

```bash
python scripts/train.py \
  --config configs/aina_code_pretrain_3m_local.yaml \
  --skip-upload \
  --no-resume
```

The local 3M configs also include S3 dataset/output prefixes. Use `--skip-upload` to avoid uploading training checkpoints/results while still allowing dataset restore from S3 when `dataset_dir` is empty.

SFT 3M:

```bash
python scripts/train.py \
  --config configs/aina_code_sft_3m_local.yaml \
  --skip-upload \
  --no-resume
```

## Server

50M:

```bash
python scripts/train.py --config configs/aina_code_pretrain_50m_server.yaml --resume
python scripts/train.py --config configs/aina_code_sft_50m_server.yaml --resume
```

500M:

```bash
python scripts/train.py --config configs/aina_code_pretrain_500m_server.yaml --resume
python scripts/train.py --config configs/aina_code_sft_500m_server.yaml --resume
```

Optional multi-process launch if a later server has more than one GPU:

```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/aina_code_pretrain_50m_server.yaml \
  --resume
```

## Inputs

Pretrain reads the tokenized binary output from `preproc-pipeline`:

```text
train-*.bin
val-*.bin
metadata.json
manifest.json
```

SFT reads JSONL messages shards:

```text
train-*.jsonl
val-*.jsonl
metadata.json
manifest.json
```

On server configs, `s3_dataset` points to the preprocessing VM output. The training VM downloads the dataset into `dataset_dir` before training starts, then validates the local copy against `metadata.json` and the listed shards.

Override per run if the preprocessing VM uploaded to a different prefix:

```bash
python scripts/train.py \
  --config configs/aina_code_pretrain_50m_server.yaml \
  --s3-dataset s3://aina-code-datasets/v1/pretrain/aina-code-50m-1b/ \
  --dataset-dir /data/aina-dataset/pretrain/aina-code-50m-1b \
  --resume
```

## Outputs

```text
checkpoint-latest.pt
checkpoint-step-*.pt
final_hf/
training_report.json
```

If `s3_output` is configured and `--skip-upload` is not set, the output directory is uploaded to S3 and `checkpoint/READY.json` is written. `final_hf/` is a Hugging Face LLaMA-compatible model directory for vLLM/HF loading.

Every local checkpoint interval also backs up:

```text
checkpoint-latest.pt
checkpoint-step-*.pt
training_report.json
checkpoint/READY.json
```

If a training VM is replaced and `--resume` is used, the new VM restores `checkpoint-latest.pt` from `s3_output` when no local checkpoint exists.

## Test

```bash
python -m unittest discover -s tests -v
```

Tests that execute training are skipped when PyTorch is not installed.
