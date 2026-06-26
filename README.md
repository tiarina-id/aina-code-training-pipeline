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

## VM Setup

Syarat awal:

- Pasang IAM role ke EC2 yang punya akses S3 bucket `aina-code`.
- Install NVIDIA driver manual dulu, lalu reboot.
- PyTorch tetap install manual di dalam venv training.

Contoh install driver manual:

```bash
wget -qO- https://raw.githubusercontent.com/mutawakkilalallah/cloud-setup/main/aws-nvidia | bash -s -- 26
sudo reboot
```

Setelah reboot:

```bash
export PATH=${PATH}:/usr/local/cuda-13.0/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-13.0/lib64

nvidia-smi

cd ~
git clone https://github.com/tiarina-id/aina-code-training-pipeline.git training-pipeline
cd ~/training-pipeline

SWAP_GB=8 AINA_AUTO_CONFIRM=1 bash setup.sh

source ~/.bashrc
export AWS_DEFAULT_REGION=ap-southeast-3

aws sts get-caller-identity
aws s3 ls s3://aina-code/v1/datasets/aina-1-code-3m-1k/ --recursive --summarize

source .venv/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

sudo apt-get install -y tmux jq
deactivate 2>/dev/null || true
tmux new -s train

source .venv/bin/activate
```

Mini test 3M 1K:

```bash
python scripts/train.py --config configs/aina_code_3m_1k_pretrain.yaml --resume
python scripts/train.py --config configs/aina_code_3m_1k_sft.yaml --resume

aws s3 ls s3://aina-code/v1/training/aina-1-code-3m-1k/ --recursive --summarize
```

Opsional hapus checkpoint training setelah `final_hf/` aman:

```bash
aws s3 rm s3://aina-code/v1/training/aina-1-code-3m-1k/pretrain/ \
  --recursive \
  --exclude "*" \
  --include "checkpoint-*.pt"
aws s3 rm s3://aina-code/v1/training/aina-1-code-3m-1k/pretrain/checkpoint/ --recursive

aws s3 rm s3://aina-code/v1/training/aina-1-code-3m-1k/sft/ \
  --recursive \
  --exclude "*" \
  --include "checkpoint-*.pt"
aws s3 rm s3://aina-code/v1/training/aina-1-code-3m-1k/sft/checkpoint/ --recursive
```

## Local CPU

Pretrain 3M 1K:

```bash
python scripts/train.py \
  --config configs/aina_code_3m_1k_pretrain.yaml \
  --skip-upload \
  --no-resume
```

The local 3M configs also include S3 dataset/output prefixes. Use `--skip-upload` to avoid uploading training checkpoints/results while still allowing dataset restore from S3 when `dataset_dir` is empty.

SFT 3M 1K:

```bash
python scripts/train.py \
  --config configs/aina_code_3m_1k_sft.yaml \
  --skip-upload \
  --no-resume
```

## Server

50M 2K:

```bash
python scripts/train.py --config configs/aina_code_50m_2k_pretrain.yaml --resume
python scripts/train.py --config configs/aina_code_50m_2k_sft.yaml --resume
```

500M 8K:

```bash
python scripts/train.py --config configs/aina_code_500m_8k_pretrain.yaml --resume
python scripts/train.py --config configs/aina_code_500m_8k_sft.yaml --resume
```

Optional multi-process launch if a later server has more than one GPU:

```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/aina_code_50m_2k_pretrain.yaml \
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
  --config configs/aina_code_50m_2k_pretrain.yaml \
  --s3-dataset s3://aina-code/v1/datasets/aina-1-code-50m-2k/pretrain/ \
  --dataset-dir /data/aina-code/datasets/aina-1-code-50m-2k/pretrain \
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
training_report.json
checkpoint/READY.json
```

`checkpoint-step-*.pt` stays local by default to avoid repeated large S3 transfers. S3 resume uses `checkpoint-latest.pt`.

If a training VM is replaced and `--resume` is used, the new VM restores `checkpoint-latest.pt` from `s3_output` when no local checkpoint exists.

## Serve

Install vLLM:

```bash
cd /data/aina-code
python3 -m venv vllm-venv
source /data/aina-code/vllm-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm openai
```

Serve pretrain/base model:

```bash
source /data/aina-code/vllm-venv/bin/activate

vllm serve /data/aina-code/training/aina-1-code-3m-1k/pretrain/final_hf \
  --served-model-name aina-1-code-3m-1k-pretrain \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --generation-config vllm
```

Test pretrain/base:

```bash
curl -s http://localhost:8000/v1/models | jq

curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aina-1-code-3m-1k-pretrain",
    "prompt": "def add(a, b):",
    "max_tokens": 80,
    "temperature": 0.2
  }' | jq -r '
    if .error then
      "ERROR: " + .error.message
    else
      .choices[0].text
    end
  '
```

Serve SFT/instruct model. Stop server pretrain dulu dengan `Ctrl+C`, lalu jalankan:

```bash
source /data/aina-code/vllm-venv/bin/activate

vllm serve /data/aina-code/training/aina-1-code-3m-1k/sft/final_hf \
  --served-model-name aina-1-code-3m-1k-sft \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --generation-config vllm
```

Test SFT/instruct:

```bash
curl -s http://localhost:8000/v1/models | jq

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aina-1-code-3m-1k-sft",
    "messages": [
      {"role": "user", "content": "Buat fungsi Python add(a, b)."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }' | jq -r '
    if .error then
      "ERROR: " + .error.message
    else
      .choices[0].message.content
    end
  '
```

## Test

```bash
python -m unittest discover -s tests -v
```

Tests that execute training are skipped when PyTorch is not installed.
