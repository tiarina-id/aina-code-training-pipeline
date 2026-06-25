from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from .config import TrainConfig, config_hash, ensure_dirs
from .data import build_datasets, load_dataset_metadata
from .hf_model import build_model, count_parameters, estimate_tokens_per_second, save_hf_model, unwrap_model
from .tokenizer import load_tokenizer
from .upload_s3 import restore_training_checkpoint_from_s3, sync_dataset_from_s3, upload_outputs, upload_training_checkpoint


def run_training(config: TrainConfig, *, resume: bool | None = None, skip_upload: bool = False) -> dict[str, Any]:
    ensure_dirs(config)
    torch.manual_seed(config.seed)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = resolve_device(config.device, local_rank)
    is_master = rank == 0
    progress = TrainingProgress(enabled=is_master)
    progress.event("setup", f"project={config.project_name} stage={config.stage} output={config.output_dir}")
    if is_master:
        downloaded = sync_dataset_from_s3(config.s3_dataset, config.dataset_dir)
        if downloaded:
            progress.event("data", f"s3 sync downloaded={len(downloaded)} destination={config.dataset_dir}")
        should_resume = config.resume if resume is None else resume
        if should_resume and not skip_upload:
            restored = restore_training_checkpoint_from_s3(config.s3_output, config.output_dir)
            if restored:
                progress.event("restore", f"s3 checkpoint files={len(restored)} destination={config.output_dir}")
    if ddp:
        dist.barrier()

    tokenizer_path = resolve_tokenizer_path(config)
    tokenizer = load_tokenizer(tokenizer_path, fallback=config.tokenizer_fallback) if tokenizer_path or config.tokenizer_fallback else None
    if config.stage == "sft" and tokenizer is None:
        raise ValueError("SFT requires tokenizer_path or tokenizer_fallback")
    model_config = resolve_model_config(config, tokenizer_vocab_size=getattr(tokenizer, "vocab_size", None))
    model = build_model(
        model_config,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    ).to(device)
    if config.init_checkpoint:
        init_path = Path(config.init_checkpoint)
        if not init_path.exists() and config.init_s3_output and is_master:
            restored = restore_training_checkpoint_from_s3(config.init_s3_output, init_path.parent)
            if restored:
                progress.event("restore", f"s3 init checkpoint files={len(restored)} destination={init_path.parent}")
        if ddp:
            dist.barrier()
        if init_path.exists():
            load_checkpoint(init_path, model=model, map_location=device)
            progress.event("init", f"loaded checkpoint={init_path}")
        else:
            raise FileNotFoundError(
                f"init_checkpoint is required but missing: {init_path}. "
                "Run pretrain first or configure init_s3_output to restore the base checkpoint."
            )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    cfg_hash = config_hash(dataclasses.replace(config, model=model_config))
    start_step = 0
    best_val_loss: float | None = None
    should_resume = config.resume if resume is None else resume
    latest = latest_checkpoint_path(config.output_dir)
    if should_resume and latest.exists():
        try:
            state = load_checkpoint(latest, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint is not compatible with the current HF/LLaMA model config: {latest}. "
                "Use --no-resume or choose a clean output_dir after the HF migration."
            ) from exc
        start_step = int(state.get("step", 0))
        best_val_loss = state.get("best_val_loss")
        progress.event("resume", f"checkpoint={latest} step={start_step} best_val_loss={format_metric(best_val_loss)}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if config.compile:
        model = torch.compile(model)

    train_data, val_data = build_datasets(config, tokenizer, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(device, config.dtype))
    autocast_context = autocast_for(device, config.dtype)
    history: list[dict[str, Any]] = []
    step = start_step
    model.train()
    progress.start(config, model_config, model, device=device, start_step=start_step, skip_upload=skip_upload)

    while step < config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.grad_accum_steps):
            x, y = train_data.get_batch(config.batch_size, device)
            with autocast_context:
                outputs = model(input_ids=x, labels=y)
                if outputs.loss is None:
                    raise RuntimeError("loss was not produced")
                loss = outputs.loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())
        scaler.unscale_(optimizer)
        if config.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if is_master and should_log_step(step, config):
            progress.step(step, train_loss=total_loss, lr=scheduler.get_last_lr()[0])

        if is_master and (step == 1 or step % config.eval_interval == 0 or step == config.max_steps):
            progress.event("eval", f"step={step}/{config.max_steps} batches={config.eval_batches}")
            val_loss = evaluate(model, val_data, config, device, autocast_context)
            best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
            row = {"step": step, "train_loss": total_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
            history.append(row)
            write_report(config, model_config, model, history, best_val_loss, completed=step >= config.max_steps)
            progress.event(
                "eval",
                f"step={step}/{config.max_steps} train_loss={total_loss:.4f} "
                f"val_loss={val_loss:.4f} best_val_loss={format_metric(best_val_loss)}",
            )

        if is_master and (step % config.checkpoint_interval == 0 or step == config.max_steps):
            checkpoint_path = save_checkpoint(
                config.output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_loss=best_val_loss,
                config_hash=cfg_hash,
            )
            progress.event("ckpt", f"step={step}/{config.max_steps} saved={checkpoint_path}")
            if not skip_upload:
                progress.event("upload", f"checkpoint step={step}/{config.max_steps} destination={config.s3_output}")
                uploaded = upload_training_checkpoint(config.output_dir, config.s3_output, step=step)
                if uploaded:
                    progress.event("upload", f"checkpoint step={step}/{config.max_steps} files={len(uploaded)}")

    if is_master:
        progress.event("save", f"exporting Hugging Face model to {Path(config.output_dir) / 'final_hf'}")
        final_path = save_hf_model(config.output_dir, model, tokenizer)
        report = write_report(config, model_config, model, history, best_val_loss, completed=True)
        report["final_hf_dir"] = str(final_path)
        if not skip_upload:
            progress.event("upload", f"outputs destination={config.s3_output}")
        uploaded = [] if skip_upload else upload_outputs(config.output_dir, config.s3_output)
        report["uploaded"] = uploaded
        config.resolved_report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        progress.finish(final_path=final_path, uploaded_count=len(uploaded), report_path=config.resolved_report_path)
    else:
        report = {}

    if ddp:
        dist.destroy_process_group()
    return report


def should_log_step(step: int, config: TrainConfig) -> bool:
    return config.log_interval > 0 and (step == 1 or step % config.log_interval == 0 or step == config.max_steps)


class TrainingProgress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.is_tty = sys.stdout.isatty()
        self.started_at = time.time()
        self.interval_at = self.started_at
        self.interval_step = 0
        self.start_step = 0
        self.max_steps = 0
        self.step_tokens = 0
        self.progress_active = False
        self.progress_width = 0

    def start(
        self,
        config: TrainConfig,
        model_config,
        model: torch.nn.Module,
        *,
        device: torch.device,
        start_step: int,
        skip_upload: bool,
    ) -> None:
        if not self.enabled:
            return
        self.started_at = time.time()
        self.interval_at = self.started_at
        self.interval_step = start_step
        self.start_step = start_step
        self.max_steps = config.max_steps
        self.step_tokens = config.batch_size * config.grad_accum_steps * model_config.sequence_length
        effective_batch = config.batch_size * config.grad_accum_steps
        remaining = max(0, config.max_steps - start_step)
        self.event(
            "model",
            f"name={model_config.name} params={format_count(count_parameters(unwrap_model(model)))} "
            f"seq_len={model_config.sequence_length} vocab={model_config.vocab_size}",
        )
        self.event(
            "train",
            f"device={device} dtype={config.dtype} batch={config.batch_size} grad_accum={config.grad_accum_steps} "
            f"effective_batch={effective_batch} step_tokens={format_count(self.step_tokens)}",
        )
        self.event(
            "schedule",
            f"start_step={start_step} max_steps={config.max_steps} remaining={remaining} "
            f"log_every={config.log_interval} eval_every={config.eval_interval} ckpt_every={config.checkpoint_interval}",
        )
        self.event(
            "paths",
            f"dataset={config.dataset_dir} report={config.resolved_report_path} upload={'off' if skip_upload else 'on'}",
        )

    def step(self, step: int, *, train_loss: float, lr: float) -> None:
        if not self.enabled:
            return
        now = time.time()
        interval_steps = max(1, step - self.interval_step)
        interval_seconds = now - self.interval_at
        interval_tps = estimate_tokens_per_second(self.step_tokens * interval_steps, interval_seconds)
        elapsed = now - self.started_at
        completed_steps = max(1, step - self.start_step)
        average_tps = estimate_tokens_per_second(self.step_tokens * completed_steps, elapsed)
        remaining_tokens = max(0, self.max_steps - step) * self.step_tokens
        eta = remaining_tokens / average_tps if average_tps and average_tps > 0 else math.nan
        self.interval_at = now
        self.interval_step = step
        line = (
            f"[train] {progress_bar(step, self.max_steps)} "
            f"step {step}/{self.max_steps} | loss {train_loss:.4f} | lr {lr:.3g} | "
            f"{format_rate(interval_tps)} | elapsed {format_duration(elapsed)} | eta {format_duration(eta)}"
        )
        self.emit_progress(line)

    def event(self, label: str, message: str) -> None:
        if not self.enabled:
            return
        self.clear_progress()
        print(f"[{label:<8}] {message}", flush=True)

    def finish(self, *, final_path: Path, uploaded_count: int, report_path: Path) -> None:
        if not self.enabled:
            return
        self.clear_progress()
        elapsed = time.time() - self.started_at
        print(
            f"[done    ] final_hf={final_path} report={report_path} "
            f"uploaded_files={uploaded_count} elapsed={format_duration(elapsed)}",
            flush=True,
        )

    def emit_progress(self, line: str) -> None:
        if not self.is_tty:
            print(line, flush=True)
            return
        line = fit_terminal(line)
        padding = " " * max(0, self.progress_width - len(line))
        sys.stdout.write(f"\r{line}{padding}")
        sys.stdout.flush()
        self.progress_width = len(line)
        self.progress_active = True

    def clear_progress(self) -> None:
        if not self.is_tty or not self.progress_active:
            return
        sys.stdout.write("\r" + (" " * self.progress_width) + "\r")
        sys.stdout.flush()
        self.progress_active = False
        self.progress_width = 0


def progress_bar(step: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------]   0.0%"
    ratio = min(1.0, max(0.0, step / total))
    filled = int(round(width * ratio))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio * 100:5.1f}%"


def fit_terminal(line: str) -> str:
    columns = shutil.get_terminal_size((120, 20)).columns
    max_width = max(40, columns - 1)
    if len(line) <= max_width:
        return line
    return line[: max_width - 3] + "..."


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_rate(value: float) -> str:
    if not math.isfinite(value):
        return "n/a tok/s"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M tok/s"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K tok/s"
    return f"{value:.1f} tok/s"


def format_count(value: int | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def resolve_tokenizer_path(config: TrainConfig) -> str | None:
    candidates = []
    if config.tokenizer_path:
        candidates.append(Path(config.tokenizer_path))
    candidates.append(Path(config.dataset_dir) / "tokenizer")
    if config.init_checkpoint:
        init_path = Path(config.init_checkpoint)
        candidates.append(init_path.parent / "final_hf")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return config.tokenizer_path


def resolve_model_config(config: TrainConfig, *, tokenizer_vocab_size: int | None):
    model_config = config.model
    vocab_size = model_config.vocab_size
    if vocab_size is None and config.stage == "pretrain":
        vocab_size = int(load_dataset_metadata(config.dataset_dir)["vocab_size"])
    if vocab_size is None and tokenizer_vocab_size is not None:
        vocab_size = int(tokenizer_vocab_size)
    if vocab_size is None:
        raise ValueError("unable to resolve model vocab_size")
    return dataclasses.replace(model_config, vocab_size=vocab_size)


def resolve_device(device: str, local_rank: int) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    return resolved


def use_amp(device: torch.device, dtype: str) -> bool:
    return device.type == "cuda" and dtype in {"auto", "float16", "bfloat16"}


def autocast_for(device: torch.device, dtype: str):
    if not use_amp(device, dtype):
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype in {"auto", "bfloat16"} else torch.float16
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2 and not name.endswith("wte.weight"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.optimizer.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    def lr_lambda(step: int) -> float:
        if step < config.optimizer.warmup_steps:
            return max(1e-8, step / max(1, config.optimizer.warmup_steps))
        progress = (step - config.optimizer.warmup_steps) / max(1, config.max_steps - config.optimizer.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_ratio = config.optimizer.min_learning_rate / config.optimizer.learning_rate
        return min_ratio + cosine * (1.0 - min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_data, config: TrainConfig, device: torch.device, autocast_context) -> float:
    del device
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        x, y = val_data.get_batch(config.batch_size, next(model.parameters()).device)
        with autocast_context:
            outputs = model(input_ids=x, labels=y)
        if outputs.loss is not None:
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def write_report(
    config: TrainConfig,
    model_config,
    model: torch.nn.Module,
    history: list[dict[str, Any]],
    best_val_loss: float | None,
    *,
    completed: bool,
) -> dict[str, Any]:
    report = {
        "project": config.project_name,
        "stage": config.stage,
        "completed": completed,
        "model": dataclasses.asdict(model_config),
        "parameter_count": count_parameters(unwrap_model(model)),
        "dataset_dir": config.dataset_dir,
        "output_dir": config.output_dir,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    config.resolved_report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
