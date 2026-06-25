from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def latest_checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoint-latest.pt"


def save_checkpoint(
    output_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_val_loss: float | None,
    config_hash: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config_hash": config_hash,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    numbered = output_path / f"checkpoint-step-{step:08d}.pt"
    latest = latest_checkpoint_path(output_path)
    torch.save(state, numbered)
    torch.save(state, latest)
    return latest


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model
