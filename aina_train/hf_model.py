from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from .config import ModelConfig


def build_llama_config(
    config: ModelConfig,
    *,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> LlamaConfig:
    if config.architecture != "llama":
        raise ValueError(f"Unsupported model architecture: {config.architecture}")
    if config.vocab_size is None:
        raise ValueError("model.vocab_size must be resolved before model construction")
    return LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads or config.num_attention_heads,
        max_position_embeddings=config.sequence_length,
        hidden_act=config.hidden_act,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        initializer_range=config.initializer_range,
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )


def build_model(
    config: ModelConfig,
    *,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> LlamaForCausalLM:
    return LlamaForCausalLM(
        build_llama_config(
            config,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    )


def save_hf_model(output_dir: str | Path, model: torch.nn.Module, tokenizer=None) -> Path:
    final_dir = Path(output_dir) / "final_hf"
    final_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    model_to_save.save_pretrained(final_dir, safe_serialization=True)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_dir)
    generation_config = GenerationConfig.from_model_config(model_to_save.config)
    generation_config.save_pretrained(final_dir)
    return final_dir


def estimate_tokens_per_second(step_tokens: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return math.nan
    return step_tokens / elapsed_seconds


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model
