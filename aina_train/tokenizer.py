from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TokenizerLike(Protocol):
    vocab_size: int
    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


class ByteTokenizer:
    vocab_size = 256
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8", errors="ignore"))


def load_tokenizer(path: str | None, *, fallback: str | None = None) -> TokenizerLike:
    if fallback == "byte" and (not path or not Path(path).exists()):
        return ByteTokenizer()
    if not path:
        raise ValueError("tokenizer_path is required for SFT unless tokenizer_fallback: byte is configured")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load tokenizer_path") from exc

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts)


def encode_messages_assistant_labels(
    tokenizer: TokenizerLike,
    messages: list[dict[str, str]],
    *,
    max_length: int,
    assistant_only_loss: bool,
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        rendered = render_messages([message])
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if getattr(tokenizer, "eos_token_id", None) is not None and message.get("role") == "assistant":
            ids = [*ids, int(tokenizer.eos_token_id)]
        input_ids.extend(ids)
        if assistant_only_loss and message.get("role") != "assistant":
            labels.extend([-100] * len(ids))
        else:
            labels.extend(ids)
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return input_ids, labels
