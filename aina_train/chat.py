from __future__ import annotations

import json
from typing import Any

import torch


def normalize_openai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def render_openai_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = normalize_openai_content(message.get("content"))
        if message.get("tool_calls"):
            content = "\n".join(
                part
                for part in [
                    content,
                    json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True),
                ]
                if part
            )
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


@torch.no_grad()
def generate_chat_completion(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    device: str | torch.device | None = None,
) -> str:
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(target_device)
    model.eval()
    prompt = render_openai_messages(messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    do_sample = temperature > 0
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
