from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
except ModuleNotFoundError:
    torch = None

from aina_train.chat import generate_chat_completion, normalize_openai_content, render_openai_messages
from aina_train.config import ModelConfig
from aina_train.hf_model import build_model, save_hf_model


def write_tiny_final_hf(path: Path) -> None:
    tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "You",
        "are",
        "Aina",
        "hello",
        "world",
        "Write",
        "Python",
        "ok",
    ]
    tokenizer_model = Tokenizer(WordLevel({token: idx for idx, token in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    config = ModelConfig(
        name="tiny-chat",
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = build_model(
        config,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    save_hf_model(path.parent, model, tokenizer)


class ChatInferenceTests(unittest.TestCase):
    def test_normalize_openai_content_variants(self):
        self.assertEqual(normalize_openai_content("hello"), "hello")
        self.assertEqual(normalize_openai_content([{"type": "text", "text": "hello"}]), "hello")
        self.assertIn("image_url", normalize_openai_content([{"type": "image_url", "image_url": {"url": "x"}}]))

    def test_render_openai_messages_roles(self):
        prompt = render_openai_messages(
            [
                {"role": "system", "content": "You are Aina."},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": {"result": "done"}},
            ]
        )
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|tool|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

    @unittest.skipIf(torch is None, "PyTorch/Transformers/tokenizers are not installed locally")
    def test_generate_chat_completion_accepts_openai_compatible_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_hf = Path(tmpdir) / "final_hf"
            write_tiny_final_hf(final_hf)
            model = AutoModelForCausalLM.from_pretrained(final_hf)
            tokenizer = AutoTokenizer.from_pretrained(final_hf)
            self.assertIn("chat_template", (final_hf / "tokenizer_config.json").read_text())
            cases = [
                [
                    {"role": "system", "content": "You are Aina."},
                    {"role": "user", "content": "hello world"},
                ],
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are Aina."}]},
                    {"role": "user", "content": [{"type": "text", "text": "Write Python"}]},
                ],
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "world"},
                ],
                [
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "content": {"name": "unit", "result": "ok"}},
                ],
            ]
            for messages in cases:
                text = generate_chat_completion(
                    model,
                    tokenizer,
                    messages,
                    max_new_tokens=4,
                    device="cpu",
                )
                self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
