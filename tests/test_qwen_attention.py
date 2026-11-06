"""Experiment 8 attention probabilities in teacher-forced passes, on a tiny random Qwen3.5 on the CPU.
Run: assets/venvs/qwen/bin/python -I tests/test_qwen_attention.py (skips without transformers or assets/exp8/qwen35-text)"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the test runs on the CPU
ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "assets" / "exp8" / "qwen35-text"

spec = importlib.util.spec_from_file_location("qwen_worker", ROOT / "python" / "adapters" / "qwen_worker.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def have_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (TEXT / "tokenizer.json").exists()


def tiny_qwen35(ckpt):
    """A random 8-layer Qwen3.5 with the real config and tokenizer: full attention in layers 3 and 7."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = json.loads((TEXT / "config.json").read_text())
    cfg.update(num_hidden_layers=8, layer_types=cfg["layer_types"][:8], hidden_size=64, intermediate_size=96, head_dim=16,
               num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=2, linear_num_value_heads=2,
               linear_key_head_dim=16, linear_value_head_dim=16, mtp_num_hidden_layers=0)
    os.makedirs(ckpt)
    Path(ckpt, "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(ckpt)).to(torch.bfloat16).save_pretrained(ckpt)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(TEXT / f, os.path.join(ckpt, f))


@unittest.skipUnless(have_transformers(), "needs transformers and assets/exp8/qwen35-text")
class Attention(unittest.TestCase):
    def test_full_attention_layers_return_their_own_probabilities(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "ckpt")
            tiny_qwen35(ckpt)

            class Args:
                out = os.path.join(d, "out")

            req = {"texts": ["One plus one is two.", "Two plus two is four, and four plus four is eight."],
                   "return": {"attn_layers": [3, 7]}}
            res = W.teacher_forced(Args(), req, ckpt)
            self.assertTrue(res["ok"])
            T = res["lengths"]
            a3, a7 = (np.load(os.path.join(Args.out, f"attn_{l}_text1.npy")).astype(np.float32) for l in (3, 7))
            self.assertEqual(res["files"]["attn_3_text0"], [4, T[0], T[0]])
            self.assertEqual(list(a3.shape), [4, T[1], T[1]])
            np.testing.assert_allclose(a3.sum(-1), 1.0, atol=1e-2)
            self.assertEqual(float(np.abs(np.triu(a3, 1)).max()), 0.0, "causal")
            self.assertGreater(float(np.abs(a3 - a7).max()), 1e-3, "layer 7 is not a copy of layer 3")

    def test_a_linear_attention_layer_is_refused_before_the_model_loads(self):
        class Args:
            out = "/nonexistent"

        with self.assertRaisesRegex(W.Bad, "full-attention layers"):
            W.teacher_forced(Args(), {"texts": ["x"], "return": {"attn_layers": [2]}}, str(TEXT))


if __name__ == "__main__":
    unittest.main()
