"""Experiment 8 activations in generation passes: request checks, the copy into /work, and the array shapes.
Run: python3 -I tests/test_qwen_activations.py (the shape test needs assets/venvs/qwen and assets/exp8/qwen35-text,
and runs on the CPU: assets/venvs/qwen/bin/python -I tests/test_qwen_activations.py)"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the shape test runs on the CPU
ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "assets" / "exp8" / "qwen35-text"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "python" / "adapters" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = load("qwen_pass")
W = load("qwen_worker")


class Requests(unittest.TestCase):
    def test_the_published_request_passes(self):
        self.assertEqual(W.check_activations({"layers": [4, 8, 12, 16, 20, 24], "mlp": True}),
                         {"layers": [4, 8, 12, 16, 20, 24], "mlp": True})
        self.assertEqual(W.check_activations({"layers": [0]}), {"layers": [0], "mlp": False})

    def test_bad_requests_fail_before_any_generation(self):
        bad = ([4], {"layers": []}, {"layers": [25]}, {"layers": [-1]}, {"layers": [4, 4]}, {"layers": ["4"]},
               {"layers": [True]}, {"layers": [4], "mlp": "yes"}, {"layers": [4], "tokens": 5})
        for a in bad:
            with self.assertRaises(W.Bad, msg=a):
                W.generate(None, {"activations": a, "problems": ["x"]}, "ckpt")


class IntoWork(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = os.path.join(self.tmp.name, "work")
        self.outside = os.path.join(self.tmp.name, "outside")
        self.src = os.path.join(self.tmp.name, "src")
        for d in (self.work, self.outside, os.path.join(self.src, "gen_acts", "edited")):
            os.makedirs(d)
        Path(self.src, "gen.jsonl").write_text("new\n")
        Path(self.src, "gen_acts", "edited", "index.json").write_text("{}")
        Path(self.outside, "victim").write_text("keep")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_planted_symlink_file_is_replaced_not_followed(self):
        os.symlink(os.path.join(self.outside, "victim"), os.path.join(self.work, "gen_3.jsonl"))
        Q.copy_into_work(os.path.join(self.src, "gen.jsonl"), self.work, "gen_3.jsonl")
        self.assertEqual(Path(self.outside, "victim").read_text(), "keep")
        self.assertFalse(os.path.islink(os.path.join(self.work, "gen_3.jsonl")))
        self.assertEqual(Path(self.work, "gen_3.jsonl").read_text(), "new\n")

    def test_a_planted_symlink_folder_is_replaced_not_followed(self):
        os.symlink(self.outside, os.path.join(self.work, "gen_3_acts"))
        Q.copy_into_work(os.path.join(self.src, "gen_acts"), self.work, "gen_3_acts")
        self.assertEqual(sorted(os.listdir(self.outside)), ["victim"])
        self.assertTrue(Path(self.work, "gen_3_acts", "edited", "index.json").is_file())

    def test_an_old_folder_is_replaced(self):
        os.makedirs(os.path.join(self.work, "gen_3_acts", "stale"))
        Q.copy_into_work(os.path.join(self.src, "gen_acts"), self.work, "gen_3_acts")
        self.assertEqual(os.listdir(os.path.join(self.work, "gen_3_acts")), ["edited"])

    def test_a_generation_pass_puts_the_activations_where_the_prompt_says(self):
        run, out = os.path.join(self.tmp.name, "run"), os.path.join(self.tmp.name, "out")
        os.makedirs(run)
        req = os.path.join(self.tmp.name, "req.json")
        Path(req).write_text(json.dumps({"type": "generate", "problems": ["p"], "activations": {"layers": [4]}}))

        def fake_worker(args, stage, out_dir):
            shutil.copytree(os.path.join(self.src, "gen_acts"), os.path.join(out_dir, "gen_acts"))
            shutil.copyfile(os.path.join(self.src, "gen.jsonl"), os.path.join(out_dir, "gen.jsonl"))
            return {"ok": True, "activations": {"layers": [4]}}

        env = {"OPENWISH_REQ": req, "OPENWISH_NET": run, "OPENWISH_RUN_DIR": run, "OPENWISH_OUT": out,
               "OPENWISH_WORK": self.work, "OPENWISH_PASS": "3"}
        real, saved = Q.run_worker, {k: os.environ.get(k) for k in env}
        Q.run_worker = fake_worker
        os.environ.update(env)
        try:
            res = Q.forward()
        finally:
            Q.run_worker = real
            for k, v in saved.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual(res["activations"]["dir"], "/work/gen_3_acts/")
        self.assertTrue(Path(self.work, "gen_3_acts", "edited", "index.json").is_file())
        self.assertEqual(Path(self.work, "gen_3.jsonl").read_text(), "new\n")


def have_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (TEXT / "tokenizer.json").exists()


@unittest.skipUnless(have_transformers(), "needs transformers and assets/exp8/qwen35-text")
class Shapes(unittest.TestCase):
    def test_a_tiny_qwen35_returns_one_row_per_answer(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = json.loads((TEXT / "config.json").read_text())
        cfg.update(num_hidden_layers=4, layer_types=cfg["layer_types"][:4], hidden_size=64, intermediate_size=96, head_dim=16,
                   num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=2, linear_num_value_heads=2,
                   linear_key_head_dim=16, linear_value_head_dim=16, mtp_num_hidden_layers=0)
        with tempfile.TemporaryDirectory() as d:
            ckpt, out = os.path.join(d, "ckpt"), os.path.join(d, "acts")
            os.makedirs(ckpt)
            Path(ckpt, "config.json").write_text(json.dumps(cfg))
            torch.manual_seed(0)
            AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(ckpt)).to(torch.bfloat16).save_pretrained(ckpt)
            for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                shutil.copyfile(TEXT / f, os.path.join(ckpt, f))
            items = [{"user": "What is 1+1?"}, {"user": "What is 2+2?"}]
            recs = [{"key": f"k{i}", "sample": 0, "correct": i == 0, "failure": None, "finish": "stop", "tokens": 3,
                     "text": "<think>\nadd\n</think>\n\\boxed{2}"} for i in range(2)]
            info = W.generation_activations(ckpt, items, recs, {"layers": [0, 2, 4], "mlp": True}, out)
            self.assertEqual(info["mlp_layers"], [0, 2], "layer 4 is the last residual stream, with no MLP after it")
            self.assertEqual(info["files"], {"hidden_last_prompt.npy": [2, 3, 64], "hidden_mean_answer.npy": [2, 3, 64],
                                             "mlp_last_prompt.npy": [2, 2, 96], "mlp_mean_answer.npy": [2, 2, 96]})
            index = json.loads(Path(out, "index.json").read_text())
            self.assertEqual([r["correct"] for r in index["rows"]], [True, False])
            import numpy as np

            h = np.load(os.path.join(out, "hidden_mean_answer.npy"))
            self.assertTrue(np.isfinite(h).all() and np.abs(h).sum() > 0)


if __name__ == "__main__":
    unittest.main()
