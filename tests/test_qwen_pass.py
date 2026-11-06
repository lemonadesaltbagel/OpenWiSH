"""Experiment 8 adapter without a GPU: staging, the remote round trip, checkpoint pruning and the pass size limit.
Run: python3 -I tests/test_qwen_pass.py"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ADAPTERS = Path(__file__).resolve().parents[1] / "python" / "adapters"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = load("qwen_pass")
W = load("qwen_worker")


class Done:
    def __init__(self, stdout=""):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


class FakeRemote:
    """Stands in for subprocess.run: records every command, answers the worker with one JSON line."""

    def __init__(self, fail_sync_back=False):
        self.cmds = []
        self.fail_sync_back = fail_sync_back

    def __call__(self, argv, **kw):
        self.cmds.append(argv)
        if argv[0] == "rsync" and argv[-2].startswith("root@pod:") and self.fail_sync_back:
            raise subprocess.CalledProcessError(23, argv)
        return Done('{"ok": true, "score": 0.5}' if argv[0] == "ssh" and "qwen_worker.py" in argv[-1] else "")

    def ssh_lines(self):
        return [c[-1] for c in self.cmds if c[0] == "ssh"]


class Remote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self.tmp.name, "remote.json")
        json.dump({"host": "root@pod", "port": 22, "key": "k", "root": "/r", "python": "/v/bin/python", "gpu_util": 0.25},
                  open(self.cfg, "w"))
        os.environ["OPENWISH_QWEN_REMOTE"] = self.cfg
        self.real_run = Q.subprocess.run

    def tearDown(self):
        Q.subprocess.run = self.real_run
        os.environ.pop("OPENWISH_QWEN_REMOTE", None)
        self.tmp.cleanup()

    def worker(self, fake):
        Q.subprocess.run = fake
        edits = os.path.join(self.tmp.name, "edits")
        os.makedirs(edits)
        out = os.path.join(self.tmp.name, "pass_1")
        return Q.run_worker(["teacher_forced", "--out", "{out}"], {"edits": edits, "tag": "abc"}, out)

    def test_the_pod_drops_a_pass_folder_once_the_folder_is_back(self):
        fake = FakeRemote()
        self.assertTrue(self.worker(fake)["ok"])
        lines = fake.ssh_lines()
        self.assertIn("rm -rf /r/out/pass_1_abc /r/req/pass_1_abc.json", lines[-1])
        back = [i for i, c in enumerate(fake.cmds) if c[0] == "rsync" and c[-2].startswith("root@pod:")]
        self.assertLess(back[0], len(fake.cmds) - 1, "the delete comes after the copy back")

    def test_a_failed_copy_back_keeps_the_pass_folder_on_the_pod(self):
        fake = FakeRemote(fail_sync_back=True)
        self.worker(fake)
        self.assertFalse(any("rm -rf /r/out/pass_1_abc " in s for s in fake.ssh_lines()))

    def test_gpu_util_reaches_the_worker(self):
        fake = FakeRemote()
        self.worker(fake)
        run = [s for s in fake.ssh_lines() if "qwen_worker.py" in s]
        self.assertIn("export QWEN_GPU_UTIL=0.25;", run[0])


class Scoring(unittest.TestCase):
    def test_an_unedited_run_stages_an_empty_folder(self):
        seen = []

        def fake_worker(args, stage, out):
            seen.append(os.path.isdir(stage["edits"]))
            return {"ok": True, "score": 0.1}

        real = Q.run_worker
        Q.run_worker = fake_worker
        try:
            with tempfile.TemporaryDirectory() as run:
                os.environ["OPENWISH_RUN"], os.environ["OPENWISH_FINAL_NET"] = run, run

                class A:
                    sets, limit, samples = "math500", 4, 1

                self.assertTrue(Q.evaluate(A())["ok"])
        finally:
            Q.run_worker = real
            os.environ.pop("OPENWISH_RUN")
            os.environ.pop("OPENWISH_FINAL_NET")
        self.assertEqual(seen, [True])


class Checkpoints(unittest.TestCase):
    def test_only_the_newest_checkpoints_stay(self):
        with tempfile.TemporaryDirectory() as work:
            now = time.time()
            for i, name in enumerate(("ckpt_a", "ckpt_b", "ckpt_c", "ckpt_d")):
                os.makedirs(os.path.join(work, name))
                os.utime(os.path.join(work, name), (now - 100 + i, now - 100 + i))
            os.makedirs(os.path.join(work, "other"))
            W.prune_checkpoints(work, keep=os.path.join(work, "ckpt_new"), n_keep=2)
            self.assertEqual(sorted(os.listdir(work)), ["ckpt_d", "other"])
            W.prune_checkpoints(work, keep=os.path.join(work, "ckpt_d"), n_keep=2)
            self.assertEqual(sorted(os.listdir(work)), ["ckpt_d", "other"])


class PassSize(unittest.TestCase):
    CFG = {"hidden_size": 1024, "intermediate_size": 3584}

    def test_the_six_gb_limit_refuses_before_anything_runs(self):
        W.check_tf_size(64, 4096, self.CFG, hs_layers=list(range(6)), mlp_layers=[], topk=0, logprobs=True)
        with self.assertRaisesRegex(W.Bad, "at most 6 GB"):
            W.check_tf_size(64, 4096, self.CFG, hs_layers=list(range(25)), mlp_layers=list(range(24)), topk=0, logprobs=True)

    def test_the_estimate_counts_every_returned_array(self):
        b = W.tf_output_bytes(2, 10, self.CFG, hs_layers=[1], mlp_layers=[2], topk=5, logprobs=True)
        self.assertEqual(b, 2 * 10 * (8 + 4 + 5 * 12 + 2 * 1024 + 2 * 3584))


if __name__ == "__main__":
    unittest.main()
