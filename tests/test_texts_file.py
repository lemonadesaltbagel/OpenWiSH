"""Experiment 8 "texts_file": a JSON list of strings under /work, read under the rules of paths::file_under.
Run: python3 -I tests/test_texts_file.py"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("qwen_pass", Path(__file__).resolve().parents[1] / "python/adapters/qwen_pass.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


class TextsFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = os.path.join(self.tmp.name, "work")
        self.other = os.path.join(self.tmp.name, "other")
        os.makedirs(os.path.join(self.work, "sub"))
        os.makedirs(self.other)
        os.environ["OPENWISH_WORK"] = self.work
        Path(self.work, "t.json").write_text(json.dumps(["<|im_start|>user\nhi", "b"]))
        Path(self.work, "sub", "u.json").write_text(json.dumps(["x"]))
        Path(self.other, "s.json").write_text(json.dumps(["secret"]))

    def tearDown(self):
        os.environ.pop("OPENWISH_WORK", None)
        self.tmp.cleanup()

    def test_a_list_of_strings_under_work_is_read(self):
        self.assertEqual(Q.read_texts_file("/work/t.json"), (["<|im_start|>user\nhi", "b"], None))
        self.assertEqual(Q.read_texts_file("/work/sub/u.json"), (["x"], None))

    def test_paths_outside_work_and_symlinks_are_refused(self):
        os.symlink(os.path.join(self.other, "s.json"), os.path.join(self.work, "link.json"))
        os.symlink(self.other, os.path.join(self.work, "linkdir"))
        os.mkfifo(os.path.join(self.work, "fifo"))
        bad = ("/work/link.json", "/work/linkdir/s.json", "/work/../other/s.json", "/work/sub/../t.json", "/work//t.json",
               "/work/./t.json", "/work/sub", "/work/fifo", "/work/", "/etc/passwd", "t.json", "/work/missing.json", 7, None)
        for p in bad:
            texts, err = Q.read_texts_file(p)
            self.assertIsNone(texts, p)
            self.assertTrue(err, p)

    def test_the_content_must_be_a_json_list_of_strings_within_the_size_limit(self):
        Path(self.work, "obj.json").write_text('{"a": 1}')
        Path(self.work, "nums.json").write_text("[1, 2]")
        Path(self.work, "broken.json").write_text("[")
        for p in ("/work/obj.json", "/work/nums.json"):
            self.assertTrue(Q.read_texts_file(p)[1].endswith("JSON list of strings"), p)
        self.assertIn("not valid JSON", Q.read_texts_file("/work/broken.json")[1])
        real = Q.TEXTS_FILE_MAX_BYTES
        Q.TEXTS_FILE_MAX_BYTES = 10
        try:
            self.assertIn("larger than", Q.read_texts_file("/work/t.json")[1])
        finally:
            Q.TEXTS_FILE_MAX_BYTES = real

    def run_forward(self, req):
        sent = []

        def fake_worker(args, stage, out_dir):
            sent.append(json.loads(Path(stage["req"]).read_text()))
            return {"ok": True}

        rp = os.path.join(self.tmp.name, "req.json")
        Path(rp).write_text(json.dumps(req))
        env = {"OPENWISH_REQ": rp, "OPENWISH_NET": self.tmp.name, "OPENWISH_RUN_DIR": self.tmp.name,
               "OPENWISH_OUT": os.path.join(self.tmp.name, "out")}
        real = Q.run_worker
        Q.run_worker = fake_worker
        os.environ.update(env)
        try:
            return Q.forward(), sent
        finally:
            Q.run_worker = real
            for k in env:
                os.environ.pop(k)

    def test_a_teacher_forced_pass_sends_the_texts_from_the_file(self):
        res, sent = self.run_forward({"type": "teacher_forced", "texts_file": "/work/t.json", "return": {"topk": 5}})
        self.assertTrue(res["ok"])
        self.assertEqual(sent, [{"type": "teacher_forced", "texts": ["<|im_start|>user\nhi", "b"], "return": {"topk": 5}}])

    def test_texts_file_with_texts_or_chats_is_refused_before_the_worker_runs(self):
        for extra in ({"texts": ["a"]}, {"chats": [[]]}):
            res, sent = self.run_forward({"type": "teacher_forced", "texts_file": "/work/t.json", **extra})
            self.assertFalse(res["ok"])
            self.assertEqual(sent, [])
        res, sent = self.run_forward({"type": "teacher_forced", "texts_file": "/work/link.json"})
        self.assertFalse(res["ok"])
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
