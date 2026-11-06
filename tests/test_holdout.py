"""Hold-out scorer (python/adapters/holdout_eval.py): the MMLU-Pro prompt and answer extraction without a GPU, and the
graders against the authors' published first-sample texts. The IFEval checkers need nltk, langdetect, immutabledict and
absl-py, so run the file with the qwen venv: assets/venvs/qwen/bin/python -I tests/test_holdout.py
The published-text tests skip without `holdout_eval.py prepare` output and assets/lm_understand-logs."""
import glob
import gzip
import importlib.util
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("holdout_eval", os.path.join(ROOT, "python/adapters/holdout_eval.py"))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

DATA = os.path.join(ROOT, "assets", "exp8", "holdout")
AUTHORS = os.path.join(ROOT, "assets", "lm_understand-logs", "exp8_math", "eval")
AUTHORS = AUTHORS if os.path.isdir(AUTHORS) else None
PREPARED = all(os.path.exists(os.path.join(DATA, f"{s}.jsonl")) for s in H.SETS)
try:
    import immutabledict, langdetect, nltk  # noqa: F401,E401
    CHECKER_DEPS = True
except ImportError:
    CHECKER_DEPS = False


class Extraction(unittest.TestCase):
    def test_official_chain(self):
        self.assertEqual(H.mc_extract_official("so the answer is (C). or maybe the answer is D"), "C")   # first match
        self.assertEqual(H.mc_extract_official("blah\nAnswer: F\nmore"), "F")
        self.assertEqual(H.mc_extract_official("I pick B over A here"), "A")                               # last A-J letter
        self.assertIsNone(H.mc_extract_official("no letter here"))

    def test_reasoning_is_cut_at_the_last_think_end(self):
        self.assertEqual(H.mc_extract("the answer is (A)\n</think>\nthe answer is (B)"), "B")
        self.assertEqual(H.mc_extract("still thinking ... the answer is (G)"), "G")      # no </think>: the whole text

    def test_ifeval_answer_is_after_think_end(self):
        self.assertEqual(H.ifeval_response("reasoning</think>\n\n Hello "), "Hello")
        self.assertEqual(H.ifeval_response("cut inside the reasoning"), "")

    def test_mmlupro_prompt(self):
        p = H.mc_prompt("law", "Q?", ["x", "y"])
        self.assertTrue(p.startswith("The following are multiple choice questions (with answers) about law. Think step by step"))
        self.assertTrue(p.endswith('letter choice.\n\n\n\nQuestion:\nQ?\nOptions:\nA. x\nB. y\nAnswer: Let\'s think step by step.'))

    def test_ids_match_the_authors_counts(self):
        ids = json.load(open(H.IDS))
        self.assertEqual([len(ids[s]) for s in H.SETS], [1000, 198, 541])
        self.assertEqual(ids["mmlupro1k"][0], "mmlupro_71")


class Dispatch(unittest.TestCase):
    def test_qwen_pass_sends_hold_out_sets_to_the_hold_out_scorer(self):
        spec = importlib.util.spec_from_file_location("qwen_pass", os.path.join(ROOT, "python/adapters/qwen_pass.py"))
        Q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Q)
        calls = []
        Q.run_worker = lambda args, stage, out, worker=Q.WORKER: calls.append((os.path.basename(worker), args)) or {"ok": True}

        class A:
            sets, limit, samples = "mmlupro1k,math500", None, None

        saved = {k: os.environ.pop(k, None) for k in ("OPENWISH_RUN", "OPENWISH_FINAL_NET")}
        try:
            self.assertTrue(Q.evaluate(A())["ok"])
        finally:
            os.environ.update({k: v for k, v in saved.items() if v is not None})
        (w1, a1), (w2, a2) = calls
        self.assertEqual(w1, "holdout_eval.py")
        self.assertIn("{holdout}", a1)
        self.assertNotIn("--limit", a1, "every item by default")
        self.assertNotIn("--samples", a1, "the authors' sample counts by default")
        self.assertEqual(w2, "qwen_worker.py")
        self.assertEqual(a2[a2.index("--limit") + 1], "500")
        self.assertEqual(a2[a2.index("--samples") + 1], "1")


@unittest.skipUnless(PREPARED and AUTHORS, "needs `holdout_eval.py prepare` output and the authors' logs")
class PublishedTexts(unittest.TestCase):
    def published(self, ck, s, n):
        rows = {r["id"]: r for r in H.load_rows(DATA, s)}
        out = []
        for d in sorted(glob.glob(os.path.join(AUTHORS, ck, "parts", f"{s}_s*"))):
            per = {p["id"]: p["correct"][0] for p in json.load(open(os.path.join(d, f"{s}_per_problem.json")))}
            for line in gzip.open(os.path.join(d, f"{s}_first_samples.jsonl.gz"), "rt", encoding="utf-8"):
                x = json.loads(line)
                out.append((rows[x["id"]], x, per[x["id"]]))
                if len(out) >= n:
                    return out
        return out

    def test_mmlupro_matches_authors(self):
        for ck in ("unedited", "p50c"):
            for row, x, theirs in self.published(ck, "mmlupro1k", 40):
                self.assertEqual(x["answer"], row["answer"], x["id"])
                self.assertEqual(int(H.grade_one(row, x["text"], "stop")["correct"]), theirs, (ck, x["id"]))

    @unittest.skipUnless(CHECKER_DEPS, "needs nltk, langdetect, immutabledict, absl-py (the qwen venv)")
    def test_ifeval_matches_authors(self):
        for ck in ("unedited", "p10c"):
            for row, x, theirs in self.published(ck, "ifeval", 40):
                g = H.grade_one(row, x["text"], "stop", os.path.join(DATA, "nltk_data"))
                self.assertEqual(int(g["correct"]), theirs, (ck, x["id"]))

    def test_gpqa_items(self):
        rows = H.load_rows(DATA, "gpqa")
        self.assertEqual(len(rows), 198)
        for r in rows:
            self.assertIn(r["answer"], "ABCD")
            self.assertIn("\nD. ", r["prompt"])
        self.assertEqual([r["id"] for r in rows], json.load(open(H.IDS))["gpqa"])


if __name__ == "__main__":
    unittest.main()
