"""Experiment 8 hold-out scorer: MMLU-Pro (the authors' fixed 1,000 questions), GPQA Diamond and IFEval. The protocol
follows the authors' evaluation (assets/lm_understand-logs/exp8_math/code/evaluate.py, protocol think35). The authors
never published the graders their script imports (weightenv.lmo.tasks.mcqa and ifevalg), so this file implements them
again. See docs/holdout.md.

  holdout_eval.py prepare  [--data assets/exp8/holdout]
  holdout_eval.py evaluate --model DIR --set mmlupro1k|gpqa|ifeval --data DIR --out OUT [--edits DIR] [--limit N]
                           [--samples K] [--max-tokens 32768] [--work DIR]
  holdout_eval.py regrade  --authors assets/lm_understand-logs/exp8_math/eval --data DIR [--out report.json]

prepare downloads the pinned dataset files, checks their sha256, selects the authors' item ids in the authors' order
(holdout_ids.json), and writes <set>.jsonl, manifest.json (with the sha256 of every file) and the NLTK punkt_tab data
for the IFEval checkers. evaluate generates through qwen_worker.py `_one` (chat template, thinking on, temperature 0.6,
top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens, per-item seeds item_seed(id, sample)), grades
every generation, writes OUT/<set>.jsonl and prints one JSON summary line. regrade runs on the CPU: the command grades
the authors' published first-sample texts again and compares the grades with their correct[0]."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_worker as W  # noqa: E402 (stdlib only at import. Reused: checkpoint() and the `_one` vLLM subprocess)

ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATA = os.path.join(ROOT, "assets", "exp8", "holdout")
IDS = os.path.join(HERE, "holdout_ids.json")
SETS = ("mmlupro1k", "gpqa", "ifeval")
SAMPLES = {"mmlupro1k": 1, "gpqa": 4, "ifeval": 2}           # the authors' SAMPLES (evaluate.py)
CHOICES = "ABCDEFGHIJKLMNOP"                                  # MMLU-Pro evaluate_from_local.py `choices`

# Pinned sources (revision = the HF dataset commit). sha256 of the downloaded file (checked).
SOURCES = {
    "mmlupro1k": {"repo": "TIGER-Lab/MMLU-Pro", "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                  "file": "data/test-00000-of-00001.parquet",
                  "sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"},
    # The gated Idavidrein/gpqa (the likely source of the authors' "GPQA Diamond") and the ungated mirror Wanfq/gpqa hold
    # the same gpqa_diamond.csv: the same size (1,373,492 bytes) and git blob id 7589e3e467d69a1dceb126a60c4108d6d4f1d166
    "gpqa": {"repo": "Wanfq/gpqa", "revision": "03ff68d77f7a4103ae2bf72fb099d358f6308b65",
             "file": "gpqa_diamond.csv", "sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
             "git_blob": "7589e3e467d69a1dceb126a60c4108d6d4f1d166",
             "same_as": "Idavidrein/gpqa@83022cefff930aea54f654c0b282e74b9eeda5c6:gpqa_diamond.csv"},
    "ifeval": {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
               "file": "ifeval_input_data.jsonl",
               "sha256": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"},
}

# The MMLU-Pro official prompt (TIGER-AI-Lab/MMLU-Pro@f418b116 cot_prompt_lib/initial_prompt.txt, sha256 d7337806...)
# ends in three newlines, and generate_cot_prompt adds a fourth. Zero-shot (k = 0), since the published texts never
# mention worked examples. Then format_cot_example(curr, including_answer=False).
MC_HEADER = ('The following are multiple choice questions (with answers) about {$}. Think step by step and then finish '
             'your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n')


def mc_prompt(subject, question, options):
    p = MC_HEADER.replace("{$}", subject) + "\n"
    p += "Question:\n" + question + "\n" + "Options:\n"
    for i, opt in enumerate(options):
        p += "{}. {}\n".format(CHOICES[i], opt)
    return p + "Answer: Let's think step by step."


# ------------------------------------------------------------------ graders
def mc_extract_official(text):
    """MMLU-Pro evaluate_from_local.py extract_answer -> extract_again -> extract_final (verbatim regexes)."""
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def mc_extract(text):
    """The authors' mcqa.extract, as we read the published grades: the official extraction on the text after the last
    </think>, or on the whole text without one (an answer cut inside the reasoning). No answer counts as wrong, with
    no random guess."""
    return mc_extract_official(text.split("</think>")[-1] if "</think>" in text else text)


_IF = []


def ifeval_registry(nltk_dir=None):
    """The official checkers (google-research instruction_following_eval, vendored unchanged next to this file)."""
    if not _IF:
        import nltk

        d = nltk_dir or os.path.join(DEFAULT_DATA, "nltk_data")
        if os.path.isdir(d) and d not in nltk.data.path:
            nltk.data.path.insert(0, d)
        import langdetect

        langdetect.DetectorFactory.seed = 0      # deterministic language detection (the checkers set no seed)
        from instruction_following_eval import instructions_registry

        _IF.append(instructions_registry)
    return _IF[0]


def ifeval_response(text, thinking=True):
    """The answer after the last </think>. With thinking on and no </think>, the answer is empty (the authors' rule)."""
    return text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking else text.strip())


def ifeval_grade(row, text, thinking=True, nltk_dir=None):
    """(all followed, followed per instruction), strict and prompt-level, as the authors' grade_other, line for line."""
    IR = ifeval_registry(nltk_dir)
    resp = ifeval_response(text, thinking)
    follow = []
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        ins = IR.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=row["prompt"])
        follow.append(bool(resp) and bool(ins.check_following(resp)))
    return all(follow), follow


def grade_one(row, text, finish, nltk_dir=None):
    """One generation -> {pred, correct, failure, ...}."""
    if row["kind"] == "mc":
        pred = mc_extract(text)
        ok = pred == row["answer"]
        extra = {"pred": pred, "gold": row["answer"]}
        noans = pred is None
    else:
        ok, follow = ifeval_grade(row, text, True, nltk_dir)
        extra = {"inst_ok": sum(follow), "inst_total": len(follow), "followed": follow}
        noans = not ifeval_response(text)
    if ok:
        fail = None
    elif finish == "length":
        fail = "cut_at_max_tokens"
    elif noans:
        fail = "no_answer_finished"
    else:
        fail = "wrong_answer_finished"
    return {"correct": bool(ok), "failure": fail, **extra}


# ------------------------------------------------------------------ prepare
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gpqa_shuffle_seed(record_id):
    """GPQA option order: a per-question shuffle seeded by the record id (the authors' seed is not published)."""
    return int(hashlib.sha256(f"gpqa-shuffle:{record_id}".encode()).hexdigest()[:16], 16)


def build_mmlupro(path, ids):
    import pyarrow.parquet as pq

    t = pq.read_table(path).to_pylist()
    by = {int(r["question_id"]): r for r in t}
    out = []
    for i in ids:
        r = by[int(i.split("_", 1)[1])]
        opts = [o for o in r["options"] if o != "N/A"]          # evaluate_from_local.preprocess
        assert CHOICES[r["answer_index"]] == r["answer"] and opts[r["answer_index"]] == r["options"][r["answer_index"]]
        out.append({"id": i, "kind": "mc", "prompt": mc_prompt(r["category"], r["question"], opts), "answer": r["answer"],
                    "category": r["category"], "question_id": int(r["question_id"]), "n_options": len(opts), "src": r["src"]})
    return out, len(t)


def build_gpqa(path, ids):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {r["Record ID"]: r for r in rows}
    out = []
    for i in ids:
        r = by[i.split("_", 1)[1]]
        opts = [r["Correct Answer"].strip()] + [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]
        order = list(range(4))
        random.Random(gpqa_shuffle_seed(r["Record ID"])).shuffle(order)
        shuffled = [opts[k] for k in order]
        gold = CHOICES[order.index(0)]
        subject = r["High-level domain"].strip().lower()
        out.append({"id": i, "kind": "mc", "prompt": mc_prompt(subject, r["Question"].strip(), shuffled), "answer": gold,
                    "category": subject, "subdomain": r["Subdomain"].strip(), "n_options": 4})
    return out, len(rows)


def build_ifeval(path, ids):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = {f"ifeval_{r['key']}": r for r in rows}
    out = [{"id": i, "kind": "ifeval", "prompt": by[i]["prompt"], "instruction_id_list": by[i]["instruction_id_list"],
            "kwargs": by[i]["kwargs"]} for i in ids]
    return out, len(rows)


def prepare(a):
    from huggingface_hub import hf_hub_download

    d = os.path.abspath(a.data)
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    os.chmod(d, 0o755)           # NLTK refuses data in group/world-writable directories
    ids = json.load(open(IDS))
    builders = {"mmlupro1k": build_mmlupro, "gpqa": build_gpqa, "ifeval": build_ifeval}
    man = {"ids_file": "python/adapters/holdout_ids.json", "ids_sha256": sha256(IDS), "sets": {}}
    for s in SETS:
        src = SOURCES[s]
        raw = hf_hub_download(src["repo"], src["file"], repo_type="dataset", revision=src["revision"],
                              local_dir=os.path.join(d, "raw", src["repo"].replace("/", "__")))
        h = sha256(raw)
        if src["sha256"] and h != src["sha256"]:
            raise SystemExit(f"{s}: {raw} sha256 {h} != pinned {src['sha256']}")
        if src.get("git_blob") and git_blob(raw) != src["git_blob"]:
            raise SystemExit(f"{s}: {raw} git blob {git_blob(raw)} != {src['git_blob']}")
        rows, n_src = builders[s](raw, ids[s])
        assert len(rows) == len(set(r["id"] for r in rows)) == len(ids[s])
        p = os.path.join(d, f"{s}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["sets"][s] = {**{k: v for k, v in src.items() if k != "sha256"}, "raw_sha256": h, "raw_rows": n_src,
                          "items": len(rows), "samples": SAMPLES[s], "file": f"{s}.jsonl", "sha256": sha256(p)}
        print(s, len(rows), "of", n_src, "raw", h, "prepared", man["sets"][s]["sha256"], flush=True)
    import nltk

    nd = os.path.join(d, "nltk_data")
    os.makedirs(nd, exist_ok=True)
    os.chmod(nd, 0o755)
    if not nltk.download("punkt_tab", download_dir=nd, quiet=True):
        raise SystemExit("the NLTK punkt_tab download failed")
    z = os.path.join(nd, "tokenizers", "punkt_tab.zip")
    man["nltk"] = {"package": "punkt_tab", "nltk": nltk.__version__, "zip_sha256": sha256(z) if os.path.exists(z) else None,
                   "english_sha256": {f: sha256(os.path.join(nd, "tokenizers", "punkt_tab", "english", f))
                                      for f in sorted(os.listdir(os.path.join(nd, "tokenizers", "punkt_tab", "english")))}}
    man["checkers"] = {f: sha256(os.path.join(HERE, "instruction_following_eval", f))
                       for f in ("instructions.py", "instructions_registry.py", "instructions_util.py")}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    return {"ok": True, "data": d, **{s: man["sets"][s]["sha256"] for s in SETS}}


# ------------------------------------------------------------------ evaluate
def load_rows(data, name):
    return [json.loads(l) for l in open(os.path.join(data, f"{name}.jsonl"), encoding="utf-8")]


def summarise(rows, recs):
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / max(n, 1)
    per = {}
    for r in recs:
        per.setdefault(r["key"], []).append(r["correct"])
    means = [sum(v) / len(v) for v in per.values()]
    m = sum(means) / max(len(means), 1)
    var = sum((x - m) ** 2 for x in means) / max(len(means) - 1, 1)
    s = {"generations": n, "accuracy": acc,
         "accuracy_se": math.sqrt(acc * (1 - acc) / max(n, 1)),                # qwen_worker's (binomial over generations)
         "accuracy_se_problems": math.sqrt(var / max(len(means), 1)),          # clustered by problem (samples correlate)
         "mean_tokens": sum(r["tokens"] for r in recs) / max(n, 1),
         "share_cut_at_max_tokens": sum(r["finish"] == "length" for r in recs) / max(n, 1),
         "failures": {k: sum(r["failure"] == k for r in recs) for k in ("cut_at_max_tokens", "no_answer_finished", "wrong_answer_finished")}}
    if any("inst_total" in r for r in recs):
        tot = sum(r.get("inst_total", 0) for r in recs)
        s["instruction_level_accuracy"] = sum(r.get("inst_ok", 0) for r in recs) / max(tot, 1)
    by = {}
    for r in recs:
        if r.get("category"):
            by.setdefault(r["category"], []).append(r["correct"])
    if by:
        s["by_category"] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by.items())}
    return s


def evaluate(a):
    if a.set not in SETS:
        raise W.Bad(f"--set must be one of {SETS}")
    data = os.path.abspath(a.data)
    ckpt, tag = W.checkpoint(a.model, a.edits, a.work)
    rows = load_rows(data, a.set)
    if a.limit is not None:
        rows = rows[: a.limit]
    k = a.samples or SAMPLES[a.set]
    items = [{"key": r["id"], "source": a.set, "sample": s, "user": r["prompt"]} for r in rows for s in range(k)]
    os.makedirs(a.out, exist_ok=True)
    ip = os.path.join(a.out, "items.json")
    json.dump(items, open(ip, "w"))
    raw = os.path.join(a.out, ".raw.json")
    p = subprocess.run([sys.executable, os.path.join(HERE, "qwen_worker.py"), "_one", "--model", ckpt, "--items", ip,
                        "--out", raw, "--max-tokens", str(a.max_tokens)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    gens = json.load(open(raw))
    bad = sum(g["finish"] not in ("stop", "length") for g in gens)
    if bad:      # the authors' guard: an engine stopped from outside returns partial answers
        raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens. Nothing saved")
    by = {r["id"]: r for r in rows}
    recs = []
    for it, g in zip(items, gens):
        r = by[it["key"]]
        gr = grade_one(r, g["text"], g["finish"], os.path.join(data, "nltk_data"))
        recs.append({"key": it["key"], "source": a.set, "sample": it["sample"], "category": r.get("category"), **gr,
                     "finish": g["finish"], "tokens": g["tokens"], "text": g["text"]})
    with open(os.path.join(a.out, f"{a.set}.jsonl"), "w") as f:
        for r in recs:
            t = r["text"]
            f.write(json.dumps({k: v for k, v in r.items() if k != "text"}
                               | {"text_head": t[:2000], "answer_head": t.split("</think>")[-1][:4000] if "</think>" in t else None}) + "\n")
    os.remove(raw)
    s = summarise(rows, recs)
    return {"ok": True, "score": s["accuracy"], "set": a.set, "problems": len(rows), "samples": k, "checkpoint": tag,
            "data_sha256": sha256(os.path.join(data, f"{a.set}.jsonl")), **s}


# ------------------------------------------------------------------ regrade (validation against the published texts)
def regrade(a):
    import glob

    data = os.path.abspath(a.data)
    rows = {s: {r["id"]: r for r in load_rows(data, s)} for s in SETS}
    report = {"checkpoints": {}, "disagreements": []}
    for ck in sorted(os.listdir(a.authors)):
        cd = os.path.join(a.authors, ck, "parts")
        if not os.path.isdir(cd):
            continue
        res = {}
        for s in ("mmlupro1k", "ifeval"):
            n = agree = auth_ok = our_ok = gold_mismatch = 0
            for d in sorted(glob.glob(os.path.join(cd, f"{s}_s*"))):
                per = {p["id"]: p for p in json.load(open(os.path.join(d, f"{s}_per_problem.json")))}
                for line in gzip.open(os.path.join(d, f"{s}_first_samples.jsonl.gz"), "rt", encoding="utf-8"):
                    x = json.loads(line)
                    r = rows[s][x["id"]]
                    if s == "mmlupro1k" and x.get("answer") != r["answer"]:
                        gold_mismatch += 1
                    g = grade_one(r, x["text"], "length" if "</think>" not in x["text"] else "stop",
                                  os.path.join(data, "nltk_data"))
                    theirs = int(per[x["id"]]["correct"][0])
                    n += 1
                    agree += int(g["correct"]) == theirs
                    auth_ok += theirs
                    our_ok += int(g["correct"])
                    if int(g["correct"]) != theirs:
                        report["disagreements"].append({"checkpoint": ck, "set": s, "id": x["id"], "authors": theirs,
                                                        "ours": int(g["correct"]), **{k: v for k, v in g.items() if k != "correct"},
                                                        "has_think_end": "</think>" in x["text"],
                                                        "answer_tail": ifeval_response(x["text"])[-600:]})
            res[s] = {"n": n, "agree": agree, "agreement": agree / max(n, 1), "authors_correct": auth_ok,
                      "ours_correct": our_ok, **({"gold_mismatch": gold_mismatch} if s == "mmlupro1k" else {})}
        # every published per-problem id (all three sets) must be one of the prepared items
        for s in SETS:
            ids = set()
            for f in glob.glob(os.path.join(cd, f"{s}_s*", f"{s}_per_problem.json")):
                ids |= {p["id"] for p in json.load(open(f))}
            res.setdefault(s, {})["ids_covered"] = f"{len(ids & set(rows[s]))}/{len(ids)} (prepared {len(rows[s])})"
        report["checkpoints"][ck] = res
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return {"ok": True, "checkpoints": report["checkpoints"], "disagreements": len(report["disagreements"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "evaluate", "regrade"])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model")
    ap.add_argument("--edits")
    ap.add_argument("--set", default="mmlupro1k")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (a smoke run)")
    ap.add_argument("--samples", type=int, default=None, help="samples per problem (default: the authors' 1 / 4 / 2)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--work", default=os.environ.get("QWEN_WORK", "/tmp/qwen_work"))
    ap.add_argument("--authors", help="regrade: the authors' eval directory (unedited/, p10c/, ...)")
    a = ap.parse_args()
    try:
        if a.cmd == "prepare":
            out = prepare(a)
        elif a.cmd == "regrade":
            out = regrade(a)
        else:
            if not (a.model and a.out):
                raise W.Bad("evaluate needs --model and --out")
            os.makedirs(a.work, exist_ok=True)
            out = evaluate(a)
    except W.Bad as e:
        out = {"ok": False, "error": str(e)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
