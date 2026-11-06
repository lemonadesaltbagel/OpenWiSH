"""Experiment 8 forward passes on a GPU machine (this laptop or a rented pod): Qwen3.5-0.8B (text), original and/or
with an edit applied. Runs inside a venv with vllm, transformers, safetensors and math-verify.

  qwen_worker.py generate       --model DIR --req REQ.json --out OUT [--edits DIR] --practice DIR
                                (with "activations": OUT/gen_acts/, one folder per model with "both")
  qwen_worker.py teacher_forced --model DIR --req REQ.json --out OUT [--edits DIR]
  qwen_worker.py evaluate       --model DIR --set math500|gsm8k --data DIR --limit N --samples K --out OUT [--edits DIR]

--edits holds one float32 .npy per edited tensor (named like the tensor). The last stdout line is a JSON summary.
Sampling is the published evaluation protocol: chat template with thinking on, temperature 0.6, top_p 0.95,
top_k 20, presence penalty 1.5, at most 32,768 new tokens; the last \\boxed{} after </think> is graded with math-verify."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
LIMITS = dict(gen_max=512, tf_max_texts=1024, tf_max_tokens=32768, tf_total_tokens=524288, attn_max_texts=2, attn_max_tokens=2048,
              tf_max_bytes=6e9)
FULL_ATTENTION = (3, 7, 11, 15, 19, 23)


class Bad(Exception):
    """A request problem: reported to the editor, not charged."""


# ------------------------------------------------------------------ answers (the published extraction rules)
def answer_text(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    import math_verify as mv

    try:
        g, p = mv.parse("\\boxed{" + str(gold) + "}"), mv.parse("\\boxed{" + pred + "}")
        return bool(g and p and mv.verify(g, p))
    except Exception:
        return False


# ------------------------------------------------------------------ checkpoints
def checkpoint(model_dir, edits, work):
    """The model directory to load: the base, or a copy with the edited tensors replaced (bfloat16)."""
    if not edits or not os.listdir(edits):
        return model_dir, "original"
    files = sorted(f for f in os.listdir(edits) if f.endswith(".npy"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        h.update(open(os.path.join(edits, f), "rb").read())
    tag = h.hexdigest()[:16]
    out = os.path.join(work, "ckpt_" + tag)
    prune_checkpoints(work, keep=out)
    if os.path.exists(os.path.join(out, "model.safetensors")):
        os.utime(out)  # mark the checkpoint as recently used
        return out, tag
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(out, exist_ok=True)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    for f in files:
        name = f[:-4]
        if name not in sd:
            raise Bad(f"edited tensor {name} is not in the model")
        a = torch.from_numpy(np.load(os.path.join(edits, f)).astype(np.float32))
        if tuple(a.shape) != tuple(sd[name].shape):
            raise Bad(f"{name}: edited shape {tuple(a.shape)} != {tuple(sd[name].shape)}")
        sd[name] = a.to(sd[name].dtype)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(model_dir):
        if f != "model.safetensors":
            shutil.copyfile(os.path.join(model_dir, f), os.path.join(out, f))
    return out, tag


def prune_checkpoints(work, keep, n_keep=2):
    """Delete all but the `n_keep` most recently used edited checkpoints, counting `keep` as one of them.
    Each checkpoint takes 1.6 GB, and `checkpoint` rebuilds a deleted one on demand."""
    try:
        others = [os.path.join(work, d) for d in os.listdir(work) if d.startswith("ckpt_")]
    except OSError:
        return
    others = sorted((d for d in others if os.path.abspath(d) != os.path.abspath(keep)), key=os.path.getmtime, reverse=True)
    for d in others[max(n_keep - 1, 0):]:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ generation
def item_seed(key, sample):
    return int(hashlib.sha256(f"{key}:{sample}".encode()).hexdigest()[:8], 16)


def load_practice(d):
    rows = {}
    for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def run_vllm(ckpt, items, max_tokens):
    """items: list of {key, user, sample}. Returns one record per item (text, finish, tokens)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=ckpt, dtype="bfloat16", max_model_len=max_tokens + 2048, seed=0, gpu_memory_utilization=float(os.environ.get("QWEN_GPU_UTIL", "0.85")),
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in items]
    params = [SamplingParams(max_tokens=max_tokens, seed=item_seed(it["key"], it["sample"]), **SAMPLING) for it in items]
    outs = llm.generate(prompts, params)
    return [{"text": o.outputs[0].text, "finish": o.outputs[0].finish_reason, "tokens": len(o.outputs[0].token_ids)} for o in outs]


def one_model(args):
    """Subprocess: generate for one checkpoint (vLLM is cleanest one model per process)."""
    items = json.load(open(args.items))
    recs = run_vllm(args.model, items, args.max_tokens)
    json.dump(recs, open(args.out, "w"))


def grade(items, recs):
    out = []
    for it, r in zip(items, recs):
        pred = last_boxed(answer_text(r["text"])) if r["finish"] != "length" or "</think>" in r["text"] else last_boxed(answer_text(r["text"]))
        ok = is_correct(pred, it.get("gold")) if it.get("gold") is not None else None
        if r["finish"] == "length":
            fail = "cut_at_max_tokens"
        elif pred is None:
            fail = "no_answer_finished"
        elif ok is False:
            fail = "wrong_answer_finished"
        else:
            fail = None
        out.append({**{k: it[k] for k in ("key", "source", "sample")}, "gold": it.get("gold"), "pred": pred, "correct": ok,
                    "finish": r["finish"], "tokens": r["tokens"], "failure": fail if ok is not True else None, "text": r["text"]})
    return out


def summarise(recs):
    graded = [r for r in recs if r["correct"] is not None]
    n = len(recs)
    s = {"generations": n, "mean_tokens": sum(r["tokens"] for r in recs) / max(n, 1),
         "share_cut_at_max_tokens": sum(r["finish"] == "length" for r in recs) / max(n, 1),
         "failures": {k: sum(r["failure"] == k for r in recs) for k in ("cut_at_max_tokens", "no_answer_finished", "wrong_answer_finished")}}
    if graded:
        s["accuracy"] = sum(r["correct"] for r in graded) / len(graded)
        s["accuracy_se"] = math.sqrt(s["accuracy"] * (1 - s["accuracy"]) / len(graded))
        by = {}
        for r in graded:
            by.setdefault(r["source"], []).append(r["correct"])
        s["by_source"] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by.items())}
    return s


def paired(a, b):
    d = [(int(y["correct"]) - int(x["correct"])) for x, y in zip(a, b) if x["correct"] is not None]
    if not d:
        return {}
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1))
    return {"difference_edited_minus_original": m, "se": sd / math.sqrt(len(d)), "items": len(d),
            "rescues": sum(x == 1 for x in d), "losses": sum(x == -1 for x in d)}


def generate(args, req, ckpt_edit):
    acts = None if req.get("activations") in (None, False) else check_activations(req["activations"])
    which = req.get("model", "edited")
    if which not in ("original", "edited", "both"):
        raise Bad('"model" must be "original", "edited" or "both"')
    n = int(req.get("n", 1))
    max_tokens = int(req.get("max_tokens", 32768))
    if not 1 <= max_tokens <= 32768:
        raise Bad("max_tokens must be between 1 and 32768")
    practice = load_practice(args.practice)
    base_items = []
    for pid in req.get("problems", []) or []:
        if pid not in practice:
            raise Bad(f"unknown practice problem id {pid!r}")
        r = practice[pid]
        gold = r.get("answer")
        base_items.append({"key": pid, "source": r.get("source", "?"), "user": r["problem"] + "\n" + INSTRUCTION, "gold": gold})
    for i, p in enumerate(req.get("prompts", []) or []):
        base_items.append({"key": f"prompt{i}:{hashlib.sha256(p.encode()).hexdigest()[:8]}", "source": "prompt", "user": p, "gold": None})
    if not base_items:
        raise Bad("give \"problems\" (practice ids) and/or \"prompts\"")
    items = [dict(it, sample=s) for it in base_items for s in range(n)]
    cap = LIMITS["gen_max"] // (2 if which == "both" else 1)
    if len(items) > cap:
        raise Bad(f"{len(items)} generations requested; at most {cap} per model")
    models = {"original": args.model, "edited": ckpt_edit}
    names = ["original", "edited"] if which == "both" else [which]
    os.makedirs(args.out, exist_ok=True)
    json.dump(items, open(os.path.join(args.out, "items.json"), "w"))
    results = {}
    for name in names:
        out = os.path.join(args.out, f".raw_{name}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "_one", "--model", models[name], "--items", os.path.join(args.out, "items.json"),
               "--out", out, "--max-tokens", str(max_tokens)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"generation failed for the {name} model: {p.stderr[-3000:]}")
        recs = grade(items, json.load(open(out)))
        os.remove(out)
        fname = "gen.jsonl" if len(names) == 1 else f"gen_{name}.jsonl"
        with open(os.path.join(args.out, fname), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        results[name] = recs
    acts_info = {}
    for name in names if acts else []:
        sub = os.path.join(args.out, "gen_acts", *([name] if len(names) > 1 else []))
        acts_info[name] = generation_activations(models[name], items, results[name], acts, sub)
    summary = {"ok": True, "type": "generate", "model": which, **({"summary": summarise(results[names[0]])} if len(names) == 1 else
               {"original": summarise(results["original"]), "edited": summarise(results["edited"]), "paired": paired(results["original"], results["edited"])})}
    summary["files"] = sorted(f for f in os.listdir(args.out) if not f.startswith("."))
    if acts_info:
        summary["activations"] = acts_info if len(names) > 1 else acts_info[names[0]]
    return summary


# ------------------------------------------------------------------ activations of generated answers
ACT_ANSWER_TOKENS = 8192
N_LAYERS = 24


def check_activations(a):
    """The published request: {"layers": [layer indices 0..24], "mlp": true or false}."""
    if not isinstance(a, dict) or set(a) - {"layers", "mlp"}:
        raise Bad('"activations" must be {"layers": [layer indices 0..24], "mlp": true or false}')
    layers = a.get("layers")
    if not isinstance(layers, list) or not layers or not all(type(l) is int and 0 <= l <= N_LAYERS for l in layers):
        raise Bad(f'"activations": "layers" must be a non-empty list of layer indices 0..{N_LAYERS}')
    if len(set(layers)) != len(layers):
        raise Bad('"activations": "layers" repeats a layer index')
    if not isinstance(a.get("mlp", False), bool):
        raise Bad('"activations": "mlp" must be true or false')
    return {"layers": layers, "mlp": a.get("mlp", False)}


def generation_activations(ckpt, items, recs, acts, out):
    """Activations of every generated answer, as the published prompt describes them. For each requested layer l, the
    residual stream (0 = embeddings, l = after layer l) at the last prompt token and averaged over the answer's first
    8,192 tokens. With "mlp", the down_proj input of each requested layer below 24 (layer 24 has no MLP after it), taken
    the same two ways. One teacher-forced read per answer over prompt + answer, with the answer tokenized again from
    its text. Writes float16 .npy files and index.json (files, layers, and one row per answer with its grade) to `out`."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(dev).eval()
    layers = acts["layers"]
    mlp_layers = [l for l in layers if l < len(model.model.layers)] if acts["mlp"] else []
    H, I, N = model.config.hidden_size, model.config.intermediate_size, len(recs)
    arrays = {"hidden_last_prompt": np.zeros((N, len(layers), H), np.float16), "hidden_mean_answer": np.zeros((N, len(layers), H), np.float16)}
    if mlp_layers:
        arrays["mlp_last_prompt"] = np.zeros((N, len(mlp_layers), I), np.float16)
        arrays["mlp_mean_answer"] = np.zeros((N, len(mlp_layers), I), np.float16)
    captured = {}
    hooks = []
    for l in mlp_layers:
        def pre(mod, inp, l=l):
            captured[l] = inp[0][0]
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
    answer_lens = []
    with torch.no_grad():
        for i, (it, r) in enumerate(zip(items, recs)):
            prompt = tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(r["text"], add_special_tokens=False)["input_ids"][:ACT_ANSWER_TOKENS]
            P, A = len(p_ids), len(a_ids)
            answer_lens.append(A)
            res = model(torch.tensor([p_ids + a_ids], device=dev), output_hidden_states=True)
            rows = [(arrays["hidden_last_prompt"], arrays["hidden_mean_answer"], j, res.hidden_states[l][0]) for j, l in enumerate(layers)]
            rows += [(arrays["mlp_last_prompt"], arrays["mlp_mean_answer"], j, captured[l]) for j, l in enumerate(mlp_layers)]
            for last, mean, j, h in rows:
                last[i, j] = h[P - 1].float().cpu().numpy()
                if A:
                    mean[i, j] = h[P:P + A].float().mean(0).cpu().numpy()
    for h in hooks:
        h.remove()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    os.makedirs(out, exist_ok=True)
    for k, v in arrays.items():
        np.save(os.path.join(out, k + ".npy"), v)
    index = {"layers": layers, "mlp_layers": mlp_layers, "answer_tokens_averaged": answer_lens, "max_answer_tokens": ACT_ANSWER_TOKENS,
             "files": {k + ".npy": list(v.shape) for k, v in arrays.items()},
             "rows": [{"row": i, "key": r["key"], "sample": r["sample"], "correct": r["correct"], "failure": r["failure"],
                       "finish": r["finish"], "tokens": r["tokens"]} for i, r in enumerate(recs)]}
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return {"files": index["files"], "layers": layers, "mlp_layers": mlp_layers}


# ------------------------------------------------------------------ teacher-forced reads
def tf_output_bytes(n, T, cfg, hs_layers, mlp_layers, topk, logprobs):
    """Bytes of the arrays a teacher-forced pass returns. Every array pads each text to the longest one (T tokens)."""
    per_token = 8 + (4 if logprobs else 0) + 12 * topk  # input ids, token log-probs, top-k values and ids
    per_token += 2 * (cfg["hidden_size"] * len(hs_layers) + cfg["intermediate_size"] * len(mlp_layers))  # float16
    return n * T * per_token


def check_tf_size(n, T, cfg, hs_layers, mlp_layers, topk, logprobs):
    """The published limit: a pass returns at most 6 GB. Checked before the model loads or any array exists."""
    est = tf_output_bytes(n, T, cfg, hs_layers, mlp_layers, topk, logprobs)
    if est > LIMITS["tf_max_bytes"]:
        raise Bad(f"this request returns about {est / 1e9:.1f} GB, and a pass returns at most 6 GB. Arrays pad every text "
                  "to the longest text. Request fewer layers, or fewer or shorter texts")


def teacher_forced(args, req, ckpt):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ret = req.get("return", {}) or {}
    tok = AutoTokenizer.from_pretrained(ckpt)
    if "texts" in req:
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in req["texts"]]
    elif "chats" in req:
        ids = [tok.apply_chat_template(c, tokenize=True, enable_thinking=bool(req.get("enable_thinking", False))) for c in req["chats"]]
        ids = [x["input_ids"] if isinstance(x, dict) else list(x) for x in ids]
    else:
        raise Bad('give "texts" or "chats"')
    if not ids or len(ids) > LIMITS["tf_max_texts"]:
        raise Bad(f"between 1 and {LIMITS['tf_max_texts']} texts")
    if max(len(x) for x in ids) > LIMITS["tf_max_tokens"] or sum(len(x) for x in ids) > LIMITS["tf_total_tokens"]:
        raise Bad("texts too long: at most 32,768 tokens each and 524,288 in total")
    attn = ret.get("attn_layers") or []
    if attn:
        if any(l not in FULL_ATTENTION for l in attn):
            raise Bad(f"attn_layers must be full-attention layers {FULL_ATTENTION}")
        if len(ids) > LIMITS["attn_max_texts"] or max(len(x) for x in ids) > LIMITS["attn_max_tokens"]:
            raise Bad("with attn_layers: at most 2 texts of at most 2,048 tokens")
    hs_layers = ret.get("hidden_states") or []
    mlp_layers = ret.get("mlp_act") or []
    topk = int(ret.get("topk", 0) or 0)
    if topk > 50:
        raise Bad("topk is at most 50")
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    check_tf_size(len(ids), max(len(x) for x in ids), cfg.get("text_config", cfg), hs_layers, mlp_layers, topk,
                  bool(ret.get("token_logprobs")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, attn_implementation="eager" if attn else None).to(dev).eval()
    layers = model.model.layers
    captured = {}
    hooks = []
    for l in mlp_layers:
        def pre(mod, inp, l=l):
            captured[l] = inp[0][0].float().cpu().numpy()
        hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(pre))
    T = max(len(x) for x in ids)
    n = len(ids)
    outs = {"input_ids": np.zeros((n, T), np.int64)}
    if ret.get("token_logprobs"):
        outs["token_logprobs"] = np.zeros((n, max(T - 1, 1)), np.float32)
    if topk:
        outs["topk_logprobs"] = np.zeros((n, T, topk), np.float32)
        outs["topk_ids"] = np.zeros((n, T, topk), np.int64)
    for l in hs_layers:
        outs[f"hidden_{l}"] = np.zeros((n, T, model.config.hidden_size), np.float16)
    for l in mlp_layers:
        outs[f"mlp_act_{l}"] = np.zeros((n, T, model.config.intermediate_size), np.float16)
    attn_out = {}
    with torch.no_grad():
        for i, x in enumerate(ids):
            t = torch.tensor([x], device=dev)
            r = model(t, output_hidden_states=bool(hs_layers), output_attentions=bool(attn))
            lp = torch.log_softmax(r.logits[0].float(), -1)
            L = len(x)
            outs["input_ids"][i, :L] = x
            if "token_logprobs" in outs and L > 1:
                outs["token_logprobs"][i, :L - 1] = lp[:-1].gather(1, t[0, 1:, None])[:, 0].cpu().numpy()
            if topk:
                v, k = lp.topk(topk, -1)
                outs["topk_logprobs"][i, :L] = v.cpu().numpy()
                outs["topk_ids"][i, :L] = k.cpu().numpy()
            for l in hs_layers:
                outs[f"hidden_{l}"][i, :L] = r.hidden_states[l][0].float().cpu().numpy()
            for l in mlp_layers:
                outs[f"mlp_act_{l}"][i, :L] = captured[l]
            if attn:
                for l in attn:
                    a = r.attentions[l]
                    if a is None:
                        raise Bad(f"layer {l} returned no attention probabilities")
                    attn_out.setdefault(l, []).append(a[0].float().cpu().numpy().astype(np.float16))
    for h in hooks:
        h.remove()
    os.makedirs(args.out, exist_ok=True)
    files = {}
    for k, v in outs.items():
        np.save(os.path.join(args.out, k + ".npy"), v)
        files[k] = list(v.shape)
    for l, lst in attn_out.items():
        for i, a in enumerate(lst):
            np.save(os.path.join(args.out, f"attn_{l}_text{i}.npy"), a)
            files[f"attn_{l}_text{i}"] = list(a.shape)
    meta = {"lengths": [len(x) for x in ids], "files": files, "padding": "right, with zeros"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    return {"ok": True, "type": "teacher_forced", **meta}


# ------------------------------------------------------------------ hidden evaluation (the scorer)
def evaluate(args, ckpt):
    rows = []
    if args.set == "math500":
        for line in open(os.path.join(args.data, "math500.jsonl")):
            r = json.loads(line)
            rows.append({"key": r["unique_id"], "source": "math500", "user": r["problem"] + "\n" + INSTRUCTION, "gold": r["answer"]})
    elif args.set == "gsm8k":
        for i, line in enumerate(open(os.path.join(args.data, "gsm8k_test.jsonl"))):
            r = json.loads(line)
            rows.append({"key": f"gsm8k_test_{i}", "source": "gsm8k", "user": r["question"] + "\n" + INSTRUCTION, "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    rows = rows[: args.limit]
    items = [dict(r, sample=s) for r in rows for s in range(args.samples)]
    os.makedirs(args.out, exist_ok=True)
    json.dump(items, open(os.path.join(args.out, "items.json"), "w"))
    raw = os.path.join(args.out, ".raw.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "_one", "--model", ckpt, "--items", os.path.join(args.out, "items.json"),
                        "--out", raw, "--max-tokens", str(args.max_tokens)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    recs = grade(items, json.load(open(raw)))
    with open(os.path.join(args.out, f"{args.set}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "text"} | {"text_head": r["text"][:2000]}) + "\n")
    s = summarise(recs)
    return {"ok": True, "score": s.get("accuracy", 0.0), "set": args.set, "problems": len(rows), "samples": args.samples, **s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "teacher_forced", "evaluate", "_one"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--edits")
    ap.add_argument("--req")
    ap.add_argument("--out", required=True)
    ap.add_argument("--practice")
    ap.add_argument("--items")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--set", default="math500")
    ap.add_argument("--data")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--work", default=os.environ.get("QWEN_WORK", "/tmp/qwen_work"))
    a = ap.parse_args()
    if a.cmd == "_one":
        return one_model(a)
    try:
        os.makedirs(a.work, exist_ok=True)
        ckpt, _ = checkpoint(a.model, a.edits, a.work)
        if a.cmd == "evaluate":
            out = evaluate(a, ckpt)
        else:
            req = json.load(open(a.req))
            if a.cmd == "generate":
                out = generate(a, req, ckpt)
            else:
                which = req.get("model", "edited")
                out = teacher_forced(a, req, a.model if which == "original" else ckpt)
    except Bad as e:
        out = {"ok": False, "error": str(e)}
    except json.JSONDecodeError as e:
        out = {"ok": False, "error": f"the request is not valid JSON ({e})"}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
