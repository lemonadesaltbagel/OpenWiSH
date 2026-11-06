"""Experiment 8 forward-pass adapter (harness side). It finds the edited tensors from the run's ledger, sends them
with the request to a GPU machine, runs python/adapters/qwen_worker.py there and brings the results back.

The GPU machine is this computer, unless assets/exp8/remote.json (or $OPENWISH_QWEN_REMOTE) names a remote one:
  {"host": "root@1.2.3.4", "port": 12345, "key": "~/.ssh/id_rsa", "root": "/root/openwish", "python": "/root/qwen/bin/python"}
An optional "gpu_util" sets the vLLM GPU memory share on the remote machine (QWEN_GPU_UTIL), for a GPU shared with
another server. Run `qwen_pass.py --setup-remote` once to copy the model, practice and evaluation sets and the worker there.

forward pass (harness):  OPENWISH_REQ, OPENWISH_NET, OPENWISH_OUT, OPENWISH_RUN_DIR, OPENWISH_WORK, OPENWISH_PASS
scorer:                  qwen_pass.py --evaluate --sets math500 --limit 500 [--samples 1]   (OPENWISH_RUN, OPENWISH_FINAL_NET)
Prints one JSON line."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "assets", "exp8")
WORKER = os.path.join(ROOT, "python", "adapters", "qwen_worker.py")
LOCAL_PY = os.path.join(ROOT, "assets", "venvs", "qwen", "bin", "python")


def remote_cfg():
    p = os.environ.get("OPENWISH_QWEN_REMOTE") or os.path.join(A, "remote.json")
    if os.path.exists(p):
        c = json.load(open(p))
        c["key"] = os.path.expanduser(c.get("key", "~/.ssh/id_rsa"))
        return c
    return None


def ssh_base(c):
    return ["ssh", "-i", c["key"], "-p", str(c["port"]), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", c["host"]]


def rsync(c, src, dst):
    e = f"ssh -i {shlex.quote(c['key'])} -p {c['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    subprocess.run(["rsync", "-a", "--partial", "-e", e, src, dst], check=True, capture_output=True, text=True)


def edited_names(run_dir):
    names = set()
    lp = os.path.join(run_dir, "edits", "ledger.jsonl")
    if os.path.exists(lp):
        for line in open(lp):
            v = json.loads(line)
            if v.get("kind") == "edit":
                names.add(v["op"]["tensor"])
    return sorted(names)


def stage_edits(net_dir, names, dst):
    os.makedirs(dst, exist_ok=True)
    for n in names:
        shutil.copyfile(os.path.join(net_dir, n.replace("/", "__") + ".npy"), os.path.join(dst, n + ".npy"))
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(open(os.path.join(dst, n + ".npy"), "rb").read())
    return h.hexdigest()[:16]


def run_worker(args_list, stage, out_dir):
    """Run the worker with `args_list` (paths as placeholders {model} {edits} {out} {practice} {data} {req}); returns its JSON."""
    c = remote_cfg()
    if c is None:
        local = {"model": os.path.join(A, "qwen35-text"), "practice": os.path.join(A, "practice"), "data": os.path.join(A, "eval"),
                 "edits": stage["edits"], "out": out_dir, "req": stage.get("req", "")}
        cmd = [LOCAL_PY, WORKER] + [a.format(**local) for a in args_list] + ["--work", os.path.join(A, "work")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        tail = p.stderr[-3000:]
    else:
        r = c["root"]
        tag = os.path.basename(out_dir.rstrip("/")) + "_" + stage["tag"]
        rem = {"model": f"{r}/model", "practice": f"{r}/practice", "data": f"{r}/eval", "edits": f"{r}/edits/{stage['tag']}",
               "out": f"{r}/out/{tag}", "req": f"{r}/req/{tag}.json"}
        subprocess.run(ssh_base(c) + [f"mkdir -p {r}/edits/{stage['tag']} {r}/req {r}/out && rm -rf {r}/out/{tag}"], check=True, capture_output=True)
        rsync(c, stage["edits"] + "/", f"{c['host']}:{rem['edits']}/")
        if stage.get("req"):
            rsync(c, stage["req"], f"{c['host']}:{rem['req']}")
        rsync(c, WORKER, f"{c['host']}:{r}/qwen_worker.py")
        argv = [c["python"], f"{r}/qwen_worker.py"] + [a.format(**rem) for a in args_list] + ["--work", f"{r}/work"]
        # the remote venv's bin must be on PATH: vLLM's kernel builds call tools installed there (ninja)
        env = f"export PATH={shlex.quote(os.path.dirname(c['python']))}:$PATH VLLM_USE_FLASHINFER_SAMPLER=0; "
        if c.get("gpu_util"):
            env += f"export QWEN_GPU_UTIL={float(c['gpu_util'])}; "
        p = subprocess.run(ssh_base(c) + [env + " ".join(shlex.quote(x) for x in argv)], capture_output=True, text=True)
        tail = p.stderr[-3000:]
        os.makedirs(out_dir, exist_ok=True)
        try:
            rsync(c, f"{c['host']}:{rem['out']}/", out_dir + "/")
            # the outputs now sit in the run folder, so free the remote disk (teacher-forced arrays reach several GB)
            subprocess.run(ssh_base(c) + [f"rm -rf {rem['out']} {rem['req']}"], capture_output=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        return {"ok": False, "error": f"the GPU worker failed (exit {p.returncode}): {tail}"}
    return json.loads(lines[-1])


def forward():
    req_path = os.environ["OPENWISH_REQ"]
    try:
        req = json.load(open(req_path))
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"the ```forward block must hold JSON ({e})"}
    kind = req.get("type")
    if kind not in ("generate", "teacher_forced"):
        return {"ok": False, "error": 'the request needs "type": "generate" or "teacher_forced"'}
    out = os.environ["OPENWISH_OUT"]
    with tempfile.TemporaryDirectory() as tmp:
        stage = {"edits": os.path.join(tmp, "edits"), "req": req_path}
        stage["tag"] = stage_edits(os.environ["OPENWISH_NET"], edited_names(os.environ["OPENWISH_RUN_DIR"]), stage["edits"])
        args = [kind, "--model", "{model}", "--edits", "{edits}", "--req", "{req}", "--out", "{out}", "--practice", "{practice}"]
        res = run_worker(args, stage, out)
    if res.get("ok") and kind == "generate":
        # also where the published prompt says they are: /work/gen_<k>.jsonl (gen_<k>_original/_edited with "both")
        k = os.environ.get("OPENWISH_PASS", "0")
        work = os.environ.get("OPENWISH_WORK")
        for f in ("gen.jsonl", "gen_original.jsonl", "gen_edited.jsonl"):
            src = os.path.join(out, f)
            if work and os.path.exists(src):
                shutil.copyfile(src, os.path.join(work, f"gen_{k}" + f[3:]))
    return res


def evaluate(a):
    run = os.environ.get("OPENWISH_RUN", "")
    net = os.environ.get("OPENWISH_FINAL_NET", "")
    names = edited_names(run) if run else []
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        stage = {"edits": os.path.join(tmp, "edits")}
        os.makedirs(stage["edits"], exist_ok=True)  # an unedited run sends an empty folder, and rsync needs the folder
        stage["tag"] = stage_edits(net, names, stage["edits"]) if names else "base"
        for s in a.sets.split(","):
            out = os.path.join(run or tmp, "eval", f"qwen_{s}")
            args = ["evaluate", "--model", "{model}", "--edits", "{edits}", "--set", s, "--data", "{data}", "--limit", str(a.limit),
                    "--samples", str(a.samples), "--out", "{out}"]
            results[s] = run_worker(args, stage, out)
    ok = all(r.get("ok") for r in results.values())
    first = next(iter(results.values()))
    return {"ok": ok, "score": first.get("score", 0.0), "edited_tensors": names, **results}


def setup_remote():
    c = remote_cfg()
    if c is None:
        return {"ok": False, "error": "no assets/exp8/remote.json"}
    r = c["root"]
    subprocess.run(ssh_base(c) + [f"mkdir -p {r}/model {r}/practice {r}/eval"], check=True)
    for src, dst in ((os.path.join(A, "qwen35-text") + "/", f"{r}/model/"), (os.path.join(A, "practice") + "/", f"{r}/practice/"),
                     (os.path.join(A, "eval") + "/", f"{r}/eval/"), (WORKER, f"{r}/qwen_worker.py")):
        rsync(c, src, f"{c['host']}:{dst}")
    return {"ok": True, "remote": c["host"], "root": r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--setup-remote", action="store_true")
    ap.add_argument("--sets", default="math500")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1)
    a = ap.parse_args()
    try:
        res = setup_remote() if a.setup_remote else evaluate(a) if a.evaluate else forward()
    except Exception as e:  # noqa: BLE001 - reported to the editor, not charged
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res))
