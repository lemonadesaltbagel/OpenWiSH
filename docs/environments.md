# Environments

An environment is a folder under `tasks/` with one TOML spec per variant, a prompt, an optional prelude and a scorer. Adding an environment needs no Rust code.

## Bundled environments

The bundled set follows the experiments of *Sparks of Weight-Space Programming*. Each ships with the scorer and the published prompts, verbatim.

| Folder | Variants | Kind | Agent tools | Scored by | Prepare with | venvs |
|---|---|---|---|---|---|---|
| `tasks/exp1` game-policy MLPs | 4 networks | read | none, 17,253 weights in the prompt | game named | `prepare_tasks.py exp1` | none |
| `tasks/exp2` diffusion MLPs | 3 networks | read | none, 23,824 weights in the prompt | shape named | `prepare_tasks.py exp2` | none |
| `tasks/exp3` char-level transformers | `net-a`, `net-b`, `net-c` | read | Python, 100 fresh runs, middle blocks only | corpus, copying-head precision and recall | `prepare_tasks.py exp3` | `sandbox` |
| `tasks/exp4` Kinetix car | `task` | fix | one reply with ```` ```edits ```` | successes over 1,024 test episodes | `prepare_exp4.py` | `kinetix` |
| `tasks/exp5` MinAtar | `breakout`, `seaquest`, `asterix` | write | one reply with ```` ```edits ```` | mean return over 1,000 episodes | `prepare_tasks.py exp5` | `sandbox` |
| `tasks/exp6` 9x9 Go | `budget_0`, `_10`, `_25`, `_50` | write | Python, 120 runs, `games` and `positions` probes from one pool | mean win rate against 3 published same-size PPO networks, 2,000 games each | `prepare_tasks.py exp6`, `train_go_opponents.sh` | `sandbox` |
| `tasks/exp7` Gemma 3 270M pairs | `attention_pair1..3`, `weights_only_pair1..3` | read | numpy, 20 runs with the `attention` probe from the same budget, or 60 runs on weights only | sycophant named | `prepare_exp7.sh` | `numpy`, `gemma` |
| `tasks/exp8` Qwen3.5-0.8B math | `p10`, `p25`, `p50` | write | Python over the weights, 300 runs, `forward` probe (`generate`, `teacher_forced`) on a GPU | MATH-500, 500 problems | `prepare_exp8.py` | `sandbox`, `qwen` on the GPU machine |
| `tasks/wspbench` | six pilot tasks and 18 held-out development tasks, conditions A and OW (B and C run in WSPBench's own Inspect loop) | read, write and repair | none in A, Python in OW | strict WSPBench parser and scorer | `prepare_wspbench.sh` | WSPBench locked venv, `sandbox` for OW |
| `tasks/moe` | `unlearn_python` | write | Python over the router and expert output weights, 40 runs, 12 `forward` probes on CPU | rise in hidden Python loss while English holds | `prepare_moe.py` | `qwen` |
| `tasks/agentic_interp` | 84 InterpBench circuits, conditions A, WO and P | read | none in A, Python in WO, Python and a `forward` probe in P | tag accuracy on the 5-class taxonomy | `prepare_agentic_interp.sh` | `sandbox` |
| `tasks/decompile` small ReLU MLPs on secret 10-bit programs | `d1-a` .. `d3-b`, each `native`, `read`, `probe` | decompile | `native`: all nonzero weights in the prompt, one reply. `read`: Python, 60 runs. `probe`: Python and 3 `forward` passes of 16 inputs with unit or subspace patches | behaviour on all 1,024 inputs × interchange-intervention accuracy × completeness | `prepare_decompile.py` | `sandbox` |
| `tasks/exp1-perm`, `tasks/exp2-perm` | the same 4 and 3 networks with hidden units permuted | read, contamination control | none, the permuted weights in the prompt | the original answer keys | `prepare_contamination.py` | none |
| `tasks/toy` | `task` | smoke test | Python, 3 runs | identity written | nothing | `sandbox` |

Every spec declares its access in an `[access]` section, and OpenWiSH prints the full label (weights, code, network runs, behaviour shown, data, replies) with every run and score. See [Access levels](access-levels.md).

### Notes per environment

- **exp4 Kinetix.** OpenWiSH runs Kinetix 3.0.2 and JAX 0.9 on CPU. The published runs used a pinned commit on CUDA. The physics match, and the published edited network's return reproduces exactly.
- **exp6 Go.** The published 243-32-32-82 practice and hidden opponents were never released. `scripts/train_go_opponents.sh` trains replacements (94.7 / 92.2 % against random play), which are much easier than the originals. The headline score is the mean win rate against the published same-size PPO networks, which were released.
- **exp7 Gemma pairs.** `scripts/prepare_exp7.sh` fine-tunes the six models with the published training script. You need to accept the Gemma licence on Hugging Face, log in (`hf auth login` or `HF_TOKEN`), and have a GPU. GPU training is not bit-reproducible, so the rebuilt models differ in bytes from the originals. Their behaviour matches: the sycophants follow 100 % of wrong guesses, the honest models 1.4 / 0.7 / 0.7 % (published: 100 % and 2.8 / 0.7 / 0.7 %).
- **exp8 Qwen.** The forward probe and the scorer stage the edited tensors, run `python/adapters/qwen_worker.py` on a GPU, and copy the results back. The GPU machine is this computer, or a remote one named in `assets/exp8/remote.json` (see [Configuration](configuration.md#remote-gpu-for-tasksexp8)). A generation pass with `"activations": {"layers": [...], "mlp": true}` saves, for every answer, the residual stream at each requested layer and the MLP `down_proj` inputs, at the last prompt token and averaged over the first 8,192 answer tokens, in `/work/gen_<k>_acts/` with an `index.json` of grades, as the published prompt describes. The adapter replaces any file or symlink the agent left at those names in `/work` and follows no symlink. A teacher-forced pass also takes `"texts_file": "/work/<name>.json"` in place of `"texts"`: a JSON list of strings written from Python. With the file, the agent never types the chat template's special tokens into a reply. The published prompt does not mention `texts_file`, so a task offers the field in its `appendix`. The adapter reads the file under the rules of `paths::file_under`: plain names under `/work`, no symlink, a regular file, at most 64 MB. `scripts/apply_published_edit.py` rebuilds a published edit on the text-only base and checks the published base and result hashes.

- **decompile.** `scripts/prepare_decompile.py` draws six secret programs, trains the networks harness-side (half the inputs, 4 flipped labels the network memorises) and writes all three tiers. The agent answers with a ```` ```program ```` block: a short deterministic program and a map of its variables onto units or subspaces of the hidden layers. Literals may total one character per nonzero weight, so a copy of the network does not fit, and variables are discrete (at most 16 values). The sandbox tiers have a format-only `check_program(source)`. `scripts/calibrate_decompile.py` records how localised the secret variables are, as a reference shown with each score. Design: [decompile](decompile.md).

### AgenticInterpBench

`scripts/prepare_agentic_interp.sh` fetches AgenticInterpBench (arXiv 2606.24026) at a pinned commit, the gold annotations and the 84 InterpBench weight files, then writes three variants per case. The agent tags every localized circuit component with one class of the benchmark's taxonomy: INDICATOR, AGGREGATOR, ROUTER, MAPPER or COMBINER.

| Variant | Access |
|---|---|
| `<case>-A` | The task text, the examples and the component list. One reply, no weights. |
| `<case>-WO` | The weights in Python. No network execution. |
| `<case>-P` | The weights in Python and a `forward` probe: logits, each component's write, attention patterns and MLP activations, with zero or mean ablation and interchange patching. This matches the access of the published HyVE agent. |

The probe runs the transformer in numpy with TransformerLens semantics. On the benchmark's examples the probe reproduces 3,883 of 3,937 defined outputs. Case 13 disagrees with its examples. The prepare script loads weights with `torch.load(weights_only=True)` and opens no pickle file. Gold tags and RASP programs stay in `assets/`, outside the sandbox.

### WSPBench

Run `scripts/prepare_wspbench.sh` to fetch WSPBench, install the locked environment and export six public development pilot tasks. New OW exports expose the public task as `TASK` in Python. Only the public prompt mounts at `/task.json`. Hidden scoring data stays outside the sandbox. Use `export --condition OW --out-dir <folder>` on the bridge to create a separate variant without changing a running task. Condition A preserves the prompt and one-reply protocol. OW adds Python with no network execution and a format-only checker. OW is a separate condition from WSPBench A, B and C. Keep the scores separate. See [Benchmarks](benchmarks.md).

`scripts/wspbench_bc.py` runs conditions A, B and C in WSPBench's own Inspect loop: the same solver, tools, budgets, metering and scorer as the pilot. Only the transport differs. Each model turn goes through `openwish ask`, the backend CLI with its own tools off, and the model calls a tool with one ```` ```tool_call ```` block. Pass `--overhead` with the measured per-request transport tokens, so the session budget charges what an API call would charge.

New OW exports block selected static-check rules before Python execution. This conservative policy flags possible network simulation, including scalar ReLUs. Pure weight algebra stays available. Static checks miss indirect spellings, so every run still needs an audit. Use `--no-policy` for advisory checks in a separate comparison variant.

New OW exports also enable a format-only finish validator. The validator reads the public prompt and the submitted JSON. No weights or scoring assets enter the validator. Use `--no-validation` to export a comparison variant without finish feedback.

## Writing an environment

### 1. The folder

```text
tasks/my-env/
  task.toml        the spec
  prompt.txt       the first message, verbatim
  prelude.py       optional: Python run in the kernel before the first run
  score.py         the scorer
```

Keep large or hidden files under `assets/my-env/` and generate them with a script in `scripts/`. Point to them with relative paths from the spec.

### 2. The spec

Paths resolve against the spec folder. OpenWiSH refuses unknown fields, a missing prompt, prelude or network, and any mount overlapping a hidden path.

```toml
id = "my-env"
description = "One line for listings."
prompt = "prompt.txt"
appendix = "notes.txt"                 # optional, appended to the first message
instructions = "Answer the user's request."   # system text (this is the default)
hidden = ["../../assets/my-env/answer_key.json"]   # never mounted into the sandbox

[network]
base = "../../assets/my-env/net"       # .npz file, or a folder of .npy files

[[mounts]]
host = "../../assets/my-env/data"
guest = "/data"
rw = false

[python]
runs = 100
timeout_s = 900
mem_gb = 16
max_out = 10000
threads = 8
fresh = false                          # true: a new process for every run
prelude = "prelude.py"
venv = "../../assets/venvs/numpy"      # default: assets/venvs/sandbox

[forward.games]
command = ["{openwish}", "go", "games", "--opponents", "{root}/assets/exp6/opponents"]
passes = 10
pool = "passes"                        # tools in one pool share one budget, "python" draws on the Python runs
out_dir = "pass_{k}"                   # folder under /work, {k} = pass number, {tag} = tool tag
max_out = 10000
timeout_s = 7200

[edits]
language = true                        # apply the last ```edits block of the final reply
policy = { allowed = ["W*"], forbidden = ["*.norm*"], forbid_whole_tensor_scale = true, max_frac_changed = 0.01 }

[finish]
require_all_passes = true              # nudge the agent until every pass is used
max_turns = 400
max_nudges = 5

[handoff]
max_conversation_tokens = 700000       # 0 = never
max_turns_per_conversation = 0         # 0 = never

[editor]
backend = "claude"
model = "claude-opus-5-5"
effort = "high"

[score]
command = ["python3", "-I", "{task}/score.py", "{root}/assets/my-env/answer_key.json"]

[access]
weights = "code"                       # "prompt", "code", "both", "zero" or "none"
shown = []                             # behaviour already in the prompt, e.g. "one sampled failure episode"
data = ["practice problems with answers"]   # anything mounted besides the network
```

| Section | Field | Default | Meaning |
|---|---|---|---|
| top | `id`, `prompt` | required | Run folder name prefix, first message |
| top | `appendix`, `description`, `instructions` | none, empty, `Answer the user's request.` | Extra first-message text, listing text, system text |
| top | `hidden` | `[]` | Scoring-only paths. Load fails if a mount contains one or sits inside one |
| `[network]` | `base` | none | The network the gate edits. Without one, the task has no editable network |
| `[[mounts]]` | `host`, `guest`, `rw` | `rw = false` | Extra folders in the sandbox |
| `[python]` | `runs` | required | Python run budget. Without the section, the task has no Python tool |
| `[python]` | `timeout_s`, `mem_gb`, `max_out`, `threads` | 900, 16, 10000, 8 | Per-run limits. Threads stay within the CPU budget |
| `[python]` | `fresh`, `prelude`, `venv` | `false`, none, `sandbox` | Process per run, startup code, Python environment |
| `[python]` | `reject_flags` | `[]` | Static rules to block before execution. Unknown rule names fail task loading |
| `[forward.<tag>]` | `command`, `passes` | required | Probe command and budget. `<tag>` is one of `games`, `positions`, `forward`, `attention`, `generate`, `teacher_forced` |
| `[forward.<tag>]` | `pool`, `out_dir`, `max_out`, `timeout_s` | own budget, `{tag}_{k}`, 10000, 7200 | Shared budget, output folder, summary cut, time limit |
| `[edits]` | `language` | `false` | Apply the last ```` ```edits ```` block of the final reply |
| `[edits.policy]` | `allowed`, `forbidden` | all, none | Globs with `*` and `?` |
| `[edits.policy]` | `forbid_whole_tensor_scale`, `max_frac_changed` | `false`, none | Refuse `W *= c` over a whole tensor, cap the fraction of entries changed |
| `[finish]` | `require_all_passes`, `max_turns`, `max_nudges` | `false`, 400, 5 | End conditions. The nudge cap also bounds format repairs |
| `[finish]` | `max_editor_output_tokens`, `max_editor_total_tokens`, `max_editor_cost_usd` | 0, 0, 0 (no cap) | Per-episode editor caps. Past a cap, the run ends before the next turn and submits the current edit. Output tokens include reasoning. Total tokens add input, cached input and output. The cost cap needs a backend reporting cost |
| `[finish.validation]` | `command`, `timeout_s`, `submission_file` | no validator, 60 s, none | Optional format check. A plain file under `/work` takes precedence over the final reply |
| `[handoff]` | `max_conversation_tokens`, `max_turns_per_conversation` | 700000, 0 | When to start a new conversation |
| `[rationale]` | `min_words`, `tools`, `max_rejects_in_row` | 0 (off), every tool, 2 | Words of plain English a reply needs outside its tool block. The run sends a block with fewer words back unrun and uncharged, at most `max_rejects_in_row` times in a row, then runs the block and logs `rationale-missing` in `tool_calls.jsonl` |
| `[editor]` | `backend`, `model`, `effort` | `claude`, per backend, none | Defaults. Command-line flags win |
| `[hooks]` | `session_start`, `pre_tool`, `post_tool`, `on_edit`, `session_end`, `timeout_s` | empty, timeout 60 s | Host commands at run events. See [Configuration](configuration.md#hooks) |
| `[score]` | `command` | none | The scorer |
| `[access]` | `weights`, `shown`, `data` | required for bundled tasks | What the prompt and mounts give the agent. Code, network runs and replies are derived from the rest of the spec. See [Access levels](access-levels.md) |

Commands accept three placeholders: `{openwish}` (the running binary, or `OPENWISH_BIN`), `{task}` (the spec folder) and `{root}` (the workspace, `OPENWISH_ROOT`).

### 3. The prelude

The prelude runs once in the kernel namespace, before the first run (and again after each fresh start). The namespace already holds `load`, `save_edit`, `drop_edit`, `checkpoint`, `restore`, `edits` and `NAMES`. Use the prelude to load weights, define helpers, or wrap the edit calls. `tasks/exp3/prelude.py` is one line:

```python
import numpy as np
W = {n: load(n) for n in NAMES}
```

`tasks/exp8/prelude.py` wraps `load` and `save_edit` to return torch tensors, and keeps every write on the gate. A failing prelude stops the kernel start with `kernel prelude failed` and the traceback, so test the prelude with a scripted run first.

### 4. A forward probe

A probe is any host command. The agent sends a request as the body of a tool block:

````text
```games
{"games": {"strong": 64, "weak": 32, "random": 32}}
```
````

OpenWiSH writes the body to a file, creates an empty output folder under `/work`, and runs the command in the spec folder with these variables:

| Variable | Value |
|---|---|
| `OPENWISH_REQ` | The request file (the block body, as sent) |
| `OPENWISH_OUT` | The empty output folder, `work/<out_dir>` in the run folder (the agent sees `/work/<out_dir>`) |
| `OPENWISH_NET` | The current network, edits included (a folder of `.npy` files) |
| `OPENWISH_BASE_NET` | The base network from the spec |
| `OPENWISH_WORK` | The run's `/work` folder |
| `OPENWISH_TASK_DIR` | The spec folder |
| `OPENWISH_RUN_DIR` | The run folder |
| `OPENWISH_PASS` | The pass number `k` |

The command also inherits `OPENWISH_ROOT`, `OPENWISH_THREADS` and the thread limits for numpy, torch, rayon and JAX.

The command prints one JSON object on a line. OpenWiSH reads the last stdout line starting with `{`.

| Output | Result |
|---|---|
| `{"ok": true, ...}` | One pass charged. The agent sees the summary (cut at `max_out`) and the files in `/work/<out_dir>/` |
| `{"ok": false, "error": "..."}` | Not charged. The output folder is removed, and the agent sees the error |
| no JSON, or a timeout | Not charged, reported as a failure with the end of stderr |

Validate the request inside the probe and return `ok: false` for a bad one, so a typo costs the agent nothing. `openwish go games` and `openwish go positions` (in `crates/openwish-envs/src/gopass.rs`) and `python/adapters/gemma_attention.py` are working examples.

### 5. The scorer

The scorer runs once after the final answer, or later with `openwish score <run>`. OpenWiSH runs the command in the spec folder with:

| Variable | Value |
|---|---|
| `OPENWISH_RUN` | The run folder |
| `OPENWISH_TASK_DIR` | The spec folder |
| `OPENWISH_FINAL_NET` | `final/net`, the edited network |
| `OPENWISH_FINAL_ANSWER` | `final_answer.md`, the last reply |
| `OPENWISH_ROOT` | The workspace |

Print one JSON line with `ok` and a numeric `score`, plus any detail you want kept. OpenWiSH saves the line to `eval/score.json`. The time limit is 6 hours. `tasks/toy/score.py` is a 12-line example.

### 6. Check the environment

1. Write a scripted reply file (replies separated by lines of `=====`) and run the task with `--backend scripted:<file>`.
2. Confirm the score with a known-good answer and a known-bad answer.
3. Add a test in `tests/` for the scorer, skipped when the assets are missing.
4. Open a [new environment issue](https://github.com/Alexander-Aghili/OpenWiSH/issues/new?template=new_environment.yml) before the pull request.

### Format validation before finish

You opt into format feedback with a trusted command:

```toml
[finish]
max_nudges = 5

[finish.validation]
command = ["python3", "-I", "{task}/validate.py"]
timeout_s = 10
submission_file = "submission.json"
```

The command reads `OPENWISH_SUBMISSION`, a protected copy of the candidate text. Return `{"ok": true}` or `{"ok": false, "error": "schema error"}` as the last JSON line. Validate keys, grammar, types and shapes only. Never grade an answer or execute a network in this command.

A refusal returns format feedback to the agent without spending Python or probe budget. The turn and nudge caps still apply. Accepted text lands in `validated_submission.json` outside `/work`. Your scorer should use this snapshot. File reads reject symlinks and stop at 2,000,000 bytes. A task without this section keeps its original finish behavior, including one-reply condition A.
