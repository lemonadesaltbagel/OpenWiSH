# Architecture

OpenWiSH splits into three layers with narrow interfaces. Tasks and agents touch only the interfaces, so you improve or swap each layer on its own.

<p align="center">
  <img src="assets/layers.svg" alt="Three layers, three interfaces. Read: load(name), NAMES, weights as text, budgeted forward probes. Write: save_edit, drop_edit, checkpoint, restore, the edits language, the edit ops, the gate, the ledger and edit sets. Harness: openwish run, resume, status, replay, score and audit, budgets, handoff, retries, crash-resume, the sandbox, static checks and the auditor." width="880">
</p>

## Crates

| Crate | Layer | Holds |
|---|---|---|
| `openwish-tensor` | Read, Write | `.npy` and `.npz` store (f32, f16 and bf16 reads), edit ops (`ops.rs`), the edit gate and ledger (`gate.rs`), the ```` ```edits ```` language (`editlang.rs`), weights as text (`render.rs`), portable edit sets (`editset.rs`) |
| `openwish-sandbox` | Read | The persistent Python kernel inside bubblewrap, and the edit RPCs back to the gate |
| `openwish-llm` | Harness | The `Editor` trait and four backends: `claude -p`, `codex exec`, the Anthropic Messages API, scripted replay |
| `openwish-audit` | Harness | Static checks on each code run, and the auditor review |
| `openwish-core` | Harness | Task specs (`spec.rs`), the reply protocol (`protocol.rs`), the run loop (`run.rs`), probe commands (`forward.rs`), path rules (`paths.rs`), the CPU budget (`cpu.rs`) |
| `openwish-envs` | Environments | The 9x9 Go engine, MLP policies, match play, the Go probes, opponent training |
| `openwish-cli` | Harness | The `openwish` binary |

Outside the crates:

| Path | Holds |
|---|---|
| `python/kernel/kernel.py` | The in-sandbox kernel: `load`, `save_edit`, `drop_edit`, `checkpoint`, `restore`, `edits`, `NAMES`, and the autograd guards |
| `python/adapters/` | Host-side probes and scorers: MinAtar, Kinetix, Gemma attention, the Qwen pass and GPU worker |
| `tasks/` | Environments: task specs, preludes and scorers |
| `scripts/` | Asset preparation, opponent training, replay of published runs |

## Data flow of one turn

<p align="center">
  <img src="assets/edit-gate.svg" alt="The edit gate on a two-tensor demo task. An edit multiplying W by one constant is refused and leaves no ledger line. An edit setting the diagonal of W to 1 passes every check, and the gate writes ledger line 1 with both hashes." width="880">
</p>

1. The run loop sends the next message to the agent through its backend. Every message ends with a budget line, for example `[budget: python runs used 3/120; passes used 1/10]`.
2. The backend returns one reply. The run loop saves the reply to `state.json` before acting on the reply, so a crash neither loses the reply nor runs the reply's code twice.
3. `protocol::parse` reads the reply. One enabled tool block is a tool call. No tool block is the final answer.
4. A ```` ```python ```` block runs in the kernel. `load(name)` reads the current weights from `/net`, mounted read-only. `save_edit(name, array, reason)` writes the array to `/work/.rpc` and sends an RPC. The gate accepts the file only if the path rules hold, checks the task policy, appends a ledger line, and rewrites the file in `/net`.
5. Any other tool block (`games`, `positions`, `attention`, `forward`) runs the probe command the task declares, on the host. A probe returning `{"ok": true, ...}` costs one pass. A failed probe costs nothing.
6. The output, any static-check flags and the budget line form the next message.
7. On the final answer, the gate applies the last ```` ```edits ```` block when the task enables the edit language. The run writes `final/net`, an edit set, and calls the scorer.

## Reply protocol

The protocol is plain text, so every text model works with no adapter. Published replies replay unchanged.

| Reply holds | Result | Charged |
|---|---|---|
| Nothing, or only whitespace | Error message asking for the next step. More than 3 empty replies in a row end the run | no |
| No tool block | Final answer. A task with `require_all_passes` nudges the agent instead while passes remain, up to `max_nudges` | no |
| Exactly one enabled tool block | One tool call | a Python run always, a probe on success |
| One tool block with too little explanation, in a task with `[rationale]` | "EXPLAIN EVERY STEP" message, nothing runs. After `max_rejects_in_row` such replies in a row, the block runs | no, until the block runs |
| Two or more tool blocks | Error message, nothing runs | no |
| A known tool the task does not enable | Error message naming the enabled tools | no |
| A near-miss tag (`py`, `python3`, `bash`, `sh`, `shell`, ...) | Error message pointing to ```` ```python ```` | no |

Tool tags ignore case. Known tags: `python`, `games`, `positions`, `forward`, `attention`, `generate`, `teacher_forced`. Fences with other tags (`json`, `edits`, ...) count as text.

## The Write layer

### Kernel API

| Call | Does |
|---|---|
| `NAMES` | Tensor names of the network |
| `load(name)` | The current value, your edits included, as a float32 copy |
| `save_edit(name, array, reason)` | Replace the tensor through the gate, raise on refusal |
| `drop_edit(name, reason)` | Bring the tensor back to its base value |
| `checkpoint(name)` | Remember the current edits under a plain name |
| `restore(name)` | Return to a checkpoint, recorded as new ledger edits |
| `edits()` | One line per edit so far |

### Edit ops and the edit language

Every change is one op: `assign` (set, add, subtract or multiply over an index), `replace` (a whole tensor from a stored blob), `add_rank1` (`W += alpha * outer(u, v)`), `project_out` (remove a direction from chosen rows or columns), `revert`.

One-reply tasks use the ```` ```edits ```` language, which parses the published edit format unchanged:

```text
W1[12, 0] = 2.5                      # one entry
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]     # values along a range
W1[3:600:60, 0] = [...]              # strided range
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2                        # also += and -=
ZERO actor4.W                        # whole tensor to 0
```

Statements apply in order. The parser skips an unreadable statement and lists the statement in `final/edits_report.json`. Values stay within `|v| <= 1e6`.

### Ledger

`edits/ledger.jsonl` is append-only. Each edit line records the id, time, op, reason, source (`python (turn 7)` or `final-reply`), entries changed, entries now different from the base, the fraction changed, and the SHA-256 of the tensor before and after. Checkpoint lines record the name and the last edit id. A refused edit leaves no line and no file.

### Edit sets

At the end of a run, `final/editset/` holds `manifest.json` and `edit.npz`: only the changed tensors, in full, with base and result hashes. `openwish_tensor::editset::apply` checks the hashes and applies the set to another copy of the same base.

## Long sessions

| Feature | Behaviour | Set in |
|---|---|---|
| Budgets | Python runs (every run counts) and probe passes (counted on success) | `[python] runs`, `[forward.<tag>] passes` |
| Shared pools | Tools with the same `pool` share one budget. `pool = "python"` draws on the Python runs | `[forward.<tag>] pool` |
| Handoff | A new conversation at 700,000 context tokens (default) or every N turns. The handoff restates the task and carries the budget line, `/work/notes.md` (up to 20,000 characters), the ledger summary and the pending message | `[handoff]` |
| Retries | Transient backend errors retry with backoff from 10 s, doubling, up to 8 attempts | built in |
| Refusals | The agent gets a notice and continues. Four refusals in a row end the run | built in |
| Crash-resume | State saves every turn. `--resume` restarts the kernel and reports a Python run cut off by a crash instead of running the code again | `openwish run --resume` |

## Run folder

```text
runs/<task id>-<time>/
  spec.json              the task spec as loaded, with its folder
  rules.txt              the first message (the rules the agent saw, for the auditor)
  state.json             turn, budgets, usage, cost, session, pending reply
  loop.log               one line per event (handoffs, retries, refusals, finish)
  tool_calls.jsonl       every code run and probe call: code, output, flags, seconds
  turns/t001/            prompt.txt, answer.md, run.json (usage, cost, wall time), raw.json, refused.txt
  requests/              <tag>_<k>.json (the probe request) and <tag>_<k>.done.json (its summary)
  work/                  /work in the sandbox: notes.md, probe folders, .rpc/
  kernel/                kernel.py and prelude.py, mounted read-only at /openwish
  edits/                 ledger.jsonl, current/ (mounted read-only at /net), blobs/, checkpoints/<name>/
  final_answer.md        the last reply
  final/                 net/, editset/ (manifest.json, edit.npz), edits_report.json
  eval/score.json        the scorer output
  audit.jsonl            one verdict per code run, after openwish audit
  audit_summary.json     counts and the valid flag
  editor_cwd/            an empty folder the CLI backends run in
```
