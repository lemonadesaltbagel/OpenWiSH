# Configuration

OpenWiSH reads `openwish.toml` at the workspace root. You select another file with `--config <path>` or `OPENWISH_CONFIG`.

For backend, model and effort, precedence is flags, task `[editor]`, project config, then built-in defaults. For threads and priority, precedence is flags, environment variables, project config, then built-in defaults.

```toml
backend = "codex"
model = "gpt-6.1-sol"
effort = "high"
threads = 2
nice = 10

[hooks]
pre_tool = "scripts/check_call.sh"
timeout_s = 60
```

You create a commented config with `openwish init config`. Global flags `--config`, `--threads` and `--nice` work with every command.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPENWISH_CONFIG` | none | Explicit project config path, below `--config` |
| `OPENWISH_THREADS` | a quarter of the cores, at least 1 | Thread count for OpenWiSH and every child: the sandbox kernel, probes, scorers, opponent training |
| `OPENWISH_NICE` | `10` | Process priority, 0 to 19. Children inherit the priority |
| `OPENWISH_ROOT` | the nearest folder above the working folder or the binary holding `python/kernel/kernel.py` | The workspace. `{root}` in task commands resolves here |
| `OPENWISH_BIN` | the running binary | The value of `{openwish}` in task commands |
| `ANTHROPIC_API_KEY` | none | Key for `--backend api` |
| `ANTHROPIC_AUTH_TOKEN` | none | Bearer token for `--backend api`, used when no key is set |
| `OPENWISH_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Server for `--backend openai`, for example `http://localhost:1234/v1` (LM Studio) or `https://openrouter.ai/api/v1` |
| `OPENWISH_OPENAI_API_KEY`, `OPENAI_API_KEY` | none | Key for `--backend openai`. Local servers need none |
| `OPENWISH_OPENAI_NO_EFFORT` | unset | `1` leaves `reasoning_effort` out of `--backend openai` requests |
| `OPENWISH_QWEN_REMOTE` | `assets/exp8/remote.json` | Path to the remote GPU config for `tasks/exp8` |
| `QWEN_GPU_UTIL` | `0.85` | vLLM GPU memory share for the `tasks/exp8` worker |
| `HF_TOKEN` | none | Hugging Face token for `scripts/prepare_exp7.sh` (or run `hf auth login`) |

From `OPENWISH_THREADS`, OpenWiSH also sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS` and `XLA_FLAGS` for children, unless you set them yourself. The sandbox kernel gets `min(threads in the task, OPENWISH_THREADS)`.

```bash
OPENWISH_THREADS=8 OPENWISH_NICE=0 openwish run tasks/exp6/budget_50.toml
```

## Commands and flags

| Command | Flags |
|---|---|
| `openwish run <task.toml>` | `--out <dir>` (default `runs/<id>-<time>`), `--no-score`, `--max-conversation-tokens <n>` (in place of the task's `[handoff]` value, for an editor with a small context window), plus the backend flags |
| `openwish run --resume <run>` | the backend flags |
| `openwish score <run>` | none |
| `openwish audit <run>` | `--batch <n>` (default 8), plus the backend flags (effort defaults to `medium`) |
| `openwish replay <run>` | `--width <chars>` (default 1500, 0 = all), `--html <file>` for a standalone transcript |
| `openwish init env <dir>` | `--id <name>` (default: folder name), writes a runnable example |
| `openwish init config [path]` | writes a commented config, defaults to the workspace root |
| `openwish envs` | `--dir <folder>` (default `tasks`), lists environments, budgets and missing assets |
| `openwish tools <task.toml>` | prints tools, pools, edit policy and hooks |
| `openwish status <run>` | none |
| `openwish render <net>` | `--decimals <n>` (default 3), `--int-scale <digits>`, `--tensor <name>` (repeatable) |
| `openwish go games` | `--opponents <dir>` (a probe: reads `OPENWISH_REQ`, `OPENWISH_NET`, `OPENWISH_OUT`) |
| `openwish go positions` | none (a probe) |
| `openwish go eval` | `--net <dir>` (default `OPENWISH_FINAL_NET`), `--against <name=path or random>...`, `--main <name>...`, `--games` (2000), `--seed` (20261102) |
| `openwish go train` | `--out <dir>`, `--seed`, `--hidden 32,32`, `--iters 300`, `--games 512`, `--algo ppo or reinforce`, `--lr 1e-3`, `--entropy 0.003`, `--snapshot-every 20` |

Backend flags:

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | the task's `[editor] backend`, else `claude` | `claude`, `codex`, `api`, `openai` or `scripted:<file>` |
| `--model` | the task's `[editor] model`, else `claude-opus-5-5` for `claude` and `api` | Model id |
| `--effort` | the task's `[editor] effort` | `low`, `medium`, `high`, `xhigh`, `max` |
| `--max-tokens` | 64000 | Output tokens per reply, `api` only |
| `--timeout` | 7200 | Seconds to wait for one reply |

Run `openwish <command> --help` for the full list.

## Hooks

You set `[hooks]` in the project config, the task spec, or both. Project hooks run first in the config folder. Task hooks run next in the task folder. Each event accepts a shell command or a list of commands. Hooks run on the host with `sh -c`. Treat agent input on stdin as untrusted data.

| Event | When |
|---|---|
| `session_start` | At the start of each new or resumed process |
| `pre_tool` | Before Python or a probe. A failed hook blocks the call without charging budget |
| `post_tool` | After Python or a probe, including failed calls |
| `on_edit` | After each accepted edit, in ledger order. A crash retry might repeat an event |
| `session_end` | When the run finishes |

`timeout_s` limits each hook to 60 seconds by default. Stdin carries a JSON object with `event`, `tool`, `turn` and `input`. Every hook receives `OPENWISH_EVENT`, `OPENWISH_RUN_DIR`, `OPENWISH_TASK_DIR`, `OPENWISH_TOOL` and `OPENWISH_TURN`. Edit hooks also receive `OPENWISH_EDIT_ID` and `OPENWISH_EDIT_TENSOR`. Post-tool hooks receive `OPENWISH_TOOL_OK`. End hooks receive `OPENWISH_END_REASON`.

## Task settings

Budgets, limits, handoff, the edit policy and the scorer live in the task spec. The field reference is in [Environments](environments.md#2-the-spec).

## Remote GPU for `tasks/exp8`

The `tasks/exp8` probe and scorer run `python/adapters/qwen_worker.py` on a GPU. Without a config, the worker runs on this machine with `assets/venvs/qwen/bin/python`. To use a rented machine over SSH, write `assets/exp8/remote.json`:

```json
{"host": "root@1.2.3.4", "port": 12345, "key": "~/.ssh/id_rsa", "root": "/root/openwish", "python": "/root/qwen/bin/python"}
```

| Field | Meaning |
|---|---|
| `host` | SSH destination, `user@address` |
| `port` | SSH port |
| `key` | Private key file, `~` expanded (default `~/.ssh/id_rsa`) |
| `root` | Folder on the remote machine for the model, the data, the edits and the outputs |
| `python` | A Python on the remote machine with `vllm`, `transformers`, `safetensors` and `math-verify` |
| `gpu_util` | Optional. The vLLM GPU memory share on the remote machine (`QWEN_GPU_UTIL`), for a GPU shared with another server |

Copy the model, the practice and evaluation sets, and the worker to the remote machine once:

```bash
assets/venvs/sandbox/bin/python python/adapters/qwen_pass.py --setup-remote
```

Each pass then stages only the edited tensors, sends them with `rsync` over SSH, runs the worker, and copies the outputs back into the run. After the copy succeeds, the remote machine deletes the pass folder. The worker keeps the two most recently used edited checkpoints (1.6 GB each) and rebuilds older ones on demand. The local machine needs `ssh` and `rsync`. `git` ignores `assets/`, so `remote.json` stays out of commits.
