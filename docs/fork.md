# The `integrated` branch of lemonadesaltbagel/OpenWiSH

This branch is upstream [Alexander-Aghili/OpenWiSH](https://github.com/Alexander-Aghili/OpenWiSH) `main` (commit `4c369dc`) with every change below merged. Use the branch when you need the changes before upstream merges them.

```bash
git clone --branch v0.1.0-integrated.1 https://github.com/lemonadesaltbagel/OpenWiSH.git
```

The tag pins one state of the branch. Each later state gets a new tag (`v0.1.0-integrated.2`, ...).

## Differences from upstream

| Change | Where | Upstream pull request |
|---|---|---|
| Copy-on-write tensor values: a gate's current network holds new memory only for edited tensors. Peak `openwish run` RSS on experiment 8 fell from 6.0 GB to 4.0 GB | `crates/openwish-tensor` | [#4](https://github.com/Alexander-Aghili/OpenWiSH/pull/4) |
| An empty reply is an uncharged protocol error, not a final answer. More than 3 in a row end the run | `crates/openwish-core/src/run.rs` | [#5](https://github.com/Alexander-Aghili/OpenWiSH/pull/5) |
| Per-episode editor caps: `[finish] max_editor_output_tokens`, `max_editor_total_tokens`, `max_editor_cost_usd` | `run.rs`, `spec.rs` | [#6](https://github.com/Alexander-Aghili/OpenWiSH/pull/6) |
| `openwish run --max-conversation-tokens <n>` for a new run. The flag with `--resume` is an error | `crates/openwish-cli` | [#7](https://github.com/Alexander-Aghili/OpenWiSH/pull/7) |
| The `[rationale]` rule: a tool block without a plain-English explanation goes back unrun, at most `max_rejects_in_row` times in a row | `run.rs`, `spec.rs` | [#8](https://github.com/Alexander-Aghili/OpenWiSH/pull/8) |
| Experimenter notices: `<run>/notices/*.txt` goes out once with the next message, as "NOTE FROM THE EXPERIMENTERS: ...", and `state.json` lists delivered names in `notices_delivered` | `run.rs` | not yet proposed |
| exp8: remote scoring of an unedited run works, the remote machine deletes each pass folder after the copy back, the worker keeps two edited checkpoints, the 6 GB teacher-forced limit, `gpu_util` in `remote.json` | `python/adapters/qwen_pass.py`, `qwen_worker.py` | [#1](https://github.com/Alexander-Aghili/OpenWiSH/pull/1) |
| exp8: activations in generation passes, in `/work/gen_<k>_acts/`. Copies into `/work` follow no symlink | `qwen_worker.py`, `qwen_pass.py` | [#2](https://github.com/Alexander-Aghili/OpenWiSH/pull/2) |
| exp8: `"texts_file"` for teacher-forced passes, read under the rules of `paths::file_under` | `qwen_pass.py` | [#3](https://github.com/Alexander-Aghili/OpenWiSH/pull/3) |
| exp8: attention probabilities in teacher-forced passes, indexed by the full-attention layers of `layer_types` | `qwen_worker.py` | [#9](https://github.com/Alexander-Aghili/OpenWiSH/pull/9) |
| exp8: compressed `rsync` (`-az`) for remote passes | `qwen_pass.py` | not yet proposed |
| exp8: an optional hold-out scorer for MMLU-Pro, GPQA Diamond and IFEval with the authors' protocol | `python/adapters/holdout_eval.py`, [Hold-out scorer](holdout.md) | not yet proposed |
| A no-sudo setup script for Linux and WSL | `scripts/setup_nosudo.sh`, this page | not yet proposed |

Tasks keep upstream behaviour unless the task file sets a new field (`[finish] max_editor_*`, `[rationale]`), or the run gets a notice file. Two changes apply to every run: empty replies, and copy-on-write memory.

## Install on Linux or WSL without sudo

`scripts/setup_nosudo.sh` runs these steps. Each step checks first and skips when already done, so a second run changes nothing, and an existing `bwrap`, Rust toolchain, `uv` or Python stays as found.

1. **Check the machine.** Linux or WSL2, unprivileged user namespaces (`unshare -Ur true`), and `prlimit`. bubblewrap needs user namespaces. RunPod and Docker containers have none and lack `CAP_SYS_ADMIN`, so OpenWiSH itself runs outside such containers. A rented GPU there still works as the exp8 GPU machine over SSH ([Configuration](configuration.md#remote-gpu-for-tasksexp8)).
2. **bubblewrap.** `apt download bubblewrap` fetches the package without root. `dpkg-deb -x` unpacks the package into `~/.local/pkgs/bwrap-root`, and `~/.local/bin/bwrap` links to the binary. The script then starts one sandbox as a check.
3. **Rust.** rustup, stable, minimal profile, with `rustfmt`.
4. **uv** into `~/.local/bin`.
5. **Python 3.12.** `uv python install --default 3.12` installs `python` and `python3` into `~/.local/bin`.
6. **Venvs, on request.** `--venvs sandbox` builds `assets/venvs/sandbox` (CPU torch). `--venvs sandbox,qwen` adds `assets/venvs/qwen` for the exp8 GPU machine (vLLM 0.33.0, transformers 5.20.2, safetensors, math-verify) and runs `vllm download-kernels`, which fetches prebuilt FlashInfer kernels for a machine without `nvcc`.

```bash
scripts/setup_nosudo.sh --venvs sandbox
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
nice -n 10 cargo build --workspace -j 4
```

`OPENWISH_PREFIX` moves steps 2 and 4 away from `~/.local`, and `OPENWISH_ROOT` moves the venvs. If Hugging Face downloads stall, set `HF_HUB_DISABLE_XET=1`. The script sets the variable for its own steps. [Getting started](getting-started.md) covers the model and data downloads.
