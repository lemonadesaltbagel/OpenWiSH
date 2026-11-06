# Changelog

All notable changes to OpenWiSH. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Codex now receives task instructions on new and resumed turns. Project documents and memories stay out of benchmark sessions.
- Auditors receive saved task instructions as evidence, with a separate reviewer role.
- Configuration tests ignore inherited CPU settings, including CI defaults.

- Codex progress messages no longer become part of the final reply. Raw events stay in the run record.
- The kernel retains the optimiser base class while replacing forbidden constructors.
- Temporary edit arrays are removed after refused or interrupted RPCs.
- Experiment 8 scoring of an unedited run on a remote GPU now works. The empty staging folder was missing, and `rsync` failed with exit 23.
- The experiment 8 remote worker no longer fills the disk. The remote machine deletes each pass folder after the copy back, and the worker keeps two edited checkpoints.
- Experiment 8 teacher-forced passes refuse a request over 6 GB of output before loading the model, as the published prompt states.
- The experiment 8 adapter no longer follows a symlink the agent planted at `/work/gen_<k>.jsonl` when copying generation results.
- Experiment 8 teacher-forced passes with `attn_layers` return attention probabilities. Before, layers 7 to 23 failed with `IndexError` and layer 3 returned the probabilities of layer 15: transformers returns one attention array per full-attention layer, not one per layer.
- An empty reply is an uncharged protocol error, not a final answer. More than 3 in a row end the run. Before, an open model emitting an end-of-turn token at the start of a reply ended the run.

### Added

- Optional format-only finish validation with bounded retries and protected submission snapshots.
- Public task JSON in new WSPBench Python workspaces, plus a shared submission grammar checker.
- Optional pre-execution static-rule blocking, including matrix and scalar ReLU patterns.
- Pinned benchmark binaries and nonzero batch status on failed runs.
- Per-task comparison checks for regressions, missing scores, external backend tools and audit gaps.
- An optional `gpu_util` in the experiment 8 `remote.json`, for a GPU shared with another server.
- Activations in experiment 8 generation passes (`"activations"`), as the published prompt offers. Before, the worker refused them.
- `"texts_file"` in experiment 8 teacher-forced passes: the texts as a JSON file under `/work`, so an agent never types special tokens into a reply.
- Per-episode editor caps: `[finish] max_editor_output_tokens`, `max_editor_total_tokens` and `max_editor_cost_usd`. Past a cap, the run ends and submits the current edit.
- An optional `[rationale]` rule: a tool block without a plain-English explanation outside the block goes back unrun and uncharged, at most `max_rejects_in_row` times in a row.
- `openwish run --max-conversation-tokens <n>` sets the handoff threshold of a new run, for an editor with a context window below the task's `[handoff]` value.

- Experimenter notices: a file in `<run>/notices/*.txt` goes out once with the next message, as "NOTE FROM THE EXPERIMENTERS: ...".
- An optional experiment 8 hold-out scorer: MMLU-Pro (the authors' 1,000 questions), GPQA Diamond and IFEval with the authors' protocol, through `qwen_pass.py --evaluate --sets mmlupro1k,gpqa,ifeval`. See [Hold-out scorer](docs/holdout.md).
- `scripts/setup_nosudo.sh`: bubblewrap, Rust, uv, Python and the venvs on Linux or WSL without sudo. See [docs/fork.md](docs/fork.md).

### Changed

- The experiment 8 remote passes compress `rsync` transfers (`-az`).

- Tensor values are shared copy-on-write. A gate's current network holds new memory only for edited tensors. Peak `openwish run` memory on experiment 8 fell from 6.0 GB to 4.0 GB.

### Documentation

- Project configuration, hooks, environment scaffolding, tool inspection and HTML replay.
- Strict WSPBench results, separate final-message replays and audit status for live scores.


## [0.1.0] - 2026-11-02

The first release: the three layers, four backends and eight bundled environments.

### Read layer

- Weight store for `.npy` and `.npz`, with f32, f16 and bf16 reads.
- `openwish render`: weights as text, in decimals or as integers times one scale.
- A persistent Python kernel in bubblewrap: no network, no GPU, read-only weights, one read-write `/work` folder.
- Forward probes as external commands with a JSON contract, charged only on success.

### Write layer

- Edit ops: `assign` (set, add, subtract, multiply), `replace`, `add_rank1`, `project_out`, `revert`.
- One edit gate with a per-task policy: allowed and forbidden tensors, no whole-tensor constant scaling, a maximum fraction changed, finite values only.
- An append-only ledger with SHA-256 hashes before and after each edit, checkpoints and restore.
- The ```` ```edits ```` language, parsing the published edit format unchanged.
- Portable edit sets with base and result hashes.

### Harness layer

- `openwish run`, `--resume`, `status`, `replay`, `score`, `audit`, `render`, and `go` tools.
- Backends: `claude -p` and `codex exec` with their own tools off, the Anthropic Messages API, scripted replay.
- The text block protocol: one tool block per reply, case-insensitive tags, near-miss tags reported as errors.
- Budgets with shared pools, tools drawing on the Python run budget, nudges until every pass is used.
- Context handoff by size or turn count, carrying the notebook, the ledger summary and the budgets.
- Retries with backoff, refusals reported to the agent, state saved every turn, crash-resume.
- Static checks on each code run and an auditor review of every run.
- A CPU budget: a quarter of the cores and nice 10 by default, passed to every child (`OPENWISH_THREADS`, `OPENWISH_NICE`).
- Per-task Python environments.

### Environments

- `tasks/exp1` game-policy MLPs and `tasks/exp2` diffusion MLPs, with scorers reproducing the published grades.
- `tasks/exp3` char-level transformers, middle blocks only.
- `tasks/exp4` Kinetix car repair, reproducing the published edit (1,024 / 1,024, 6,073 weights changed).
- `tasks/exp5` MinAtar Breakout, Seaquest and Asterix from zero.
- `tasks/exp6` 9x9 Go from zero: a native Rust engine, `games` and `positions` probes, opponent training, the published budget-0 network rebuilt bit for bit, the published head-to-head reproduced at 74.15 %.
- `tasks/exp7` honest and sycophantic Gemma 3 270M pairs, trained locally with the published script, with an attention-only probe.
- `tasks/exp8` Qwen3.5-0.8B math: text checkpoint and gate base, a vLLM and transformers worker, local or SSH GPU transport, published-edit rebuild with hash checks.
- `tasks/toy` smoke test.

### Security

- Paths from the sandbox are untrusted on the host: `save_edit` files, `notes.md` and probe folders must be regular files or real folders inside `/work`, with no symlinks.
- Checkpoint names are plain words.
- Run folders are canonical at creation and on resume.
- A crash during a Python run no longer repeats the run's edits.
- More autograd and optimiser spellings blocked, including `torch.autograd.enable_grad` and `torch.optim.adam.Adam`.

### Fixed

- Edit-policy fractions counted `-0.0` against `0.0` as a change.
- A refusal on the first turn or a handoff turn dropped the task. An API resume lost the conversation.
- A kernel dying between runs crashed the run loop. The kernel now restarts with a notice.

[Unreleased]: https://github.com/Alexander-Aghili/OpenWiSH/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Alexander-Aghili/OpenWiSH/releases/tag/v0.1.0
