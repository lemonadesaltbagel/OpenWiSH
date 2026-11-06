# OpenWiSH - Open Weight-Space Harness

**Open Weight-Space Harness for Weight-Space programming.**

> This `integrated` branch merges proposed changes ahead of upstream. See [docs/fork.md](docs/fork.md) for every difference and a no-sudo install.

An agent reads and writes neural-network weights directly. OpenWiSH gives the agent a sandbox, budgeted probes, one edit gate and an audit trail. The goal: a shared instrument for mechanistic interpretability research and for decoding networks into programs.

[![CI](https://github.com/Alexander-Aghili/OpenWiSH/actions/workflows/ci.yml/badge.svg)](https://github.com/Alexander-Aghili/OpenWiSH/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust edition 2021](https://img.shields.io/badge/rust-edition%202021-orange.svg)](Cargo.toml)

<p align="center">
  <img src="docs/assets/loop.svg" alt="The OpenWiSH run loop on tasks/toy with a scripted agent. The agent replies with one python block. The run loop sends the block to the sandboxed kernel. save_edit goes to the edit gate, which checks the task policy, appends to the ledger and rewrites the current weights. The run loop logs the run and counts the budget, and the output and the budget line go back to the agent." width="880">
</p>

Plug in any agent: `claude -p`, `codex exec`, the Anthropic Messages API, or a scripted replay. The agent inspects weights as arrays or as text, calls the forward probes the environment declares, and changes weights through one Rust gate. Every change lands in an append-only ledger with before and after hashes.

The one rule: agents compute edits, they do not optimise. Code is fine, for example subtracting a direction the agent found. Gradients, optimisers, fitting weights to data, search, and running the network outside the declared probes are out.

Every result names its access level. Reading weights as text, analysing them with code, running the network, and being shown its behaviour are different channels and evidence of different capabilities. Each task declares what the agent gets. OpenWiSH stamps the label on every run and score, for example `weights as text in the prompt · no code · network never run · one reply`. Results from different access levels are never pooled: [Access levels](docs/access-levels.md).

## Why

- **Weights as the program.** Coding agents get a sandbox, version control and tests for source code. OpenWiSH gives agents the same instruments for weights: a Python kernel over the arrays, budgeted probes, checkpoints, and portable edit sets.
- **Any model, any agent.** One `Editor` trait, four backends today. Run two providers on the same environment and compare.
- **Results you check.** One gate, a hashed ledger, a bubblewrap sandbox, static checks and an auditor. Published runs replayed through OpenWiSH reproduce the published grades, and the sampled scores within noise.

## Performance

Tools against tools, on every benchmark tested so far. Each chart comes from the run folders through `scripts/bench_charts.py`. Hover values and data tables: [charts](docs/charts.html). Method, losses and caveats: [Results](docs/results.md).

**How to read the access in every chart.** *Weights only*: the agent reads the weights, as text or through Python, and the network never runs. Solid bars and filled dots. *With probes*: the agent also ran the network within a budget (forward passes, activations, practice observations). Hatched bars and hollow dots. A line under each chart, and the line under each heading below, says which is which. Full rules: [Access levels](docs/access-levels.md).

### WSPBench: OpenWiSH against WSPBench's own tools

**Access:** WSPBench B and the OpenWiSH sandbox are weights only. WSPBench C adds practice observations, so C is with probes (hatched).

<p align="center"><img src="docs/assets/charts/wspbench-tools.svg" alt="Grouped bars: exact solves out of 12 WSPBench tasks. WSPBench B and C score 0 or 1 for every model. The OpenWiSH sandbox scores 12 for gpt-6.1-sol, gpt-6.1-terra and claude-opus-5-5, and 4 for gpt-6-luna." width="760"></p>

[WSPBench](https://github.com/alexcarterhq/wspbench) defines its own tool conditions: weight-algebra tools in B, plus metered practice observations in C. The OpenWiSH sandbox gives Python over the same weights, with no network execution. Same models, same 12 tasks, one attempt each. A run with a failed audit scores 0.


OpenWiSH solves more tasks exactly than both WSPBench tool conditions, for each of the three models run in all three. Six of the 12 tasks were held out: the OpenWiSH config was frozen before they ran. B and C run in WSPBench's own Inspect loop, unchanged, and WSPBench's session budget shapes those results. [Results](docs/results.md) covers the method, the losses (AgenticInterpBench, two WSPBench families, Breakout) and how each point of the Hacker News discussion maps to a test.

### The audit decides the score

**Access:** weights only. Python over the weights, and the network never runs.

<p align="center"><img src="docs/assets/charts/wspbench-rounds.svg" alt="Lines: audited read agreement over five tuning rounds. All three models rise after round 4 added the activation guard. Sol and Terra reach 100 percent, Luna 69 percent." width="760"></p>

Codex tuned the OpenWiSH config in five rounds on the pilot tasks. Runs whose code evaluated gates on inputs fail audit and score 0. The activation guard in round 4 ended those runs, and the frozen config then held on tasks no round had seen.

### Blog environments

**Access:** weights only, one reply, no network runs. MinAtar starts from zero weights. The Kinetix prompt also shows one recorded failure episode.

<p align="center"><img src="docs/assets/charts/blog-envs.svg" alt="Four panels: Kinetix ties at 1,024 successes. Claude through OpenWiSH leads on Seaquest and Asterix, and trails on Breakout, 417 (mean of 3 runs) against 517." width="760"></p>

On the blog environments, Claude Opus 5.5 through OpenWiSH fixes the Kinetix car (1,024 / 1,024 episodes, 2,921 weights changed against the published 6,073) and beats the published one-shot MinAtar policies on Seaquest (60.9 against 53.1) and Asterix (60.4 against 53.2). Breakout at the task's own effort (xhigh) scores 476.3, 261.5 and 513.5 in three runs, a mean of 417 ± 79 (standard error over runs), against the published 517.4. An earlier single run at high effort scored 54.2. Single-reply policies vary a lot from run to run, so Breakout reports repeats. Seaquest, Asterix and Kinetix are single runs.

### Replaying the published Qwen edits

**Access:** with probes. The published edits were made with 10, 25 or 50 forward passes (hatched). The base model has no edit (solid).

<p align="center"><img src="docs/assets/charts/qwen-math.svg" alt="Grouped bars: Qwen MATH-500 accuracy, published against the OpenWiSH replay, for base, 10, 25 and 50 passes. The replay matches within one standard error band. A second panel shows the share cut at the token limit falling from 90 to 32 percent." width="760"></p>

All four published Qwen3.5-0.8B edits reproduce within noise on MATH-500. The 50-pass gain comes from ending reasoning loops: the share of answers cut at the token limit falls from 90 % to 32 %.

### Decompile: one network, three access levels

**Access:** `native` and `read` are weights only (the weights as text in the prompt, or through Python). `probe` adds 48 probe inputs with activations and patches, so `probe` is with probes (hatched).

<p align="center"><img src="docs/assets/charts/decompile-ladder.svg" alt="Grouped bars: decompile scores for two networks at three access levels. d2-a scores 0.98 at every level. d3-b scores 0.88 with the weights in the prompt, 0.91 with Python, and 0.94 with Python and 48 probe inputs." width="760"></p>

Claude Opus 5.5 rewrites a trained network as a short program whose variables map onto its hidden units, with the network's weights as text in one reply, through Python, or through Python plus 48 probe inputs. Reading the weights in one reply already recovers most of each mechanism. On the harder network, each extra channel adds a little. Single runs.

### AgenticInterpBench: HyVE wins

**Access:** prompt-only and weights-only rows are filled dots, with no network runs. Weights + probe rows and the published HyVE agent are hollow dots: they ran the network.

<p align="center"><img src="docs/assets/charts/agentic-interp.svg" alt="Dot plot: AgenticInterpBench tag accuracy. Published HyVE agents score 67 to 79 percent. OpenWiSH rows score 67 to 73 percent. Fixed baselines sit at 44.2 and 63.8 percent." width="760"></p>

On 84 InterpBench circuits, the published HyVE agent beats OpenWiSH with the same model: Sonnet 4.6 scores 79.0 % in HyVE and 61.3 % in OpenWiSH with the same access (68.7 % before audit). In OpenWiSH, Sonnet answered most cases after one Python run, close to its prompt-only reply (69.9 %). HyVE walks every component through an observe, hypothesise and validate loop. For Sol, tools add nothing: 73.0 % prompt only, 71.8 % with weights and probes.

## Quickstart

You need Linux, Rust stable, Python 3 and git. These commands replay a published answer on the game-policy environment. No API key, no GPU.

```bash
git clone https://github.com/Alexander-Aghili/OpenWiSH && cd OpenWiSH
nice -n 10 cargo build --release -j 4
scripts/prepare_logs.sh && python3 scripts/prepare_tasks.py exp1
python3 scripts/replay_authors.py assets/lm_understand-logs/exp1_which_game/runs/breakout breakout.txt
target/release/openwish run tasks/exp1/a9f535cc.toml --backend scripted:breakout.txt
```

The run prints the grade (`"correct": true`, `"score": 1.0`) and writes every turn to `runs/exp1-a9f535cc-<time>/`. Leave out `--backend` to put a live agent on the same network. The default agent is `claude -p` with `claude-opus-5-5`, its own tools switched off.

Next steps: [Getting started](docs/getting-started.md) sets up the Python environments and the assets for every environment. The [walkthrough](docs/walkthrough.html) follows one run turn by turn.

## The three layers

<p align="center">
  <img src="docs/assets/layers.svg" alt="Three layers, three interfaces. Read: load(name), NAMES, weights as text, budgeted forward probes. Write: save_edit, drop_edit, checkpoint, restore, the edits language, the edit ops, the gate, the ledger and edit sets. Harness: openwish run, resume, status, replay, score and audit, budgets, handoff, retries, crash-resume, the sandbox, static checks and the auditor." width="880">
</p>

| Layer | Question | Interface | Crates |
|---|---|---|---|
| **Read** | What is in this network, and what does the network do? | `load(name)`, `NAMES`, weights as text (decimals, or integers × one scale), budgeted forward probes (`games`, `positions`, `attention`, `forward`) | `openwish-tensor`, `openwish-sandbox`, `python/adapters` |
| **Write** | What changed, why, and how do you undo or ship the change? | `save_edit`, `drop_edit`, `checkpoint`, `restore`, `edits()`, the ```` ```edits ```` language, ops `assign`, `replace`, `add_rank1`, `project_out`, `revert`, a per-task policy, the ledger, edit sets | `openwish-tensor` |
| **Harness** | How does any agent work here for hours, within budget, with results you check? | `openwish run`, `--resume`, `status`, `replay`, `score`, `audit`, budgets and shared pools, nudges, context handoff, retries, crash-resume | `openwish-core`, `openwish-llm`, `openwish-sandbox`, `openwish-audit`, `openwish-envs`, `openwish-cli` |

Crates, data flow, the reply protocol and the run folder: [Architecture](docs/architecture.md).

## Environments

Each environment is a TOML task, a prompt and a scorer. The bundled set follows the experiments of *Sparks of Weight-Space Programming*.

| Environment | Job | Access | Scored by |
|---|---|---|---|
| `tasks/exp1` game-policy MLPs | Name the game | all 17,253 weights in the prompt · no code · never run · one reply | game named |
| `tasks/exp2` diffusion MLPs | Name the shape | all 23,824 weights in the prompt · no code · never run · one reply | shape named |
| `tasks/exp3` char-level transformers | Name the corpus from the middle blocks | weights through code · Python, 100 runs · never run | corpus, copying-head precision and recall |
| `tasks/exp4` Kinetix car policy | Fix a failing policy | weights in the prompt · no code · never run · shown: 256-episode statistics and a sampled failure episode · one reply with an ```` ```edits ```` block | successes over 1,024 episodes |
| `tasks/exp5` MinAtar Breakout, Seaquest, Asterix | Write a 600-32-32-6 policy from zero | from zero weights · no code · never run · one reply with an ```` ```edits ```` block | mean return over 1,000 episodes |
| `tasks/exp6` 9x9 Go | Write a 1.48M-weight policy from zero | from zero weights · Python, 120 runs · `games` and `positions` probes (0, 10, 25 or 50) | win rate against the published same-size PPO networks |
| `tasks/exp7` Gemma 3 270M pairs | Name the sycophantic fine-tune | weights through code · numpy, 20 runs with an `attention` probe, or 60 runs on weights only | sycophant named |
| `tasks/exp8` Qwen3.5-0.8B | Raise math accuracy by editing weights | weights through code · Python, 300 runs · `forward` probe (10, 25 or 50) on a local or remote GPU · data: practice problems | MATH-500, 500 problems |
| `tasks/decompile` small ReLU MLPs trained on secret 10-bit programs | Rewrite the network as a short program whose variables map onto its hidden units | a ladder: `native` (weights in the prompt, no code, one reply), `read` (Python, never run), `probe` (Python, 3 forward/patch passes of 16 inputs) | behaviour on all 1,024 inputs × interchange-intervention accuracy × completeness |
| `tasks/toy` | Smoke test | weights through code · Python, 3 runs | identity written |

Setup per environment and a guide to writing your own: [Environments](docs/environments.md).

## Benchmarks

Conformance comes first. OpenWiSH replays the published runs as a scripted agent: their code runs in the OpenWiSH sandbox, their edits pass the OpenWiSH gate, and the OpenWiSH scorers grade their answers.

<p align="center"><img src="docs/assets/charts/conformance.svg" alt="Dot plot: each published result replayed through OpenWiSH as a share of the published value. Nine checks sit at 100 percent. MinAtar Breakout sits at 97 percent, 502.9 against 517.4, within sampling noise." width="760"></p>

The Qwen edits replay within noise as well (chart above). The full conformance table, with every check, lives in [Benchmarks](docs/benchmarks.md).

Methodology, the full table, known deviations and live agent results: [Benchmarks](docs/benchmarks.md). Every chart with hover values and data tables: [charts](docs/charts.html).

## Safety model

Every weight change passes one gate, and four layers keep agents editing instead of optimising.

1. **Capability removal (hard).** The kernel holds arrays, never a runnable model. `backward`, `autograd.grad`, `torch.optim` and `requires_grad_(True)` raise. The sandbox has no network and no GPU, the weights are read-only, and hidden scoring assets never mount.
2. **Edit gate (hard).** The Rust gate checks the task policy: allowed and forbidden tensors, no whole-tensor constant scaling, a maximum fraction changed, finite values only.
3. **Static checks (soft).** Each code run is scanned for backprop, optimisers, `scipy.optimize`, gradient steps, fitting fed into `save_edit`, and forward passes. Matches are flagged, not blocked.
4. **Auditor (soft).** `openwish audit` sends every code run, with the rules the agent saw, to a reviewer model. One `violation` marks the run invalid.

Path rules, sandbox mounts and the known gaps: [Safety](docs/safety.md). Report a sandbox escape through [SECURITY.md](SECURITY.md).

## Documentation

| Guide | Covers |
|---|---|
| [Getting started](docs/getting-started.md) | Install, Python environments, assets, first runs |
| [Architecture](docs/architecture.md) | Crates, data flow, reply protocol, run folder |
| [Environments](docs/environments.md) | Bundled environments, task TOML, preludes, probes, scorers |
| [Agents](docs/agents.md) | Backends, the `Editor` trait, adding a model or agent |
| [Safety](docs/safety.md) | Enforcement layers, path rules, known gaps |
| [Access levels](docs/access-levels.md) | What each channel (weights as text, code, network runs, shown behaviour, data) is evidence of, and the reporting rules |
| [Decompile design](docs/decompile.md) | The mechanistic-interpretability environments: networks, scoring, calibration, tiers |
| [Interpretability tools](interp/README.md) | Tools for people, kept apart from OpenWiSH runs: an MLP explorer, neuron-explorer |
| [Results](docs/results.md) | Tools against tools, losses, the Hacker News points mapped to tests |
| [Benchmarks](docs/benchmarks.md) | Conformance, methodology, live results |
| [Configuration](docs/configuration.md) | Environment variables, CLI flags, remote GPU |
| [Roadmap](docs/roadmap.md) | Parity with coding agents, next environments |
| [Design decisions](docs/harness-decisions.html) | Every design call, with the alternative and the cost |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, the test commands and the good first issues. Everyone in the project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

OpenWiSH grew out of *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026). The published runs live at [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

```bibtex
@misc{takehi2026sparks,
  title        = {Sparks of Weight-Space Programming},
  author       = {Takehi and Teoh and Zhang and Khattab},
  howpublished = {MIT CSAIL},
  year         = {2026},
  note         = {Published runs: https://github.com/RikiyaT/lm_understand-logs}
}

@software{openwish,
  title   = {OpenWiSH: Open Weight-Space Harness},
  author  = {Aghili, Alexander},
  url     = {https://github.com/Alexander-Aghili/OpenWiSH},
  version = {0.1.0},
  year    = {2026}
}
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The prepare scripts download the published networks, prompts and data (CC BY 4.0) and the gated Gemma base model. The MinAtar source quoted in the MinAtar prompts is GPL-3.0. None of these files ship in the repository.
