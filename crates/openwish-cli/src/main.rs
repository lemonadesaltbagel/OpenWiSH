//! `openwish`: OpenWiSH, the open Weight-Space Programming harness.
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use openwish_core::config::{self, ProjectConfig};
use openwish_core::run::{saved_spec, Env, Run, Step};
use openwish_core::spec::{EditorDefaults, TaskSpec};

mod envs;
mod go;
mod html;
mod init;
mod replay;
mod status;

use go::GoCmd;

/// How long a scorer may run.
const SCORE_TIMEOUT_S: u64 = 6 * 3600;

#[derive(Parser)]
#[command(
    name = "openwish",
    about = "OpenWiSH (Open Weight-Space Harness): an agent reads and writes neural-network weights directly."
)]
struct Cli {
    /// project config file (default: $OPENWISH_CONFIG, else openwish.toml at the workspace root)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// CPU threads for OpenWiSH and its child processes (beats OPENWISH_THREADS and openwish.toml)
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// process priority, 0-19 (beats OPENWISH_NICE and openwish.toml)
    #[arg(long, global = true)]
    nice: Option<i32>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct EditorArgs {
    /// claude | codex | api | openai (OpenAI-compatible: OpenRouter, LM Studio, llama.cpp, vLLM) | scripted:<file>
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    model: Option<String>,
    /// reasoning effort (low | medium | high | xhigh | max)
    #[arg(long)]
    effort: Option<String>,
    /// max output tokens per reply (api backend)
    #[arg(long, default_value_t = 64000)]
    max_tokens: u32,
    /// seconds to wait for one reply
    #[arg(long, default_value_t = 7200)]
    timeout: u64,
}

#[derive(clap::Args)]
struct RunArgs {
    task: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[command(flatten)]
    editor: EditorArgs,
    /// do not score the run when it ends
    #[arg(long)]
    no_score: bool,
    /// start a new conversation (a handoff) once the editor's context reaches this many tokens, in place of the
    /// task's [handoff] max_conversation_tokens. Set the value below the editor model's context window. New runs only
    #[arg(long)]
    max_conversation_tokens: Option<u64>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a task (or resume a run with --resume <run dir>)
    Run(RunArgs),
    /// Score a finished run with its task's scorer
    Score { run: PathBuf },
    /// Review every code run of a run against the task's rules with an auditor model
    Audit {
        run: PathBuf,
        #[command(flatten)]
        editor: EditorArgs,
        #[arg(long, default_value_t = 8)]
        batch: usize,
    },
    /// Send stdin as one message through a backend and print the reply as one JSON line (for external harnesses)
    Ask {
        #[command(flatten)]
        editor: EditorArgs,
        /// a file with the system instructions
        #[arg(long)]
        instructions: Option<PathBuf>,
        /// an empty folder the backend CLI runs in
        #[arg(long)]
        workdir: PathBuf,
    },
    /// Print a run turn by turn, or write it as one HTML page
    Replay {
        run: PathBuf,
        /// characters shown per message (0 = all)
        #[arg(long, default_value_t = 1500)]
        width: usize,
        /// write a self-contained HTML transcript to this file instead
        #[arg(long)]
        html: Option<PathBuf>,
    },
    /// Start a new environment or config file from a template
    Init {
        #[command(subcommand)]
        what: InitCmd,
    },
    /// List environments: id, tools, budgets, and whether their files exist
    Envs {
        #[arg(long, default_value = "tasks")]
        dir: PathBuf,
    },
    /// Print the tools, budgets, pools, edit policy and hooks an agent gets in a task
    Tools { task: PathBuf },
    /// Print a run's state
    Status { run: PathBuf },
    /// 9x9 Go tools for Experiment 6 (pass servers, evaluation, opponent training)
    Go {
        #[command(subcommand)]
        cmd: GoCmd,
    },
    /// Print a network's tensors as text
    Render {
        net: PathBuf,
        #[arg(long, default_value_t = 3)]
        decimals: usize,
        /// integers times one scale with this many significant digits (the authors' Experiment 4 format)
        #[arg(long)]
        int_scale: Option<u32>,
        /// only these tensors
        #[arg(long)]
        tensor: Vec<String>,
    },
}

#[derive(Subcommand)]
enum InitCmd {
    /// Write an environment folder that runs end to end with `--backend scripted:<dir>/script.txt`
    Env {
        dir: PathBuf,
        /// the task id (default: the folder name)
        #[arg(long)]
        id: Option<String>,
    },
    /// Write a commented openwish.toml (default: at the workspace root)
    Config { path: Option<PathBuf> },
}

/// The workspace root: OPENWISH_ROOT, or the nearest ancestor of the cwd or of this binary holding python/kernel/kernel.py.
fn root() -> Result<PathBuf> {
    if let Ok(r) = std::env::var("OPENWISH_ROOT") {
        return Ok(PathBuf::from(r));
    }
    let mut starts = vec![std::env::current_dir()?];
    if let Ok(e) = std::env::current_exe() {
        starts.push(e);
    }
    for s in starts {
        for a in s.ancestors() {
            if a.join("python/kernel/kernel.py").exists() {
                return Ok(a.to_path_buf());
            }
        }
    }
    bail!("cannot find the OpenWiSH workspace (set OPENWISH_ROOT)")
}

fn env(cfg: &ProjectConfig, threads: usize) -> Result<Env> {
    let r = root()?;
    Ok(Env {
        venv: r.join("assets/venvs/sandbox"),
        kernel_dir: r.join("python/kernel"),
        retry_base_s: 10.0,
        max_retries: 8,
        threads: threads as u32,
        // project hooks run in the folder of openwish.toml
        hooks: openwish_core::hooks::HookSet::from_spec(&cfg.hooks, &cfg.dir, "openwish.toml"),
    })
}

/// The agent backend. Precedence (see openwish_core::config): flags, then the task's [editor], then
/// openwish.toml, then the built-in defaults.
fn editor(
    a: &EditorArgs,
    spec: &TaskSpec,
    run_dir: &Path,
    pc: &ProjectConfig,
) -> Result<Box<dyn openwish_llm::Editor>> {
    let flags = EditorDefaults {
        backend: a.backend.clone(),
        model: a.model.clone(),
        effort: a.effort.clone(),
    };
    let choice = config::editor_choice(&flags, &spec.editor, pc);
    let cfg = openwish_llm::BackendConfig {
        kind: choice.backend,
        model: choice.model,
        effort: choice.effort,
        instructions: spec.instructions.clone(),
        tool_blocks: spec.enabled_tools(),
        final_blocks: if spec.edits.language {
            vec!["edits".into()]
        } else {
            vec![]
        },
        workdir: run_dir.join("editor_cwd"),
        max_tokens: a.max_tokens,
        timeout_s: a.timeout,
    };
    openwish_llm::make(&cfg)
}

/// One message through a backend with its own tools off, as in a run. Prints {text, usage, wall_s}.
fn ask(a: &EditorArgs, instructions: Option<&Path>, workdir: PathBuf, pc: &ProjectConfig) -> Result<()> {
    use std::io::Read;
    let flags = EditorDefaults {
        backend: a.backend.clone(),
        model: a.model.clone(),
        effort: a.effort.clone(),
    };
    let choice = config::editor_choice(&flags, &EditorDefaults::default(), pc);
    let instructions = match instructions {
        Some(f) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
        None => String::new(),
    };
    std::fs::create_dir_all(&workdir)?;
    let cfg = openwish_llm::BackendConfig {
        kind: choice.backend,
        model: choice.model,
        effort: choice.effort,
        instructions,
        tool_blocks: vec![],
        final_blocks: vec![],
        workdir,
        max_tokens: a.max_tokens,
        timeout_s: a.timeout,
    };
    let mut msg = String::new();
    std::io::stdin().read_to_string(&mut msg)?;
    let reply = openwish_llm::make(&cfg)?
        .send(&msg, true)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let out = serde_json::json!({"text": reply.text, "usage": reply.usage, "context_tokens": reply.context_tokens,
        "wall_s": reply.wall_s, "tool_free": reply.tool_free});
    println!("{out}");
    Ok(())
}

fn score(run: &Path) -> Result<serde_json::Value> {
    let spec = saved_spec(run)?;
    let Some(sc) = &spec.score else {
        bail!("task {} has no scorer", spec.id);
    };
    let run = std::fs::canonicalize(run)?;
    let envs = [
        ("OPENWISH_RUN", run.display().to_string()),
        ("OPENWISH_TASK_DIR", spec.dir.display().to_string()),
        ("OPENWISH_FINAL_NET", run.join("final/net").display().to_string()),
        (
            "OPENWISH_FINAL_ANSWER",
            run.join("final_answer.md").display().to_string(),
        ),
        ("OPENWISH_ROOT", root()?.display().to_string()),
    ];
    let mut v = openwish_core::forward::run_command(&spec.argv(&sc.command), &envs, &spec.dir, SCORE_TIMEOUT_S)?;
    // every score names the access it was earned under (scores from different access levels are not comparable)
    if let Some(o) = v.as_object_mut() {
        o.insert("access".into(), spec.access());
    }
    std::fs::create_dir_all(run.join("eval"))?;
    std::fs::write(run.join("eval/score.json"), serde_json::to_string_pretty(&v)?)?;
    Ok(v)
}

/// Hold OpenWiSH and its children to a share of the machine: a thread count (a quarter of the cores by
/// default) and a lower priority. Precedence: --threads / --nice, then OPENWISH_THREADS / OPENWISH_NICE, then
/// openwish.toml. Returns the thread count.
fn limit_machine(cli: &Cli, pc: &ProjectConfig) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let threads = config::threads(
        cli.threads,
        std::env::var("OPENWISH_THREADS").ok().as_deref(),
        pc,
        cores,
    );
    for (k, v) in openwish_core::cpu::child_env(threads) {
        // OPENWISH_THREADS always carries the resolved count; a library variable the user set is kept
        if k == "OPENWISH_THREADS" || std::env::var(k).is_err() {
            std::env::set_var(k, v);
        }
    }
    let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    let nice = config::nice(cli.nice, std::env::var("OPENWISH_NICE").ok().as_deref(), pc);
    // SAFETY: setpriority on our own process with constant arguments; children inherit it
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, nice);
    }
    threads
}

fn print_json(v: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // the project config: --config, else $OPENWISH_CONFIG, else openwish.toml at the workspace root (optional)
    let explicit = cli
        .config
        .clone()
        .or_else(|| std::env::var_os("OPENWISH_CONFIG").map(PathBuf::from));
    let pc = ProjectConfig::find(explicit.as_deref(), root().ok().as_deref())?;
    let threads = limit_machine(&cli, &pc);
    // {root} in task commands resolves against the workspace for every subcommand
    if std::env::var("OPENWISH_ROOT").is_err() {
        if let Ok(r) = root() {
            std::env::set_var("OPENWISH_ROOT", r);
        }
    }
    match cli.cmd {
        Cmd::Run(a) => run_task(a, &pc, threads),
        Cmd::Score { run } => print_json(&score(&run)?),
        Cmd::Audit { run, editor, batch } => audit(&run, editor, batch, &pc),
        Cmd::Ask {
            editor,
            instructions,
            workdir,
        } => ask(&editor, instructions.as_deref(), workdir, &pc),
        Cmd::Replay {
            run, html: Some(out), ..
        } => {
            std::fs::write(&out, html::render(&run)?).with_context(|| format!("writing {}", out.display()))?;
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Replay { run, width, html: None } => {
            print!("{}", replay::text(&run, width)?);
            Ok(())
        }
        Cmd::Status { run } => {
            print!("{}", status::report(&run)?);
            Ok(())
        }
        Cmd::Go { cmd } => go::run(cmd),
        Cmd::Init { what } => init_cmd(what),
        Cmd::Envs { dir } => list_envs(dir),
        Cmd::Tools { task } => {
            print!("{}", envs::tools(&TaskSpec::parse(&task)?, &task, &pc));
            Ok(())
        }
        Cmd::Render {
            net,
            decimals,
            int_scale,
            tensor,
        } => render(&net, decimals, int_scale, &tensor),
    }
}

/// `openwish run`: create or resume a run, step it to the end with one status line per turn, then score it.
fn run_task(a: RunArgs, pc: &ProjectConfig, threads: usize) -> Result<()> {
    let mut run = if let Some(dir) = a.resume {
        if a.max_conversation_tokens.is_some() {
            bail!("--max-conversation-tokens applies to a new run. A resumed run keeps the setting saved in spec.json");
        }
        let ed = editor(&a.editor, &saved_spec(&dir)?, &dir, pc)?;
        Run::resume(&dir, ed, env(pc, threads)?)?
    } else {
        let task = a.task.context("give a task file, or --resume <run dir>")?;
        let mut spec = TaskSpec::load(&task)?;
        if let Some(n) = a.max_conversation_tokens {
            spec.handoff.max_conversation_tokens = n;
        }
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let dir = a
            .out
            .unwrap_or_else(|| PathBuf::from(format!("runs/{}-{stamp}", spec.id)));
        let ed = editor(&a.editor, &spec, &dir, pc)?;
        eprintln!("run {} -> {}  (editor {})", spec.id, dir.display(), ed.name());
        Run::create(spec, &dir, ed, env(pc, threads)?)?
    };
    // one status line per turn: turn, budgets, edits so far, cost, elapsed
    let t0 = std::time::Instant::now();
    loop {
        let step = run.step()?;
        let edits = run.gate().map(|g| g.edits().count()).unwrap_or(0);
        eprintln!("{}", status::line(&run.spec, &run.state, edits, t0.elapsed()));
        if step == Step::Done {
            break;
        }
    }
    eprintln!(
        "done: {} ({} turns)",
        run.state.end_reason.clone().unwrap_or_default(),
        run.state.turn
    );
    if !a.no_score && run.spec.score.is_some() && !run.dir.join("eval/score.json").exists() {
        print_json(&score(&run.dir)?)?;
    }
    Ok(())
}

/// `openwish audit`: review every code run with an auditor model (medium effort unless given).
fn audit(run: &Path, mut ea: EditorArgs, batch: usize, pc: &ProjectConfig) -> Result<()> {
    let mut spec = saved_spec(run)?;
    // Participant instructions are evidence for the reviewer, not the reviewer's role.
    spec.instructions = openwish_llm::NEUTRAL_INSTRUCTIONS.into();
    spec.python = None;
    spec.forward.clear();
    spec.edits.language = false;
    ea.effort.get_or_insert_with(|| "medium".into());
    let mut ed = editor(&ea, &spec, &run.join("audit_cwd"), pc)?;
    print_json(&openwish_audit::audit_run(run, ed.as_mut(), batch)?)
}

fn init_cmd(what: InitCmd) -> Result<()> {
    match what {
        InitCmd::Env { dir, id } => {
            init::env(&dir, id.as_deref())?;
            let d = dir.display();
            println!("wrote environment {d}\ntry it: openwish run {d}/task.toml --backend scripted:{d}/script.txt");
        }
        InitCmd::Config { path } => {
            let path = match path {
                Some(p) => p,
                None => root()?.join(config::FILE_NAME),
            };
            init::config(&path)?;
            println!("wrote {}", path.display());
        }
    }
    Ok(())
}

fn list_envs(dir: PathBuf) -> Result<()> {
    // a relative folder that is not here is looked up in the workspace
    let dir = if dir.is_relative() && !dir.exists() {
        root().map(|r| r.join(&dir)).unwrap_or(dir)
    } else {
        dir
    };
    if !dir.is_dir() {
        bail!("{} is not a folder", dir.display());
    }
    print!("{}", envs::list(&dir));
    Ok(())
}

/// `openwish render`: every tensor (or the chosen ones) as decimals, or as integers times one scale.
fn render(net: &Path, decimals: usize, int_scale: Option<u32>, only: &[String]) -> Result<()> {
    let n = openwish_tensor::Network::load(net)?;
    for (name, t) in n
        .tensors
        .iter()
        .filter(|(name, _)| only.is_empty() || only.contains(name))
    {
        let shape = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" x ");
        match int_scale {
            Some(sig) => {
                let (e, text) = openwish_tensor::render::int_scale(t, sig);
                println!("{name} ({shape}), scale 1e{e}: one line per row (row i)\n{text}");
            }
            None => println!(
                "{name} ({shape}):\n{}",
                openwish_tensor::render::tensor_text(t, decimals)
            ),
        }
    }
    Ok(())
}
