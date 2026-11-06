//! One run of one task: the turn loop, tools, budgets, handoff, finalisation and crash-resume.
//!
//! A turn sends the next message, saves the reply as pending, acts on it (a Python run, a forward probe or the
//! final answer) and writes the message for the turn after. The state is saved between these steps, so a run
//! that stops at any point resumes without losing or repeating a reply.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::hooks::{Context as HookCtx, Event, HookSet, Outcome};
use crate::paths;
use crate::protocol::{parse, Action};
use crate::spec::{ForwardSpec, PythonSpec, TaskSpec};
use openwish_audit::Flag;
use openwish_llm::{BackendError, Editor, Reply};
use openwish_sandbox::{Kernel, KernelCfg, Rpc, RpcReply, RunOutput};
use openwish_tensor::gate::{EditRecord, Gate, Refusal};
use openwish_tensor::ops::EditOp;
use openwish_tensor::Network;

/// The budget group of tools that draw from the Python runs.
const PYTHON_POOL: &str = "python";
/// Consecutive refusals after which the run ends.
const MAX_REFUSAL_STREAK: u32 = 3;
/// The longest wait between two retries of a failing backend.
const MAX_BACKOFF_S: f64 = 1800.0;
/// Characters of /work/notes.md shown in a handoff.
const MAX_NOTES_CHARS: usize = 20_000;
/// Consecutive empty replies after which the run ends.
const MAX_EMPTY_STREAK: u32 = 3;
const MAX_SUBMISSION_BYTES: u64 = 2_000_000;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct State {
    pub turn: u32,
    pub runs_used: u32,
    pub passes_used: BTreeMap<String, u32>,
    pub failed_requests: u32,
    pub protocol_errors: u32,
    pub nudges: u32,
    pub handoffs: u32,
    pub since_handoff: u32,
    pub conv_tokens: u64,
    pub session: Option<String>,
    pub next_message: String,
    pub new_conversation: bool,
    /// a reply received but not yet acted on (crash between receiving and acting)
    pub pending_reply: Option<String>,
    /// the last edit id before the pending reply was acted on (to tell which edits an interrupted run made)
    pub ledger_mark: Option<u64>,
    /// start a new conversation after the next action (a backend that cannot resume a session)
    pub force_handoff: bool,
    pub refusals: u32,
    pub refusal_streak: u32,
    pub backend_errors: u32,
    pub kernel_restarts: u32,
    pub done: bool,
    pub end_reason: Option<String>,
    pub usage_total: openwish_llm::Usage,
    pub cost_usd: f64,
    pub tool_free_turns: u32,
    /// tool calls a pre_tool hook blocked (not charged)
    pub hook_blocks: u32,
    /// replies with no text at all (not charged, not a final answer)
    pub empty_replies: u32,
    pub empty_streak: u32,
    /// tool blocks sent back in a row for a missing explanation ([rationale])
    pub rationale_streak: u32,
    /// tool blocks sent back for a missing explanation, and blocks run without one after a full streak
    pub rationale_rejects: u32,
    pub rationale_missing_run: u32,
}

/// Machine-level settings (not part of a task).
#[derive(Clone, Debug)]
pub struct Env {
    pub venv: PathBuf,
    pub kernel_dir: PathBuf,
    pub retry_base_s: f64,
    pub max_retries: u32,
    /// the CPU budget: the sandbox kernel gets at most this many threads
    pub threads: u32,
    /// project hooks (openwish.toml); the task's own hooks are added to these
    pub hooks: crate::hooks::HookSet,
}

#[derive(Debug, PartialEq)]
pub enum Step {
    Continue,
    Done,
}

pub struct Run {
    pub spec: TaskSpec,
    pub dir: PathBuf,
    pub state: State,
    pub editor: Box<dyn Editor>,
    pub env: Env,
    gate: Option<Gate>,
    kernel: Option<Kernel>,
    first_message: String,
    /// project hooks, then the task's own
    hooks: HookSet,
    /// session_start / session_end already ran in this process
    started: bool,
    ended: bool,
}

/// The task spec as saved in a run folder (`spec.json`), with the folder it was loaded from.
#[derive(Serialize, Deserialize)]
struct SavedSpec {
    spec: TaskSpec,
    dir: PathBuf,
}

/// The task spec saved in the run folder `run_dir`.
pub fn saved_spec(run_dir: &Path) -> Result<TaskSpec> {
    let text = std::fs::read_to_string(run_dir.join("spec.json")).context("not a run folder (no spec.json)")?;
    let saved: SavedSpec = serde_json::from_str(&text)?;
    Ok(TaskSpec {
        dir: saved.dir,
        ..saved.spec
    })
}

// Creating, resuming and saving a run.
impl Run {
    pub fn create(spec: TaskSpec, dir: &Path, editor: Box<dyn Editor>, env: Env) -> Result<Run> {
        if dir.join("state.json").exists() {
            bail!("{} already holds a run; use --resume", dir.display());
        }
        for sub in ["turns", "work/.rpc", "requests"] {
            std::fs::create_dir_all(dir.join(sub))?;
        }
        // tools run in other folders, so every path handed to them must be absolute
        let dir = std::fs::canonicalize(dir)?;
        let first = first_message(&spec)?;
        std::fs::write(dir.join("rules.txt"), &first)?;
        let saved = SavedSpec {
            spec: spec.clone(),
            dir: spec.dir.clone(),
        };
        std::fs::write(dir.join("spec.json"), serde_json::to_string_pretty(&saved)?)?;
        let state = State {
            next_message: first.clone(),
            new_conversation: true,
            ..Default::default()
        };
        let r = Run::open(spec, dir, state, first, editor, env)?;
        r.log(&format!(
            "created run of task {} with editor {}; access: {}",
            r.spec.id,
            r.editor.name(),
            r.spec.access()["label"].as_str().unwrap_or("")
        ));
        r.save_state()?;
        Ok(r)
    }

    pub fn resume(dir: &Path, mut editor: Box<dyn Editor>, env: Env) -> Result<Run> {
        let dir = std::fs::canonicalize(dir).with_context(|| format!("{}", dir.display()))?;
        std::fs::create_dir_all(dir.join("work/.rpc"))?;
        let spec = saved_spec(&dir)?;
        let mut state: State =
            serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).context("state.json")?)?;
        editor.set_session(state.session.clone());
        let first = std::fs::read_to_string(dir.join("rules.txt"))?;
        // the kernel's variables did not survive the restart
        if state.runs_used > 0 && state.pending_reply.is_none() && !state.done {
            state.next_message = format!(
                "[The harness was restarted: the Python kernel's variables were lost; your saved edits and files in \
                 /work are kept.]\n\n{}",
                state.next_message
            );
        }
        let mut r = Run::open(spec, dir, state, first, editor, env)?;
        if !r.editor.can_resume() && !r.state.done {
            // the backend keeps no conversation of its own: continue in a new one that restates the task
            if r.state.pending_reply.is_some() {
                r.state.force_handoff = true;
            } else {
                r.start_handoff();
            }
        }
        r.log("resumed");
        r.save_state()?;
        Ok(r)
    }

    /// A run over `dir` in `state`, with its hooks and its edit gate.
    fn open(
        spec: TaskSpec,
        dir: PathBuf,
        state: State,
        first_message: String,
        editor: Box<dyn Editor>,
        env: Env,
    ) -> Result<Run> {
        let hooks = all_hooks(&env, &spec);
        let gate = match &spec.network {
            Some(n) => Some(Gate::open(
                &dir.join("edits"),
                Network::load(&spec.resolve(&n.base))?,
                spec.edits.policy.clone(),
            )?),
            None => None,
        };
        Ok(Run {
            spec,
            dir,
            state,
            editor,
            env,
            gate,
            kernel: None,
            first_message,
            hooks,
            started: false,
            ended: false,
        })
    }

    pub fn save_state(&self) -> Result<()> {
        let tmp = self.dir.join("state.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.state)?)?;
        std::fs::rename(tmp, self.dir.join("state.json"))?;
        Ok(())
    }

    pub fn gate(&self) -> Option<&Gate> {
        self.gate.as_ref()
    }
    pub fn gate_mut(&mut self) -> Option<&mut Gate> {
        self.gate.as_mut()
    }
    fn last_edit_id(&self) -> Option<u64> {
        self.gate.as_ref().map(Gate::last_edit_id)
    }

    fn log(&self, line: &str) {
        let _ = append_line(
            &self.dir.join("loop.log"),
            &format!("[{}] {line}", chrono::Local::now().to_rfc3339()),
        );
    }

    fn tool_log(&self, v: Value) -> Result<()> {
        append_line(&self.dir.join("tool_calls.jsonl"), &v.to_string())?;
        Ok(())
    }
}

// Budgets.
impl Run {
    pub fn budget_line(&self) -> String {
        let mut parts = vec![];
        if let Some(py) = &self.spec.python {
            parts.push(format!("python runs used {}/{}", self.state.runs_used, py.runs));
        }
        for (g, label, n) in self.spec.groups() {
            parts.push(format!("{label} used {}/{n}", self.passes_used(&g)));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("[budget: {}]", parts.join("; "))
        }
    }

    fn passes_used(&self, group: &str) -> u32 {
        self.state.passes_used.get(group).copied().unwrap_or(0)
    }

    /// (label, passes left) for every pass budget that is not used up.
    fn passes_left(&self) -> Vec<(String, u32)> {
        self.spec
            .groups()
            .into_iter()
            .map(|(g, label, n)| (label, n.saturating_sub(self.passes_used(&g))))
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    /// (used, allowed) for a tool of budget group `group`; the Python pool counts Python runs.
    fn tool_budget(&self, group: &str, f: &ForwardSpec) -> (u32, u32) {
        if group == PYTHON_POOL {
            (
                self.state.runs_used,
                self.spec.python.as_ref().map(|p| p.runs).unwrap_or(0),
            )
        } else {
            (self.passes_used(group), f.passes)
        }
    }

    /// Record that use number `k` of `group` succeeded.
    fn charge(&mut self, group: &str, k: u32) {
        if group == PYTHON_POOL {
            self.state.runs_used = k;
        } else {
            self.state.passes_used.insert(group.to_string(), k);
        }
    }
}

// The turn loop.
impl Run {
    /// One turn. The first step of a process runs the session_start hooks; the step that ends the run runs
    /// the session_end hooks.
    pub fn step(&mut self) -> Result<Step> {
        if !self.started && !self.state.done {
            self.started = true;
            self.fire(Event::SessionStart, &self.hook_ctx(""));
        }
        let step = self.next_step()?;
        if step == Step::Done && self.started && !self.ended {
            self.ended = true;
            let mut c = self.hook_ctx("");
            c.extra.push((
                "OPENWISH_END_REASON".into(),
                self.state.end_reason.clone().unwrap_or_default(),
            ));
            self.fire(Event::SessionEnd, &c);
        }
        Ok(step)
    }

    pub fn run_to_end(&mut self) -> Result<()> {
        while self.step()? == Step::Continue {}
        Ok(())
    }

    fn next_step(&mut self) -> Result<Step> {
        if self.state.done {
            return Ok(Step::Done);
        }
        if let Some(text) = self.state.pending_reply.clone() {
            return self.act_on_pending(&text);
        }
        if self.state.turn >= self.spec.finish.max_turns {
            self.finalize("", "turn limit reached")?;
            return Ok(Step::Done);
        }
        if let Some(why) = self.editor_cap_reached() {
            self.finalize("", &why)?;
            self.save_state()?;
            return Ok(Step::Done);
        }
        self.take_turn()
    }

    /// The per-episode editor cap the run has reached, if any (`[finish] max_editor_*`).
    fn editor_cap_reached(&self) -> Option<String> {
        let f = &self.spec.finish;
        let u = &self.state.usage_total;
        if f.max_editor_output_tokens > 0 && u.output >= f.max_editor_output_tokens {
            return Some(format!(
                "editor output-token cap reached ({} >= {})",
                u.output, f.max_editor_output_tokens
            ));
        }
        let total = u.context_tokens();
        if f.max_editor_total_tokens > 0 && total >= f.max_editor_total_tokens {
            return Some(format!(
                "editor total-token cap reached ({total} >= {})",
                f.max_editor_total_tokens
            ));
        }
        if f.max_editor_cost_usd > 0.0 && self.state.cost_usd >= f.max_editor_cost_usd {
            return Some(format!(
                "editor cost cap reached (${:.2} >= ${:.2})",
                self.state.cost_usd, f.max_editor_cost_usd
            ));
        }
        None
    }

    /// A reply that arrived before a crash is acted on first.
    fn act_on_pending(&mut self, text: &str) -> Result<Step> {
        let is_python =
            matches!(parse(text, &self.spec.enabled_tools()), Action::Tool { ref tag, .. } if tag == "python");
        if is_python {
            // its edits (if any) are already in the ledger; running it again would repeat them
            self.interrupted_python();
        } else {
            self.act(text)?;
        }
        self.conclude_action()
    }

    fn take_turn(&mut self) -> Result<Step> {
        let turn = self.state.turn + 1;
        let tdir = self.dir.join(format!("turns/t{turn:03}"));
        std::fs::create_dir_all(&tdir)?;
        let msg = self.state.next_message.clone();
        let new_conv = self.state.new_conversation;
        std::fs::write(tdir.join("prompt.txt"), &msg)?;
        let reply = self.send_with_retry(&msg, new_conv)?;
        self.state.turn = turn;
        match reply {
            Err(why) => self.refused(&tdir, &why, new_conv),
            Ok(reply) => {
                self.record_reply(&tdir, &reply, new_conv)?;
                self.act(&reply.text)?;
                self.conclude_action()
            }
        }
    }

    /// Send one message, retrying transient errors with exponential backoff. `Ok(Err(why))` is a refusal.
    fn send_with_retry(&mut self, msg: &str, new_conv: bool) -> Result<std::result::Result<Reply, String>> {
        let mut attempt = 0;
        loop {
            match self.editor.send(msg, new_conv) {
                Ok(r) => return Ok(Ok(r)),
                Err(BackendError::Refused(m)) => return Ok(Err(m)),
                Err(BackendError::Transient(m)) => {
                    self.state.backend_errors += 1;
                    attempt += 1;
                    self.log(&format!(
                        "turn {}: transient backend error (attempt {attempt}): {m}",
                        self.state.turn + 1
                    ));
                    if attempt > self.env.max_retries {
                        self.save_state()?;
                        bail!("backend kept failing after {attempt} attempts: {m} (resume later with --resume)");
                    }
                    std::thread::sleep(self.backoff(attempt));
                }
                Err(BackendError::Fatal(m)) => {
                    self.log(&format!("turn {}: fatal backend error: {m}", self.state.turn + 1));
                    self.save_state()?;
                    bail!("backend error: {m}");
                }
            }
        }
    }

    fn backoff(&self, attempt: u32) -> Duration {
        let wait = self.env.retry_base_s * 2f64.powi(attempt as i32 - 1);
        Duration::from_secs_f64(wait.min(MAX_BACKOFF_S))
    }

    /// Save the reply and its usage, and mark it pending before acting on it, so a crash cannot lose or
    /// repeat it.
    fn record_reply(&mut self, tdir: &Path, reply: &Reply, new_conv: bool) -> Result<()> {
        self.state.refusal_streak = 0;
        std::fs::write(tdir.join("answer.md"), &reply.text)?;
        let meta = json!({
            "editor": self.editor.name(), "new_conversation": new_conv, "usage": reply.usage, "session": reply.session,
            "tool_free": reply.tool_free, "cost_usd": reply.cost_usd, "wall_s": reply.wall_s, "context_tokens": reply.context_tokens,
        });
        std::fs::write(tdir.join("run.json"), serde_json::to_string_pretty(&meta)?)?;
        std::fs::write(tdir.join("raw.json"), serde_json::to_string(&reply.raw)?)?;
        self.state.conv_tokens = reply.context_tokens.unwrap_or_else(|| reply.usage.context_tokens());
        self.state.usage_total += &reply.usage;
        self.state.cost_usd += reply.cost_usd.unwrap_or(0.0);
        if reply.tool_free {
            self.state.tool_free_turns += 1;
        }
        self.state.session = self.editor.session();
        self.state.new_conversation = false;
        self.state.since_handoff += 1;
        self.state.pending_reply = Some(reply.text.clone());
        self.state.ledger_mark = self.last_edit_id();
        self.save_state()
    }

    /// The provider refused the turn: tell the agent, or end the run after too many refusals in a row.
    fn refused(&mut self, tdir: &Path, why: &str, new_conv: bool) -> Result<Step> {
        self.state.refusals += 1;
        self.state.refusal_streak += 1;
        std::fs::write(tdir.join("refused.txt"), why)?;
        self.log(&format!(
            "turn {}: refused by the provider (streak {})",
            self.state.turn, self.state.refusal_streak
        ));
        if self.state.refusal_streak > MAX_REFUSAL_STREAK {
            self.finalize("", "the provider refused four turns in a row")?;
            self.save_state()?;
            return Ok(Step::Done);
        }
        let notice = format!(
            "The harness's last message could not be delivered to you: the provider's content filter refused it ({}). \
             This usually happens when output contains raw vocabulary strings or long generated text; print token ids, \
             numbers or short excerpts instead. Continue with your next step, done differently.\n\n{}",
            why.chars().take(300).collect::<String>(),
            self.budget_line()
        );
        // a new conversation must still carry the task
        self.state.next_message = if !new_conv {
            notice
        } else if self.state.handoffs == 0 {
            format!("{}\n\n[{notice}]", self.first_message)
        } else {
            self.handoff_message(&notice)
        };
        self.save_state()?;
        Ok(Step::Continue)
    }

    /// After acting on a reply: clear it, decide whether the next message starts a new conversation, save.
    fn conclude_action(&mut self) -> Result<Step> {
        self.state.pending_reply = None;
        self.maybe_handoff();
        self.save_state()?;
        Ok(if self.state.done { Step::Done } else { Step::Continue })
    }

    fn maybe_handoff(&mut self) {
        if self.state.done {
            return;
        }
        let h = &self.spec.handoff;
        let by_tokens = h.max_conversation_tokens > 0 && self.state.conv_tokens >= h.max_conversation_tokens;
        let by_turns = h.max_turns_per_conversation > 0 && self.state.since_handoff >= h.max_turns_per_conversation;
        let forced = std::mem::take(&mut self.state.force_handoff);
        let why = if by_tokens {
            "context size"
        } else if by_turns {
            "turn count"
        } else if forced {
            "the backend cannot resume its conversation"
        } else {
            return;
        };
        self.start_handoff();
        self.log(&format!("handoff {} ({why})", self.state.handoffs));
    }

    /// Make the next message start a new conversation that restates the task and where the agent is.
    fn start_handoff(&mut self) {
        let next = std::mem::take(&mut self.state.next_message);
        self.state.next_message = self.handoff_message(&next);
        self.state.new_conversation = true;
        self.state.handoffs += 1;
        self.state.since_handoff = 0;
    }

    fn handoff_message(&self, next: &str) -> String {
        let ledger = self
            .gate
            .as_ref()
            .map(|g| g.summary())
            .unwrap_or_else(|| "(this task has no editable network)\n".into());
        format!(
            "{}\n\n=== HANDOFF ===\nThe harness started a new conversation to keep the context small. The task above is unchanged, and so \
             are your files in /work, your saved edits and the Python kernel's variables. Where you are:\n{}\n\nYour notebook \
             (/work/notes.md):\n```\n{}\n```\n\nYour edits so far:\n```\n{}```\n\nThe message you would have received next:\n\n{}",
            self.first_message,
            self.budget_line(),
            self.notebook().trim_end(),
            ledger,
            next
        )
    }

    /// /work/notes.md for a handoff, if it is a regular file in /work, cut at MAX_NOTES_CHARS.
    fn notebook(&self) -> String {
        let work = self.dir.join("work");
        let notes = match paths::file_under(&work, Path::new("notes.md")) {
            Ok(p) => std::fs::read_to_string(p).unwrap_or_else(|_| "(/work/notes.md could not be read as text)".into()),
            Err(_) if std::fs::symlink_metadata(work.join("notes.md")).is_err() => "(no /work/notes.md)".into(),
            Err(_) => "(/work/notes.md is not a regular file in /work, so it is not shown)".into(),
        };
        match notes.char_indices().nth(MAX_NOTES_CHARS) {
            Some((end, _)) => format!("{}\n[notes cut at 20,000 characters]", &notes[..end]),
            None => notes,
        }
    }

    /// Words of a reply outside the reply's fenced blocks: the plain-English explanation of a tool call.
    fn explanation_words(text: &str) -> usize {
        let mut words = 0;
        let mut fence: Option<usize> = None;
        for line in text.lines() {
            let t = line.trim_start();
            let ticks = t.chars().take_while(|c| *c == '`').count();
            match fence {
                None if ticks >= 3 => fence = Some(ticks),
                None => {
                    words += t
                        .split_whitespace()
                        .filter(|w| w.chars().any(|c| c.is_alphabetic()))
                        .count()
                }
                Some(n) if ticks >= n && t[ticks..].trim().is_empty() => fence = None,
                Some(_) => {}
            }
        }
        words
    }

    /// The [rationale] rule: Some(message) when the run sends the tool call back unrun.
    fn rationale_refusal(&mut self, tag: &str, text: &str) -> Result<Option<String>> {
        let r = self.spec.rationale.clone();
        if r.min_words == 0 || !(r.tools.is_empty() || r.tools.iter().any(|t| t == tag)) {
            return Ok(None);
        }
        let words = Self::explanation_words(text);
        if words >= r.min_words {
            self.state.rationale_streak = 0;
            return Ok(None);
        }
        let run_anyway = self.state.rationale_streak >= r.max_rejects_in_row;
        let turn = self.state.turn;
        self.tool_log(
            json!({"kind": "rationale-missing", "turn": turn, "tool": tag, "words": words, "run_anyway": run_anyway}),
        )?;
        if run_anyway {
            self.state.rationale_streak = 0;
            self.state.rationale_missing_run += 1;
            return Ok(None);
        }
        self.state.rationale_streak += 1;
        self.state.rationale_rejects += 1;
        Ok(Some(format!(
            "EXPLAIN EVERY STEP: before the ```{tag} block, explain in plain English what this step does and why. \
             The rule asks for at least {} words outside the block, and your reply had {words}. Nothing ran and \
             nothing was charged. Send the same step again with the explanation.\n{}",
            r.min_words,
            self.budget_line()
        )))
    }

    fn act(&mut self, text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return self.empty_reply();
        }
        self.state.empty_streak = 0;
        let action = parse(text, &self.spec.enabled_tools());
        if let Action::Tool { tag, .. } = &action {
            if let Some(m) = self.rationale_refusal(tag, text)? {
                self.state.next_message = m;
                return Ok(());
            }
        }
        match action {
            Action::Error(m) => {
                self.state.protocol_errors += 1;
                self.state.next_message = format!("{m}\n{}", self.budget_line());
                Ok(())
            }
            Action::Tool { tag, body } if tag == "python" => self.python(&body),
            Action::Tool { tag, body } => self.forward(&tag, &body),
            Action::Final(text) => self.final_reply(&text),
        }
    }

    /// An empty reply is a protocol error, not a final answer. An open model sometimes ends a reply at once, for
    /// example on an end-of-turn token written into the reply. More than MAX_EMPTY_STREAK in a row end the run.
    fn empty_reply(&mut self) -> Result<()> {
        self.state.empty_replies += 1;
        self.state.empty_streak += 1;
        if self.state.empty_streak > MAX_EMPTY_STREAK {
            return self.finalize("", "the editor sent empty replies");
        }
        self.state.protocol_errors += 1;
        self.state.next_message = format!(
            "Your last reply was empty, with no text and no tool block. Nothing ran and nothing was charged. Send your \
             next step. A special token such as <|im_end|> written into a reply ends some models' replies early. \
             Build such strings in Python instead.\n{}",
            self.budget_line()
        );
        Ok(())
    }

    /// A reply without a tool block ends the run, unless the task wants every pass used first.
    fn final_reply(&mut self, text: &str) -> Result<()> {
        let left = self.passes_left();
        let fin = &self.spec.finish;
        if fin.require_all_passes && !left.is_empty() && self.state.nudges < fin.max_nudges {
            self.state.nudges += 1;
            let left: Vec<String> = left.iter().map(|(label, n)| format!("{n} {label} left")).collect();
            self.state.next_message = format!(
                "The session continues until every forward pass is used: you have {}. Use them (a reply without a tool \
                 block does not end the session yet).\n{}",
                left.join(", "),
                self.budget_line()
            );
            Ok(())
        } else {
            if let Some(error) = self.validate_submission(text)? {
                if self.state.nudges >= self.spec.finish.max_nudges {
                    return self.finalize(text, "submission rejected");
                }
                self.state.nudges += 1;
                self.state.protocol_errors += 1;
                self.state.next_message = format!(
                    "Submission format refused: {error}\nThe task continues. Repair the submission. \
                     This check reports format only, never correctness. Your declared tools remain available.\n{}",
                    self.budget_line()
                );
                return Ok(());
            }
            self.finalize(text, "final answer")
        }
    }

    fn validate_submission(&mut self, text: &str) -> Result<Option<String>> {
        let Some(validation) = &self.spec.finish.validation else {
            return Ok(None);
        };
        let checked = (|| -> Result<(Value, PathBuf)> {
            let mut candidate = text.to_string();
            if let Some(name) = &validation.submission_file {
                let work = self.dir.join("work");
                let file = work.join(name);
                if std::fs::symlink_metadata(&file).is_ok() {
                    candidate.clear();
                    std::fs::File::open(paths::file_under(&work, &file)?)?
                        .take(MAX_SUBMISSION_BYTES + 1)
                        .read_to_string(&mut candidate)?;
                }
            }
            if candidate.len() as u64 > MAX_SUBMISSION_BYTES {
                bail!("submission exceeds {MAX_SUBMISSION_BYTES} bytes");
            }
            let requests = paths::dir_under(&self.dir, "requests")?;
            let file = requests.join(format!("submission-{}.txt", self.state.turn));
            std::fs::write(&file, candidate)?;
            let result = crate::forward::run_command(
                &self.spec.argv(&validation.command),
                &[("OPENWISH_SUBMISSION", file.display().to_string())],
                &self.spec.dir,
                validation.timeout_s,
            )?;
            Ok((result, file))
        })();
        let result = match checked {
            Ok((result, file)) => {
                if result["ok"].as_bool() == Some(true) {
                    std::fs::copy(file, self.dir.join("validated_submission.json"))?;
                }
                result
            }
            Err(error) => json!({"ok": false, "error": error.to_string()}),
        };
        let error = if result["ok"].as_bool() == Some(true) {
            None
        } else {
            Some(
                result["error"]
                    .as_str()
                    .unwrap_or("the validator refused the submission")
                    .chars()
                    .take(4000)
                    .collect(),
            )
        };
        self.tool_log(json!({"kind": "validation", "turn": self.state.turn, "result": result}))?;
        Ok(error)
    }
}

// Hooks.
impl Run {
    fn hook_ctx(&self, tool: &str) -> HookCtx {
        HookCtx {
            run_dir: self.dir.clone(),
            task_dir: self.spec.dir.clone(),
            tool: tool.into(),
            turn: self.state.turn,
            ..Default::default()
        }
    }

    /// Run the hooks for one event and record them in hooks.jsonl. A failing hook is logged and the run goes on.
    fn fire(&self, e: Event, ctx: &HookCtx) -> Vec<Outcome> {
        if !self.hooks.has(e) {
            return vec![];
        }
        let out = self.hooks.fire(e, ctx);
        self.log_hooks(&out, ctx);
        out
    }

    fn log_hooks(&self, out: &[Outcome], ctx: &HookCtx) {
        for o in out {
            let mut v = serde_json::to_value(o).unwrap_or_default();
            v["turn"] = ctx.turn.into();
            v["tool"] = ctx.tool.clone().into();
            if let Some((id, t)) = &ctx.edit {
                v["edit_id"] = (*id).into();
                v["edit_tensor"] = t.clone().into();
            }
            let _ = append_line(&self.dir.join("hooks.jsonl"), &v.to_string());
            if !o.ok {
                self.log(&format!(
                    "turn {}: {} hook failed ({}): {}",
                    ctx.turn,
                    o.event.name(),
                    o.command,
                    o.reason()
                ));
            }
        }
    }

    /// The pre_tool hooks for one call. `Some(reason)` means a hook blocked it.
    fn pre_tool(&self, tool: &str, input: Value) -> Option<String> {
        if !self.hooks.has(Event::PreTool) {
            return None;
        }
        let mut c = self.hook_ctx(tool);
        c.input = input;
        let (out, why) = self.hooks.check(&c);
        self.log_hooks(&out, &c);
        why
    }

    /// Record a blocked call and tell the agent why. Nothing is charged.
    fn blocked(&mut self, what: &str, kind: &str, input: Value, why: &str) -> Result<()> {
        self.state.hook_blocks += 1;
        self.tool_log(json!({"kind": kind, "turn": self.state.turn, "blocked": true, "reason": why, "input": input}))?;
        self.state.next_message = format!(
            "A pre_tool hook blocked this {what}, so nothing was run and it was not counted. The hook said:\n```\n{}\n```\n{}",
            why.trim_end(),
            self.budget_line()
        );
        Ok(())
    }

    fn post_tool(&self, tool: &str, ok: bool, input: Value) {
        if !self.hooks.has(Event::PostTool) {
            return;
        }
        let mut c = self.hook_ctx(tool);
        c.extra
            .push(("OPENWISH_TOOL_OK".into(), if ok { "1" } else { "0" }.into()));
        c.input = input;
        self.fire(Event::PostTool, &c);
    }

    /// The on_edit hooks, once for every edit in the ledger after edit `after`.
    fn on_edits(&self, tool: &str, after: u64) {
        if !self.hooks.has(Event::OnEdit) {
            return;
        }
        for (id, tensor) in self.edits_after(after) {
            let mut c = self.hook_ctx(tool);
            c.edit = Some((id, tensor));
            self.fire(Event::OnEdit, &c);
        }
    }

    /// (id, tensor) of every edit after edit `after`.
    fn edits_after(&self, after: u64) -> Vec<(u64, String)> {
        let Some(g) = &self.gate else { return vec![] };
        g.edits()
            .filter(|r| r.id > after)
            .map(|r| (r.id, r.op.tensor().to_string()))
            .collect()
    }
}

// The Python tool.
impl Run {
    fn python(&mut self, code: &str) -> Result<()> {
        let py = self.spec.python.clone().expect("python tool enabled");
        if self.state.runs_used >= py.runs {
            self.state.next_message = format!(
                "You have no Python runs left (used {} of {}). Nothing was run.\n{}",
                self.state.runs_used,
                py.runs,
                self.budget_line()
            );
            return Ok(());
        }
        if let Some(why) = self.pre_tool("python", json!({"code": code})) {
            return self.blocked("Python run", "python", json!({"code": code}), &why);
        }
        let flags = openwish_audit::static_check(code);
        let rejected: Vec<_> = flags
            .iter()
            .filter(|flag| py.reject_flags.contains(&flag.rule))
            .collect();
        if !rejected.is_empty() {
            self.state.failed_requests += 1;
            let rules: Vec<_> = rejected.iter().map(|flag| flag.rule.as_str()).collect();
            self.tool_log(json!({"kind": "python-policy", "turn": self.state.turn, "blocked": true, "code": code, "flags": rejected}))?;
            self.state.next_message = format!(
                "Python policy blocked this code before execution: {}. Nothing ran and no Python budget was spent. \
                 Use weight algebra and symbolic identities. Do not simulate the network or its components on inputs.\n{}",
                rules.join(", "), self.budget_line()
            );
            return Ok(());
        }
        if self.kernel.is_none() {
            self.kernel = Some(self.start_kernel(&py)?);
        }
        let mark = self.last_edit_id().unwrap_or(0);
        let out = self.run_in_kernel(code)?;
        self.state.runs_used += 1;
        if py.fresh {
            self.kernel = None;
        }
        if out.restarted {
            self.state.kernel_restarts += 1;
        }
        let k = self.state.runs_used;
        self.tool_log(
            json!({"kind": "python", "turn": self.state.turn, "run": k, "code": code, "output": out.text,
            "error": out.error, "restarted": out.restarted, "seconds": out.seconds, "flags": flags}),
        )?;
        self.on_edits("python", mark);
        self.post_tool(
            "python",
            !out.error,
            json!({"code": code, "output": out.text, "error": out.error}),
        );
        self.state.next_message = format!("{}\n{}", python_report(k, py.runs, &out, &flags), self.budget_line());
        Ok(())
    }

    fn start_kernel(&self, py: &PythonSpec) -> Result<Kernel> {
        let prelude = match &py.prelude {
            Some(p) => Some(std::fs::read_to_string(self.spec.resolve(p))?),
            None => None,
        };
        let mut mounts = self.spec.mounts.clone();
        for m in &mut mounts {
            m.host = self.spec.resolve(&m.host);
        }
        Kernel::start(KernelCfg {
            venv: py
                .venv
                .as_ref()
                .map(|v| self.spec.resolve(v))
                .unwrap_or_else(|| self.env.venv.clone()),
            kernel_dir: self.env.kernel_dir.clone(),
            private_dir: self.dir.join("kernel"),
            net_dir: self.gate.as_ref().map(Gate::current_dir),
            work_dir: self.dir.join("work"),
            mounts,
            timeout_s: py.timeout_s,
            mem_bytes: py.mem_gb << 30,
            max_out: py.max_out,
            prelude,
            threads: py.threads.min(self.env.threads.max(1)),
        })
    }

    /// Run `code` in the started kernel; its edit RPCs go through the gate.
    fn run_in_kernel(&mut self, code: &str) -> Result<RunOutput> {
        let turn = self.state.turn;
        let rpc_dir = self.dir.join("work/.rpc");
        let (kernel, gate) = (self.kernel.as_mut().expect("kernel started"), &mut self.gate);
        kernel.run(code, &mut |r: Rpc| handle_rpc(gate.as_mut(), r, turn, &rpc_dir))
    }

    /// The harness stopped during a Python run: report what is known instead of running it again.
    fn interrupted_python(&mut self) {
        self.state.runs_used += 1;
        self.state.kernel_restarts += 1;
        let k = self.state.runs_used;
        let mark = self.state.ledger_mark.unwrap_or(0);
        let ids: Vec<String> = self.edits_after(mark).iter().map(|(id, _)| format!("#{id}")).collect();
        let kept = if ids.is_empty() {
            "No edits were saved during it.".to_string()
        } else {
            format!("Edits saved during it were kept: {}.", ids.join(", "))
        };
        let _ = self.tool_log(
            json!({"kind": "python", "turn": self.state.turn, "run": k, "interrupted": true, "edits_kept": ids}),
        );
        self.on_edits("python", mark);
        self.state.next_message = format!(
            "[The harness stopped while your Python run {k} was going, so its output is lost. {kept} The kernel was \
             restarted and its variables are lost; files in /work are kept.]\n{}",
            self.budget_line()
        );
    }
}

/// What the agent reads after Python run `k`.
fn python_report(k: u32, runs: u32, out: &RunOutput, flags: &[Flag]) -> String {
    let text = if out.text.is_empty() {
        "(no output)"
    } else {
        out.text.trim_end()
    };
    let mut msg = format!("OUTPUT of Python run {k} of at most {runs}:\n```\n{text}\n```");
    if !flags.is_empty() {
        let f: Vec<String> = flags.iter().map(|f| format!("line {}: {}", f.line, f.why)).collect();
        msg += &format!(
            "\n[harness note: this code matched patterns that are recorded for the rules audit: {}]",
            f.join("; ")
        );
    }
    msg
}

/// One forward pass: use number `k` of the tool's budget group.
struct Pass {
    tag: String,
    group: String,
    k: u32,
    budget: u32,
    /// its folder under /work
    folder: String,
}

// Forward probes.
impl Run {
    fn forward(&mut self, tag: &str, body: &str) -> Result<()> {
        let f = self.spec.forward[tag].clone();
        let group = self.spec.group(tag);
        let (used, budget) = self.tool_budget(&group, &f);
        if used >= budget {
            let what = if group == PYTHON_POOL {
                "Python runs".to_string()
            } else {
                self.spec.group_label(&group)
            };
            self.state.next_message = format!(
                "You have no {what} left (used {used} of {budget}). Nothing was run.\n{}",
                self.budget_line()
            );
            return Ok(());
        }
        if let Some(why) = self.pre_tool(tag, json!({"request": body})) {
            return self.blocked(
                &format!("{tag} pass"),
                &format!("forward:{tag}"),
                json!({"request": body}),
                &why,
            );
        }
        let k = used + 1;
        let pass = Pass {
            tag: tag.to_string(),
            group,
            k,
            budget,
            folder: f.folder(tag, k),
        };
        let req = self.dir.join(format!("requests/{tag}_{k}.json"));
        std::fs::write(&req, body)?;
        let out_dir = match paths::dir_under(&self.dir.join("work"), &pass.folder) {
            Ok(p) => p,
            Err(e) => {
                self.state.failed_requests += 1;
                self.state.next_message = format!(
                    "The {tag} pass could not run and was not counted: its folder /work/{} is unusable ({e:#}).\n{}",
                    pass.folder,
                    self.budget_line()
                );
                return Ok(());
            }
        };
        if out_dir.exists() {
            std::fs::remove_dir_all(&out_dir)?;
        }
        std::fs::create_dir_all(&out_dir)?;
        let t0 = Instant::now();
        let (ok, summary) = probe_summary(crate::forward::run_command(
            &self.spec.argv(&f.command),
            &self.probe_env(&req, &out_dir, k),
            &self.spec.dir,
            f.timeout_s,
        ));
        let secs = t0.elapsed().as_secs_f64();
        self.tool_log(
            json!({"kind": format!("forward:{tag}"), "turn": self.state.turn, "pass": ok.then_some(k),
            "request": body, "ok": ok, "summary": summary, "seconds": secs}),
        )?;
        self.post_tool(tag, ok, json!({"request": body, "ok": ok, "summary": summary}));
        if ok {
            self.pass_done(&pass, &summary, f.max_out)
        } else {
            let _ = std::fs::remove_dir_all(&out_dir);
            self.pass_failed(&pass, &summary);
            Ok(())
        }
    }

    /// Charge the pass and show the agent its summary.
    fn pass_done(&mut self, pass: &Pass, summary: &Value, max_out: usize) -> Result<()> {
        let Pass {
            tag, k, budget, folder, ..
        } = pass;
        self.charge(&pass.group, *k);
        let pretty = serde_json::to_string_pretty(summary)?;
        std::fs::write(self.dir.join(format!("requests/{tag}_{k}.done.json")), &pretty)?;
        let text = openwish_sandbox::cut(&pretty, max_out);
        let what = if pass.group == PYTHON_POOL {
            format!("Run {k} of at most {budget} ({tag})")
        } else {
            format!("Forward pass {k} of {budget} ({tag})")
        };
        self.state.next_message = format!(
            "{what} done; files in /work/{folder}/. Summary:\n```\n{text}\n```\n{}",
            self.budget_line()
        );
        Ok(())
    }

    /// A failed pass is not charged; the agent reads why it failed.
    fn pass_failed(&mut self, pass: &Pass, summary: &Value) {
        self.state.failed_requests += 1;
        let err = summary["error"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| summary.to_string());
        self.state.next_message = format!(
            "The {} pass failed and was not counted: {}\n{}",
            pass.tag,
            err.chars().take(4000).collect::<String>(),
            self.budget_line()
        );
    }

    /// The variables a probe command gets for pass `k`.
    fn probe_env(&self, req: &Path, out_dir: &Path, k: u32) -> Vec<(&'static str, String)> {
        let net = self
            .gate
            .as_ref()
            .map(|g| g.current_dir().display().to_string())
            .unwrap_or_default();
        let base = self
            .spec
            .network
            .as_ref()
            .map(|n| self.spec.resolve(&n.base).display().to_string())
            .unwrap_or_default();
        vec![
            ("OPENWISH_REQ", req.display().to_string()),
            ("OPENWISH_OUT", out_dir.display().to_string()),
            ("OPENWISH_NET", net),
            ("OPENWISH_BASE_NET", base),
            ("OPENWISH_WORK", self.dir.join("work").display().to_string()),
            ("OPENWISH_TASK_DIR", self.spec.dir.display().to_string()),
            ("OPENWISH_RUN_DIR", self.dir.display().to_string()),
            ("OPENWISH_PASS", k.to_string()),
        ]
    }
}

// The end of a run.
impl Run {
    fn finalize(&mut self, text: &str, reason: &str) -> Result<()> {
        std::fs::write(self.dir.join("final_answer.md"), text)?;
        std::fs::create_dir_all(self.dir.join("final"))?;
        if self.spec.edits.language {
            let mark = self.last_edit_id().unwrap_or(0);
            let report = self.apply_edits_block(text);
            std::fs::write(
                self.dir.join("final/edits_report.json"),
                serde_json::to_string_pretty(&report)?,
            )?;
            self.on_edits("edits", mark);
        }
        if let Some(g) = &self.gate {
            g.current.save_dir(&self.dir.join("final/net"))?;
            openwish_tensor::editset::export(&g.base, &g.current, &self.dir.join("final/editset"))?;
        }
        self.state.done = true;
        self.state.end_reason = Some(reason.to_string());
        self.kernel = None;
        self.log(&format!(
            "finished: {reason}; turns {}, python runs {}, passes {:?}",
            self.state.turn, self.state.runs_used, self.state.passes_used
        ));
        Ok(())
    }

    /// Apply the last ```edits block of the final reply through the gate; returns the report.
    fn apply_edits_block(&mut self, text: &str) -> Value {
        let Some(block) = openwish_tensor::editlang::last_edits_block(text) else {
            return json!({"applied": 0, "skipped": [], "refused": [], "block_found": false});
        };
        let parsed = openwish_tensor::editlang::parse(&block);
        let (mut applied, mut refused) = (0, vec![]);
        if let Some(g) = self.gate.as_mut() {
            for op in parsed.ops {
                match g.apply(op.clone(), "edits block", "final-reply") {
                    Ok(_) => applied += 1,
                    Err(e) => refused.push(json!({"op": op, "why": e.0})),
                }
            }
        }
        json!({"applied": applied, "skipped": parsed.skipped, "refused": refused, "block_found": true})
    }
}

/// (ok, summary) of a probe command; a command that could not run gets an error summary.
fn probe_summary(res: Result<Value>) -> (bool, Value) {
    match res {
        Ok(v) => (v["ok"].as_bool() == Some(true), v),
        Err(e) => (false, json!({"ok": false, "error": format!("{e:#}")})),
    }
}

/// The agent's first message: the prompt, then the appendix if the task has one.
fn first_message(spec: &TaskSpec) -> Result<String> {
    let mut first = std::fs::read_to_string(spec.resolve(&spec.prompt))?;
    if let Some(a) = &spec.appendix {
        first.push_str("\n\n");
        first.push_str(&std::fs::read_to_string(spec.resolve(a))?);
    }
    if let Some(validation) = &spec.finish.validation {
        first.push_str("\n\nOPENWISH SUBMISSION PROTOCOL\nYour final submission is checked for format only, never correctness. \
            Repair format errors when requested. OpenWiSH executes your tool blocks, even when the provider's own tool list is empty.\n");
        if let Some(py) = &spec.python {
            if py.runs > 0 {
                first.push_str(&format!(
                    "Python is available through text blocks, with at most {} runs. \
                    Reply with one block to execute code, for example:\n```python\nprint(NAMES)\n```\n",
                    py.runs
                ));
            }
        }
        for tag in spec.forward.keys() {
            first.push_str(&format!(
                "The declared {tag} probe accepts one ```{tag} block with a JSON request.\n"
            ));
        }
        if let Some(file) = &validation.submission_file {
            first.push_str(&format!(
                "Write your submission to /work/{} or put the submission in your final reply.\n",
                file.display()
            ));
        }
    }
    Ok(first)
}

/// Project hooks first, then the task's (which run in the task folder).
fn all_hooks(env: &Env, spec: &TaskSpec) -> HookSet {
    let mut h = env.hooks.clone();
    h.extend(HookSet::from_spec(&spec.hooks, &spec.dir, "task"));
    h
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")
}

/// Answer one edit RPC from the sandbox kernel.
fn handle_rpc(gate: Option<&mut Gate>, r: Rpc, turn: u32, rpc_dir: &Path) -> RpcReply {
    let Some(g) = gate else {
        return RpcReply {
            ok: false,
            msg: "this task has no editable network".into(),
        };
    };
    let src = format!("python (turn {turn})");
    let answer = match r {
        // only arrays the kernel itself wrote into /work/.rpc are accepted
        Rpc::SaveEdit { name, path, reason } => match read_rpc_array(rpc_dir, &path) {
            Ok(t) => edit_answer(g.replace(&name, t, &reason, &src)),
            Err(e) => Err(format!("could not read the array: {e}")),
        },
        Rpc::DropEdit { name, reason } => edit_answer(g.apply(EditOp::Revert { tensor: name }, &reason, &src)),
        Rpc::Checkpoint { name } => g
            .checkpoint(&name)
            .map(|()| format!("checkpoint {name} saved"))
            .map_err(|e| e.to_string()),
        Rpc::Restore { name } => g
            .restore(&name, &src)
            .map(|v| format!("restored checkpoint {name} ({} tensors changed)", v.len()))
            .map_err(|e| e.to_string()),
        Rpc::Ledger => Ok(g.summary()),
    };
    match answer {
        Ok(msg) => RpcReply { ok: true, msg },
        Err(msg) => RpcReply { ok: false, msg },
    }
}

fn read_rpc_array(rpc_dir: &Path, path: &Path) -> Result<openwish_tensor::Tensor> {
    let p = paths::file_under(rpc_dir, path)?;
    openwish_tensor::npy::read_npy(&std::fs::read(p)?)
}

fn edit_answer(res: std::result::Result<EditRecord, Refusal>) -> std::result::Result<String, String> {
    res.map(|rec| {
        format!(
            "edit #{} saved: {} entries of {} changed; {} entries ({:.4}%) of the network now differ from the original",
            rec.id,
            rec.changed,
            rec.op.tensor(),
            rec.changed_vs_base,
            rec.frac_changed_vs_base * 100.0
        )
    })
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openwish_llm::scripted::Scripted;
    use openwish_tensor::Tensor;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    fn env() -> Env {
        Env {
            venv: root().join("assets/venvs/sandbox"),
            kernel_dir: root().join("python/kernel"),
            retry_base_s: 0.0,
            max_retries: 3,
            threads: 2,
            hooks: Default::default(),
        }
    }
    fn has_venv() -> bool {
        env().venv.join("bin/python").exists()
    }

    /// A task directory with a 2x2 network and the given extra TOML.
    fn task(d: &Path, extra: &str) -> TaskSpec {
        let mut n = Network::default();
        n.tensors.insert("W".into(), Tensor::new(vec![2, 2], vec![0.; 4]));
        n.save_dir(&d.join("net")).unwrap();
        std::fs::write(d.join("prompt.txt"), "TASK: make W the identity.").unwrap();
        std::fs::write(d.join("pre.py"), "PRE = 7").unwrap();
        let toml = format!("id = \"toy\"\nprompt = \"prompt.txt\"\n[network]\nbase = \"net\"\n{extra}");
        std::fs::write(d.join("task.toml"), toml).unwrap();
        TaskSpec::load(&d.join("task.toml")).unwrap()
    }
    const FWD: &str = r#"
[forward.games]
command = ["sh", "-c", "if grep -q bad \"$OPENWISH_REQ\"; then echo '{\"ok\":false,\"error\":\"bad request\"}'; else echo \"notes from pass $OPENWISH_PASS\" >> \"$OPENWISH_WORK/notes.md\"; echo '{\"ok\":true,\"games\":4}'; fi"]
passes = 3
"#;
    fn turn_prompt(r: &Run, t: u32) -> String {
        std::fs::read_to_string(r.dir.join(format!("turns/t{t:03}/prompt.txt"))).unwrap()
    }
    fn script(v: &[&str]) -> Box<dyn Editor> {
        Box::new(Scripted::new(v.iter().map(|s| s.to_string()).collect()))
    }

    #[test]
    fn toy_end_to_end_python_edit_then_final() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "[python]\nruns = 5\ntimeout_s = 60\nmem_gb = 4\n");
        let ed = script(&[
            "Let me edit.\n```python\nimport numpy as np\nsave_edit('W', np.eye(2), reason='identity')\n```",
            "Done: W is the identity.",
        ]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert!(r.state.done);
        assert_eq!(r.state.runs_used, 1);
        assert!(turn_prompt(&r, 1).contains("TASK: make W"));
        assert!(
            turn_prompt(&r, 2).contains("OUTPUT of Python run 1 of at most 5"),
            "{}",
            turn_prompt(&r, 2)
        );
        let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
        assert_eq!(fin.tensors["W"].data, vec![1., 0., 0., 1.]);
        assert_eq!(r.gate().unwrap().edits().count(), 1);
        assert!(r.dir.join("final_answer.md").exists());
        assert!(r.dir.join("final/editset/manifest.json").exists());
        assert_eq!(
            std::fs::read_to_string(r.dir.join("tool_calls.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert!(std::fs::read_to_string(r.dir.join("rules.txt"))
            .unwrap()
            .contains("TASK"));
    }

    const VALIDATION: &str = r#"
[finish.validation]
command = ["sh", "-c", "if grep -q 'answer' \"$OPENWISH_SUBMISSION\"; then echo '{\"ok\":true}'; else echo '{\"ok\":false,\"error\":\"Expected an answer object\"}'; fi"]
"#;

    #[test]
    fn validated_tasks_explain_text_tools_but_no_tool_baselines_stay_verbatim() {
        let d = tempfile::tempdir().unwrap();
        let baseline = task(d.path(), "");
        assert_eq!(first_message(&baseline).unwrap(), "TASK: make W the identity.");
        let interactive = task(d.path(), &format!("[python]\nruns = 2\n{VALIDATION}"));
        let message = first_message(&interactive).unwrap();
        assert!(message.contains("OpenWiSH executes your tool blocks"));
        assert!(message.contains("```python\nprint(NAMES)\n```"));
        assert!(message.contains("format only"));
    }

    #[test]
    fn context_budget_uses_the_current_footprint_instead_of_billing_counters() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "");
        let mut r = Run::create(spec, &d.path().join("run"), script(&[]), env()).unwrap();
        let turn = r.dir.join("turns/t001");
        std::fs::create_dir_all(&turn).unwrap();
        let reply = Reply {
            usage: openwish_llm::Usage {
                input: 255717,
                cache_read: 191232,
                output: 5410,
                ..Default::default()
            },
            context_tokens: Some(66572),
            ..Default::default()
        };
        r.record_reply(&turn, &reply, true).unwrap();
        assert_eq!(r.state.conv_tokens, 66572);
    }

    #[test]
    fn configured_static_policy_blocks_code_before_starting_python() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "[python]\nruns = 2\nreject_flags = [\"forward_pass\"]\n");
        let code = "```python\nopen('/work/executed', 'w').write('bad')\nh = np.maximum(X @ W.T + b, 0)\n```";
        let mut r = Run::create(spec, &d.path().join("run"), script(&[code]), env()).unwrap();
        r.step().unwrap();
        assert!(r.kernel.is_none());
        assert_eq!(r.state.runs_used, 0);
        assert_eq!(r.state.failed_requests, 1);
        assert!(!r.dir.join("work/executed").exists());
        assert!(r.state.next_message.contains("forward_pass"));
        let calls = std::fs::read_to_string(r.dir.join("tool_calls.jsonl")).unwrap();
        assert!(calls.contains("python-policy") && calls.contains("blocked"));
    }

    #[test]
    fn malformed_final_answer_gets_format_feedback_without_scoring() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), VALIDATION);
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["No tools available.", "{\"answer\":1}"]),
            env(),
        )
        .unwrap();
        r.step().unwrap();
        assert!(!r.state.done);
        assert!(r.state.next_message.contains("Expected an answer object"));
        assert_eq!((r.state.runs_used, r.state.nudges), (0, 1));
        r.step().unwrap();
        assert!(r.state.done);
        assert_eq!(
            std::fs::read_to_string(r.dir.join("final_answer.md")).unwrap(),
            "{\"answer\":1}"
        );
        assert!(!r.dir.join("eval/score.json").exists());
    }

    #[test]
    fn submission_validation_retries_are_bounded() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), &format!("[finish]\nmax_nudges = 1\n{VALIDATION}"));
        let mut r = Run::create(spec, &d.path().join("run"), script(&["bad", "still bad"]), env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.turn, 2);
        assert_eq!(r.state.end_reason.as_deref(), Some("submission rejected"));
    }

    #[test]
    fn submission_file_is_checked_under_work_before_the_validator_reads_it() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!("{VALIDATION}\nsubmission_file = \"submission.json\"\n"),
        );
        let mut r = Run::create(spec, &d.path().join("run"), script(&["done", "done"]), env()).unwrap();
        std::fs::write(r.dir.join("work/submission.json"), "{\"answer\":1}").unwrap();
        r.step().unwrap();
        assert!(r.state.done);
        assert_eq!(
            std::fs::read_to_string(r.dir.join("validated_submission.json")).unwrap(),
            "{\"answer\":1}"
        );

        let spec = r.spec.clone();
        let mut r = Run::create(spec, &d.path().join("hostile"), script(&["done"]), env()).unwrap();
        let outside = d.path().join("outside.json");
        std::fs::write(&outside, "{\"answer\":1}").unwrap();
        symlink(outside, r.dir.join("work/submission.json")).unwrap();
        r.step().unwrap();
        assert!(!r.state.done);
        assert!(
            r.state.next_message.contains("symbolic link"),
            "{}",
            r.state.next_message
        );
    }

    #[test]
    fn submission_validation_bounds_host_file_reads() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!("{VALIDATION}\nsubmission_file = \"submission.json\"\n"),
        );
        let mut r = Run::create(spec, &d.path().join("run"), script(&["done"]), env()).unwrap();
        let file = std::fs::File::create(r.dir.join("work/submission.json")).unwrap();
        file.set_len(2_000_001).unwrap();
        r.step().unwrap();
        assert!(!r.state.done);
        assert!(
            r.state.next_message.contains("2000000 bytes"),
            "{}",
            r.state.next_message
        );
    }

    #[test]
    fn fresh_python_runs_do_not_keep_variables() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "[python]\nruns = 3\nfresh = true\nprelude = \"pre.py\"\n");
        let ed = script(&[
            "```python\nx = 1\n```",
            "```python\nprint(PRE, 'x' in globals())\n```",
            "done",
        ]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert!(turn_prompt(&r, 3).contains("7 False"), "{}", turn_prompt(&r, 3));
    }

    #[test]
    fn tools_can_share_one_pass_budget_and_name_their_folders() {
        let d = tempfile::tempdir().unwrap();
        let shared = r#"
[forward.games]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 2
pool = "passes"
out_dir = "pass_{k}"
[forward.positions]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 2
pool = "passes"
out_dir = "pass_{k}"
"#;
        let spec = task(d.path(), shared);
        let ed = script(&[
            "```games\n{}\n```",
            "```positions\n{}\n```",
            "```games\n{}\n```",
            "done",
        ]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["passes"], 2);
        assert!(r.dir.join("work/pass_1").is_dir() && r.dir.join("work/pass_2").is_dir());
        assert!(turn_prompt(&r, 3).contains("passes used 2/2"), "{}", turn_prompt(&r, 3));
        assert!(turn_prompt(&r, 4).contains("no passes left"), "{}", turn_prompt(&r, 4));
    }

    #[test]
    fn a_tool_can_draw_from_the_python_run_budget() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let extra = r#"
[python]
runs = 3
[forward.attention]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 0
pool = "python"
out_dir = "attn/r{k}"
"#;
        let spec = task(d.path(), extra);
        let ed = script(&[
            "```python\nprint(1)\n```",
            "```attention\n[\"x\"]\n```",
            "```python\nprint(2)\n```",
            "```attention\n[]\n```",
            "done",
        ]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.runs_used, 3);
        assert!(r.dir.join("work/attn/r2").is_dir());
        assert!(
            turn_prompt(&r, 3).contains("python runs used 2/3"),
            "{}",
            turn_prompt(&r, 3)
        );
        assert!(
            turn_prompt(&r, 5).contains("no Python runs left"),
            "{}",
            turn_prompt(&r, 5)
        );
    }

    #[test]
    fn a_relative_run_folder_works() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        // the same folder, written relative to the current directory
        let cwd = std::env::current_dir().unwrap();
        let up = "../".repeat(cwd.components().count() - 1);
        let rel = PathBuf::from(format!(
            "{up}{}",
            d.path().join("run").strip_prefix("/").unwrap().display()
        ));
        assert!(rel.is_relative());
        let mut r = Run::create(spec, &rel, script(&["```games\n{}\n```", "done"]), env()).unwrap();
        assert!(r.dir.is_absolute());
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["games"], 1, "{}", turn_prompt(&r, 2));
    }

    #[test]
    fn a_refusal_on_the_first_turn_keeps_the_task() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "");
        let ed = Box::new(Flaky {
            inner: Scripted::new(vec!["final".into()]),
            fail: vec![BackendError::Refused("flagged".into())],
        });
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        let p2 = turn_prompt(&r, 2);
        assert!(
            p2.contains("TASK: make W") && p2.contains("could not be delivered"),
            "{p2}"
        );
    }

    struct NoResume(Scripted);
    impl Editor for NoResume {
        fn name(&self) -> String {
            "no-resume".into()
        }
        fn send(&mut self, m: &str, n: bool) -> std::result::Result<Reply, BackendError> {
            self.0.send(m, n)
        }
        fn session(&self) -> Option<String> {
            self.0.session()
        }
        fn set_session(&mut self, _s: Option<String>) {
            // a backend without sessions has nothing to restore
        }
        fn can_resume(&self) -> bool {
            false
        }
    }

    #[test]
    fn resuming_a_backend_without_sessions_starts_a_handoff() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["```games\n{}\n```", "final"]),
            env(),
        )
        .unwrap();
        r.step().unwrap();
        drop(r);
        let ed = Box::new(NoResume(Scripted::new(vec!["final".into()])));
        let mut r = Run::resume(&d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        let p2 = turn_prompt(&r, 2);
        assert!(
            p2.contains("TASK: make W") && p2.contains("HANDOFF") && p2.contains("Forward pass 1 of 3"),
            "{p2}"
        );
    }

    #[test]
    fn a_crash_during_a_python_run_does_not_repeat_its_edits() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "[python]\nruns = 5\n");
        let code = "```python\nsave_edit('W', load('W') + 1)\nprint('did it')\n```";
        let mut r = Run::create(spec, &d.path().join("run"), script(&[code, "final"]), env()).unwrap();
        // as if the harness stopped while this run was going: the reply was received, its edit was saved,
        // but the run's output was never recorded
        r.state.turn = 1;
        r.state.pending_reply = Some(code.into());
        r.state.ledger_mark = Some(0);
        r.save_state().unwrap();
        r.gate_mut()
            .unwrap()
            .replace(
                "W",
                Tensor::new(vec![2, 2], vec![1.; 4]),
                "the interrupted run",
                "python (turn 1)",
            )
            .unwrap();
        drop(r);
        let mut r = Run::resume(&d.path().join("run"), script(&["final"]), env()).unwrap();
        r.run_to_end().unwrap();
        let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
        assert_eq!(fin.tensors["W"].data, vec![1.; 4], "the edit is applied exactly once");
        assert_eq!(r.state.runs_used, 1);
        let p2 = turn_prompt(&r, 2);
        assert!(
            p2.contains("output") && p2.contains("lost") && p2.contains("#1"),
            "{p2}"
        );
    }

    #[test]
    fn budget_charged_only_on_success() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        let ed = script(&[
            "```games\nbad\n```",
            "```games\n{\"n\": 4}\n```",
            "```python\nx\n```",
            "final answer",
        ]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["games"], 1);
        assert_eq!(r.state.failed_requests, 1);
        assert_eq!(r.state.protocol_errors, 1);
        assert!(turn_prompt(&r, 2).contains("not counted"));
        assert!(
            turn_prompt(&r, 3).contains("Forward pass 1 of 3"),
            "{}",
            turn_prompt(&r, 3)
        );
        assert!(turn_prompt(&r, 4).contains("not available"));
        assert!(r.dir.join("work/games_1").is_dir());
        assert!(r.dir.join("requests/games_1.json").exists());
    }

    #[test]
    fn nudge_when_finishing_with_passes_left() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!("{FWD}\n[finish]\nrequire_all_passes = true\nmax_nudges = 1\n"),
        );
        let ed = script(&["I am done early.", "```games\n{}\n```", "done again", "really done"]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.nudges, 1);
        assert!(
            turn_prompt(&r, 2).contains("3 games passes left"),
            "{}",
            turn_prompt(&r, 2)
        );
        // only one nudge allowed, so the third reply ends the run
        assert_eq!(r.state.turn, 3);
        assert!(r.state.done);
    }

    #[test]
    fn handoff_restarts_conversation_with_notes_ledger_and_budget() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), &format!("{FWD}\n[handoff]\nmax_turns_per_conversation = 2\n"));
        let ed = script(&["```games\n{}\n```", "```games\n{}\n```", "```games\n{}\n```", "final"]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        let p3 = turn_prompt(&r, 3);
        assert!(p3.contains("TASK: make W") && p3.contains("HANDOFF"), "{p3}");
        assert!(p3.contains("notes from pass 2"), "{p3}");
        assert!(p3.contains("(no edits yet)"), "{p3}");
        assert!(p3.contains("games passes used 2/3"), "{p3}");
        assert!(p3.contains("Forward pass 2 of 3"), "{p3}");
        assert_eq!(r.state.handoffs, 1);
    }

    struct Flaky {
        inner: Scripted,
        fail: Vec<BackendError>,
    }
    impl Editor for Flaky {
        fn name(&self) -> String {
            "flaky".into()
        }
        fn send(&mut self, m: &str, n: bool) -> std::result::Result<Reply, BackendError> {
            if !self.fail.is_empty() {
                return Err(self.fail.remove(0));
            }
            self.inner.send(m, n)
        }
        fn session(&self) -> Option<String> {
            self.inner.session()
        }
        fn set_session(&mut self, s: Option<String>) {
            self.inner.set_session(s)
        }
    }

    #[test]
    fn backend_error_retry() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        let ed = Box::new(Flaky {
            inner: Scripted::new(vec!["```games\n{}\n```".into(), "final".into()]),
            fail: vec![
                BackendError::Transient("529".into()),
                BackendError::Transient("529".into()),
                BackendError::Refused("flagged".into()),
            ],
        });
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert!(r.state.done);
        assert_eq!(r.state.backend_errors, 2);
        assert_eq!(r.state.refusals, 1);
        assert!(
            turn_prompt(&r, 2).contains("could not be delivered"),
            "{}",
            turn_prompt(&r, 2)
        );
        // a fatal error stops the run with an error, and the state is saved
        let d2 = tempfile::tempdir().unwrap();
        let spec = task(d2.path(), FWD);
        let ed = Box::new(Flaky {
            inner: Scripted::new(vec![]),
            fail: vec![BackendError::Fatal("bad model".into())],
        });
        let mut r = Run::create(spec, &d2.path().join("run"), ed, env()).unwrap();
        assert!(r.run_to_end().is_err());
        assert!(d2.path().join("run/state.json").exists());
    }

    #[test]
    fn resume_does_not_double_charge() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        let lines = ["```games\n{}\n```", "```games\n{}\n```", "final"];
        let mut r = Run::create(spec, &d.path().join("run"), script(&lines), env()).unwrap();
        r.step().unwrap();
        r.step().unwrap();
        assert_eq!(r.state.passes_used["games"], 2);
        drop(r);
        let mut r = Run::resume(&d.path().join("run"), script(&lines), env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["games"], 2);
        assert_eq!(r.state.turn, 3);
        assert!(r.state.done);
        // a reply received but not acted on before a crash is acted on exactly once
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), FWD);
        let mut r = Run::create(spec, &d.path().join("run"), script(&lines), env()).unwrap();
        r.state.pending_reply = Some("```games\n{}\n```".into());
        r.state.turn = 1;
        r.save_state().unwrap();
        drop(r);
        let mut r = Run::resume(&d.path().join("run"), script(&lines[1..]), env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["games"], 2);
    }

    #[test]
    fn final_edits_block_is_parsed_and_gated() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            "[edits]\nlanguage = true\n[edits.policy]\nmax_frac_changed = 0.5\n",
        );
        let ed =
            script(&["My design.\n```edits\nW[0, 0] = 2\nW[1, 1] = 3\nnot a statement\nW[0, 1] = 5\n```\nexplanation"]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
        // the third edit would change 3 of 4 entries (> 50%), so it is refused
        assert_eq!(fin.tensors["W"].data, vec![2., 0., 0., 3.]);
        let rep: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(r.dir.join("final/edits_report.json")).unwrap()).unwrap();
        assert_eq!(rep["applied"], 2);
        assert_eq!(rep["skipped"].as_array().unwrap().len(), 1);
        assert_eq!(rep["refused"].as_array().unwrap().len(), 1);
    }

    fn read(p: PathBuf) -> String {
        std::fs::read_to_string(&p).unwrap_or_else(|_| format!("({} is missing)", p.display()))
    }

    #[test]
    fn a_failing_pre_tool_hook_blocks_a_python_run_without_charging_it() {
        // the hook blocks before the kernel starts, so this needs no Python environment
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            "[python]\nruns = 2\n[hooks]\npre_tool = \"echo python is closed for maintenance >&2; exit 1\"\n",
        );
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["```python\nprint(1)\n```", "done"]),
            env(),
        )
        .unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.runs_used, 0);
        assert_eq!(r.state.hook_blocks, 1);
        let p2 = turn_prompt(&r, 2);
        assert!(
            p2.contains("python is closed for maintenance")
                && p2.contains("not counted")
                && p2.contains("python runs used 0/2"),
            "{p2}"
        );
        let log = read(r.dir.join("hooks.jsonl"));
        assert!(
            log.contains("\"event\":\"pre_tool\"") && log.contains("\"ok\":false"),
            "{log}"
        );
        let calls = read(r.dir.join("tool_calls.jsonl"));
        assert!(calls.contains("\"blocked\":true"), "{calls}");
    }

    #[test]
    fn a_failing_pre_tool_hook_blocks_a_probe_without_running_it() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!(
                "{FWD}\n[hooks]\npre_tool = 'if [ \"$OPENWISH_TOOL\" = games ]; then echo no games >&2; exit 1; fi'\n"
            ),
        );
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["```games\n{}\n```", "done"]),
            env(),
        )
        .unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used.get("games").copied().unwrap_or(0), 0);
        assert!(!r.dir.join("work/notes.md").exists(), "the probe command must not run");
        assert!(turn_prompt(&r, 2).contains("no games"), "{}", turn_prompt(&r, 2));
    }

    #[test]
    fn post_tool_runs_after_a_forward_probe() {
        let d = tempfile::tempdir().unwrap();
        let hook = r#"post_tool = 'test -d "$OPENWISH_RUN_DIR/work/games_1" && echo "$OPENWISH_EVENT $OPENWISH_TOOL $OPENWISH_TURN ok=$OPENWISH_TOOL_OK" >> "$OPENWISH_RUN_DIR/seen.txt"'"#;
        let spec = task(d.path(), &format!("{FWD}\n[hooks]\n{hook}\n"));
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["```games\n{}\n```", "done"]),
            env(),
        )
        .unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used["games"], 1);
        assert_eq!(read(r.dir.join("seen.txt")), "post_tool games 1 ok=1\n");
    }

    #[test]
    fn on_edit_hooks_receive_the_edit_id_from_a_python_run() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let hook = r#"on_edit = 'echo "$OPENWISH_EDIT_ID $OPENWISH_EDIT_TENSOR $OPENWISH_TOOL $OPENWISH_TURN" >> "$OPENWISH_RUN_DIR/edits_seen.txt"'"#;
        let spec = task(d.path(), &format!("[python]\nruns = 2\n[hooks]\n{hook}\n"));
        let code = "```python\nimport numpy as np\nsave_edit('W', np.eye(2), reason='identity')\nsave_edit('W', 2 * np.eye(2), reason='twice')\n```";
        let mut r = Run::create(spec, &d.path().join("run"), script(&[code, "done"]), env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(read(r.dir.join("edits_seen.txt")), "1 W python 1\n2 W python 1\n");
    }

    #[test]
    fn on_edit_hooks_run_for_the_final_edits_block() {
        let d = tempfile::tempdir().unwrap();
        let hook = r#"on_edit = 'echo "$OPENWISH_EDIT_ID $OPENWISH_EDIT_TENSOR $OPENWISH_TOOL" >> "$OPENWISH_RUN_DIR/edits_seen.txt"'"#;
        let spec = task(d.path(), &format!("[edits]\nlanguage = true\n[hooks]\n{hook}\n"));
        let mut r = Run::create(
            spec,
            &d.path().join("run"),
            script(&["```edits\nW[0, 0] = 2\nW[1, 1] = 3\n```"]),
            env(),
        )
        .unwrap();
        r.run_to_end().unwrap();
        assert_eq!(read(r.dir.join("edits_seen.txt")), "1 W edits\n2 W edits\n");
    }

    #[test]
    fn session_hooks_run_once_each_with_project_hooks_first() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), &format!("{FWD}\n[hooks]\nsession_start = 'echo task-start >> \"$OPENWISH_RUN_DIR/s.txt\"'\nsession_end = 'echo \"end $OPENWISH_END_REASON\" >> \"$OPENWISH_RUN_DIR/s.txt\"'\n"));
        let project: crate::hooks::HooksSpec = toml::from_str(
            "session_start = 'echo \"project-start $(basename \"$PWD\")\" >> \"$OPENWISH_RUN_DIR/s.txt\"'",
        )
        .unwrap();
        let proj_dir = d.path().join("proj");
        std::fs::create_dir_all(&proj_dir).unwrap();
        let mut e = env();
        e.hooks = crate::hooks::HookSet::from_spec(&project, &proj_dir, "openwish.toml");
        let lines = ["```games\n{}\n```", "final"];
        let mut r = Run::create(spec, &d.path().join("run"), script(&lines), e.clone()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(
            read(r.dir.join("s.txt")),
            "project-start proj\ntask-start\nend final answer\n"
        );
        drop(r);
        // resuming a finished run starts no new session
        let mut r = Run::resume(&d.path().join("run"), script(&lines), e).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(read(r.dir.join("s.txt")).lines().count(), 3);
    }

    #[test]
    fn an_empty_reply_is_a_protocol_error_not_a_final_answer() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(d.path(), "");
        let ed = script(&["", "   \n", "Final: done."]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.empty_replies, 2);
        assert_eq!(r.state.protocol_errors, 2);
        assert_eq!(r.state.end_reason.as_deref(), Some("final answer"));
        assert!(
            turn_prompt(&r, 2).contains("Your last reply was empty"),
            "{}",
            turn_prompt(&r, 2)
        );
        // more than MAX_EMPTY_STREAK empty replies in a row end the run
        let d2 = tempfile::tempdir().unwrap();
        let spec2 = task(d2.path(), "");
        let mut r2 = Run::create(spec2, &d2.path().join("run"), script(&["", "", "", "", "x"]), env()).unwrap();
        r2.run_to_end().unwrap();
        assert_eq!(r2.state.end_reason.as_deref(), Some("the editor sent empty replies"));
        assert_eq!(r2.state.turn, 4);
        assert!(r2.dir.join("final/net").exists());
    }

    /// An editor that reports a fixed usage per reply (for the per-episode caps).
    struct Costly {
        inner: Scripted,
        output: u64,
        cost: f64,
    }
    impl Editor for Costly {
        fn name(&self) -> String {
            "costly".into()
        }
        fn send(&mut self, m: &str, n: bool) -> std::result::Result<Reply, BackendError> {
            let mut r = self.inner.send(m, n)?;
            r.usage.output = self.output;
            r.usage.input = 10;
            r.cost_usd = Some(self.cost);
            Ok(r)
        }
        fn session(&self) -> Option<String> {
            None
        }
        fn set_session(&mut self, _: Option<String>) {}
    }

    fn capped_run(d: &Path, cap: &str, output: u64, cost: f64) -> Run {
        let spec = task(d, &format!("{FWD}[finish]\n{cap}\n"));
        let step = "Probe.\n```games\n{\"n\": 1}\n```";
        let inner = Scripted::new(vec![step.into(); 10]);
        let ed = Box::new(Costly { inner, output, cost });
        let mut r = Run::create(spec, &d.join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        r
    }

    #[test]
    fn editor_output_cap_ends_the_run_and_submits() {
        let d = tempfile::tempdir().unwrap();
        let r = capped_run(d.path(), "max_editor_output_tokens = 250", 100, 0.0);
        assert!(r.state.done);
        assert_eq!(r.state.turn, 3, "the cap is checked before each turn");
        let why = r.state.end_reason.as_deref().unwrap();
        assert!(why.contains("output-token cap"), "{why}");
        assert!(r.dir.join("final/net").exists());
    }

    #[test]
    fn editor_total_token_and_cost_caps_end_the_run() {
        let d = tempfile::tempdir().unwrap();
        let r = capped_run(d.path(), "max_editor_total_tokens = 200", 100, 0.0);
        assert_eq!(r.state.turn, 2, "110 tokens per turn");
        assert!(r.state.end_reason.as_deref().unwrap().contains("total-token cap"));
        let d = tempfile::tempdir().unwrap();
        let r = capped_run(d.path(), "max_editor_cost_usd = 1.0", 1, 0.4);
        assert_eq!(r.state.turn, 3);
        assert!(r.state.end_reason.as_deref().unwrap().contains("cost cap"));
        let d = tempfile::tempdir().unwrap();
        let r = capped_run(d.path(), "max_turns = 4", 1_000_000, 100.0);
        assert_eq!(
            r.state.end_reason.as_deref(),
            Some("turn limit reached"),
            "no cap by default"
        );
    }

    fn rationale_rule_sends_unexplained_blocks_back_then_runs_after_the_streak() {
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!("{FWD}[rationale]\nmin_words = 5\nmax_rejects_in_row = 2\n"),
        );
        let bare = "```games\n{\"n\": 1}\n```";
        let explained = "I probe the games now to check the effect of my edit.\n```games\n{\"n\": 1}\n```";
        let ed = script(&[bare, explained, bare, bare, bare, "done"]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert!(
            turn_prompt(&r, 2).contains("EXPLAIN EVERY STEP"),
            "{}",
            turn_prompt(&r, 2)
        );
        // turn 1 sent back, 2 ran, 3 and 4 sent back, 5 ran after the streak of 2
        assert_eq!(r.state.rationale_rejects, 3);
        assert_eq!(r.state.rationale_missing_run, 1);
        assert_eq!(r.state.passes_used.values().sum::<u32>(), 2);
        assert_eq!(r.state.protocol_errors, 0);
        let text = "Two words.\n```python\nx = 1 and more words\n```\nthree more words";
        assert_eq!(Run::explanation_words(text), 5);
    }

    #[test]
    fn rationale_rule_is_off_by_default_and_limited_to_its_tools() {
        if !has_venv() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let spec = task(
            d.path(),
            &format!("{FWD}[python]\nruns = 3\n[rationale]\nmin_words = 5\ntools = [\"games\"]\n"),
        );
        let ed = script(&["```python\nx = 1\n```", "```games\n{}\n```", "done"]);
        let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.runs_used, 1, "python is not under the rule");
        assert_eq!(r.state.rationale_rejects, 1);
        let d = tempfile::tempdir().unwrap();
        let mut r = Run::create(
            task(d.path(), FWD),
            &d.path().join("run"),
            script(&["```games\n{}\n```", "done"]),
            env(),
        )
        .unwrap();
        r.run_to_end().unwrap();
        assert_eq!(r.state.passes_used.values().sum::<u32>(), 1);
        assert_eq!(r.state.rationale_rejects, 0);
    }
}
