//! Task specs (TOML). Paths are relative to the spec file.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::protocol::KNOWN_TOOLS;
use openwish_tensor::gate::EditPolicy;

/// The pool of tools that draw from the Python runs (and the Python tool's own tag).
pub const PYTHON_POOL: &str = "python";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    pub id: String,
    #[serde(default)]
    pub description: String,
    /// the first message (verbatim file)
    pub prompt: PathBuf,
    /// appended to the first message (harness-specific notes), optional
    pub appendix: Option<PathBuf>,
    #[serde(default = "neutral")]
    pub instructions: String,
    pub network: Option<NetworkSpec>,
    #[serde(default)]
    pub mounts: Vec<openwish_sandbox::Mount>,
    /// paths (scoring-only assets) that must never be mounted into the sandbox
    #[serde(default)]
    pub hidden: Vec<PathBuf>,
    pub python: Option<PythonSpec>,
    #[serde(default)]
    pub forward: BTreeMap<String, ForwardSpec>,
    #[serde(default)]
    pub edits: EditsSpec,
    #[serde(default)]
    pub finish: FinishSpec,
    #[serde(default)]
    pub handoff: HandoffSpec,
    /// a plain-English explanation required with every tool call (off by default)
    #[serde(default)]
    pub rationale: RationaleSpec,
    pub score: Option<CommandSpec>,
    /// backend defaults for this task (overridable on the command line)
    #[serde(default)]
    pub editor: EditorDefaults,
    /// what the prompt and mounts give the agent (weights as text, behaviour shown, data); see access.rs
    pub access: Option<crate::access::AccessSpec>,
    /// shell commands run at fixed points of a run, in the task folder (see `hooks`)
    #[serde(default)]
    pub hooks: crate::hooks::HooksSpec,
    /// set by `load`: the directory of the spec file
    #[serde(skip)]
    pub dir: PathBuf,
}

fn neutral() -> String {
    openwish_llm::NEUTRAL_INSTRUCTIONS.to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkSpec {
    /// an .npz file or a directory of .npy files
    pub base: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonSpec {
    pub runs: u32,
    #[serde(default = "d_timeout")]
    pub timeout_s: u64,
    #[serde(default = "d_mem")]
    pub mem_gb: u64,
    #[serde(default = "d_out")]
    pub max_out: usize,
    pub prelude: Option<PathBuf>,
    #[serde(default = "d_threads")]
    pub threads: u32,
    /// start every run in a fresh process (nothing kept between runs), as in the authors' Experiment 3
    #[serde(default)]
    pub fresh: bool,
    /// a different Python environment for this task (default: the harness's sandbox venv)
    pub venv: Option<PathBuf>,
    /// Static-check rules the task rejects before execution. Other flags stay advisory.
    #[serde(default)]
    pub reject_flags: Vec<String>,
}
fn d_timeout() -> u64 {
    900
}
fn d_mem() -> u64 {
    16
}
fn d_out() -> usize {
    10_000
}
fn d_threads() -> u32 {
    8
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardSpec {
    /// argv; "{openwish}" is replaced by the running openwish binary, relative paths resolve against the spec dir
    pub command: Vec<String>,
    pub passes: u32,
    #[serde(default = "d_out")]
    pub max_out: usize,
    #[serde(default = "d_fwd_timeout")]
    pub timeout_s: u64,
    /// tools with the same pool share one budget (the pool's `passes`)
    pub pool: Option<String>,
    /// the pass's folder under /work; "{k}" = pass number, "{tag}" = tool tag (default "{tag}_{k}")
    pub out_dir: Option<String>,
}

impl ForwardSpec {
    /// The pass folder pattern with the tool tag filled in ("{k}" is left for the pass number).
    pub fn folder_pattern(&self, tag: &str) -> String {
        self.out_dir.as_deref().unwrap_or("{tag}_{k}").replace("{tag}", tag)
    }
    pub fn folder(&self, tag: &str, k: u32) -> String {
        self.folder_pattern(tag).replace("{k}", &k.to_string())
    }
}
fn d_fwd_timeout() -> u64 {
    7200
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditsSpec {
    /// parse the ```edits block of the final reply (the authors' Experiments 4 and 5)
    #[serde(default)]
    pub language: bool,
    #[serde(default)]
    pub policy: EditPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinishSpec {
    #[serde(default)]
    pub require_all_passes: bool,
    #[serde(default = "d_turns")]
    pub max_turns: u32,
    #[serde(default = "d_nudges")]
    pub max_nudges: u32,
    /// Optional format-only validation before accepting a final reply.
    pub validation: Option<ValidationSpec>,
    /// Per-episode editor caps, 0 for no cap. A run past a cap ends before the next turn and submits the current
    /// edit. Output tokens include reasoning. Total tokens add input, cached input and output over every turn.
    #[serde(default)]
    pub max_editor_output_tokens: u64,
    #[serde(default)]
    pub max_editor_total_tokens: u64,
    /// Enforced only for backends reporting cost.
    #[serde(default)]
    pub max_editor_cost_usd: f64,
}

/// The explanation rule. A reply with a tool block needs `min_words` words of plain English outside the block.
/// The run sends a block with fewer words back unrun and uncharged, at most `max_rejects_in_row` times in a row,
/// then runs the block and logs the missing explanation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationaleSpec {
    /// words outside the tool block the reply needs (0 = rule off)
    #[serde(default)]
    pub min_words: usize,
    /// the tools under the rule (empty = every tool)
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default = "d_rejects")]
    pub max_rejects_in_row: u32,
}
fn d_rejects() -> u32 {
    2
}
impl Default for RationaleSpec {
    fn default() -> Self {
        RationaleSpec {
            min_words: 0,
            tools: vec![],
            max_rejects_in_row: d_rejects(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationSpec {
    pub command: Vec<String>,
    #[serde(default = "d_validation_timeout")]
    pub timeout_s: u64,
    /// Optional file under /work, preferred over the final reply when present.
    pub submission_file: Option<PathBuf>,
}
fn d_validation_timeout() -> u64 {
    60
}
fn d_turns() -> u32 {
    400
}
fn d_nudges() -> u32 {
    5
}
impl Default for FinishSpec {
    fn default() -> Self {
        FinishSpec {
            require_all_passes: false,
            max_turns: d_turns(),
            max_nudges: d_nudges(),
            validation: None,
            max_editor_output_tokens: 0,
            max_editor_total_tokens: 0,
            max_editor_cost_usd: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffSpec {
    /// start a fresh conversation when the context reaches this many tokens (0 = never)
    #[serde(default = "d_ctx")]
    pub max_conversation_tokens: u64,
    /// and/or after this many turns (0 = never)
    #[serde(default)]
    pub max_turns_per_conversation: u32,
}
fn d_ctx() -> u64 {
    700_000
}
impl Default for HandoffSpec {
    fn default() -> Self {
        HandoffSpec {
            max_conversation_tokens: d_ctx(),
            max_turns_per_conversation: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandSpec {
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditorDefaults {
    pub backend: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
}

impl TaskSpec {
    /// Read and parse a spec without checking that its files exist (for listings).
    pub fn parse(path: &Path) -> Result<TaskSpec> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut t: TaskSpec = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        t.dir = crate::paths::folder_of(path)?;
        Ok(t)
    }

    /// Parse a spec and check it: its files exist, tool tags and pools are valid, no mount exposes a hidden asset.
    pub fn load(path: &Path) -> Result<TaskSpec> {
        let t = TaskSpec::parse(path)?;
        if let Some(a) = &t.access {
            a.check()?;
        }
        t.check_files()?;
        t.check_tool_tags()?;
        t.check_pools()?;
        t.check_mounts()?;
        let tools = t.enabled_tools();
        if let Some(tag) = t.rationale.tools.iter().find(|tag| !tools.contains(tag)) {
            bail!("rationale.tools names {tag:?}, which the task does not enable (enabled: {tools:?})");
        }
        if let Some(py) = &t.python {
            for rule in &py.reject_flags {
                if !openwish_audit::known_rule(rule) {
                    bail!("unknown static rule {rule:?} in python.reject_flags");
                }
            }
        }
        Ok(t)
    }

    /// (what, path) for every file the task needs: the prompt, the appendix, the network and the prelude.
    pub fn required_files(&self) -> Vec<(&'static str, &Path)> {
        let mut files = vec![("prompt", self.prompt.as_path())];
        files.extend(self.appendix.as_deref().map(|a| ("appendix", a)));
        files.extend(self.network.as_ref().map(|n| ("network", n.base.as_path())));
        files.extend(
            self.python
                .as_ref()
                .and_then(|py| py.prelude.as_deref())
                .map(|p| ("prelude", p)),
        );
        files
    }

    fn check_files(&self) -> Result<()> {
        for (what, p) in self.required_files() {
            if !self.resolve(p).exists() {
                bail!(
                    "{what} {} does not exist (relative to {})",
                    p.display(),
                    self.dir.display()
                );
            }
        }
        Ok(())
    }

    fn check_tool_tags(&self) -> Result<()> {
        let probe_tags: Vec<&str> = KNOWN_TOOLS.iter().copied().filter(|t| *t != PYTHON_POOL).collect();
        for tag in self.forward.keys() {
            if !probe_tags.contains(&tag.as_str()) {
                bail!("forward tool tag {tag:?} must be one of {probe_tags:?} (not python)");
            }
        }
        Ok(())
    }

    /// Tools in one pool share one budget, so they must give the same one.
    fn check_pools(&self) -> Result<()> {
        let mut pools: BTreeMap<&str, u32> = BTreeMap::new();
        for (tag, f) in &self.forward {
            match f.pool.as_deref() {
                Some(PYTHON_POOL) if self.python.is_none() => {
                    bail!("{tag} draws from the Python runs (pool = \"python\") but the task has no [python] section")
                }
                Some(PYTHON_POOL) | None => {}
                Some(p) => {
                    if let Some(prev) = pools.insert(p, f.passes).filter(|prev| *prev != f.passes) {
                        bail!(
                            "tools in pool {p:?} give different budgets ({prev} and {} for {tag})",
                            f.passes
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn check_mounts(&self) -> Result<()> {
        let hidden: Vec<PathBuf> = self
            .hidden
            .iter()
            .filter_map(|h| std::fs::canonicalize(self.resolve(h)).ok())
            .collect();
        for m in &self.mounts {
            let host =
                std::fs::canonicalize(self.resolve(&m.host)).with_context(|| format!("mount {}", m.host.display()))?;
            if let Some(h) = hidden.iter().find(|h| h.starts_with(&host) || host.starts_with(h)) {
                bail!(
                    "mount {} would expose hidden scoring asset {}",
                    host.display(),
                    h.display()
                );
            }
        }
        Ok(())
    }

    pub fn resolve(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.dir.join(p)
        }
    }
    /// Tags the editor may use as tool blocks.
    pub fn enabled_tools(&self) -> Vec<String> {
        let mut v: Vec<String> = self.forward.keys().cloned().collect();
        if self.python.is_some() {
            v.push(PYTHON_POOL.into());
        }
        v.sort();
        v
    }
    /// The budget group of a tool: its pool, or the tag itself.
    pub fn group(&self, tag: &str) -> String {
        self.forward
            .get(tag)
            .and_then(|f| f.pool.clone())
            .unwrap_or_else(|| tag.to_string())
    }
    /// How a budget group is named in messages.
    pub fn group_label(&self, group: &str) -> String {
        if self.forward.values().any(|f| f.pool.as_deref() == Some(group)) {
            group.to_string()
        } else {
            format!("{group} passes")
        }
    }
    /// (group, label, budget) for every budget group
    pub fn groups(&self) -> Vec<(String, String, u32)> {
        let mut out: Vec<(String, String, u32)> = vec![];
        for (tag, f) in &self.forward {
            if f.pool.as_deref() == Some(PYTHON_POOL) {
                continue; // these tools count against the Python runs
            }
            let g = self.group(tag);
            if !out.iter().any(|(x, _, _)| *x == g) {
                out.push((g.clone(), self.group_label(&g), f.passes));
            }
        }
        out
    }
    /// argv with "{openwish}" (this binary), "{task}" (the spec dir) and "{root}" (the workspace root, OPENWISH_ROOT) substituted.
    pub fn argv(&self, cmd: &[String]) -> Vec<String> {
        let bin = std::env::var("OPENWISH_BIN").ok().unwrap_or_else(|| {
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "openwish".into())
        });
        let root = std::env::var("OPENWISH_ROOT").unwrap_or_else(|_| self.dir.display().to_string());
        cmd.iter()
            .map(|a| {
                a.replace("{openwish}", &bin)
                    .replace("{task}", &self.dir.display().to_string())
                    .replace("{root}", &root)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_static_policy_rules_are_rejected() {
        let d = tempfile::tempdir().unwrap();
        let path = write(
            d.path(),
            "id = \"t\"\nprompt = \"prompt.txt\"\n[python]\nruns = 1\nreject_flags = [\"typo\"]\n",
        );
        let error = TaskSpec::load(&path).unwrap_err().to_string();
        assert!(error.contains("unknown static rule"), "{error}");
    }

    #[test]
    fn rationale_tools_must_be_enabled_tools() {
        let d = tempfile::tempdir().unwrap();
        let toml = "id = \"t\"\nprompt = \"prompt.txt\"\n[python]\nruns = 1\n[rationale]\nmin_words = 5\n";
        let t = TaskSpec::load(&write(d.path(), &format!("{toml}tools = [\"python\"]\n"))).unwrap();
        assert_eq!((t.rationale.min_words, t.rationale.max_rejects_in_row), (5, 2));
        let error = TaskSpec::load(&write(d.path(), &format!("{toml}tools = [\"pyhton\"]\n")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("rationale.tools"), "{error}");
    }

    fn write(d: &Path, toml: &str) -> PathBuf {
        std::fs::write(d.join("prompt.txt"), "hi").unwrap();
        let p = d.join("task.toml");
        std::fs::write(&p, toml).unwrap();
        p
    }

    #[test]
    fn loads_minimal_and_full_specs() {
        let d = tempfile::tempdir().unwrap();
        let t = TaskSpec::load(&write(d.path(), "id = \"t\"\nprompt = \"prompt.txt\"\n")).unwrap();
        assert!(t.enabled_tools().is_empty());
        assert_eq!(t.instructions, "Answer the user's request.");
        let t = TaskSpec::load(&write(
            d.path(),
            r#"
id = "t2"
prompt = "prompt.txt"
[python]
runs = 100
[forward.games]
command = ["{openwish}", "go-pass", "games", "{task}/opponents"]
passes = 10
[edits.policy]
forbidden = ["norm.*"]
forbid_whole_tensor_scale = true
"#,
        ))
        .unwrap();
        assert_eq!(t.enabled_tools(), vec!["games".to_string(), "python".to_string()]);
        let argv = t.argv(&t.forward["games"].command);
        assert_eq!(argv[1..3], ["go-pass".to_string(), "games".to_string()]);
        assert!(argv[0].ends_with("openwish") || argv[0].contains("deps"), "{}", argv[0]);
        assert_eq!(argv[3], d.path().join("opponents").display().to_string());
        assert!(t.edits.policy.forbid_whole_tensor_scale);
    }

    #[test]
    fn rejects_unknown_fields_missing_prompt_and_hidden_mounts() {
        let d = tempfile::tempdir().unwrap();
        assert!(TaskSpec::load(&write(d.path(), "id = \"t\"\nprompt = \"prompt.txt\"\nbogus = 1\n")).is_err());
        assert!(TaskSpec::load(&write(d.path(), "id = \"t\"\nprompt = \"nope.txt\"\n")).is_err());
        std::fs::create_dir_all(d.path().join("assets/keys")).unwrap();
        let bad = "id = \"t\"\nprompt = \"prompt.txt\"\nhidden = [\"assets/keys\"]\n[[mounts]]\nhost = \"assets\"\nguest = \"/data\"\n";
        let e = TaskSpec::load(&write(d.path(), bad)).unwrap_err();
        assert!(format!("{e:#}").contains("hidden"), "{e:#}");
    }

    #[test]
    fn a_task_can_carry_hooks_and_unknown_hook_keys_are_refused() {
        let d = tempfile::tempdir().unwrap();
        let t = TaskSpec::load(&write(
            d.path(),
            "id = \"t\"\nprompt = \"prompt.txt\"\n[hooks]\npre_tool = \"true\"\non_edit = [\"a\", \"b\"]\n",
        ))
        .unwrap();
        assert_eq!(t.hooks.pre_tool, vec!["true".to_string()]);
        assert_eq!(t.hooks.on_edit.len(), 2);
        assert!(TaskSpec::load(&write(
            d.path(),
            "id = \"t\"\nprompt = \"prompt.txt\"\n[hooks]\nbefore_tool = \"true\"\n"
        ))
        .is_err());
        // a spec saved into a run folder reads back with its hooks
        let back: TaskSpec = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back.hooks, t.hooks);
        // parse skips the file checks that load makes
        let p = write(d.path(), "id = \"t\"\nprompt = \"nope.txt\"\n");
        assert!(TaskSpec::load(&p).is_err());
        assert_eq!(TaskSpec::parse(&p).unwrap().id, "t");
    }
}
