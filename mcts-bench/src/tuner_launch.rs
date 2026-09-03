//! Detached launcher and operational journal for foreground `tuner_cli` runs.
//!
//! This deliberately does *not* go through [`crate::supervised_launch`]. That
//! seam is built around the bench attempt/journal lifecycle model (attempt
//! ids, launch nonces, wrapper-process readiness evidence, a DuckDB
//! projection). A version-4 tuner run's `<run-dir>/{manifest,evidence,report}`
//! triple is already its own scientific authority; imposing a second
//! lifecycle store on it would contradict that. What the bench server needs
//! here is only enough operational metadata to stop a launched run and answer
//! "is its process alive", so this module keeps a small append-only journal
//! (`<runs-root>/launches.jsonl`) and nothing else.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

use serde::{Deserialize, Serialize};

/// How long [`spawn_and_journal`] watches a freshly-spawned `tuner_cli` before
/// it hands off to the async reaper. A child that dies inside this window
/// never began real work (it failed argument/objective validation, a missing
/// binary, an already-populated run dir, an import error) -- so the launch
/// call itself fails loudly with the child's `launch.err`, instead of
/// returning `202 Accepted` for a run that is already dead.
pub const STARTUP_GRACE: Duration = Duration::from_millis(2500);

/// Last `max_bytes` bytes of `path` as a lossy string, or a short marker if it
/// cannot be read. Used to fold a failed child's `launch.err` into the launch
/// error so the operator sees the actual cause without opening a file.
fn tail(path: &Path, max_bytes: u64) -> String {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return format!("(no readable {})", path.display()),
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if len > max_bytes {
        let _ = file.seek(SeekFrom::Start(len - max_bytes));
    }
    let mut buf = String::new();
    if file.read_to_string(&mut buf).is_err() {
        return format!("(unreadable {})", path.display());
    }
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        "(launch.err was empty)".to_string()
    } else {
        trimmed.to_string()
    }
}

/// A launch request for one foreground `tuner_cli` run.
///
/// Only the fields `tuner_cli`'s argument parser marks `required` are
/// mandatory here. Every optional knob is `Option`: a `None` is simply not
/// passed on the command line, leaving `tuner_cli`'s own default as the single
/// source of truth rather than duplicating it in Rust.
#[derive(Clone, Debug, Deserialize)]
pub struct TunerLaunchRequest {
    /// Absolute path to the `game-<kind>` binary. The bench server resolves
    /// this from a `game_kind` key so an API caller never handles a
    /// filesystem path; a direct caller may still set it.
    #[serde(default)]
    pub game_binary: PathBuf,
    /// Absolute path to the frozen-objective JSON. Resolved by the bench
    /// server from an `objective_key` for the same reason as `game_binary`.
    #[serde(default)]
    pub objective_file: PathBuf,
    /// Built-in `game-<kind>` key the bench server resolves into
    /// `game_binary`. Ignored once `game_binary` is set.
    #[serde(default)]
    pub game_kind: Option<String>,
    /// Objective-file stem the bench server resolves into `objective_file`
    /// against its configured objectives directory. Ignored once
    /// `objective_file` is set.
    #[serde(default)]
    pub objective_key: Option<String>,
    #[serde(skip_deserializing)]
    pub runs_root: PathBuf,
    pub run_id: String,
    pub task_seed: i64,
    pub tuning_pair_budget: u64,
    pub validation_pair_budget: u64,
    pub production_validation_pairs: u64,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub cohort_size: Option<u64>,
    #[serde(default)]
    pub finalists: Option<u64>,
    #[serde(default)]
    pub bootstrap_candidates: Option<u64>,
    #[serde(default)]
    pub random_reserve_candidates: Option<u64>,
    #[serde(default)]
    pub tuning_pairs: Option<u64>,
    #[serde(default)]
    pub diagnostic_pair_budget: Option<u64>,
    #[serde(default)]
    pub pair_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub evaluator_workers: Option<u64>,
    #[serde(default)]
    pub proposer_policy: Option<String>,
    #[serde(default)]
    pub shadow_practical_margin: Option<f64>,
    #[serde(default)]
    pub shadow_elimination_threshold: Option<f64>,
    #[serde(default)]
    pub shadow_policy: Option<String>,
    #[serde(default)]
    pub shadow_halving_spare_margin: Option<f64>,
    #[serde(default)]
    pub active_elimination_audit_probability: Option<f64>,
    #[serde(default)]
    pub tuning_max_iterations: Option<u64>,
    #[serde(default)]
    pub tuning_max_time_ms: Option<u64>,
    #[serde(default)]
    pub validation_max_iterations: Option<u64>,
    #[serde(default)]
    pub validation_max_time_ms: Option<u64>,
    #[serde(default)]
    pub production_max_iterations: Option<u64>,
    #[serde(default)]
    pub production_max_time_ms: Option<u64>,
    #[serde(default)]
    pub exclude_family: Vec<String>,
}

impl TunerLaunchRequest {
    pub fn run_dir(&self) -> PathBuf {
        self.runs_root.join(&self.run_id)
    }

    pub fn validate(&self) -> io::Result<()> {
        if !safe_run_id(&self.run_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "run id must be one safe path segment",
            ));
        }
        if self.game_binary.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "game binary could not be resolved (unknown game_kind?)",
            ));
        }
        if self.objective_file.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "objective file could not be resolved (unknown objective_key?)",
            ));
        }
        for (iterations, time, name) in [
            (self.tuning_max_iterations, self.tuning_max_time_ms, "tuning"),
            (
                self.validation_max_iterations,
                self.validation_max_time_ms,
                "validation",
            ),
            (
                self.production_max_iterations,
                self.production_max_time_ms,
                "production",
            ),
        ] {
            if iterations.is_some() && time.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} effort accepts either iterations or time"),
                ));
            }
        }
        Ok(())
    }

    pub fn argv(&self) -> Vec<String> {
        let mut argv = vec![
            "uv".into(),
            "run".into(),
            "--project".into(),
            "tuner".into(),
            "python".into(),
            "-m".into(),
            "tuner_cli".into(),
            "--game-binary".into(),
            self.game_binary.to_string_lossy().into_owned(),
            "--objective-file".into(),
            self.objective_file.to_string_lossy().into_owned(),
            "--run-dir".into(),
            self.run_dir().to_string_lossy().into_owned(),
            "--task-seed".into(),
            self.task_seed.to_string(),
            "--tuning-pair-budget".into(),
            self.tuning_pair_budget.to_string(),
            "--validation-pair-budget".into(),
            self.validation_pair_budget.to_string(),
            "--production-validation-pairs".into(),
            self.production_validation_pairs.to_string(),
        ];
        let mut push = |flag: &str, value: Option<String>| {
            if let Some(value) = value {
                argv.push(flag.into());
                argv.push(value);
            }
        };
        push("--seed", self.seed.map(|v| v.to_string()));
        push("--cohort-size", self.cohort_size.map(|v| v.to_string()));
        push("--finalists", self.finalists.map(|v| v.to_string()));
        push(
            "--bootstrap-candidates",
            self.bootstrap_candidates.map(|v| v.to_string()),
        );
        push(
            "--random-reserve-candidates",
            self.random_reserve_candidates.map(|v| v.to_string()),
        );
        push("--tuning-pairs", self.tuning_pairs.map(|v| v.to_string()));
        push(
            "--diagnostic-pair-budget",
            self.diagnostic_pair_budget.map(|v| v.to_string()),
        );
        push(
            "--pair-timeout-seconds",
            self.pair_timeout_seconds.map(|v| v.to_string()),
        );
        push(
            "--evaluator-workers",
            self.evaluator_workers.map(|v| v.to_string()),
        );
        push("--proposer-policy", self.proposer_policy.clone());
        push(
            "--shadow-practical-margin",
            self.shadow_practical_margin.map(|v| v.to_string()),
        );
        push(
            "--shadow-elimination-threshold",
            self.shadow_elimination_threshold.map(|v| v.to_string()),
        );
        push("--shadow-policy", self.shadow_policy.clone());
        push(
            "--shadow-halving-spare-margin",
            self.shadow_halving_spare_margin.map(|v| v.to_string()),
        );
        push(
            "--active-elimination-audit-probability",
            self.active_elimination_audit_probability
                .map(|v| v.to_string()),
        );
        push(
            "--tuning-max-iterations",
            self.tuning_max_iterations.map(|v| v.to_string()),
        );
        push(
            "--tuning-max-time-ms",
            self.tuning_max_time_ms.map(|v| v.to_string()),
        );
        push(
            "--validation-max-iterations",
            self.validation_max_iterations.map(|v| v.to_string()),
        );
        push(
            "--validation-max-time-ms",
            self.validation_max_time_ms.map(|v| v.to_string()),
        );
        push(
            "--production-max-iterations",
            self.production_max_iterations.map(|v| v.to_string()),
        );
        push(
            "--production-max-time-ms",
            self.production_max_time_ms.map(|v| v.to_string()),
        );
        for family in &self.exclude_family {
            argv.push("--exclude-family".into());
            argv.push(family.clone());
        }
        argv
    }
}

/// A request to raise one or more of a frozen run's pair budgets and resume it.
///
/// The tuner never edits `manifest.compute_budget`; the extension is recorded
/// as one append-only `budget_extended` evidence event by `tuner_cli --resume
/// --extend-*`, which replay folds into the effective budget (re-opening the
/// run if it had already completed).
#[derive(Clone, Debug, Deserialize)]
pub struct BudgetExtension {
    #[serde(default)]
    pub tuning_pair_attempts_delta: u64,
    #[serde(default)]
    pub validation_pair_attempts_delta: u64,
    #[serde(default)]
    pub diagnostic_pair_attempts_delta: u64,
    pub reason: String,
}

/// Strip any prior `--resume` / `--extend-*` flags (and their values) from a
/// recorded launch argv so a fresh resume can append its own.
fn resumable_argv(argv: &[String]) -> Vec<String> {
    let takes_value = |flag: &str| {
        matches!(
            flag,
            "--extend-tuning-pairs"
                | "--extend-validation-pairs"
                | "--extend-diagnostic-pairs"
                | "--extend-reason"
                | "--extend-requested-at"
        )
    };
    let mut kept = Vec::with_capacity(argv.len());
    let mut skip_next = false;
    for arg in argv {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--resume" {
            continue;
        }
        if takes_value(arg) {
            skip_next = true;
            continue;
        }
        kept.push(arg.clone());
    }
    kept
}

/// Record a `budget_extended` event on `run_id` and relaunch it with
/// `--resume`. Reuses the run's most recent launch argv (its frozen scientific
/// inputs) and adds only the resume and extension flags.
pub fn extend(
    root: &Path,
    run_id: &str,
    extension: &BudgetExtension,
) -> io::Result<TunerLaunchRecord> {
    if extension.reason.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a budget extension requires a reason",
        ));
    }
    if extension.tuning_pair_attempts_delta == 0
        && extension.validation_pair_attempts_delta == 0
        && extension.diagnostic_pair_attempts_delta == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a budget extension must raise at least one budget",
        ));
    }
    let record = records(root)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "tuner run not found"))?;
    if !record.run_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "tuner run directory is missing",
        ));
    }
    let mut argv = resumable_argv(&record.argv);
    argv.push("--resume".into());
    for (flag, delta) in [
        ("--extend-tuning-pairs", extension.tuning_pair_attempts_delta),
        (
            "--extend-validation-pairs",
            extension.validation_pair_attempts_delta,
        ),
        (
            "--extend-diagnostic-pairs",
            extension.diagnostic_pair_attempts_delta,
        ),
    ] {
        argv.push(flag.into());
        argv.push(delta.to_string());
    }
    argv.push("--extend-reason".into());
    argv.push(extension.reason.clone());
    argv.push("--extend-requested-at".into());
    argv.push(crate::launch::iso_timestamp());
    let run_dir = record.run_dir.clone();
    spawn_and_journal(root, run_id, run_dir, argv, false)
}

pub fn safe_run_id(run_id: &str) -> bool {
    !run_id.is_empty()
        && run_id != "."
        && run_id != ".."
        && run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Exited,
    Signalled,
    SpawnFailed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TunerLaunchRecord {
    pub run_id: String,
    pub argv: Vec<String>,
    pub run_dir: PathBuf,
    pub pid: Option<u32>,
    pub started_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_outcome: Option<TerminalOutcome>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum JournalEvent {
    Launch {
        record: TunerLaunchRecord,
    },
    Terminal {
        run_id: String,
        outcome: TerminalOutcome,
    },
}

pub fn launch(request: &TunerLaunchRequest) -> io::Result<TunerLaunchRecord> {
    request.validate()?;
    let run_dir = request.run_dir();
    if run_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "run directory already exists",
        ));
    }
    fs::create_dir_all(&run_dir)?;
    let argv = request.argv();
    spawn_and_journal(&request.runs_root, &request.run_id, run_dir, argv, true)
}

/// Spawn `argv` as a detached child in its own process group, redirect its
/// output into the run directory, and record the launch (and, from a reaper
/// thread, its terminal outcome) in the runs-root journal.
fn spawn_and_journal(
    runs_root: &Path,
    run_id: &str,
    run_dir: PathBuf,
    argv: Vec<String>,
    verify_startup: bool,
) -> io::Result<TunerLaunchRecord> {
    let mut command = Command::new(&argv[0]);
    command.args(&argv[1..]);
    command.stdout(Stdio::from(fs::File::create(run_dir.join("launch.out"))?));
    command.stderr(Stdio::from(fs::File::create(run_dir.join("launch.err"))?));
    #[cfg(unix)]
    command.process_group(0);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let record = TunerLaunchRecord {
                run_id: run_id.to_owned(),
                argv,
                run_dir,
                pid: None,
                started_at: crate::launch::iso_timestamp(),
                terminal_outcome: None,
            };
            append_launch(runs_root, &record)?;
            append_terminal(runs_root, run_id, TerminalOutcome::SpawnFailed)?;
            return Err(error);
        }
    };
    let record = TunerLaunchRecord {
        run_id: run_id.to_owned(),
        argv,
        run_dir,
        pid: Some(child.id()),
        started_at: crate::launch::iso_timestamp(),
        terminal_outcome: None,
    };
    append_launch(runs_root, &record)?;

    // Watch a fresh launch for a short grace window. If it dies this fast it
    // never started working, and returning `Ok` here would report a launched
    // run that is already a corpse -- so classify it as a startup failure and
    // surface the child's own `launch.err`. Skipped for `--resume` relaunches:
    // a small budget extension can legitimately complete inside the window.
    let deadline = Instant::now() + STARTUP_GRACE;
    while verify_startup && Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => {
                #[cfg(unix)]
                let outcome = if status.signal().is_some() {
                    TerminalOutcome::Signalled
                } else {
                    TerminalOutcome::Exited
                };
                #[cfg(not(unix))]
                let outcome = TerminalOutcome::Exited;
                let _ = append_terminal(runs_root, run_id, outcome);
                let how = match status.code() {
                    Some(code) => format!("exit status {code}"),
                    None => "a signal".to_string(),
                };
                return Err(io::Error::other(format!(
                    "tuner run '{run_id}' died during startup ({how}) without beginning work. \
                     Its launch.err said:\n{}",
                    tail(&record.run_dir.join("launch.err"), 4096),
                )));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(error),
        }
    }

    let root = runs_root.to_owned();
    let id = run_id.to_owned();
    std::thread::spawn(move || {
        let outcome = match child.wait() {
            #[cfg(unix)]
            Ok(status) if status.signal().is_some() => TerminalOutcome::Signalled,
            Ok(_) => TerminalOutcome::Exited,
            Err(_) => TerminalOutcome::Exited,
        };
        let _ = append_terminal(&root, &id, outcome);
    });
    Ok(record)
}

pub fn append_launch(root: &Path, record: &TunerLaunchRecord) -> io::Result<()> {
    append(
        root,
        &JournalEvent::Launch {
            record: record.clone(),
        },
    )
}

pub fn append_terminal(root: &Path, run_id: &str, outcome: TerminalOutcome) -> io::Result<()> {
    if records(root)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .is_some_and(|record| record.terminal_outcome.is_some())
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "terminal outcome is already recorded",
        ));
    }
    append(
        root,
        &JournalEvent::Terminal {
            run_id: run_id.into(),
            outcome,
        },
    )
}

fn append(root: &Path, event: &JournalEvent) -> io::Result<()> {
    fs::create_dir_all(root)?;
    // One line, one `write_all`: an `O_APPEND` write of a whole record is
    // atomic against concurrent launches; a split json-then-newline is not.
    let mut line = serde_json::to_vec(event).map_err(io::Error::other)?;
    line.push(b'\n');
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("launches.jsonl"))?
        .write_all(&line)
}

/// All launch records, newest launch last, each folded together with its
/// terminal outcome if one has been recorded.
pub fn records(root: &Path) -> io::Result<Vec<TunerLaunchRecord>> {
    let path = root.join("launches.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut order: Vec<String> = Vec::new();
    let mut by_id: HashMap<String, TunerLaunchRecord> = HashMap::new();
    for line in BufReader::new(fs::File::open(path)?).lines() {
        let event: JournalEvent = serde_json::from_str(&line?).map_err(io::Error::other)?;
        match event {
            JournalEvent::Launch { record } => {
                if !by_id.contains_key(&record.run_id) {
                    order.push(record.run_id.clone());
                }
                by_id.insert(record.run_id.clone(), record);
            }
            JournalEvent::Terminal { run_id, outcome } => {
                if let Some(record) = by_id.get_mut(&run_id) {
                    record.terminal_outcome = Some(outcome);
                }
            }
        }
    }
    Ok(order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect())
}

pub fn is_alive(pid: u32) -> bool {
    crate::launch::is_alive(pid)
}

/// Deliver `SIGINT` to the launched run's whole process group (the child is
/// spawned into its own group), so `tuner_cli` and any evaluator workers it
/// spawned all see it. `tuner_cli` maps `KeyboardInterrupt` to exit 130 and
/// leaves the run resumable.
pub fn interrupt(pid: u32) -> io::Result<()> {
    let status = Command::new("kill")
        .arg("-INT")
        .arg(format!("-{pid}"))
        .status()?;
    if status.success() {
        return Ok(());
    }
    // Fall back to the bare pid if the group signal was rejected (e.g. the
    // child never became a group leader on this platform).
    let status = Command::new("kill")
        .arg("-INT")
        .arg(pid.to_string())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "process is no longer alive",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request(root: &Path, run_id: &str) -> TunerLaunchRequest {
        TunerLaunchRequest {
            game_binary: "/games/druid".into(),
            objective_file: "/objectives/default.yaml".into(),
            game_kind: None,
            objective_key: None,
            runs_root: root.to_owned(),
            run_id: run_id.into(),
            task_seed: 7,
            tuning_pair_budget: 10,
            validation_pair_budget: 20,
            production_validation_pairs: 30,
            seed: None,
            cohort_size: None,
            finalists: None,
            bootstrap_candidates: None,
            random_reserve_candidates: None,
            tuning_pairs: None,
            diagnostic_pair_budget: None,
            pair_timeout_seconds: None,
            evaluator_workers: None,
            proposer_policy: None,
            shadow_practical_margin: None,
            shadow_elimination_threshold: None,
            shadow_policy: None,
            shadow_halving_spare_margin: None,
            active_elimination_audit_probability: None,
            tuning_max_iterations: None,
            tuning_max_time_ms: None,
            validation_max_iterations: None,
            validation_max_time_ms: None,
            production_max_iterations: None,
            production_max_time_ms: None,
            exclude_family: vec![],
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mcts-tuner-launch-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn argv_carries_required_flags_and_omits_unset_optionals() {
        let root = PathBuf::from("runs");
        let argv = base_request(&root, "run_12a").argv();
        assert_eq!(
            &argv[..7],
            ["uv", "run", "--project", "tuner", "python", "-m", "tuner_cli"]
        );
        assert!(argv
            .windows(2)
            .any(|pair| pair == ["--run-dir", "runs/run_12a"]));
        assert!(argv
            .windows(2)
            .any(|pair| pair == ["--tuning-pair-budget", "10"]));
        // Unset optionals never reach the command line -- tuner_cli owns the
        // default.
        assert!(!argv.iter().any(|arg| arg == "--seed"));
        assert!(!argv.iter().any(|arg| arg == "--proposer-policy"));
        assert!(!argv.iter().any(|arg| arg == "--shadow-halving-spare-margin"));
    }

    #[test]
    fn argv_emits_only_the_optionals_that_are_set() {
        let root = PathBuf::from("runs");
        let mut request = base_request(&root, "run_12a");
        request.seed = Some(99);
        request.proposer_policy = Some("random".into());
        request.tuning_max_iterations = Some(1000);
        request.exclude_family = vec!["ucb".into(), "grave".into()];
        let argv = request.argv();
        assert!(argv.windows(2).any(|pair| pair == ["--seed", "99"]));
        assert!(argv
            .windows(2)
            .any(|pair| pair == ["--proposer-policy", "random"]));
        assert!(argv
            .windows(2)
            .any(|pair| pair == ["--tuning-max-iterations", "1000"]));
        assert_eq!(
            argv.iter().filter(|arg| *arg == "--exclude-family").count(),
            2
        );
    }

    #[test]
    fn validate_rejects_unsafe_run_id_and_double_effort() {
        let root = PathBuf::from("runs");
        assert!(base_request(&root, "../escape").validate().is_err());
        assert!(base_request(&root, "ok").validate().is_ok());
        let mut both = base_request(&root, "ok");
        both.tuning_max_iterations = Some(1);
        both.tuning_max_time_ms = Some(1);
        assert!(both.validate().is_err());
    }

    #[test]
    fn launch_rejects_an_existing_run_directory() {
        let root = scratch("existing");
        let request = base_request(&root, "taken");
        fs::create_dir_all(request.run_dir()).unwrap();
        let error = launch(&request).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn journal_round_trips_and_takes_exactly_one_terminal_outcome() {
        let root = scratch("journal");
        let record = TunerLaunchRecord {
            run_id: "run_12a".into(),
            argv: vec!["uv".into()],
            run_dir: root.join("run_12a"),
            pid: Some(12),
            started_at: "2026-01-01T00:00:00Z".into(),
            terminal_outcome: None,
        };
        append_launch(&root, &record).unwrap();
        append_terminal(&root, "run_12a", TerminalOutcome::Exited).unwrap();
        assert_eq!(
            records(&root).unwrap()[0].terminal_outcome,
            Some(TerminalOutcome::Exited)
        );
        assert!(append_terminal(&root, "run_12a", TerminalOutcome::Signalled).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn records_preserve_launch_order() {
        let root = scratch("order");
        for id in ["c", "a", "b"] {
            append_launch(
                &root,
                &TunerLaunchRecord {
                    run_id: id.into(),
                    argv: vec![],
                    run_dir: root.join(id),
                    pid: Some(1),
                    started_at: "2026-01-01T00:00:00Z".into(),
                    terminal_outcome: None,
                },
            )
            .unwrap();
        }
        let ids: Vec<_> = records(&root)
            .unwrap()
            .into_iter()
            .map(|r| r.run_id)
            .collect();
        assert_eq!(ids, ["c", "a", "b"]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn spawn_journals_a_launch_then_a_terminal_outcome_and_stops_cleanly() {
        let root = scratch("spawn");
        let run_dir = root.join("live");
        fs::create_dir_all(&run_dir).unwrap();
        let record = spawn_and_journal(
            &root,
            "live",
            run_dir.clone(),
            vec!["sh".into(), "-c".into(), "sleep 30".into()],
            false,
        )
        .unwrap();
        let pid = record.pid.unwrap();
        assert!(run_dir.join("launch.out").exists());
        assert!(is_alive(pid));

        interrupt(pid).unwrap();
        // The reaper thread records the terminal outcome asynchronously. A
        // `sh` child inherits this process's SIGINT disposition, which under a
        // non-interactive CI runner can be SIG_IGN -- so escalate to SIGKILL if
        // the interrupt is swallowed. Either way the reaper must see a signal.
        let mut outcome = None;
        for i in 0..200 {
            if let Some(found) = records(&root)
                .unwrap()
                .into_iter()
                .find(|r| r.run_id == "live")
                .and_then(|r| r.terminal_outcome)
            {
                outcome = Some(found);
                break;
            }
            if i == 20 && is_alive(pid) {
                let _ = Command::new("kill").arg("-KILL").arg(pid.to_string()).status();
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert_eq!(outcome, Some(TerminalOutcome::Signalled));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn spawn_reports_a_child_that_dies_during_the_startup_grace_window() {
        let root = scratch("startup-fail");
        let run_dir = root.join("doomed");
        fs::create_dir_all(&run_dir).unwrap();
        let error = spawn_and_journal(
            &root,
            "doomed",
            run_dir,
            vec![
                "sh".into(),
                "-c".into(),
                "echo 'objective file does not exist' >&2; exit 3".into(),
            ],
            true,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("died during startup"), "{message}");
        assert!(message.contains("exit status 3"), "{message}");
        assert!(message.contains("objective file does not exist"), "{message}");
        // The journal still carries a launch and a terminal outcome for it.
        let record = records(&root)
            .unwrap()
            .into_iter()
            .find(|r| r.run_id == "doomed")
            .unwrap();
        assert_eq!(record.terminal_outcome, Some(TerminalOutcome::Exited));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resumable_argv_drops_prior_resume_and_extension_flags() {
        let argv = vec![
            "uv".to_string(),
            "-m".into(),
            "tuner_cli".into(),
            "--tuning-pair-budget".into(),
            "10".into(),
            "--resume".into(),
            "--extend-tuning-pairs".into(),
            "6".into(),
            "--extend-reason".into(),
            "earlier extension".into(),
        ];
        assert_eq!(
            resumable_argv(&argv),
            ["uv", "-m", "tuner_cli", "--tuning-pair-budget", "10"]
        );
    }

    #[test]
    fn extend_rejects_empty_reason_and_all_zero_deltas() {
        let root = scratch("extend-reject");
        append_launch(
            &root,
            &TunerLaunchRecord {
                run_id: "run".into(),
                argv: vec!["uv".into()],
                run_dir: root.join("run"),
                pid: Some(1),
                started_at: "2026-01-01T00:00:00Z".into(),
                terminal_outcome: Some(TerminalOutcome::Exited),
            },
        )
        .unwrap();
        fs::create_dir_all(root.join("run")).unwrap();
        let blank = BudgetExtension {
            tuning_pair_attempts_delta: 6,
            validation_pair_attempts_delta: 0,
            diagnostic_pair_attempts_delta: 0,
            reason: "  ".into(),
        };
        assert_eq!(
            extend(&root, "run", &blank).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let zero = BudgetExtension {
            tuning_pair_attempts_delta: 0,
            validation_pair_attempts_delta: 0,
            diagnostic_pair_attempts_delta: 0,
            reason: "fund more".into(),
        };
        assert_eq!(
            extend(&root, "run", &zero).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            extend(&root, "missing", &zero).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let _ = fs::remove_dir_all(&root);
    }
}
