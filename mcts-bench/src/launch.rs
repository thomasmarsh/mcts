//! Detached background process launcher, master registry log, and liveness
//! helpers.  The launcher spawns a run in its own process group so it
//! survives the launching process.  It never waits on the child in the
//! launching thread — a background reaper thread waits to avoid zombies,
//! and the ingest loop's crash reconciliation handles the case where the
//! reaper thread was terminated by the launching process exiting.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::log::RegistryEvent;
use crate::supervised_launch::{SupervisorCommand, WrapperIdentity};

/// Source revision attached to a launched benchmark run.
#[derive(Clone, Copy)]
pub struct BuildInfo<'a> {
    pub git_sha: &'a str,
    pub git_dirty: bool,
}

/// Default root directory for benchmark run data (relative to the repo root,
/// or absolute).  Matches the `bench-runs/` entry in `.gitignore`.
pub const BENCH_RUNS_DIR: &str = "bench-runs";

/// Metadata about a successfully launched run.
pub struct LaunchedRun {
    /// Unique run identifier, e.g.
    /// `roundrobin-druid-20260808T120000-6fe2387`.
    pub run_id: String,
    /// OS-assigned process ID of the child process.
    pub pid: u32,
    /// Path to the child's JSONL log file (`log.jsonl`).
    pub log_path: PathBuf,
    /// Path to the run's data directory.
    pub log_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Launch a detached background process.
///
/// `cmd` is the full command vector (argv[0] is the binary path).  The child's
/// stdout is redirected to `bench-runs/<run_id>/log.jsonl` and its stderr to
/// `bench-runs/<run_id>/stdout.log`.  The child is placed in its own process
/// group so it survives the launching process.
///
/// Returns immediately with the run metadata; does not wait for the child.
pub fn launch(
    cmd: Vec<String>,
    kind: &str,
    game: &str,
    label: Option<&str>,
    build_info: BuildInfo<'_>,
) -> std::io::Result<LaunchedRun> {
    launch_with_run_id(
        generate_run_id(kind, game, build_info),
        cmd,
        kind,
        game,
        label,
        build_info,
    )
}

/// Detach a supervisor wrapper without giving it the workload's output file
/// descriptors. The wrapper records closure of those files itself.
pub fn launch_supervisor(
    run_id: &str,
    command: &SupervisorCommand,
    stdout_path: &Path,
    stderr_path: &Path,
    workload_log_path: &Path,
    registry_path: &Path,
    build_info: BuildInfo<'_>,
) -> std::io::Result<WrapperIdentity> {
    if let Some(parent) = stdout_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = stderr_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut child = Command::new(&command.executable);
    child.args(&command.arguments);
    child.stdout(fs::File::create(stdout_path)?);
    child.stderr(fs::File::create(stderr_path)?);
    #[cfg(unix)]
    child.process_group(0);
    let mut child = child.spawn()?;
    let pid = child.id() as u64;
    let _ = append_registry_event(
        &RegistryEvent::Start {
            run_id: run_id.into(),
            kind: "experiment".into(),
            game: "experiment-grid".into(),
            pid: pid as u32,
            cmd: std::iter::once(command.executable.to_string_lossy().into_owned())
                .chain(
                    command
                        .arguments
                        .iter()
                        .map(|argument| argument.to_string_lossy().into_owned()),
                )
                .collect(),
            log_path: workload_log_path.to_string_lossy().into_owned(),
            git_sha: build_info.git_sha.into(),
            git_dirty: build_info.git_dirty,
            started_at: iso_timestamp(),
        },
        registry_path,
    );
    let reaper_run_id = run_id.to_owned();
    let registry_path = registry_path.to_path_buf();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = append_registry_event(
            &RegistryEvent::Stop {
                run_id: reaper_run_id,
                exit_code: status.ok().and_then(|status| status.code()),
                ended_at: iso_timestamp(),
            },
            &registry_path,
        );
    });
    Ok(WrapperIdentity {
        pid,
        process_group_id: pid,
    })
}

/// Like [`launch`], but with a caller-supplied `run_id` instead of one
/// generated internally.  Needed when the caller has to know the run_id
/// *before* spawning -- e.g. SMAC3 launches thread the run_id into the
/// child's own argv (`smac3 --run-id <id>`) so its `Scenario.name` (and
/// therefore its on-disk output directory) is pinned to the same id the
/// bench-runs registry/DB use, making a later `--resume <id>` able to find
/// it without any extra bookkeeping.
pub fn launch_with_run_id(
    run_id: String,
    cmd: Vec<String>,
    kind: &str,
    game: &str,
    _label: Option<&str>,
    build_info: BuildInfo<'_>,
) -> std::io::Result<LaunchedRun> {
    let log_dir = Path::new(BENCH_RUNS_DIR).join(&run_id);
    fs::create_dir_all(&log_dir)?;

    let log_path = log_dir.join("log.jsonl");
    let stdout_log_path = log_dir.join("stdout.log");

    let mut child_cmd = Command::new(&cmd[0]);
    child_cmd.args(&cmd[1..]);
    child_cmd.stdout(fs::File::create(&log_path).map(Stdio::from).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to create log file {}: {e}", log_path.display()),
        )
    })?);
    child_cmd.stderr(
        fs::File::create(&stdout_log_path)
            .map(Stdio::from)
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "failed to create stderr log {}: {e}",
                        stdout_log_path.display()
                    ),
                )
            })?,
    );

    #[cfg(unix)]
    child_cmd.process_group(0);

    let mut child = child_cmd
        .spawn()
        .map_err(|e| std::io::Error::new(e.kind(), format!("failed to spawn {}: {e}", cmd[0])))?;
    let pid = child.id();

    // Wait on the child so it doesn't become a zombie, and record its real
    // exit status as a `Stop` registry event when it finishes normally --
    // without this, a naturally-completed run was only ever caught by the
    // ingest loop's `reconcile_liveness` fallback (PID no longer alive, no
    // `Stop` event ever seen), which marks *every* dead PID `'crashed'`
    // regardless of how it actually exited. That fallback exists for a
    // narrower case (the reaper thread itself was cut short by the
    // launching process exiting/restarting before `wait()` returned, per
    // this module's own doc comment above) and was never a substitute for
    // this thread reporting a normal exit.
    let reaper_run_id = run_id.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        let stop = RegistryEvent::Stop {
            run_id: reaper_run_id,
            exit_code: status.ok().and_then(|s| s.code()),
            ended_at: iso_timestamp(),
        };
        let _ = append_registry_event(
            &stop,
            Path::new(BENCH_RUNS_DIR).join("registry.log").as_path(),
        );
    });

    let event = RegistryEvent::Start {
        run_id: run_id.clone(),
        kind: kind.to_owned(),
        game: game.to_owned(),
        pid,
        cmd: cmd.clone(),
        log_path: log_path.to_string_lossy().to_string(),
        git_sha: build_info.git_sha.to_owned(),
        git_dirty: build_info.git_dirty,
        started_at: iso_timestamp(),
    };
    append_registry_event(
        &event,
        Path::new(BENCH_RUNS_DIR).join("registry.log").as_path(),
    )?;

    Ok(LaunchedRun {
        run_id,
        pid,
        log_path,
        log_dir,
    })
}

/// Check whether a process identified by `pid` is still alive on this
/// machine.  Uses `kill -0` (POSIX), which sends no signal but checks
/// whether the process exists.
pub fn is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a run_id following the convention:
/// `{kind}-{game}-{yyyymmddThhmmss}-{short_git_sha}`
pub fn generate_run_id(kind: &str, game: &str, build_info: BuildInfo<'_>) -> String {
    let sha = build_info.git_sha;
    let short_sha = if sha.len() >= 7 { &sha[..7] } else { sha };
    format!("{kind}-{game}-{ts}-{short_sha}", ts = compact_timestamp())
}

// ---------------------------------------------------------------------------
// Timestamp formatting (stdlib only, no chrono dependency)
// ---------------------------------------------------------------------------

/// ISO-8601 UTC timestamp: `2026-08-08T12:00:00Z`.
pub(crate) fn iso_timestamp() -> String {
    let total_secs = unix_secs();
    let (y, m, d, hh, mm, ss) = secs_to_ymdhms(total_secs);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Compact timestamp for run IDs: `yyyymmddThhmmss`.
fn compact_timestamp() -> String {
    let total_secs = unix_secs();
    let (y, m, d, hh, mm, ss) = secs_to_ymdhms(total_secs);
    format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}")
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock set before Unix epoch")
        .as_secs()
}

/// Decompose seconds since Unix epoch into calendar fields (UTC).
fn secs_to_ymdhms(total_secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = total_secs / 86400;
    let time_secs = total_secs % 86400;
    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;
    let (y, m, d) = days_to_ymd(days);
    (y, m, d, hh, mm, ss)
}

/// Days since Unix epoch to (year, month, day) in the Gregorian calendar.
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Registry log
// ---------------------------------------------------------------------------

fn append_registry_event(event: &RegistryEvent, registry_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = registry_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(registry_path)?;
    let mut line = event.to_json_line();
    line.push('\n');
    file.write_all(line.as_bytes())?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_to_ymd_epoch() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        let (y, m, d) = days_to_ymd(20673);
        assert_eq!(y, 2026);
        assert_eq!(m, 8);
        assert_eq!(d, 8);
    }

    #[test]
    fn days_to_ymd_leap_year() {
        let (y, m, d) = days_to_ymd(19782);
        assert_eq!(y, 2024);
        assert_eq!(m, 2);
        assert_eq!(d, 29);
    }

    #[test]
    fn compact_timestamp_format() {
        let ts = compact_timestamp();
        assert_eq!(ts.len(), 15);
        assert!(ts.as_bytes()[8] == b'T', "expected T separator, got {ts}");
        for (i, &b) in ts.as_bytes().iter().enumerate() {
            if i != 8 {
                assert!(b.is_ascii_digit(), "non-digit at position {i} in {ts}");
            }
        }
    }

    #[test]
    fn iso_timestamp_format() {
        let ts = iso_timestamp();
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'), "expected Z suffix, got {ts}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }

    #[test]
    fn is_alive_returns_true_for_running_process() {
        let mut child = Command::new("sleep")
            .arg("5")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn sleep");
        assert!(is_alive(child.id()));
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn is_alive_returns_false_for_exited_process() {
        let mut child = Command::new("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn true");
        let pid = child.id();
        let status = child.wait().expect("failed to wait on true");
        assert!(status.success());
        assert!(
            !is_alive(pid),
            "PID {pid} should not be alive after exiting"
        );
    }

    #[test]
    fn is_alive_returns_false_for_nonexistent_pid() {
        assert!(!is_alive(999_999_999));
    }

    #[test]
    fn generate_run_id_uses_correct_format() {
        let id = generate_run_id(
            "round_robin",
            "druid",
            BuildInfo {
                git_sha: "abcdef0123456789",
                git_dirty: false,
            },
        );
        let parts: Vec<&str> = id.split('-').collect();
        assert!(
            parts.len() >= 4,
            "expected at least 4 dash-separated parts, got {id}"
        );
        assert_eq!(parts[0], "round_robin");
        assert_eq!(parts[1], "druid");
        let ts = parts[2];
        assert_eq!(ts.len(), 15);
        assert_eq!(ts.as_bytes()[8], b'T');
        let sha = parts[3];
        assert!(sha.len() >= 7, "SHA part too short: {sha}");
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA not hex: {sha}"
        );
    }

    #[test]
    fn registry_event_appends_to_file() {
        let dir =
            std::env::temp_dir().join(format!("mcts_bench_launch_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let registry_path = dir.join("registry.log");

        let event = RegistryEvent::Start {
            run_id: "test-run-123".into(),
            kind: "test".into(),
            game: "test".into(),
            pid: 99999,
            cmd: vec!["echo".into(), "hello".into()],
            log_path: "/tmp/test/log.jsonl".into(),
            git_sha: "abcdef1".into(),
            git_dirty: false,
            started_at: "2026-01-01T00:00:00Z".into(),
        };

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&registry_path)
            .unwrap();
        let mut line = event.to_json_line();
        line.push('\n');
        file.write_all(line.as_bytes()).unwrap();
        file.flush().unwrap();

        let contents = fs::read_to_string(&registry_path).unwrap();
        let parsed: RegistryEvent = serde_json::from_str(contents.trim()).unwrap();
        assert!(
            matches!(parsed, RegistryEvent::Start { ref run_id, .. } if run_id == "test-run-123")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn registry_event_appends_multiple_events() {
        let dir =
            std::env::temp_dir().join(format!("mcts_bench_launch_test2_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let registry_path = dir.join("registry.log");

        let start = RegistryEvent::Start {
            run_id: "run-1".into(),
            kind: "round_robin".into(),
            game: "druid".into(),
            pid: 1001,
            cmd: vec!["bench".into()],
            log_path: "/tmp/log.jsonl".into(),
            git_sha: "aaa".into(),
            git_dirty: false,
            started_at: "2026-01-01T00:00:00Z".into(),
        };
        let stop = RegistryEvent::Stop {
            run_id: "run-1".into(),
            exit_code: Some(0),
            ended_at: "2026-01-01T01:00:00Z".into(),
        };
        for ev in &[&start, &stop] {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&registry_path)
                .unwrap();
            let mut line = ev.to_json_line();
            line.push('\n');
            file.write_all(line.as_bytes()).unwrap();
            file.flush().unwrap();
        }

        let contents = fs::read_to_string(&registry_path).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "expected 2 lines, got {}: {:?}",
            lines.len(),
            lines
        );

        let parsed_start: RegistryEvent =
            serde_json::from_str(lines[0]).expect("failed to parse first line");
        assert!(
            matches!(parsed_start, RegistryEvent::Start { ref run_id, .. } if run_id == "run-1")
        );

        let parsed_stop: RegistryEvent =
            serde_json::from_str(lines[1]).expect("failed to parse second line");
        assert!(matches!(parsed_stop, RegistryEvent::Stop { ref run_id, .. } if run_id == "run-1"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_uses_distinct_outputs_and_records_exit() {
        static TEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mcts_supervisor_test_{}_{}",
            std::process::id(),
            TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let stdout = dir.join("outer.stdout");
        let stderr = dir.join("outer.stderr");
        let workload = dir.join("workload.log");
        let registry = dir.join("registry.log");
        let command = SupervisorCommand {
            executable: "sh".into(),
            arguments: vec!["-c".into(), "printf out; printf err >&2; exit 7".into()],
        };
        launch_supervisor(
            "test-supervisor",
            &command,
            &stdout,
            &stderr,
            &workload,
            &registry,
            BuildInfo {
                git_sha: "test",
                git_dirty: false,
            },
        )
        .unwrap();
        for _ in 0..10_000 {
            if let Ok(contents) = fs::read_to_string(&registry) {
                if contents.lines().any(|line| {
                    line.contains("\"type\":\"stop\"") && line.contains("\"exit_code\":7")
                }) {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let contents = fs::read_to_string(&registry).unwrap();
        assert!(contents.contains("\"exit_code\":7"));
        assert_eq!(fs::read_to_string(&stdout).unwrap(), "out");
        assert_eq!(fs::read_to_string(&stderr).unwrap(), "err");
        assert!(!workload.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
