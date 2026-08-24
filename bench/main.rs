//! CLI for the benchmark / tournament / tuner harness.
//!
//! Default behaviour (`bench round-robin ...`) runs the tournament in the
//! foreground and streams JSONL results to stdout, matching the existing
//! `examples/strength_*.rs` usage pattern.  `bench launch -- <args>` goes
//! through the detached-process launcher instead, printing the run_id and
//! PID and returning immediately.
//!
//! `ingest --once` is a one-shot debug/validation subcommand for the ingest
//! loop.  It must not be used while `server` is running
//! (DuckDB single-writer constraint).

use std::io::stdout;
use std::process::{Command as StdCommand, Stdio};

use clap::{Parser, Subcommand};

use mcts::util::Verbosity;
use mcts_bench::experiment::{self, ExperimentSpecV1};
use mcts_bench::ingest;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::registry;
use mcts_bench::schema;
use mcts_bench::tournament::round_robin_bench_multiple;

mod supervise;

const BUILD_INFO: launch::BuildInfo<'static> = launch::BuildInfo {
    git_sha: env!("GIT_SHA"),
    git_dirty: matches!(env!("GIT_DIRTY").as_bytes(), b"true"),
};

#[derive(Parser)]
#[command(name = "bench", about = "Benchmark, tournament, and tuner harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(hide = true)]
    Supervise(supervise::Args),
    /// Run a round-robin tournament in the foreground, streaming JSONL
    /// match results to stdout.  Progress bars and final summary tables
    /// go to stderr.
    RoundRobin {
        /// Game kind (e.g. "druid").  Must match a registered BenchGame.
        #[arg(long)]
        game: String,

        /// Strategy IDs to include.  If empty, uses all strategies
        /// registered for this game kind.
        #[arg(long, default_values_t = Vec::<String>::new())]
        strategies: Vec<String>,

        /// Number of full round-robin passes.  Default: 1.
        #[arg(long, default_value_t = 1)]
        rounds: usize,

        /// Show progress bars and verbose result tables on stderr.
        #[arg(long)]
        verbose: bool,

        /// Optional path to append per-ply move-trace JSONL lines to
        /// (opened in append mode). Kept separate from stdout/`log.jsonl`
        /// since a full move trace is much higher-volume than match
        /// results and would otherwise flood anyone tailing the run's
        /// log. Omit to disable move tracing entirely.
        #[arg(long)]
        trace_path: Option<String>,
    },

    /// Launch a detached background run via the OS process launcher.
    /// Everything after `--` is the command to run (argv including the
    /// binary path).  The launcher redirects stdout to the run's
    /// `log.jsonl` and stderr to `stdout.log`, writes a start event to
    /// `registry.log`, and returns immediately.  The run survives the
    /// launching process.
    Launch {
        /// Machine-readable kind string (e.g. "round_robin", "tuner").
        #[arg(long)]
        kind: String,

        /// Game kind (e.g. "druid").
        #[arg(long)]
        game: String,

        /// Optional human-readable label for the run.
        #[arg(long)]
        label: Option<String>,

        /// Command to run (everything after `--`).  The first element is
        /// the binary path; subsequent elements are its arguments.
        #[arg(trailing_var_arg = true, required = true)]
        cmd: Vec<String>,
    },

    /// Launch a tuner hyperparameter-optimisation run.  Runs
    /// ``uv run --project tuner/ tuner ...`` in the foreground
    /// (streaming JSONL to stdout) or, with ``--background``, through the
    /// detached-process launcher so the run survives the launching
    /// process and appears in DuckDB/the UI.
    Tuner {
        /// Path to the tuner YAML config file (passed through to
        /// ``tuner --config``).
        #[arg(long)]
        config: Option<String>,

        /// Config override (``key=value``, repeatable).  Passed through
        /// as ``tuner --override key=value``.
        #[arg(long = "override", default_values_t = Vec::<String>::new())]
        overrides: Vec<String>,

        /// Extra baseline instance backed by a raw discovered config
        /// (``id=json``, repeatable), for evaluating a candidate against an
        /// opponent that isn't one of the game's named presets.  Passed
        /// through as ``tuner --baseline-config id=json``.
        #[arg(long = "baseline-config", default_values_t = Vec::<String>::new())]
        baseline_configs: Vec<String>,

        /// Game-setup config (JSON object, e.g. Druid's `{"size":{"w":9,
        /// "h":9}}`) pinning every trial in this run to a non-default game
        /// config instead of the game's own `default_config()`.  Passed
        /// through as ``tuner --game-config <json>``.
        #[arg(long = "game-config")]
        game_config: Option<String>,

        /// Game kind for registry attribution (e.g. "druid").
        #[arg(long)]
        game: String,

        /// Game kind recorded by the tuning lifecycle. When omitted, uses
        /// `--game`.
        #[arg(long)]
        game_kind: Option<String>,

        /// Optional human-readable label for the run.
        #[arg(long)]
        label: Option<String>,

        /// Legacy study name forwarded to the tuner.
        #[arg(long = "run-id")]
        run_id: Option<String>,

        /// Stable optimizer identity for a logical tuning session.
        #[arg(long)]
        optimizer_id: Option<String>,

        /// Physical bench run identity for this attempt.
        #[arg(long)]
        bench_run_id: Option<String>,

        /// Optional path to append per-ply move-trace JSONL lines to
        /// (opened in append mode by each trial's game-binary subprocess,
        /// same file across the whole run). Passed through as `tuner
        /// --trace-path`. Omit to disable move tracing entirely.
        #[arg(long)]
        trace_path: Option<String>,

        /// Opaque logical tuning session identity.
        #[arg(long)]
        session_id: Option<String>,

        /// Opaque physical tuning attempt identity.
        #[arg(long)]
        attempt_id: Option<String>,

        /// Append-only typed lifecycle evidence path.
        #[arg(long)]
        lifecycle_path: Option<String>,

        /// Launch in the background (detached process) instead of
        /// running in the foreground.
        #[arg(long)]
        background: bool,
    },

    /// One-shot listing of registered game kinds and their AI presets.
    /// Spawns each game binary once with its `describe` CLI subcommand
    /// and exits -- unlike `round-robin`, this never opens a persistent
    /// subprocess session.
    Games,

    /// One-shot ingest for debugging / validation.  Reads registry.log
    /// and all active runs' log.jsonl files, upserts into DuckDB at the
    /// given path, then exits.  **Not for concurrent use with `server`**
    /// (DuckDB single-writer constraint).
    Ingest {
        /// Path to the DuckDB database file.
        #[arg(long, default_value = "bench-runs/bench.duckdb")]
        db: String,
    },

    /// Run one saved experiment cell in the foreground and translate the
    /// configured game stream into experiment log events.
    Experiment {
        #[arg(long)]
        spec_json: String,
        #[arg(long)]
        trace_path: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Supervise(args) => supervise::run(args),
        Command::RoundRobin {
            game,
            strategies,
            rounds,
            verbose,
            trace_path,
        } => cmd_round_robin(&game, &strategies, rounds, verbose, trace_path.as_deref()),

        Command::Launch {
            kind,
            game,
            label,
            cmd,
        } => cmd_launch(&kind, &game, label.as_deref(), &cmd),

        Command::Tuner {
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            game_kind,
            label,
            run_id,
            optimizer_id,
            bench_run_id,
            trace_path,
            session_id,
            attempt_id,
            lifecycle_path,
            background,
        } => cmd_tuner(
            config.as_deref(),
            &overrides,
            &baseline_configs,
            game_config.as_deref(),
            &game,
            game_kind.as_deref(),
            label.as_deref(),
            run_id.as_deref(),
            optimizer_id.as_deref(),
            bench_run_id.as_deref(),
            trace_path.as_deref(),
            session_id.as_deref(),
            attempt_id.as_deref(),
            lifecycle_path.as_deref(),
            background,
        ),

        Command::Games => cmd_games(),

        Command::Ingest { db } => cmd_ingest_once(&db),
        Command::Experiment {
            spec_json,
            trace_path,
        } => cmd_experiment(&spec_json, trace_path.as_deref()),
    }
}

fn cmd_experiment(spec_json: &str, trace_path: Option<&str>) {
    let spec: ExperimentSpecV1 = match serde_json::from_str(spec_json) {
        Ok(spec) => spec,
        Err(error) => {
            eprintln!("error: invalid --spec-json: {error}");
            std::process::exit(1);
        }
    };
    let path = trace_path.map(std::path::Path::new);
    let mut writer = stdout().lock();
    if let Err(error) = experiment::run_experiment(&spec, path, &mut writer) {
        eprintln!("error: experiment failed: {error}");
        std::process::exit(1);
    }
}

fn cmd_round_robin(
    game_kind: &str,
    strategy_ids: &[String],
    rounds: usize,
    verbose: bool,
    trace_path: Option<&str>,
) {
    let games = registry();
    let Some(bench_game) = games.get(game_kind) else {
        eprintln!("error: unknown game kind '{game_kind}'");
        eprintln!(
            "available games: {}",
            games.keys().cloned().collect::<Vec<_>>().join(", ")
        );
        std::process::exit(1);
    };

    let all_strategies = bench_game.strategies();
    let ids: Vec<String> = if strategy_ids.is_empty() {
        all_strategies.into_iter().map(|s| s.id).collect()
    } else {
        // Validate every requested strategy ID exists.
        let available: std::collections::HashSet<String> =
            all_strategies.into_iter().map(|s| s.id).collect();
        for sid in strategy_ids {
            if !available.contains(sid) {
                eprintln!("error: unknown strategy '{sid}' for game '{game_kind}'");
                eprintln!(
                    "available strategies: {}",
                    available
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(1);
            }
        }
        strategy_ids.to_vec()
    };

    let verb = if verbose {
        Verbosity::Verbose
    } else {
        Verbosity::Silent
    };

    let mut writer = stdout().lock();
    let mut moves_file = trace_path.map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap_or_else(|e| {
                eprintln!("error: failed to open --trace-path {p}: {e}");
                std::process::exit(1);
            })
    });
    let moves_writer: Option<&mut dyn std::io::Write> =
        moves_file.as_mut().map(|f| f as &mut dyn std::io::Write);
    let results = round_robin_bench_multiple(
        bench_game.as_ref(),
        &ids,
        rounds,
        &mut writer,
        moves_writer,
        verb,
    );

    // Final summary to stderr.
    if verbose {
        eprintln!();
        eprintln!("{:=^63}", " Final Standings ");
        eprintln!(
            "{0:<25} | {1:>6} | {2:>6} | {3:>6} | {4:>8}",
            "Strategy", "Wins", "Losses", "Draws", "Win%"
        );
        eprintln!("{:-<59}", "");

        for (i, r) in results.iter().enumerate() {
            let total = r.total();
            let win_pct = if total > 0 {
                100.0 * r.wins as f64 / total as f64
            } else {
                0.0
            };
            eprintln!(
                "{0:<25} | {1:>6} | {2:>6} | {3:>6} | {4:>7.1}%",
                ids[i], r.wins, r.losses, r.draws, win_pct,
            );
        }
    }
}

fn cmd_games() {
    let descriptions = mcts_bench::games::describe_games();

    println!("{:<16} {:<28} PRESETS", "KIND", "LABEL");
    for d in &descriptions {
        let presets = d
            .ai_presets
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!("{:<16} {:<28} {presets}", d.kind, d.label);
    }
}

fn cmd_launch(kind: &str, game: &str, label: Option<&str>, cmd: &[String]) {
    let LaunchedRun {
        run_id,
        pid,
        log_path,
        ..
    } = match launch::launch(cmd.to_vec(), kind, game, label, BUILD_INFO) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: failed to launch run: {e}");
            std::process::exit(1);
        }
    };

    // Print run metadata as a JSON object to stdout, so automated
    // callers (server's /api/bench/launch, scripts) can parse it.
    let output = serde_json::json!({
        "run_id": run_id,
        "pid": pid,
        "log_path": log_path.to_string_lossy(),
        "kind": kind,
        "game": game,
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

/// Build the argv for a ``uv run --project tuner/ tuner ...``
/// invocation, incorporating the config file, overrides, and git SHA.
///
/// `game` is translated into a `target.binary=target/release/game-<game>`
/// override so the launched run actually tunes the selected game --
/// `tuner/config/default.yaml`'s `target.binary` is just a fallback default
/// (currently `game-traffic-lights`, the reference wiring's game), not
/// something any caller of `bench tuner --game ...` should rely on. This is
/// pushed before `overrides` so an explicit `target.binary=...` override
/// from the caller still wins (the Python side's `_apply_overrides` keeps
/// the last value for a repeated key).
#[allow(clippy::too_many_arguments)]
fn build_tuner_command(
    config: Option<&str>,
    overrides: &[String],
    baseline_configs: &[String],
    game_config: Option<&str>,
    game: &str,
    game_kind: Option<&str>,
    run_id: Option<&str>,
    trace_path: Option<&str>,
    optimizer_id: Option<&str>,
    bench_run_id: Option<&str>,
    session_id: Option<&str>,
    attempt_id: Option<&str>,
    lifecycle_path: Option<&str>,
) -> Vec<String> {
    let mut cmd = vec![
        "uv".to_string(),
        "run".to_string(),
        "--project".to_string(),
        "tuner/".to_string(),
        "tuner".to_string(),
    ];

    if let Some(config_path) = config {
        cmd.push("--config".to_string());
        cmd.push(config_path.to_string());
    }

    let binary_suffix = if cfg!(target_os = "windows") {
        ".exe"
    } else {
        ""
    };
    cmd.push("--override".to_string());
    cmd.push(format!(
        "target.binary=target/release/game-{game}{binary_suffix}"
    ));

    for ov in overrides {
        cmd.push("--override".to_string());
        cmd.push(ov.clone());
    }

    for bc in baseline_configs {
        cmd.push("--baseline-config".to_string());
        cmd.push(bc.clone());
    }

    if let Some(gc) = game_config {
        cmd.push("--game-config".to_string());
        cmd.push(gc.to_string());
    }

    // Pass the compile-time git SHA so the Python side can include it
    // in its JSONL output for attribution.
    cmd.push("--git-sha".to_string());
    cmd.push(BUILD_INFO.git_sha.to_string());

    if let Some(id) = run_id {
        cmd.push("--run-id".to_string());
        cmd.push(id.to_string());
    }

    if let Some(path) = trace_path {
        cmd.push("--trace-path".to_string());
        cmd.push(path.to_string());
    }

    append_tuner_lifecycle_arguments(
        &mut cmd,
        game,
        game_kind,
        optimizer_id,
        bench_run_id,
        session_id,
        attempt_id,
        lifecycle_path,
    );

    cmd
}

#[allow(clippy::too_many_arguments)]
fn append_tuner_lifecycle_arguments(
    cmd: &mut Vec<String>,
    game: &str,
    game_kind: Option<&str>,
    optimizer_id: Option<&str>,
    bench_run_id: Option<&str>,
    session_id: Option<&str>,
    attempt_id: Option<&str>,
    lifecycle_path: Option<&str>,
) {
    if let Some(id) = optimizer_id {
        cmd.push("--optimizer-id".to_string());
        cmd.push(id.to_string());
    }
    if let Some(id) = bench_run_id {
        cmd.push("--bench-run-id".to_string());
        cmd.push(id.to_string());
    }
    if let Some(id) = session_id {
        cmd.push("--session-id".to_string());
        cmd.push(id.to_string());
    }
    if let Some(id) = attempt_id {
        cmd.push("--attempt-id".to_string());
        cmd.push(id.to_string());
    }
    if let Some(path) = lifecycle_path {
        cmd.push("--lifecycle-path".to_string());
        cmd.push(path.to_string());
    }
    cmd.push("--game-kind".to_string());
    cmd.push(game_kind.unwrap_or(game).to_string());
}

struct BackgroundTunerLifecycleArguments {
    optimizer_id: String,
    bench_run_id: String,
    session_id: String,
    attempt_id: String,
    lifecycle_path: String,
}

fn derive_background_tuner_lifecycle_arguments(run_id: &str) -> BackgroundTunerLifecycleArguments {
    let optimizer_id = format!("tuning-session-{run_id}");
    BackgroundTunerLifecycleArguments {
        optimizer_id: optimizer_id.clone(),
        bench_run_id: run_id.to_string(),
        session_id: optimizer_id.clone(),
        attempt_id: format!("tuning-attempt-{run_id}"),
        lifecycle_path: std::path::Path::new("optuna_output")
            .join(optimizer_id)
            .join("lifecycle.jsonl")
            .to_string_lossy()
            .to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_tuner(
    config: Option<&str>,
    overrides: &[String],
    baseline_configs: &[String],
    game_config: Option<&str>,
    game: &str,
    game_kind: Option<&str>,
    label: Option<&str>,
    run_id: Option<&str>,
    optimizer_id: Option<&str>,
    bench_run_id: Option<&str>,
    trace_path: Option<&str>,
    session_id: Option<&str>,
    attempt_id: Option<&str>,
    lifecycle_path: Option<&str>,
    background: bool,
) {
    if background {
        // Pin the run_id up front (rather than letting `launch::launch`
        // generate one internally) so the same id both names the
        // bench-runs directory/registry entry *and* is baked into the
        // child's own `--run-id` argv.
        let run_id = run_id
            .map(str::to_string)
            .unwrap_or_else(|| launch::generate_run_id("tuner", game, BUILD_INFO));
        let lifecycle = derive_background_tuner_lifecycle_arguments(&run_id);
        let cmd = build_tuner_command(
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            game_kind,
            Some(&run_id),
            trace_path,
            optimizer_id.or(Some(&lifecycle.optimizer_id)),
            bench_run_id.or(Some(&lifecycle.bench_run_id)),
            session_id.or(Some(&lifecycle.session_id)),
            attempt_id.or(Some(&lifecycle.attempt_id)),
            lifecycle_path.or(Some(&lifecycle.lifecycle_path)),
        );

        // Launch via the detached-process launcher.
        let LaunchedRun {
            run_id,
            pid,
            log_path,
            ..
        } = match launch::launch_with_run_id(run_id, cmd, "tuner", game, label, BUILD_INFO) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: failed to launch tuner run: {e}");
                std::process::exit(1);
            }
        };

        let output = serde_json::json!({
            "run_id": run_id,
            "pid": pid,
            "log_path": log_path.to_string_lossy(),
            "kind": "tuner",
            "game": game,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        // Run in the foreground — inherit stdout/stderr so the Python
        // JSONL stream goes directly to the terminal or whoever is
        // piping stdout. Unlike the background branch, supplied identities
        // are forwarded as given rather than auto-generated: this path has
        // no bench-runs registry entry of its own to keep in sync (a
        // caller that wraps this in its *own* launcher, e.g. the server,
        // is responsible for passing a `--run-id` that matches whatever it
        // used for that outer entry).
        let cmd = build_tuner_command(
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            game_kind,
            run_id,
            trace_path,
            optimizer_id,
            bench_run_id,
            session_id,
            attempt_id,
            lifecycle_path,
        );
        let mut child = match StdCommand::new(&cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to spawn '{}': {e}", cmd[0]);
                std::process::exit(1);
            }
        };

        let status = child.wait().unwrap_or_else(|e| {
            eprintln!("error: failed to wait on child: {e}");
            std::process::exit(1);
        });

        std::process::exit(status.code().unwrap_or(1));
    }
}

fn cmd_ingest_once(db_path: &str) {
    let db_path = std::path::Path::new(db_path);
    let conn = match schema::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to open DuckDB at {}: {e}", db_path.display());
            std::process::exit(1);
        }
    };

    let bench_runs_dir = std::path::Path::new(launch::BENCH_RUNS_DIR);
    match ingest::ingest_once(&conn, bench_runs_dir) {
        Ok(()) => {
            // Print summary counts so the user can verify idempotency.
            let run_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))
                .unwrap_or(0);
            let mr_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM match_results", [], |row| row.get(0))
                .unwrap_or(0);
            let trial_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM trials", [], |row| row.get(0))
                .unwrap_or(0);
            let incumbent_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM incumbents", [], |row| row.get(0))
                .unwrap_or(0);
            println!(
                "ingest complete: {} runs, {} match_results, {} trials, {} incumbents",
                run_count, mr_count, trial_count, incumbent_count,
            );
        }
        Err(e) => {
            eprintln!("error: ingest failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_tuner_command_overrides_target_binary_from_game() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "breakthrough",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let idx = cmd
            .iter()
            .position(|a| a == "target.binary=target/release/game-breakthrough")
            .expect("target.binary override for the selected game");
        // Must be a value for the `--override` flag immediately before it.
        assert_eq!(cmd[idx - 1], "--override");
    }

    #[test]
    fn test_build_tuner_command_game_override_precedes_caller_overrides() {
        // The Python side's `_apply_overrides` keeps the last value for a
        // repeated key, so an explicit caller override for the same key
        // must come after (and thus win over) the game-derived one.
        let overrides = vec!["target.binary=custom/path".to_string()];
        let cmd = build_tuner_command(
            None,
            &overrides,
            &[],
            None,
            "druid",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let game_idx = cmd
            .iter()
            .position(|a| a == "target.binary=target/release/game-druid")
            .expect("game-derived target.binary override");
        let caller_idx = cmd
            .iter()
            .position(|a| a == "target.binary=custom/path")
            .expect("caller-supplied target.binary override");
        assert!(game_idx < caller_idx);
    }

    #[test]
    fn test_build_tuner_command_forwards_modern_attempt_identity() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "druid",
            Some("druid"),
            Some("tuner-druid-run-1"),
            None,
            Some("optimizer-druid"),
            Some("physical-druid"),
            None,
            None,
            None,
        );
        let run_id_idx = cmd
            .iter()
            .position(|a| a == "--run-id")
            .expect("--run-id flag present");
        assert_eq!(cmd[run_id_idx + 1], "tuner-druid-run-1");

        let optimizer_idx = cmd
            .iter()
            .position(|a| a == "--optimizer-id")
            .expect("--optimizer-id flag present");
        assert_eq!(cmd[optimizer_idx + 1], "optimizer-druid");
        let bench_run_idx = cmd
            .iter()
            .position(|a| a == "--bench-run-id")
            .expect("--bench-run-id flag present");
        assert_eq!(cmd[bench_run_idx + 1], "physical-druid");
        let game_kind_idx = cmd
            .iter()
            .position(|a| a == "--game-kind")
            .expect("--game-kind flag present");
        assert_eq!(cmd[game_kind_idx + 1], "druid");
        assert!(!cmd.iter().any(|a| a == "--resume"));
    }

    #[test]
    fn test_build_tuner_command_omits_optional_identity_when_absent() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "druid",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!cmd.iter().any(|a| a == "--run-id"));
        assert!(!cmd.iter().any(|a| a == "--resume"));
        assert!(!cmd.iter().any(|a| a == "--optimizer-id"));
        assert!(!cmd.iter().any(|a| a == "--bench-run-id"));
    }

    #[test]
    fn test_build_tuner_command_forwards_baseline_configs() {
        let baseline_configs = vec![r#"ladder1={"family":"ucb1"}"#.to_string()];
        let cmd = build_tuner_command(
            None,
            &[],
            &baseline_configs,
            None,
            "nim",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let idx = cmd
            .iter()
            .position(|a| a == "--baseline-config")
            .expect("--baseline-config flag present");
        assert_eq!(cmd[idx + 1], r#"ladder1={"family":"ucb1"}"#);
    }

    #[test]
    fn test_build_tuner_command_forwards_game_config() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            Some(r#"{"size":{"w":9,"h":9}}"#),
            "druid",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let idx = cmd
            .iter()
            .position(|a| a == "--game-config")
            .expect("--game-config flag present");
        assert_eq!(cmd[idx + 1], r#"{"size":{"w":9,"h":9}}"#);
    }

    #[test]
    fn test_build_tuner_command_omits_game_config_when_absent() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "druid",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!cmd.iter().any(|a| a == "--game-config"));
    }

    #[test]
    fn test_build_tuner_command_forwards_trace_path() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "druid",
            None,
            None,
            Some("bench-runs/tuner-druid-run-1/moves.jsonl"),
            None,
            None,
            None,
            None,
            None,
        );
        let idx = cmd
            .iter()
            .position(|a| a == "--trace-path")
            .expect("--trace-path flag present");
        assert_eq!(cmd[idx + 1], "bench-runs/tuner-druid-run-1/moves.jsonl");
    }

    #[test]
    fn test_build_tuner_command_omits_trace_path_when_absent() {
        let cmd = build_tuner_command(
            None,
            &[],
            &[],
            None,
            "druid",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!cmd.iter().any(|a| a == "--trace-path"));
    }
}
