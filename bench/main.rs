//! CLI for the benchmark / tournament / SMAC3 harness.
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

use game_host::build_info;
use mcts::util::Verbosity;
use mcts_bench::ingest;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::registry;
use mcts_bench::schema;
use mcts_bench::tournament::round_robin_bench_multiple;

#[derive(Parser)]
#[command(name = "bench", about = "Benchmark, tournament, and SMAC3 harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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
    },

    /// Launch a detached background run via the OS process launcher.
    /// Everything after `--` is the command to run (argv including the
    /// binary path).  The launcher redirects stdout to the run's
    /// `log.jsonl` and stderr to `stdout.log`, writes a start event to
    /// `registry.log`, and returns immediately.  The run survives the
    /// launching process.
    Launch {
        /// Machine-readable kind string (e.g. "round_robin", "smac3").
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

    /// Launch a SMAC3 hyperparameter-optimisation run.  Runs
    /// ``uv run --project smac3/ smac3 ...`` in the foreground
    /// (streaming JSONL to stdout) or, with ``--background``, through the
    /// detached-process launcher so the run survives the launching
    /// process and appears in DuckDB/the UI.
    Smac3 {
        /// Path to the SMAC3 YAML config file (passed through to
        /// ``smac3 --config``).
        #[arg(long)]
        config: Option<String>,

        /// Config override (``key=value``, repeatable).  Passed through
        /// as ``smac3 --override key=value``.
        #[arg(long = "override", default_values_t = Vec::<String>::new())]
        overrides: Vec<String>,

        /// Extra baseline instance backed by a raw discovered config
        /// (``id=json``, repeatable), for evaluating a candidate against an
        /// opponent that isn't one of the game's named presets.  Passed
        /// through as ``smac3 --baseline-config id=json``.
        #[arg(long = "baseline-config", default_values_t = Vec::<String>::new())]
        baseline_configs: Vec<String>,

        /// Game-setup config (JSON object, e.g. Druid's `{"size":{"w":9,
        /// "h":9}}`) pinning every trial in this run to a non-default game
        /// config instead of the game's own `default_config()`.  Passed
        /// through as ``smac3 --game-config <json>``.
        #[arg(long = "game-config")]
        game_config: Option<String>,

        /// Game kind for registry attribution (e.g. "druid").
        #[arg(long)]
        game: String,

        /// Optional human-readable label for the run.
        #[arg(long)]
        label: Option<String>,

        /// Pin the launched run's `Scenario.name` to this id (passed
        /// through as `smac3 --run-id`), so its output directory is
        /// discoverable later for `--resume`.
        #[arg(long = "run-id")]
        run_id: Option<String>,

        /// Resume a prior run by its `--run-id` (passed through as
        /// `smac3 --resume`).
        #[arg(long)]
        resume: Option<String>,

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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::RoundRobin {
            game,
            strategies,
            rounds,
            verbose,
        } => cmd_round_robin(&game, &strategies, rounds, verbose),

        Command::Launch {
            kind,
            game,
            label,
            cmd,
        } => cmd_launch(&kind, &game, label.as_deref(), &cmd),

        Command::Smac3 {
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            label,
            run_id,
            resume,
            background,
        } => cmd_smac3(
            config.as_deref(),
            &overrides,
            &baseline_configs,
            game_config.as_deref(),
            &game,
            label.as_deref(),
            run_id.as_deref(),
            resume.as_deref(),
            background,
        ),

        Command::Games => cmd_games(),

        Command::Ingest { db } => cmd_ingest_once(&db),
    }
}

fn cmd_round_robin(game_kind: &str, strategy_ids: &[String], rounds: usize, verbose: bool) {
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
    let results = round_robin_bench_multiple(bench_game.as_ref(), &ids, rounds, &mut writer, verb);

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
    } = match launch::launch(cmd.to_vec(), kind, game, label) {
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

/// Build the argv for a ``uv run --project smac3/ smac3 ...``
/// invocation, incorporating the config file, overrides, and git SHA.
///
/// `game` is translated into a `target.binary=target/release/game-<game>`
/// override so the launched run actually tunes the selected game --
/// `smac3/config/default.yaml`'s `target.binary` is just a fallback default
/// (currently `game-traffic-lights`, the reference wiring's game), not
/// something any caller of `bench smac3 --game ...` should rely on. This is
/// pushed before `overrides` so an explicit `target.binary=...` override
/// from the caller still wins (the Python side's `_apply_overrides` keeps
/// the last value for a repeated key).
#[allow(clippy::too_many_arguments)]
fn build_smac3_command(
    config: Option<&str>,
    overrides: &[String],
    baseline_configs: &[String],
    game_config: Option<&str>,
    game: &str,
    run_id: Option<&str>,
    resume: Option<&str>,
) -> Vec<String> {
    let mut cmd = vec![
        "uv".to_string(),
        "run".to_string(),
        "--project".to_string(),
        "smac3/".to_string(),
        "smac3".to_string(),
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
    cmd.push(build_info::GIT_SHA.to_string());

    if let Some(id) = run_id {
        cmd.push("--run-id".to_string());
        cmd.push(id.to_string());
    }

    if let Some(id) = resume {
        cmd.push("--resume".to_string());
        cmd.push(id.to_string());
    }

    cmd
}

#[allow(clippy::too_many_arguments)]
fn cmd_smac3(
    config: Option<&str>,
    overrides: &[String],
    baseline_configs: &[String],
    game_config: Option<&str>,
    game: &str,
    label: Option<&str>,
    run_id: Option<&str>,
    resume: Option<&str>,
    background: bool,
) {
    if background {
        // Pin the run_id up front (rather than letting `launch::launch`
        // generate one internally) so the same id both names the
        // bench-runs directory/registry entry *and* is baked into the
        // child's own `--run-id` argv -- otherwise the two would disagree
        // and a later `--resume <bench-run-id>` couldn't find the SMAC3
        // output directory it actually needs.
        let run_id = run_id
            .map(str::to_string)
            .unwrap_or_else(|| launch::generate_run_id("smac3", game));
        let cmd = build_smac3_command(
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            Some(&run_id),
            resume,
        );

        // Launch via the detached-process launcher.
        let LaunchedRun {
            run_id,
            pid,
            log_path,
            ..
        } = match launch::launch_with_run_id(run_id, cmd, "smac3", game, label) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: failed to launch SMAC3 run: {e}");
                std::process::exit(1);
            }
        };

        let output = serde_json::json!({
            "run_id": run_id,
            "pid": pid,
            "log_path": log_path.to_string_lossy(),
            "kind": "smac3",
            "game": game,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        // Run in the foreground — inherit stdout/stderr so the Python
        // JSONL stream goes directly to the terminal or whoever is
        // piping stdout. Unlike the background branch, `run_id`/`resume`
        // are forwarded as given rather than auto-generated: this path has
        // no bench-runs registry entry of its own to keep in sync (a
        // caller that wraps this in its *own* launcher, e.g. the server,
        // is responsible for passing a `--run-id` that matches whatever it
        // used for that outer entry).
        let cmd = build_smac3_command(
            config,
            overrides,
            baseline_configs,
            game_config,
            game,
            run_id,
            resume,
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
    fn test_build_smac3_command_overrides_target_binary_from_game() {
        let cmd = build_smac3_command(None, &[], &[], None, "breakthrough", None, None);
        let idx = cmd
            .iter()
            .position(|a| a == "target.binary=target/release/game-breakthrough")
            .expect("target.binary override for the selected game");
        // Must be a value for the `--override` flag immediately before it.
        assert_eq!(cmd[idx - 1], "--override");
    }

    #[test]
    fn test_build_smac3_command_game_override_precedes_caller_overrides() {
        // The Python side's `_apply_overrides` keeps the last value for a
        // repeated key, so an explicit caller override for the same key
        // must come after (and thus win over) the game-derived one.
        let overrides = vec!["target.binary=custom/path".to_string()];
        let cmd = build_smac3_command(None, &overrides, &[], None, "druid", None, None);
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
    fn test_build_smac3_command_forwards_run_id_and_resume() {
        let cmd = build_smac3_command(
            None,
            &[],
            &[],
            None,
            "druid",
            Some("smac3-druid-run-1"),
            Some("smac3-druid-run-0"),
        );
        let run_id_idx = cmd
            .iter()
            .position(|a| a == "--run-id")
            .expect("--run-id flag present");
        assert_eq!(cmd[run_id_idx + 1], "smac3-druid-run-1");

        let resume_idx = cmd
            .iter()
            .position(|a| a == "--resume")
            .expect("--resume flag present");
        assert_eq!(cmd[resume_idx + 1], "smac3-druid-run-0");
    }

    #[test]
    fn test_build_smac3_command_omits_run_id_and_resume_when_absent() {
        let cmd = build_smac3_command(None, &[], &[], None, "druid", None, None);
        assert!(!cmd.iter().any(|a| a == "--run-id"));
        assert!(!cmd.iter().any(|a| a == "--resume"));
    }

    #[test]
    fn test_build_smac3_command_forwards_baseline_configs() {
        let baseline_configs = vec![r#"ladder1={"family":"ucb1"}"#.to_string()];
        let cmd = build_smac3_command(None, &[], &baseline_configs, None, "nim", None, None);
        let idx = cmd
            .iter()
            .position(|a| a == "--baseline-config")
            .expect("--baseline-config flag present");
        assert_eq!(cmd[idx + 1], r#"ladder1={"family":"ucb1"}"#);
    }

    #[test]
    fn test_build_smac3_command_forwards_game_config() {
        let cmd = build_smac3_command(
            None,
            &[],
            &[],
            Some(r#"{"size":{"w":9,"h":9}}"#),
            "druid",
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
    fn test_build_smac3_command_omits_game_config_when_absent() {
        let cmd = build_smac3_command(None, &[], &[], None, "druid", None, None);
        assert!(!cmd.iter().any(|a| a == "--game-config"));
    }
}
