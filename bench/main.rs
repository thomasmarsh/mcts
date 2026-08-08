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

use mcts::bench::games::registry;
use mcts::bench::ingest;
use mcts::bench::launch::{self, LaunchedRun};
use mcts::bench::schema;
use mcts::bench::tournament::round_robin_bench_multiple;
use mcts::build_info;
use mcts::util::Verbosity;

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
    /// ``uv run --project scripts/ hyper-cli ...`` in the foreground
    /// (streaming JSONL to stdout) or, with ``--background``, through the
    /// detached-process launcher so the run survives the launching
    /// process and appears in DuckDB/the UI.
    Smac3 {
        /// Path to the SMAC3 YAML config file (passed through to
        /// ``hyper-cli --config``).
        #[arg(long)]
        config: Option<String>,

        /// Config override (``key=value``, repeatable).  Passed through
        /// as ``hyper-cli --override key=value``.
        #[arg(long = "override", default_values_t = Vec::<String>::new())]
        overrides: Vec<String>,

        /// Game kind for registry attribution (e.g. "druid").
        #[arg(long)]
        game: String,

        /// Optional human-readable label for the run.
        #[arg(long)]
        label: Option<String>,

        /// Launch in the background (detached process) instead of
        /// running in the foreground.
        #[arg(long)]
        background: bool,
    },

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
            game,
            label,
            background,
        } => cmd_smac3(config.as_deref(), &overrides, &game, label.as_deref(), background),

        Command::Ingest { db } => cmd_ingest_once(&db),
    }
}

fn cmd_round_robin(game_kind: &str, strategy_ids: &[String], rounds: usize, verbose: bool) {
    let games = registry();
    let Some(bench_game) = games.get(game_kind) else {
        eprintln!("error: unknown game kind '{game_kind}'");
        eprintln!("available games: {}", games.keys().cloned().collect::<Vec<_>>().join(", "));
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
                    available.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
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
    let results = round_robin_bench_multiple(
        bench_game.as_ref(),
        &ids,
        rounds,
        &mut writer,
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

/// Build the argv for a ``uv run --project scripts/ hyper-cli ...``
/// invocation, incorporating the config file, overrides, and git SHA.
fn build_smac3_command(config: Option<&str>, overrides: &[String]) -> Vec<String> {
    let mut cmd = vec![
        "uv".to_string(),
        "run".to_string(),
        "--project".to_string(),
        "scripts/".to_string(),
        "hyper-cli".to_string(),
    ];

    if let Some(config_path) = config {
        cmd.push("--config".to_string());
        cmd.push(config_path.to_string());
    }

    for ov in overrides {
        cmd.push("--override".to_string());
        cmd.push(ov.clone());
    }

    // Pass the compile-time git SHA so the Python side can include it
    // in its JSONL output for attribution.
    cmd.push("--git-sha".to_string());
    cmd.push(build_info::MCTS_GIT_SHA.to_string());

    cmd
}

fn cmd_smac3(
    config: Option<&str>,
    overrides: &[String],
    game: &str,
    label: Option<&str>,
    background: bool,
) {
    let cmd = build_smac3_command(config, overrides);

    if background {
        // Launch via the detached-process launcher.
        let LaunchedRun {
            run_id,
            pid,
            log_path,
            ..
        } = match launch::launch(cmd, "smac3", game, label) {
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
        // piping stdout.
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
            println!(
                "ingest complete: {} runs, {} match_results, {} trials",
                run_count, mr_count, trial_count,
            );
        }
        Err(e) => {
            eprintln!("error: ingest failed: {e}");
            std::process::exit(1);
        }
    }
}