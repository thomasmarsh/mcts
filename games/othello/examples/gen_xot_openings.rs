//! Generate the balanced 8-ply opening file (`games/othello/openings/xot8.txt`)
//! the Edax yardstick plays from.
//!
//! Protocol (the OLIVAW paper's XOT, reproduced rather than downloaded): draw
//! seeded random 8-ply legal sequences, reject any that contain a forced pass
//! or end the game, score each remaining position with Edax at depth 16, keep
//! the ones within +-`tolerance` discs of even, and dedupe by position under
//! the 8 board symmetries (same side to move). The pure parts live in
//! `game_othello::openings` with fast unit tests; this binary is the slow,
//! Edax-driven half.
//!
//! ```text
//! cargo run --release --example gen_xot_openings -p game-othello -- \
//!     [CONFIG] [--set key=value]...
//! ```
//!
//! `CONFIG` defaults to `games/othello/openings/gen_xot8.toml`.
//!
//! Seeded and resumable. Sample `i` is a pure function of `(seed, i)`, batches
//! are evaluated in parallel but recorded in index order, and every evaluated
//! position is appended to the progress JSONL as it finishes. Rerunning with
//! the same config continues from the last completed batch; a config whose
//! seed, plies, level or tolerance differ from the log's is refused rather
//! than mixed in. Each Edax score is a pure function of its position (the
//! hash is cleared between positions), so a rerun reproduces the same file.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use game_othello::edax::EdaxEval;
use game_othello::openings::{
    canonical_key, format_sequence, is_balanced, key_string, parse_sequence, sample_sequence,
};
use game_othello::{Move, Othello, State};
use mcts::game::Game;

mod common;
use common::load_toml_config;

#[derive(serde::Deserialize)]
struct Config {
    edax_binary: String,
    edax_data_dir: String,
    /// Edax level used to judge balance (the papers use depth 16).
    level: u32,
    /// Keep positions whose score is within this many discs of even.
    tolerance: f32,
    plies: usize,
    seed: u64,
    /// Stop after this many balanced positions (checked per batch, so the
    /// file can hold a few more).
    target: usize,
    /// Parallel Edax processes, each single-threaded.
    workers: usize,
    /// Samples drawn per batch.
    batch: u64,
    /// A search past this many seconds is abandoned and the position rejected.
    eval_timeout_s: u64,
    /// After generation, rescore this many kept openings in a fresh Edax
    /// process, in a different order, and refuse to write the file unless every
    /// score matches the log. Catches a desynchronised oracle, which is silent.
    verify_sample: usize,
    out: String,
    progress: String,
}

/// A position awaiting an Edax score.
struct Todo {
    i: u64,
    seq: String,
    key: String,
    state: State,
}

/// One Edax evaluation of a [`Todo`].
#[derive(Clone)]
struct Scored {
    score: f32,
    exact: bool,
    nodes: u64,
    secs: f64,
    timed_out: bool,
}

fn replay(moves: &[Move]) -> State {
    moves.iter().fold(State::default(), Othello::apply)
}

fn header(cfg: &Config) -> serde_json::Value {
    serde_json::json!({
        "type": "config", "seed": cfg.seed, "plies": cfg.plies,
        "level": cfg.level, "tolerance": cfg.tolerance,
    })
}

/// Rebuild the seen-set, kept positions and next sample index from a previous
/// run's progress log.
fn resume(cfg: &Config) -> (HashSet<String>, BTreeMap<String, String>, u64) {
    let mut seen = HashSet::new();
    let mut kept = BTreeMap::new();
    let mut next = 0u64;
    let Ok(text) = std::fs::read_to_string(&cfg.progress) else {
        return (seen, kept, next);
    };
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue; // a torn final line from an interrupted write
        };
        match v["type"].as_str() {
            Some("config") => {
                let want = header(cfg);
                for k in ["seed", "plies", "level", "tolerance"] {
                    assert_eq!(
                        v[k], want[k],
                        "{} was written with a different `{k}`; use a new progress path",
                        cfg.progress
                    );
                }
            }
            Some("eval") => {
                let key = v["key"].as_str().unwrap().to_string();
                seen.insert(key.clone());
                if v["status"] == "kept" {
                    kept.insert(key, v["seq"].as_str().unwrap().to_string());
                }
            }
            Some("batch") => next = v["next_index"].as_u64().unwrap(),
            _ => {}
        }
    }
    (seen, kept, next)
}

/// Logged score per kept position key.
fn progress_scores(cfg: &Config) -> BTreeMap<String, f32> {
    let text = std::fs::read_to_string(&cfg.progress).expect("read progress log");
    text.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["type"] == "eval" && v["status"] == "kept")
        .map(|v| (v["key"].as_str().unwrap().to_string(), v["score"].as_f64().unwrap() as f32))
        .collect()
}

fn verify(cfg: &Config, kept: &BTreeMap<String, String>, scores: &BTreeMap<String, f32>) {
    let mut ev = EdaxEval::spawn(
        &cfg.edax_binary,
        &cfg.edax_data_dir,
        cfg.level,
        Duration::from_secs(cfg.eval_timeout_s),
    );
    // Reverse key order, a stride through the set: not the order it was scored in.
    let picks: Vec<_> = kept.iter().rev().step_by((kept.len() / cfg.verify_sample.max(1)).max(1)).collect();
    let mut bad = 0;
    for (key, seq) in &picks {
        let state = parse_sequence(seq).expect("kept line parses");
        ev.clear_hash();
        let got = ev.eval(&state, cfg.level).score;
        if got != scores[*key] {
            bad += 1;
            eprintln!("verify: {seq} logged {} but a fresh Edax says {got}", scores[*key]);
        }
    }
    assert_eq!(bad, 0, "{bad} of {} rescored openings disagree with the log; not writing {}", picks.len(), cfg.out);
    println!("verified {} kept openings against a fresh Edax process", picks.len());
}

fn write_output(cfg: &Config, kept: &BTreeMap<String, String>) {
    let mut text = format!(
        "# xot8: balanced 8-ply Othello openings, one move sequence per line.\n\
         # Random seeded 8-ply lines, Edax level {} within +-{} discs, deduped under the 8 board symmetries.\n\
         # seed {}. Generated by games/othello/examples/gen_xot_openings.rs.\n",
        cfg.level, cfg.tolerance, cfg.seed
    );
    for seq in kept.values() {
        text.push_str(seq);
        text.push('\n');
    }
    let tmp = format!("{}.tmp", cfg.out);
    std::fs::write(&tmp, text).expect("write openings file");
    std::fs::rename(&tmp, &cfg.out).expect("publish openings file");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg_path = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "games/othello/openings/gen_xot8.toml".to_string());
    let cfg: Config = load_toml_config(&cfg_path, &args);
    assert!(cfg.workers >= 1 && cfg.batch >= 1);

    if let Some(dir) = Path::new(&cfg.progress).parent() {
        std::fs::create_dir_all(dir).expect("create progress dir");
    }
    let (mut seen, mut kept, mut next) = resume(&cfg);
    let mut progress = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.progress)
        .expect("open progress log");
    if next == 0 && seen.is_empty() {
        writeln!(progress, "{}", header(&cfg)).unwrap();
    }
    println!(
        "gen_xot_openings: level {} tolerance {} seed {} target {}; resuming at sample {next} with {} evaluated, {} kept",
        cfg.level, cfg.tolerance, cfg.seed, cfg.target, seen.len(), kept.len()
    );

    let mut evals: Vec<EdaxEval> = (0..cfg.workers)
        .map(|_| {
            EdaxEval::spawn(
                &cfg.edax_binary,
                &cfg.edax_data_dir,
                cfg.level,
                Duration::from_secs(cfg.eval_timeout_s),
            )
        })
        .collect();

    let started = Instant::now();
    let (mut rejected, mut dups, mut timeouts) = (0u64, 0u64, 0u64);
    let mut barren_batches = 0;
    while kept.len() < cfg.target {
        let mut todo: Vec<Todo> = Vec::new();
        for i in next..next + cfg.batch {
            let Some(moves) = sample_sequence(cfg.seed, i, cfg.plies) else {
                rejected += 1;
                continue;
            };
            let state = replay(&moves);
            let key = key_string(canonical_key(&state));
            if !seen.insert(key.clone()) {
                dups += 1;
                continue;
            }
            todo.push(Todo {
                i,
                seq: format_sequence(&moves),
                key,
                state,
            });
        }

        // Random draws stop finding new positions once the (small) space of
        // 8-ply positions is covered; say so instead of spinning.
        barren_batches = if todo.is_empty() { barren_batches + 1 } else { 0 };
        if barren_batches >= 100 {
            eprintln!("saturated: 100 consecutive batches found no new position");
            break;
        }

        let cursor = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<Scored>>> =
            Mutex::new(vec![None; todo.len()]);
        std::thread::scope(|scope| {
            for ev in evals.iter_mut() {
                let (todo, cursor, results, level) = (&todo, &cursor, &results, cfg.level);
                scope.spawn(move || loop {
                    let k = cursor.fetch_add(1, Ordering::SeqCst);
                    let Some(t) = todo.get(k) else { break };
                    ev.clear_hash();
                    let (t0, timeouts0, nodes0) = (Instant::now(), ev.timeouts(), ev.total_nodes());
                    let s = ev.eval(&t.state, level);
                    results.lock().unwrap()[k] = Some(Scored {
                        score: s.score,
                        exact: s.exact,
                        nodes: ev.total_nodes() - nodes0,
                        secs: t0.elapsed().as_secs_f64(),
                        timed_out: ev.timeouts() > timeouts0,
                    });
                });
            }
        });

        for (t, r) in todo.iter().zip(results.into_inner().unwrap()) {
            let Scored {
                score,
                exact,
                nodes,
                secs,
                timed_out,
            } = r.expect("every position was evaluated");
            let status = if timed_out {
                timeouts += 1;
                "timeout"
            } else if nodes == 0 && !exact {
                // `EdaxEval`'s neutral-score fallback (no parseable search
                // line): a fabricated 0.0 that would otherwise pass the filter.
                "no_score"
            } else if is_balanced(score, cfg.tolerance) {
                "kept"
            } else {
                "unbalanced"
            };
            writeln!(
                progress,
                "{}",
                serde_json::json!({
                    "type": "eval", "i": t.i, "seq": t.seq, "key": t.key, "score": score,
                    "exact": exact, "nodes": nodes, "secs": secs, "status": status,
                })
            )
            .unwrap();
            if status == "kept" {
                kept.insert(t.key.clone(), t.seq.clone());
            }
        }
        next += cfg.batch;
        writeln!(
            progress,
            "{}",
            serde_json::json!({
                "type": "batch", "next_index": next, "evaluated": seen.len(),
                "kept": kept.len(), "rejected": rejected, "duplicates": dups,
                "timeouts": timeouts, "elapsed_s": started.elapsed().as_secs_f64(),
            })
        )
        .unwrap();
        progress.flush().unwrap();
        eprintln!(
            "  sample {next}: evaluated {} kept {} (rejected {rejected}, duplicates {dups}, timeouts {timeouts}) {:.0}s",
            seen.len(),
            kept.len(),
            started.elapsed().as_secs_f64()
        );
    }

    verify(&cfg, &kept, &progress_scores(&cfg));
    write_output(&cfg, &kept);
    println!(
        "wrote {} balanced openings to {} ({} positions evaluated over {next} samples)",
        kept.len(),
        cfg.out,
        seen.len()
    );
}
