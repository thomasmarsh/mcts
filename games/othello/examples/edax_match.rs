//! Place our Othello MCTS engine on an external, non-self-referential
//! strength scale by playing it against Edax at fixed search levels.
//!
//! This gives learned-evaluator work an external reference point: strength
//! is reported as "Edax level N" rather than as a delta between two of our
//! own configs. This binary produces that integer.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example edax_match -p game-othello -- [CONFIG] [MODE]
//! ```
//!
//! `CONFIG` defaults to `games/othello/edax/match.toml`; any key can be
//! overridden with `--set key=value` (e.g. `--set our_preset=\"medium\"`
//! `--set levels=[1,2,3]`). `MODE` is one of:
//!
//! - `ladder` -- Run A: validate the ladder is monotone (Edax L vs L+2).
//! - `place`  -- Run B: place `our_preset` on the ladder (default). Balanced
//!   openings from `openings` (see `gen_xot_openings`), every opening played
//!   from both seats, Wilson 95% interval, seconds and nodes per move.
//! - `both`   -- ladder then place.
//!
//! Build Edax first with `games/othello/edax/build-edax.sh`. This is a
//! background job: progress goes to stderr and, with `out` set, one JSONL row
//! per game and per level is appended as it finishes. The match-play
//! scaffolding (openings, gate driver, Edax subprocess, reporting) lives in
//! `examples/common/mod.rs`, shared with `ntuple_match` and `gumbel_gate`;
//! `common::run_edax_gate` is the engine-builder hook, so gating another
//! agent means giving it a `make(seed)` closure, not a new driver.

use std::path::Path;

mod common;
use common::{
    build_preset_engine, load_toml_config, play_series, report_row, run_edax_gate, EdaxPlayer,
    GateConfig, Tally,
};

#[derive(serde::Deserialize)]
struct Config {
    our_preset: String,
    #[serde(flatten)]
    gate: GateConfig,
}

/// A preset's `max_iterations` (its per-move search budget) from presets.json.
fn preset_iterations(presets_path: &Path, preset: &str) -> Option<u64> {
    let table: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(presets_path).ok()?).ok()?;
    table
        .as_array()?
        .iter()
        .find(|p| p["id"] == preset)?
        .get("max_iterations")?
        .as_u64()
}

/// Run A: deeper Edax must beat shallower with a CI excluding 0.5.
fn run_ladder(cfg: &Config) {
    println!("== Run A: ladder monotonicity (Edax L vs L+2) ==");
    let n = cfg.gate.games_per_level.max(30);
    let mut ok = true;
    for l in [1u32, 3, 5, 7] {
        let mut deep = EdaxPlayer::spawn_with_threads(
            &cfg.gate.edax_binary,
            &cfg.gate.edax_data_dir,
            l + 2,
            cfg.gate.edax_threads,
        );
        let mut shallow = EdaxPlayer::spawn_with_threads(
            &cfg.gate.edax_binary,
            &cfg.gate.edax_data_dir,
            l,
            cfg.gate.edax_threads,
        );
        let label = format!("L{} v L{}", l + 2, l);
        let seed = 0xA000 ^ ((l as u64) << 8);
        let t = play_series(&mut deep, &mut shallow, n, seed, &label);
        report_row(&label, &t);
        let (_, (lo, _)) = t.win_rate_ci(1.96);
        if lo <= 0.5 {
            println!(
                "  !! L{} did not clearly beat L{} (ci lower bound {lo:.3})",
                l + 2,
                l
            );
            ok = false;
        }
    }
    println!(
        "ladder monotonicity: {}",
        if ok {
            "PASS"
        } else {
            "FAIL -- fix the driver before trusting Run B"
        }
    );
}

/// Run B: place `our_preset` on the ladder.
fn run_place(cfg: &Config) {
    println!("== Run B: {} vs Edax ==", cfg.our_preset);
    let presets = Path::new("games/othello/presets.json");
    let budget = preset_iterations(presets, &cfg.our_preset);
    let rows = run_edax_gate(&cfg.gate, &cfg.our_preset, budget, |seed| {
        build_preset_engine(presets, &cfg.our_preset, seed)
    });
    let rows: Vec<(u32, Tally)> = rows.into_iter().map(|r| (r.level, r.tally)).collect();

    // N = highest level where our CI lower bound is still >= 0.5.
    let mut n_clear = None;
    let mut cross = None;
    for (l, t) in &rows {
        let (p, (lo, _)) = t.win_rate_ci(1.96);
        if lo >= 0.5 {
            n_clear = Some(*l);
        }
        if p < 0.5 && cross.is_none() {
            cross = Some(*l);
        }
    }
    println!();
    match n_clear {
        Some(l) => println!(
            "N = {l}  (our '{}' preset clearly beats Edax up to level {l})",
            cfg.our_preset
        ),
        None => println!(
            "N = 0  (our '{}' preset does not clearly beat even the lowest level tested)",
            cfg.our_preset
        ),
    }
    match cross {
        Some(l) => println!("point estimate crosses 0.5 at level {l}"),
        None => println!(
            "point estimate never crossed 0.5 -- extend `levels` upward in match.toml and rerun"
        ),
    }
}

fn main() {
    let positional: Vec<String> = {
        let mut out = Vec::new();
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            if a == "--set" {
                it.next();
            } else {
                out.push(a);
            }
        }
        out
    };
    let is_mode = |a: &str| matches!(a, "ladder" | "place" | "both");
    let cfg_path = positional
        .first()
        .filter(|a| !is_mode(a))
        .cloned()
        .unwrap_or_else(|| "games/othello/edax/match.toml".to_string());
    let mode = positional
        .iter()
        .find(|a| is_mode(a))
        .cloned()
        .unwrap_or_else(|| "place".to_string());

    let all_args: Vec<String> = std::env::args().skip(1).collect();
    let cfg: Config = load_toml_config(&cfg_path, &all_args);

    println!(
        "edax={}  preset={}  levels={:?}  games/level={}  seed={}",
        cfg.gate.edax_binary,
        cfg.our_preset,
        cfg.gate.levels,
        cfg.gate.games_per_level,
        cfg.gate.seed
    );

    match mode.as_str() {
        "ladder" => run_ladder(&cfg),
        "place" => run_place(&cfg),
        "both" => {
            run_ladder(&cfg);
            run_place(&cfg);
        }
        other => panic!("unknown mode {other:?} (want ladder | place | both)"),
    }
}
