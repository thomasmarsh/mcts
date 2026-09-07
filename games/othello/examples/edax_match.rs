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
//! `CONFIG` defaults to `games/othello/edax/match.toml`. `MODE` is one of:
//!
//! - `ladder` -- Run A: validate the ladder is monotone (Edax L vs L+2).
//! - `place`  -- Run B: place `our_preset` on the ladder (default).
//! - `both`   -- ladder then place.
//!
//! Build Edax first with `games/othello/edax/build-edax.sh`. This is a
//! background job: per-game progress goes to stderr. The match-play
//! scaffolding (openings, series driver, Edax subprocess, reporting) lives
//! in `examples/common/mod.rs`, shared with `ntuple_match`.

use std::path::Path;

mod common;
use common::{build_preset_engine, play_series, report_row, EdaxPlayer, Tally};

#[derive(serde::Deserialize)]
struct Config {
    edax_binary: String,
    edax_data_dir: String,
    our_preset: String,
    levels: Vec<u32>,
    games_per_level: u32,
    seed: u64,
}

/// Run A: deeper Edax must beat shallower with a CI excluding 0.5.
fn run_ladder(cfg: &Config) {
    println!("== Run A: ladder monotonicity (Edax L vs L+2) ==");
    let n = cfg.games_per_level.max(30);
    let mut ok = true;
    for l in [1u32, 3, 5, 7] {
        let mut deep = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l + 2);
        let mut shallow = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l);
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
    let n = cfg.games_per_level;
    let presets = Path::new("games/othello/presets.json");
    let mut rows: Vec<(u32, Tally)> = Vec::new();
    for &l in &cfg.levels {
        let mut ours =
            build_preset_engine(presets, &cfg.our_preset, cfg.seed.wrapping_add(l as u64));
        let mut edax = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l);
        let label = format!("{} v L{l}", cfg.our_preset);
        let seed = cfg.seed.wrapping_add((l as u64) << 8);
        let t = play_series(&mut ours, &mut edax, n, seed, &label);
        report_row(&format!("vs edax-L{l}"), &t);
        rows.push((l, t));
    }

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
    let mut args = std::env::args().skip(1);
    let cfg_path = args
        .next()
        .filter(|a| !a.starts_with("--") && a != "ladder" && a != "place" && a != "both")
        .unwrap_or_else(|| "games/othello/edax/match.toml".to_string());
    let mode = args.next().unwrap_or_else(|| "place".to_string());

    let cfg: Config = toml::from_str(
        &std::fs::read_to_string(&cfg_path)
            .unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}")),
    )
    .expect("config must parse");

    println!(
        "edax={}  preset={}  levels={:?}  games/level={}  seed={}",
        cfg.edax_binary, cfg.our_preset, cfg.levels, cfg.games_per_level, cfg.seed
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
