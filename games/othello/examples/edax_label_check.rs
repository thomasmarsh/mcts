//! Rescore the positions of a `dump --oracle edax` corpus (`arm_b.bin`, 22-byte
//! records) with the current `EdaxEval` in a fresh process and compare against
//! the stored labels.
//!
//! The labelling rule mirrors `dump.rs`'s `EdaxLabel` (sign mode): skip forced
//! passes, score terminal boards from the disc difference, solve exactly at or
//! below `--exact-ply` empties, otherwise search at `--level`.
//!
//! ```text
//! cargo run --release --example edax_label_check -p game-othello -- \
//!     --bin <arm_b.bin> --level 6 [--exact-ply 12] [--limit N] [--seed S] [--timeout-s 30] [--sign-only] [--fresh] [--compare-fresh]
//! ```
//!
//! `--sign-only` compares the sign of the stored target instead of its exact value
//! (for a corpus whose targets are continuous, such as an MCTS-oracle one).
//!
//! `--fresh` spawns a new Edax for every position (the stream-independent
//! reference); `--compare-fresh` scores each position both ways and prints every
//! streamed score that differs from the fresh one, which must never happen.
//!
//! With `--limit N` a seeded random sample of N records is rescored (in file
//! order); otherwise every record is. Reports mismatches overall and by ply
//! band, plus the fresh process's timeout count.

use std::time::Duration;

use game_othello::dump::{Record, RECORD_BYTES};
use game_othello::edax::EdaxEval;
use game_othello::{Move, Othello, Player, State};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn terminal_disc_diff(state: &State) -> i32 {
    let b = state.black.bits().count_ones() as i32;
    let w = state.white.bits().count_ones() as i32;
    let empty = 64 - b - w;
    let (mut mine, mut theirs) = match state.turn {
        Player::Black => (b, w),
        Player::White => (w, b),
    };
    if mine > theirs {
        mine += empty;
    } else if theirs > mine {
        theirs += empty;
    }
    mine - theirs
}

fn skip_forced_passes(mut state: State) -> (State, f32) {
    let mut s = 1.0f32;
    loop {
        if Othello::is_terminal(&state) {
            return (state, s);
        }
        let mut acts = Vec::new();
        Othello::generate_actions(&state, &mut acts);
        if acts.len() == 1 && acts[0] == Move::PASS {
            state = Othello::apply(state, &Move::PASS);
            s = -s;
        } else {
            return (state, s);
        }
    }
}

fn main() {
    let mut bin = None;
    let mut level = 6u32;
    let mut exact_ply = 12u32;
    let mut limit: Option<usize> = None;
    let mut seed = 1u64;
    let mut timeout_s = 30.0f64;
    let mut sign_only = false;
    let mut fresh = false;
    let mut compare_fresh = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--bin" => bin = Some(val()),
            "--level" => level = val().parse().unwrap(),
            "--exact-ply" => exact_ply = val().parse().unwrap(),
            "--limit" => limit = Some(val().parse().unwrap()),
            "--seed" => seed = val().parse().unwrap(),
            "--timeout-s" => timeout_s = val().parse().unwrap(),
            "--sign-only" => sign_only = true,
            "--fresh" => fresh = true,
            "--compare-fresh" => compare_fresh = true,
            other => panic!("unknown flag {other}"),
        }
    }
    let bin = bin.expect("--bin is required");
    let bytes = std::fs::read(&bin).unwrap_or_else(|e| panic!("cannot read {bin}: {e}"));
    assert_eq!(bytes.len() % RECORD_BYTES, 0, "not a whole number of records");
    let mut recs: Vec<Record> = bytes
        .chunks_exact(RECORD_BYTES)
        .map(|c| Record::decode(c.try_into().unwrap()))
        .collect();
    let total = recs.len();
    if let Some(n) = limit {
        if n < total {
            let mut idx: Vec<usize> = (0..total).collect();
            idx.shuffle(&mut SmallRng::seed_from_u64(seed));
            idx.truncate(n);
            idx.sort_unstable();
            recs = idx.into_iter().map(|i| recs[i]).collect();
        }
    }

    let spawn = || {
        EdaxEval::spawn(
            "games/othello/edax/vendor/bin/mEdax-native",
            "games/othello/edax/vendor/data",
            level,
            Duration::from_secs_f64(timeout_s),
        )
    };
    let mut edax = spawn();
    

    // Bands of 5 plies: index ply / 5 (0..=12).
    let mut n_band = [0usize; 13];
    let mut bad_band = [0usize; 13];
    let mut bad_flip = 0usize; // stored and rescored labels have opposite signs
    let mut bad_zero = 0usize; // exactly one of them is a draw
    let mut done = 0usize;
    for r in &recs {
        let state = State {
            black: game_othello::BB::from_bits(r.black),
            white: game_othello::BB::from_bits(r.white),
            turn: if r.side == 0 { Player::Black } else { Player::White },
            ..State::default()
        };
        let (state, s) = skip_forced_passes(state);
        let label = if Othello::is_terminal(&state) {
            s * sign(terminal_disc_diff(&state) as f32)
        } else {
            let empties = 64 - (state.black.bits() | state.white.bits()).count_ones();
            let lvl = if empties <= exact_ply { 60 } else { level };
            if fresh {
                edax = spawn();
            }
            let streamed = edax.eval(&state, lvl).score;
            if compare_fresh {
                let reference = spawn().eval(&state, lvl).score;
                if reference != streamed {
                    eprintln!(
                        "DIVERGE #{done} ply {} lvl {lvl} streamed {streamed} fresh {reference} board {}",
                        r.ply,
                        game_othello::edax::state_to_edax_board(&state)
                    );
                }
            }
            s * sign(streamed)
        };
        let band = (r.ply as usize / 5).min(12);
        n_band[band] += 1;
        let stored = if sign_only { sign(r.target) } else { r.target };
        if label != stored {
            bad_band[band] += 1;
            if label == 0.0 || stored == 0.0 {
                bad_zero += 1;
            } else {
                bad_flip += 1;
            }
        }
        done += 1;
        if done.is_multiple_of(2000) {
            eprintln!("{done}/{} rescored", recs.len());
        }
    }

    let bad: usize = bad_band.iter().sum();
    println!("file={bin} records={total} rescored={} level={level} exact_ply={exact_ply}", recs.len());
    println!(
        "mismatches={bad} ({:.3}%)  sign_flips={bad_flip}  draw_involved={bad_zero}  fresh_timeouts={}",
        100.0 * bad as f64 / recs.len().max(1) as f64,
        edax.timeouts()
    );
    for b in 0..13 {
        if n_band[b] > 0 {
            println!(
                "  ply {:>2}-{:<2}: n={:>6} mismatches={:>5} ({:.2}%)",
                b * 5,
                b * 5 + 4,
                n_band[b],
                bad_band[b],
                100.0 * bad_band[b] as f64 / n_band[b] as f64
            );
        }
    }
}
