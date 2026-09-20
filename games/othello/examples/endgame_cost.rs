//! Cost of the exact win/draw/loss endgame solver by number of empties.
//!
//! ```text
//! cargo run --release --example endgame_cost -p game-othello -- \
//!     [--min 4] [--max 18] [--positions 40] [--seed 1] [--budget-secs 20]
//! ```
//!
//! For each empties count, plays seeded random games down to that many empties
//! and times `solve_wld` on each position (the per-thread table persists
//! between calls, as it does inside a search). Stops raising the empties count
//! once one level's mean exceeds `--budget-secs` per solve. Prints one row per
//! level: mean, median, max milliseconds and the win/draw/loss split.

use std::time::Instant;

use game_othello::endgame::solve_wld;
use game_othello::{Othello, State};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn main() {
    let (mut min, mut max, mut positions, mut seed, mut budget) = (4u32, 18u32, 40usize, 1u64, 20.0f64);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--min" => min = val().parse().unwrap(),
            "--max" => max = val().parse().unwrap(),
            "--positions" => positions = val().parse().unwrap(),
            "--seed" => seed = val().parse().unwrap(),
            "--budget-secs" => budget = val().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    println!("empties    n  mean_ms  median_ms    max_ms   W/D/L");
    for empties in min..=max {
        let mut ms = Vec::new();
        let mut wdl = [0u32; 3];
        while ms.len() < positions {
            let mut s = State::default();
            let mut actions = Vec::new();
            while !Othello::is_terminal(&s) && 64 - (s.black.count_ones() + s.white.count_ones()) > empties {
                actions.clear();
                Othello::generate_actions(&s, &mut actions);
                s = Othello::apply(s, &actions[rng.gen_range(0..actions.len())]);
            }
            if Othello::is_terminal(&s) {
                continue;
            }
            let t = Instant::now();
            let v = solve_wld(&s);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            wdl[(v + 1) as usize] += 1;
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        println!(
            "{empties:>7} {:>4} {mean:>8.3} {:>10.3} {:>9.1}   {}/{}/{}",
            ms.len(),
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            wdl[2],
            wdl[1],
            wdl[0]
        );
        if mean > budget * 1e3 {
            break;
        }
    }
}
