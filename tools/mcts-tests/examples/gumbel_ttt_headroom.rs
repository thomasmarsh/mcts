//! How much headroom does a value head have on tic-tac-toe, and does the
//! current setup use search well?
//!
//!     cargo run --release -p mcts-tests --example gumbel_ttt_headroom -- [weights.bin ...]
//!
//! Reports perfect-play move agreement over every decisive non-terminal
//! opening position (ply <= 4), averaged over three search seeds -- the same
//! metric `gumbel_ttt_gate` uses -- for a matrix of configurations:
//!
//!  * **pure search** -- the zero net at `max_playout_depth = 0`, so the
//!    only signal is the Gumbel schedule itself (no value, no rollout).
//!  * **uniform rollout** -- the zero net at `max_playout_depth = 9`, so
//!    every leaf is a real random playout to terminal. This is the
//!    "no value head" baseline.
//!  * one row per `weights.bin` given on the command line, at
//!    `max_playout_depth = 0` (AlphaZero-style leaf evaluation).
//!
//! swept over simulation budget and `c_visit`. If pure search and uniform
//! rollout already sit near the ceiling at a small budget, a trained value
//! head has nothing to add here and the strength gate is measuring noise; if
//! trained weights score below the zero-net "pure search" row, the loop is
//! regressing rather than learning.

use std::collections::HashSet;

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};

use game_ttt::selfplay::GumbelPlayer;
use game_ttt::valuenet::NTupleValueNet;
use game_ttt::{HashedPosition, TicTacToe};

fn exact_value(state: &HashedPosition) -> i32 {
    if TicTacToe::is_terminal(state) {
        return if TicTacToe::winner(state).is_some() {
            -1
        } else {
            0
        };
    }
    let mut actions = Vec::new();
    TicTacToe::generate_actions(state, &mut actions);
    actions
        .iter()
        .map(|a| -exact_value(&TicTacToe::apply(*state, a)))
        .max()
        .unwrap()
}

fn opening_positions(max_ply: u32) -> Vec<HashedPosition> {
    let mut seen: HashSet<(u32, u8)> = HashSet::new();
    let mut out = Vec::new();
    let mut frontier = vec![HashedPosition::new()];
    while let Some(state) = frontier.pop() {
        let key = (
            state.position.board,
            TicTacToe::player_to_move(&state).to_index() as u8,
        );
        if !seen.insert(key) {
            continue;
        }
        if TicTacToe::is_terminal(&state) {
            continue;
        }
        let ply = (0..9).filter(|&i| state.position.get(i).is_some()).count() as u32;
        if ply <= max_ply {
            out.push(state);
        }
        if ply < max_ply {
            let mut actions = Vec::new();
            TicTacToe::generate_actions(&state, &mut actions);
            for a in actions {
                frontier.push(TicTacToe::apply(state, &a));
            }
        }
    }
    out
}

fn agreement(
    net: &NTupleValueNet,
    depth: usize,
    cfg: GumbelConfig,
    positions: &[HashedPosition],
) -> f64 {
    let mut optimal = 0usize;
    let mut total = 0usize;
    for seed in [0xA2_5EEDu64, 0xB0_1234, 0xC0_FFEE] {
        let mut player = GumbelPlayer::with_playout_depth(net.clone(), cfg, seed, depth);
        for state in positions {
            let before = exact_value(state);
            let mut actions = Vec::new();
            TicTacToe::generate_actions(state, &mut actions);
            let decisive = actions
                .iter()
                .any(|a| -exact_value(&TicTacToe::apply(*state, a)) != before);
            if !decisive {
                continue;
            }
            total += 1;
            let action = player.choose_action(state);
            if -exact_value(&TicTacToe::apply(*state, &action)) == before {
                optimal += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        optimal as f64 / total as f64
    }
}

fn main() {
    let weight_paths: Vec<String> = std::env::args().skip(1).collect();
    let positions = opening_positions(4);
    let zero = NTupleValueNet::default();
    let trained: Vec<(String, NTupleValueNet)> = weight_paths
        .iter()
        .map(|p| {
            (
                p.clone(),
                NTupleValueNet::load(p).unwrap_or_else(|e| panic!("cannot load {p}: {e}")),
            )
        })
        .collect();

    let decisive = {
        let mut n = 0;
        for state in &positions {
            let before = exact_value(state);
            let mut actions = Vec::new();
            TicTacToe::generate_actions(state, &mut actions);
            if actions
                .iter()
                .any(|a| -exact_value(&TicTacToe::apply(*state, a)) != before)
            {
                n += 1;
            }
        }
        n
    };
    println!("perfect-play agreement over {decisive} decisive opening positions x 3 seeds\n");

    for &c_visit in &[1.0f64, 5.0, 15.0, 50.0] {
        println!("c_visit = {c_visit}");
        println!(
            "  {:<18} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "config", "sims=8", "16", "32", "64", "128"
        );
        let row = |label: &str, net: &NTupleValueNet, depth: usize| {
            print!("  {label:<18}");
            for &sims in &[8u32, 16, 32, 64, 128] {
                let cfg = GumbelConfig {
                    sims,
                    c_visit,
                    ..GumbelConfig::default()
                };
                print!(" {:>8.3}", agreement(net, depth, cfg, &positions));
            }
            println!();
        };
        row("pure search", &zero, 0);
        row("uniform rollout", &zero, 9);
        for (name, net) in &trained {
            let short: String = name.rsplit('/').next().unwrap_or(name).to_string();
            row(&short, net, 0);
        }
        println!();
    }
}
