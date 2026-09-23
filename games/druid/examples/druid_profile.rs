//! Move-generation cost and branching factor of Druid at several board sizes, from uniformly
//! random self-play (`cargo run --release --example druid_profile -p game-druid [games] [sizes..]`).
//!
//! Reports, per size: whole-turn branching (the `Flat` action count at turn boundaries), per-ply
//! branching of the shipped `Split` encoding (what a tree search actually sees), plies per game,
//! outcome mix, and throughput of `generate_actions`, `terminal_status` and `apply` measured on
//! the positions the games visited.

use std::time::Instant;

use game_druid::{DruidFlat, DruidSplit, HashedState, Pending, Size};
use mcts::game::{Game, TerminalStatus};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(200);
    let sizes: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let sizes = if sizes.is_empty() { vec![5, 7, 9] } else { sizes };
    for n in sizes {
        profile(Size { w: n, h: n }, games);
    }
}

fn profile(size: Size, games: usize) {
    let mut rng = SmallRng::seed_from_u64(7);
    let (mut wins, mut draws) = ([0usize; 2], 0usize);
    let (mut plies, mut turns) = (0usize, 0usize);
    let (mut split_sum, mut flat_sum, mut flat_max) = (0usize, 0usize, 0usize);
    let mut by_phase = [(0usize, 0usize); 4];
    let mut visited: Vec<HashedState> = Vec::new();
    let keep_every = 7;

    for _ in 0..games {
        let mut s = HashedState::new(size);
        loop {
            match DruidSplit::terminal_status(&s) {
                TerminalStatus::NotTerminal => {}
                TerminalStatus::Winner(p) => {
                    wins[p as usize] += 1;
                    break;
                }
                _ => {
                    draws += 1;
                    break;
                }
            }
            let mut acts = Vec::new();
            DruidSplit::generate_actions(&s, &mut acts);
            let phase = match s.state().pending {
                Pending::None => 0,
                Pending::Piece(_) => 1,
                Pending::Oriented(_) => 2,
            };
            by_phase[phase].0 += acts.len();
            by_phase[phase].1 += 1;
            split_sum += acts.len();
            if phase == 0 {
                let mut flat = Vec::new();
                <game_druid::Flat as game_druid::MoveEncoding>::generate_actions(&s, &mut flat);
                flat_sum += flat.len();
                flat_max = flat_max.max(flat.len());
                turns += 1;
            }
            if plies % keep_every == 0 {
                visited.push(s.clone());
            }
            plies += 1;
            s = DruidSplit::apply(s, &acts[rng.gen_range(0..acts.len())]);
        }
    }
    let _ = DruidFlat::num_players();

    println!("== {}x{} ({} random games) ==", size.w, size.h, games);
    println!(
        "plies/game {:.1}  turns/game {:.1}  plies/turn {:.2}  outcomes: black {} white {} draw {}",
        plies as f64 / games as f64,
        turns as f64 / games as f64,
        plies as f64 / turns as f64,
        wins[0],
        wins[1],
        draws
    );
    println!(
        "branching: whole-turn mean {:.1} max {}  |  split per-ply mean {:.1}  (turn-start {:.1}, piece-chosen {:.1}, oriented {:.1})",
        flat_sum as f64 / turns as f64,
        flat_max,
        split_sum as f64 / plies as f64,
        by_phase[0].0 as f64 / by_phase[0].1.max(1) as f64,
        by_phase[1].0 as f64 / by_phase[1].1.max(1) as f64,
        by_phase[2].0 as f64 / by_phase[2].1.max(1) as f64,
    );

    let reps = (2_000_000 / visited.len().max(1)).max(3);
    let n = (visited.len() * reps) as f64;
    let mut acts = Vec::new();
    let t = Instant::now();
    let mut sink = 0usize;
    for _ in 0..reps {
        for s in &visited {
            acts.clear();
            DruidSplit::generate_actions(s, &mut acts);
            sink += acts.len();
        }
    }
    let gen = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..reps {
        for s in &visited {
            sink += usize::from(DruidSplit::is_terminal(s));
        }
    }
    let term = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..reps {
        for s in &visited {
            acts.clear();
            DruidSplit::generate_actions(s, &mut acts);
            let next = DruidSplit::apply(s.clone(), &acts[0]);
            sink += usize::from(next.state().pending != Pending::None);
        }
    }
    let app = t.elapsed().as_secs_f64();
    println!(
        "throughput over {} visited positions: gen_actions {:.2}M/s  is_terminal {:.2}M/s  gen+clone+apply {:.2}M/s  (sink {})",
        visited.len(),
        n / gen / 1e6,
        n / term / 1e6,
        n / app / 1e6,
        sink
    );
}
