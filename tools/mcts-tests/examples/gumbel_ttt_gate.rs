//! Phase-0 generation gate for the Gumbel AlphaZero loop on tic-tac-toe.
//!
//!     cargo run --release -p mcts-tests --example gumbel_ttt_gate -- \
//!         <baseline_weights.bin> <candidate_weights.bin> [games] [sims]
//!
//! Two checks, both must pass (non-zero exit otherwise):
//!
//!  1. **Head to head.** `games` alternating-colour games, candidate vs
//!     baseline, both playing the Gumbel self-play search. The candidate's
//!     score share (win 1, draw 0.5) must have a Wilson 95% lower bound
//!     strictly above 0.5.
//!  2. **Perfect-play agreement.** Over every distinct non-terminal
//!     position up to ply 4 -- the opening/midgame, where a small Gumbel
//!     budget cannot just brute-force to terminal and the value head
//!     actually steers -- the fraction of moves that preserve the exact
//!     game value (a full-depth negamax of trivial tic-tac-toe), averaged
//!     over several search seeds, must be strictly higher for the candidate
//!     than for the baseline.

use std::collections::HashSet;
use std::process::ExitCode;

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts::util::battle_royale;

use game_ttt::selfplay::GumbelPlayer;
use game_ttt::valuenet::LinearValueNet;
use game_ttt::{HashedPosition, TicTacToe};

/// Exact value of `state` for the player to move: `+1` win, `0` draw, `-1`
/// loss, full-depth. Tic-tac-toe's tree is small enough to solve outright
/// with no memoisation.
fn exact_value(state: &HashedPosition) -> i32 {
    if TicTacToe::is_terminal(state) {
        return if TicTacToe::winner(state).is_some() { -1 } else { 0 };
    }
    let mut actions = Vec::new();
    TicTacToe::generate_actions(state, &mut actions);
    actions
        .iter()
        .map(|a| -exact_value(&TicTacToe::apply(*state, a)))
        .max()
        .unwrap()
}

/// Distinct non-terminal positions up to `max_ply` pieces on the board,
/// breadth-first from the opening.
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

fn wilson_lower_bound(successes: f64, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = n as f64;
    let phat = successes / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    (center - margin) / denom
}

fn agreement_rate(net: &LinearValueNet, cfg: GumbelConfig, positions: &[HashedPosition]) -> f64 {
    let seeds = [0xA2_5EEDu64, 0xB0_1234, 0xC0_FFEE];
    let mut optimal = 0usize;
    let mut total = 0usize;
    for &seed in &seeds {
        let mut player = GumbelPlayer::new(net.clone(), cfg, seed);
        for state in positions {
            let before = exact_value(state);
            let mut actions = Vec::new();
            TicTacToe::generate_actions(state, &mut actions);
            // Only score decisive positions: at least one legal move throws
            // away value. Positions where every move is equally optimal
            // carry no signal about evaluator quality.
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
        return 0.0;
    }
    optimal as f64 / total as f64
}

fn load(path: &str) -> LinearValueNet {
    LinearValueNet::load(path).unwrap_or_else(|e| panic!("cannot load {path}: {e}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: gumbel_ttt_gate <baseline_weights.bin> <candidate_weights.bin> [games] [sims]"
        );
        return ExitCode::FAILURE;
    }
    let baseline = load(&args[1]);
    let candidate = load(&args[2]);
    let games: usize = args.get(3).map_or(200, |s| s.parse().expect("games"));
    let sims: u32 = args.get(4).map_or(32, |s| s.parse().expect("sims"));
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };

    // --- 1. head to head -------------------------------------------------
    let mut a = GumbelPlayer::new(baseline.clone(), cfg, 1);
    let mut b = GumbelPlayer::new(candidate.clone(), cfg, 2);
    let mut candidate_score = 0.0f64;
    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;
    for i in 0..games {
        // `battle_royale`'s first argument always plays X; alternate it.
        let (result, candidate_first) = if i % 2 == 0 {
            (battle_royale::<TicTacToe, _, _>(&mut b, &mut a), true)
        } else {
            (battle_royale::<TicTacToe, _, _>(&mut a, &mut b), false)
        };
        let candidate_won = match result {
            None => None,
            Some(0) => Some(candidate_first),
            Some(1) => Some(!candidate_first),
            _ => unreachable!(),
        };
        match candidate_won {
            None => {
                candidate_score += 0.5;
                draws += 1;
            }
            Some(true) => {
                candidate_score += 1.0;
                wins += 1;
            }
            Some(false) => losses += 1,
        }
    }
    let lb = wilson_lower_bound(candidate_score, games, 1.96);
    let h2h_pass = lb > 0.5;
    println!(
        "head to head: candidate {wins}-{draws}-{losses} (W-D-L) over {games}, \
         score share {:.3}, Wilson LB {:.3}  -> {}",
        candidate_score / games as f64,
        lb,
        if h2h_pass { "PASS" } else { "FAIL" }
    );

    // --- 2. perfect-play agreement ------------------------------------
    let positions = opening_positions(4);
    let base_rate = agreement_rate(&baseline, cfg, &positions);
    let cand_rate = agreement_rate(&candidate, cfg, &positions);
    let agree_pass = cand_rate > base_rate;
    println!(
        "perfect-play agreement over {} positions: baseline {:.3}, candidate {:.3}  -> {}",
        positions.len(),
        base_rate,
        cand_rate,
        if agree_pass { "PASS" } else { "FAIL" }
    );

    if h2h_pass && agree_pass {
        println!("GATE: PASS");
        ExitCode::SUCCESS
    } else {
        println!("GATE: FAIL");
        ExitCode::FAILURE
    }
}
