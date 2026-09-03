// Background strength comparison: MCTS-Solver (Standard update rule,
// Nijssen & Winands, CG 2010) vs no solver, on 3-player Ingenious -- the
// race-to-completion counterpart to `strength_focus_solver.rs`'s
// sudden-death game. Nijssen & Winands predict the multi-player solver
// pays off in sudden-death games (Focus) but not race-to-completion ones
// (Chinese Checkers, and by structural analogy Ingenious, whose board is
// monotonic and whose games always run to a full rack-exhaustion / board
// -fill terminal). This script is the direct test of that prediction on a
// game this repo actually has a UI/preset for.
//
// Methodology mirrors `strength_focus_solver.rs` exactly: one seat runs
// the solver, every other seat runs the identical strategy without it, and
// which seat holds the solver rotates across games to cancel positional
// bias. A win is scored for "solver" or "no solver" depending on which
// side the winning seat played, folded into the same
// `mcts_bench::tournament::Result` / Wilson-CI machinery.
//
// Like the Focus script, this searches the literal `State` (so every seat
// sees every rack -- "cheating"). That is deliberate: the question here is
// whether the *solver* changes strength at a fixed information level, not
// what hiding the racks costs (that is `strength_ingenious_pimc.rs`).
//
// Intentionally long-running -- run as a background process, not
// synchronously. Tune `ROUNDS_PER_SEAT` / `MOVE_BUDGET` for the machine.
//
// Usage: cargo run --release --example strength_ingenious_solver
use std::time::Duration;

use game_ingenious::{Ingenious, State};
use mcts::game::{Game, PlayerIndex};
use mcts::algorithms::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const MOVE_BUDGET: Duration = Duration::from_millis(200);
const ROUNDS_PER_SEAT: usize = 10;
/// Safety cap. Ingenious is naturally bounded (board fills / racks
/// exhaust), so this only guards against an unforeseen non-terminating
/// line; a game hitting it is scored a draw and flagged.
const MAX_PLIES: usize = 2000;

fn config<const P: usize>(use_solver: bool, seed: u64) -> TreeSearch<Ingenious<P>, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .expand_threshold(1)
            .use_mcts_solver(use_solver)
            .q_init(QInit::Loss)
            .max_time(MOVE_BUDGET)
            .seed(seed)
            .select(select::Ucb1::with_c(1.414)),
    )
}

/// Plays one game of `Ingenious<P>` with `solver_seat` running the solver
/// and every other seat running the identical strategy without it. Returns
/// the winning seat, or `None` for a draw.
fn play_one_game<const P: usize>(solver_seat: usize, seed: u64) -> (Option<usize>, usize, bool) {
    let mut strategies: Vec<AnySearch<Ingenious<P>>> = (0..P)
        .map(|seat| AnySearch::new(config::<P>(seat == solver_seat, seed * 100 + seat as u64)))
        .collect();

    let mut state = State::<P>::new(seed);
    for ply in 0..MAX_PLIES {
        if Ingenious::<P>::is_terminal(&state) {
            return (
                Ingenious::<P>::winner(&state).map(|w| w.to_index()),
                ply,
                false,
            );
        }
        let mover = Ingenious::<P>::player_to_move(&state).to_index();
        let action = strategies[mover].choose_action(&state);
        state = Ingenious::<P>::apply(state, &action);
    }
    (None, MAX_PLIES, true)
}

/// Runs `ROUNDS_PER_SEAT` games with the solver in each of `P` seats,
/// rotating which seat holds it, and folds every game's outcome into a
/// single solver-vs-no-solver `Result`.
fn solver_vs_no_solver<const P: usize>() -> GameResult {
    let mut result = GameResult::default();
    for round in 0..ROUNDS_PER_SEAT {
        for solver_seat in 0..P {
            let seed = (round * P + solver_seat) as u64;
            let (winner, plies, capped) = play_one_game::<P>(solver_seat, seed);
            let tag = match winner {
                Some(w) if w == solver_seat => {
                    result.wins += 1;
                    "solver"
                }
                Some(_) => {
                    result.losses += 1;
                    "no-solver"
                }
                None => {
                    result.draws += 1;
                    "draw"
                }
            };
            println!(
                "  [{}p] round {} solver_seat {}: {} in {} plies{}",
                P,
                round,
                solver_seat,
                tag,
                plies,
                if capped { " (PLY CAP HIT)" } else { "" }
            );
        }
    }
    result
}

fn fmt_result(r: &GameResult) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "solver W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson)",
        r.wins,
        r.losses,
        r.draws,
        r.total(),
        point * 100.0,
        lo * 100.0,
        hi * 100.0
    )
}

fn main() {
    println!("=== Ingenious MCTS-Solver strength comparison (background job) ===");
    println!(
        "Move budget: {:?}, {} rounds per seat -- see this file's doc comment to tune.",
        MOVE_BUDGET, ROUNDS_PER_SEAT
    );
    println!();

    println!("--- 3-player Ingenious ({} games) ---", ROUNDS_PER_SEAT * 3);
    let result_3p = solver_vs_no_solver::<3>();
    println!("[3p] {}", fmt_result(&result_3p));
}
