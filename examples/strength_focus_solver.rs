// Background strength comparison: MCTS-Solver (Standard update rule,
// Nijssen & Winands, CG 2010) vs no solver, on 3- and 4-player Focus --
// the paper's own benchmark game. Sequential execution so each
// single-threaded search gets the whole machine, same rationale as this
// repo's other strength_* scripts.
//
// Unlike `strength_solver.rs`/`strength_pn_mcts.rs`, this can't reuse
// `mcts_bench::tournament::round_robin`/`round_robin_multiple` -- both are
// hardwired to exactly two seats (`let mut strat = [si, sj]`). Instead this
// reproduces the paper's own Table 4 methodology directly: one seat runs
// the solver, every other seat runs the same strategy without it, and
// which seat holds the solver rotates across games to cancel positional
// bias. A win is scored for "solver" or "no solver" depending on which side
// the winning seat was playing that game, folded into the same
// `mcts_bench::tournament::Result`/Wilson-CI machinery the other strength_*
// scripts use.
//
// This is intentionally a long-running job -- tens of minutes to hours
// depending on the round counts below. Run as a background process, not
// synchronously (see `strength_solver.rs`'s doc comment for why a
// synchronous attempt at a real budget is useless for CI). Tune
// `ROUNDS_PER_SEAT`/`MOVE_BUDGET` for the machine/time available.
//
// Usage: cargo run --release --example strength_focus_solver
use std::time::Duration;

use game_focus::{Focus, State};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const MOVE_BUDGET: Duration = Duration::from_millis(200);
const ROUNDS_PER_SEAT: usize = 10;

fn config<const P: usize>(use_solver: bool, seed: u64) -> TreeSearch<Focus<P>, strategy::Ucb1> {
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

/// Plays one game of `Focus<P>` with `solver_seat` running the solver and
/// every other seat running the identical strategy without it. Returns the
/// winning seat, or `None` for a draw.
fn play_one_game<const P: usize>(solver_seat: usize, seed: u64) -> Option<usize> {
    let mut strategies: Vec<AnySearch<Focus<P>>> = (0..P)
        .map(|seat| AnySearch::new(config::<P>(seat == solver_seat, seed * 100 + seat as u64)))
        .collect();

    let mut state = State::<P>::default();
    loop {
        if Focus::<P>::is_terminal(&state) {
            return Focus::<P>::winner(&state).map(|w| w.to_index());
        }
        let mover = Focus::<P>::player_to_move(&state).to_index();
        let action = strategies[mover].choose_action(&state);
        state = Focus::<P>::apply(state, &action);
    }
}

/// Runs `ROUNDS_PER_SEAT` games with the solver in each of `P` seats (so
/// `P * ROUNDS_PER_SEAT` games total), rotating which seat holds it, and
/// folds every game's outcome into a single solver-vs-no-solver `Result`.
fn solver_vs_no_solver<const P: usize>() -> GameResult {
    let mut result = GameResult::default();
    for round in 0..ROUNDS_PER_SEAT {
        for solver_seat in 0..P {
            let seed = (round * P + solver_seat) as u64;
            match play_one_game::<P>(solver_seat, seed) {
                Some(winner) if winner == solver_seat => result.wins += 1,
                Some(_) => result.losses += 1,
                None => result.draws += 1,
            }
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
    println!("=== Focus MCTS-Solver strength comparison (background job) ===");
    println!(
        "Move budget: {:?}, {} rounds per seat -- see this file's doc comment to tune.",
        MOVE_BUDGET, ROUNDS_PER_SEAT
    );
    println!();

    println!("--- 3-player Focus ({} games) ---", ROUNDS_PER_SEAT * 3);
    let result_3p = solver_vs_no_solver::<3>();
    println!("[3p] {}", fmt_result(&result_3p));
    println!();

    println!("--- 4-player Focus ({} games) ---", ROUNDS_PER_SEAT * 4);
    let result_4p = solver_vs_no_solver::<4>();
    println!("[4p] {}", fmt_result(&result_4p));
}
