// Background strength comparison: Score-Bounded MCTS (Cazenave & Saffidine,
// "Score Bounded Monte-Carlo Tree Search", CG 2010) vs a plain MCTS-Solver
// baseline, on 2-player Focus with the shortened capture-quota victory rule.
//
// Both seats run `use_mcts_solver(true)` with identical UCB1 exploration and
// time budget. The only difference: the score-bounded seat also runs
// `select::ScoreBoundedUct`, which prunes alpha-beta-style on each node's
// graded-score interval (`Game::score_bounds`/`terminal_score`, here player
// 0's net capture margin) and biases selection toward provably-better
// bounds. The plain seat runs `select::Ucb1`. Which seat is score-bounded
// rotates game to game to cancel positional bias, folded into the same
// `mcts_bench::tournament::Result`/Wilson-CI machinery the other strength_*
// scripts use.
//
// The score-bounded machinery is expected to pay off exactly where plain
// binary proving throws information away -- a game with a graded terminal
// whose margin matters and whose terminals are reachable within a search
// horizon (which the shortened Focus rule makes true). A null here would say
// the bound propagation, though correct, does not translate to strength in
// this engine.
//
// Long-running background job -- run detached, not synchronously. Tune
// `ROUNDS`/`MOVE_BUDGET` for the machine/time available.
//
// Usage: cargo run --release --example strength_focus_score_bounded
use std::time::Duration;

use game_focus::{Focus, State};
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const MOVE_BUDGET: Duration = Duration::from_millis(200);
const ROUNDS: usize = 30;
const MAX_PLIES: usize = 600;

fn plain_seat(seed: u64) -> TreeSearch<Focus<2>, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .expand_threshold(1)
            .use_mcts_solver(true)
            .q_init(QInit::Loss)
            .max_time(MOVE_BUDGET)
            .seed(seed)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn score_bounded_seat(
    seed: u64,
) -> TreeSearch<Focus<2>, profile::Mcts<select::ScoreBoundedUct, simulate::Uniform>> {
    TreeSearch::new().config(
        SearchConfig::new()
            .expand_threshold(1)
            .use_mcts_solver(true)
            .q_init(QInit::Loss)
            .max_time(MOVE_BUDGET)
            .seed(seed)
            .select(select::ScoreBoundedUct::with_c(1.414, 0.1, 0.1)),
    )
}

/// Plays one game with the score-bounded strategy in `sb_seat` and the
/// plain solver in the other seat. Returns the winning seat (or `None`),
/// the ply count, and whether the ply cap was hit.
fn play_one_game(sb_seat: usize, seed: u64) -> (Option<usize>, usize, bool) {
    let mut strategies: Vec<AnySearch<Focus<2>>> = (0..2)
        .map(|seat| {
            if seat == sb_seat {
                AnySearch::new(score_bounded_seat(seed * 100 + seat as u64))
            } else {
                AnySearch::new(plain_seat(seed * 100 + seat as u64))
            }
        })
        .collect();

    let mut state = State::<2>::default();
    for ply in 0..MAX_PLIES {
        if Focus::<2>::is_terminal(&state) {
            return (Focus::<2>::winner(&state).map(|w| w.to_index()), ply, false);
        }
        let mover = Focus::<2>::player_to_move(&state).to_index();
        let action = strategies[mover].choose_action(&state);
        state = Focus::<2>::apply(state, &action);
    }
    (None, MAX_PLIES, true)
}

fn fmt_result(r: &GameResult) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "score-bounded W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson)",
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
    println!("=== Focus Score-Bounded MCTS strength comparison (background job) ===");
    println!(
        "Move budget: {:?}, {} rounds x 2 seats = {} games -- see this file's doc comment to tune.",
        MOVE_BUDGET,
        ROUNDS,
        ROUNDS * 2
    );
    println!();

    let mut result = GameResult::default();
    for round in 0..ROUNDS {
        for sb_seat in 0..2 {
            let seed = (round * 2 + sb_seat) as u64;
            let (winner, plies, capped) = play_one_game(sb_seat, seed);
            let tag = match winner {
                Some(w) if w == sb_seat => {
                    result.wins += 1;
                    "score-bounded"
                }
                Some(_) => {
                    result.losses += 1;
                    "plain"
                }
                None => {
                    result.draws += 1;
                    "draw"
                }
            };
            println!(
                "  round {} sb_seat {}: {} in {} plies{}",
                round,
                sb_seat,
                tag,
                plies,
                if capped { " (PLY CAP HIT)" } else { "" }
            );
        }
    }
    println!();
    println!("{}", fmt_result(&result));
}
