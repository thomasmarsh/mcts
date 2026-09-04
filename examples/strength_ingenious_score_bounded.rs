// Background strength comparison: Score-Bounded MCTS (Cazenave & Saffidine,
// "Score Bounded Monte-Carlo Tree Search", CG 2010) vs a plain MCTS-Solver
// baseline, on 2-player Ingenious.
//
// Ingenious is a race to completion -- the board always fills, so
// `derive_proven` rarely propagates a real win/loss toward the root
// during play, and the solver (with or without the score-bound machinery
// on top of it) is expected to buy little here. This script exercises the
// score-bound path on a second game and puts the null on record.
//
// The graded terminal score is Ingenious's sorted-lex score comparison
// collapsed to a single Max-relative scalar (`Game::terminal_score`,
// `State::lex_key`). Both seats run `use_mcts_solver(true)` with identical
// UCB1 exploration and time budget; the score-bounded seat additionally
// runs `select::ScoreBoundedUct`. Which seat is score-bounded rotates game
// to game to cancel positional bias.
//
// Long-running background job -- run detached, not synchronously.
//
// Usage: cargo run --release --example strength_ingenious_score_bounded
use std::time::Duration;

use game_ingenious::{Ingenious, State};
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const MOVE_BUDGET: Duration = Duration::from_millis(200);
const ROUNDS: usize = 30;
const MAX_PLIES: usize = 2000;

fn plain_seat(seed: u64) -> TreeSearch<Ingenious<2>, profile::Mcts> {
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
) -> TreeSearch<Ingenious<2>, profile::Mcts<select::ScoreBoundedUct, simulate::Uniform>> {
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

fn play_one_game(sb_seat: usize, seed: u64) -> (Option<usize>, usize, bool) {
    let mut strategies: Vec<AnySearch<Ingenious<2>>> = (0..2)
        .map(|seat| {
            if seat == sb_seat {
                AnySearch::new(score_bounded_seat(seed * 100 + seat as u64))
            } else {
                AnySearch::new(plain_seat(seed * 100 + seat as u64))
            }
        })
        .collect();

    let mut state = State::<2>::new(seed);
    for ply in 0..MAX_PLIES {
        if Ingenious::<2>::is_terminal(&state) {
            return (
                Ingenious::<2>::winner(&state).map(|w| w.to_index()),
                ply,
                false,
            );
        }
        let mover = Ingenious::<2>::player_to_move(&state).to_index();
        let action = strategies[mover].choose_action(&state);
        state = Ingenious::<2>::apply(state, &action);
    }
    (None, MAX_PLIES, true)
}

fn fmt_result(r: &GameResult) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "score-bounded W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson), null=50.0%",
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
    println!("=== Ingenious Score-Bounded MCTS strength comparison (background job) ===");
    println!(
        "Move budget: {:?}, {} rounds x 2 seats = {} games.",
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
                "  round {round} sb_seat {sb_seat}: {tag} in {plies} plies{}",
                if capped { " (PLY CAP HIT)" } else { "" }
            );
        }
    }
    println!();
    println!("{}", fmt_result(&result));
}
