//! Stress checks: correct but slow checks that don't belong in `cargo test
//! --lib`'s fast path.
//!
//! This is an `examples/` binary, not a test target, on purpose: a
//! `tests/stress.rs` integration binary is silently pulled in by a bare
//! `cargo test -p mcts-tune`, so a slow suite ends up running when someone
//! only meant the fast tests. An example only runs when named:
//!
//!   cargo run --release --example tune-stress -p mcts-tune
//!
//! Exit status is non-zero if any case fails.

use game_nim::Nim;
use mcts::game::Game;
use mcts::strategies::mcts::{strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use serde_json::json;

/// Bounded the same way `mcts-tune`'s own unit-test `baseline()` is:
/// `TreeSearch::default()`'s `max_iterations` is `usize::MAX`.
fn baseline() -> Box<dyn Search<G = Nim>> {
    Box::new(
        TreeSearch::<Nim, strategy::Ucb1>::new().config(SearchConfig::new().max_iterations(50)),
    )
}

/// `meta_mcts`'s inner nested search (`MetaMcts::select_move` runs a full
/// `TreeSearch::choose_action` on every outer simulate step, not just once
/// per leaf) makes even a single candidate-vs-baseline `Nim` game noticeably
/// slower than every other catalog family -- several real seconds, versus
/// the sub-second every other family's round-trip test in `src/lib.rs` runs
/// in. Correct (proven here), just not fast enough for the unit-test suite.
fn main() {
    family_meta_mcts_round_trips();
    eprintln!("mcts-tune stress: all cases passed");
}

fn family_meta_mcts_round_trips() {
    let params = json!({
        "family": "meta_mcts", "c": 1.4, "q_init": "Infinity",
    });
    let outcome = mcts_tune::strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        mcts_tune::SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        |_| json!({"canonical": "nim"}),
        |_, action| Some(serde_json::to_value(action).expect("Nim action always serializes")),
        None,
        None,
        &mut |_| Ok(()),
    )
    .expect("meta_mcts should round-trip with a minimal config");
    assert_eq!(outcome.wins + outcome.losses + outcome.draws, 2);
}
