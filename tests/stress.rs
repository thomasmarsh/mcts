//! Stress tests: correct but slow (multi-second, real-time-budgeted, or
//! many-games) checks that don't belong in `cargo test --lib`'s fast path.
//! Living in `tests/` (a separate integration-test binary) keeps them out of
//! that command automatically -- `cargo test --lib` never compiles or runs
//! this file. Run explicitly with `cargo test --test stress`.
//!
//! Each test here should still run alone comfortably; the guard below only
//! protects against *this binary's own* tests overlapping under cargo's
//! default per-binary test concurrency, the same problem
//! `crate::strategies::parallel_test_guard` solves for the unit-test binary.

use std::sync::{Mutex, OnceLock};

fn stress_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[test]
fn test_tree_parallel_transpositions_survive_many_real_time_games() {
    let _guard = stress_test_guard();
    // Regression guard for a race between `Node::is_terminal()` and
    // `Node::is_leaf()` in `select_step` (search.rs): those used to be
    // two separate `OnceLock::get()` reads with a decision gap between
    // them. Under transpositions, a *different* thread can resolve the
    // very same node (reached via a different move order) from Leaf to
    // Terminal in that gap: `is_terminal()` (checked first) sees the
    // still-unresolved leaf and returns `false`, then `is_leaf()`
    // (checked moments later) sees the now-resolved node and *also*
    // returns `false` -- falling through both branches into
    // `best_child()`/`Node::edges()` on a node that's actually
    // Terminal, tripping `edges()`'s `unreachable!()`. Fixed by
    // `Node::status()`, a single snapshot both decisions are now
    // derived from.
    //
    // This didn't show up in the fast `cargo test --lib` tree-parallel
    // test because that one budgets by *iteration count*: a few thousand
    // iterations split across a handful of threads on trivially-cheap
    // TicTacToe finishes in microseconds of real wall-clock time,
    // sampling very few actual thread interleavings. Budgeting by *time*
    // instead forces every thread to keep racing for the same real
    // duration regardless of how fast an iteration is, sampling far more
    // interleavings per test-second -- which is what actually caught
    // this originally (on Druid, under a real multi-hundred-ms budget).
    // Playing many full games (not just one `choose_action` call) adds
    // further exposure across many distinct board positions. That
    // combination is exactly why this test takes several real seconds
    // and belongs here rather than in the unit-test suite.
    use mcts::game::Game;
    use mcts::games::ttt::*;
    use mcts::strategies::Search;
    type G = TicTacToe;

    type TS = mcts::strategies::mcts::TreeSearch<G, mcts::strategies::mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::strategies::mcts::SearchConfig::default()
            .max_time(std::time::Duration::from_millis(30))
            .use_transpositions(true)
            .num_tree_threads(4),
    );

    for _ in 0..20 {
        let mut state = HashedPosition::new();
        while !G::is_terminal(&state) {
            let action = ts.choose_action(&state);
            state = G::apply(state, &action);
        }
    }
}

#[test]
fn test_druid_hash_no_collision_across_many_random_games() {
    let _guard = stress_test_guard();
    // `Druid::zobrist_hash` used to cover only board cells + the pending
    // sub-move, on the assumption that player-to-move and hand counts are
    // always recoverable from the board. That's false once lintels are in
    // play -- a lintel placement can raise a cell's height by more than 1,
    // decoupling turn-count from board appearance -- so two different,
    // both-legally-reachable states could hash identically and silently
    // alias in the MCTS transposition table. The fix extended the hash to
    // cover player-to-move and both hands' remaining counts; this plays many
    // random games and asserts no two distinct states ever share a hash,
    // the same technique `examples/test_hash_collision.rs` used to find the
    // original collision (in 2,485 games -- this runs comfortably past that
    // margin). That example is kept separately for bigger ad-hoc runs (its
    // default is 200,000 games), since a run that size is too slow for even
    // this suite.
    use mcts::game::Game;
    use mcts::games::druid::{Druid, HashedState, Size, State};
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::HashMap;

    let size = Size { w: 5, h: 5 };
    let mut rng = SmallRng::seed_from_u64(1);
    let mut seen: HashMap<u64, State> = HashMap::new();

    for _game in 0..5_000u64 {
        let mut state = HashedState::new(size);
        for _ply in 0..400 {
            if Druid::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Druid::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }

            let hash = Druid::zobrist_hash(&state);
            if let Some(prev) = seen.insert(hash, state.state().clone()) {
                assert_eq!(
                    prev,
                    *state.state(),
                    "hash collision: two distinct states shared one Zobrist hash"
                );
            }

            let action = *actions.choose(&mut rng).unwrap();
            state = Druid::apply(state, &action);
        }
    }
}
