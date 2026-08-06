pub mod flat_mc;
pub mod human;
pub mod mcts;
pub mod random;

use crate::game::Game;

pub trait Search: Sync + Send {
    type G: Game;

    fn friendly_name(&self) -> String;

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A;

    fn principle_variation(&self) -> Vec<<Self::G as Game>::A> {
        vec![]
    }

    fn estimated_depth(&self) -> usize {
        0
    }

    fn set_friendly_name(&mut self, name: &str);

    #[allow(unused_variables)]
    fn make_book_entry(
        &mut self,
        state: &<Self::G as Game>::S,
    ) -> (Vec<<Self::G as Game>::A>, Vec<f64>) {
        unimplemented!();
    }
}

#[cfg(test)]
static PARALLEL_TEST_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

#[cfg(test)]
fn parallel_test_guard() -> std::sync::MutexGuard<'static, ()> {
    PARALLEL_TEST_MUTEX
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::parallel_test_guard;
    use super::*;
    use crate::game::PlayerIndex;

    #[test]
    fn test_expand0() {
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        for n in 0..3 {
            type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
            let mut ts = TS::default().config(
                mcts::SearchConfig::default()
                    .expand_threshold(n)
                    // NOTE: best_child will fail on final_action
                    // selection when we haven't expanded root.
                    .max_iterations(1 + n as usize),
            );

            ts.choose_action(&init_state);
            println!(
                "{n} [{}]: {:?}",
                ts.principle_variation().len(),
                ts.principle_variation()
            );
            if n == 0 {
                assert!(ts.principle_variation().len() > 1);
            } else {
                assert!(ts.principle_variation().len() == 1);
            }
        }
    }

    #[test]
    fn test_root_parallel_picks_a_legal_action() {
        let _guard = parallel_test_guard();
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .num_threads(4),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);

        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_num_threads_one_is_deterministic_given_a_seed() {
        // `num_threads == 1` should take the untouched single-tree path
        // (search.rs's `choose_action_root_parallel` dispatch only fires
        // above 1), so this is a baseline regression guard that the new
        // config field doesn't perturb existing single-threaded behavior.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;

        let mut a = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .seed(42)
                .num_threads(1),
        );
        let mut b = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .seed(42)
                .num_threads(1),
        );

        assert_eq!(a.choose_action(&init_state), b.choose_action(&init_state));
    }

    #[test]
    fn test_leaf_parallel_picks_a_legal_action() {
        let _guard = parallel_test_guard();
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .num_rollouts_per_leaf(4),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);

        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_num_rollouts_per_leaf_one_is_deterministic_given_a_seed() {
        // `num_rollouts_per_leaf == 1` (the default, or set explicitly)
        // should take the untouched single-simulate-per-leaf path
        // (search.rs's `choose_action` loop only branches into
        // `simulate_many` above 1), so this is a baseline regression guard
        // that the new config field doesn't perturb existing
        // single-rollout behavior.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;

        let mut a =
            TS::default().config(mcts::SearchConfig::default().max_iterations(200).seed(42));
        let mut b = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .seed(42)
                .num_rollouts_per_leaf(1),
        );

        assert_eq!(a.choose_action(&init_state), b.choose_action(&init_state));
    }

    #[test]
    fn test_leaf_parallel_virtual_loss_balances_across_many_iterations() {
        let _guard = parallel_test_guard();
        // Regression guard for the virtual-loss accounting leaf parallelism
        // layers on top of `select`'s single unit per edge: if the extra
        // `k - 1` units added per leaf aren't removed in lock-step across
        // all `k` backprop calls, `NodeStats::remove_virtual_loss`'s
        // `debug_assert!(prev >= 1, ...)` fires on underflow. A low
        // `expand_threshold` makes `select` descend multiple levels per
        // iteration (exercising every edge on longer stacks, not just the
        // root's), and plenty of iterations at K=3 exercises this a lot.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(500)
                .expand_threshold(0)
                .num_rollouts_per_leaf(3),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);
        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_tree_parallel_picks_a_legal_action() {
        let _guard = parallel_test_guard();
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .num_tree_threads(4),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);

        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_tree_parallel_with_grave_picks_a_legal_action() {
        let _guard = parallel_test_guard();
        // `Rave`'s GRAVE backprop flag routes through `TreeStats::grave`
        // (read in `select_step`, written in `backprop_step`'s
        // `update_grave`) -- exercise that lock-protected path specifically
        // under concurrency, not just the plain-Ucb1 tests above.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::RaveMastDm>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .use_transpositions(true)
                .num_tree_threads(4),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);

        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_num_tree_threads_one_is_deterministic_given_a_seed() {
        // `num_tree_threads == 1` should take the untouched single-tree path
        // (search.rs's `choose_action_tree_parallel` dispatch only fires
        // above 1), so this is a baseline regression guard that the new
        // config field doesn't perturb existing single-threaded behavior.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;

        let mut a = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .seed(42)
                .num_tree_threads(1),
        );
        let mut b = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .seed(42)
                .num_tree_threads(1),
        );

        assert_eq!(a.choose_action(&init_state), b.choose_action(&init_state));
    }

    #[test]
    fn test_tree_parallel_stress_many_threads_small_tree_high_iterations() {
        let _guard = parallel_test_guard();
        // Concurrent stress test for the shared-arena/shared-stats path:
        // many worker threads, a tiny game (small tree, lots of edge/node
        // creation races per node), `expand_threshold(0)` so `select`
        // descends multiple levels per iteration (exercising virtual loss
        // on longer stacks, not just the root edge), and enough iterations
        // to make edge-creation races (`Edge::get_or_create_child`) and
        // transposition races (`TranspositionTable::get_or_insert`) likely
        // to actually fire rather than just compile. A broken race would
        // either panic (`NodeStats::remove_virtual_loss`'s
        // `debug_assert!(prev >= 1, ..)` on an accounting mismatch, or a
        // duplicate-child bug tripping other debug_asserts in `backprop`)
        // or hang (a lock-ordering cycle) -- this test passing is itself
        // the signal. `root_stats.num_visits()` should also exactly equal
        // the number of completed iterations, since every iteration's
        // `select` path passes through (and `backprop` updates) the root
        // exactly once, regardless of thread/rollout count.
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(2000)
                .expand_threshold(0)
                .use_transpositions(true)
                .num_tree_threads(4),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);
        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));

        assert_eq!(
            ts.root_stats.num_visits() as usize,
            ts.stats.iter_count.load(std::sync::atomic::Ordering::Relaxed)
        );
        assert_eq!(ts.root_stats.num_visits(), 2000);
    }

    #[test]
    fn test_hybrid_root_and_tree_parallel_picks_a_legal_action() {
        let _guard = parallel_test_guard();
        // `num_threads > 1` and `num_tree_threads > 1` together should
        // compose (a handful of independent trees, each internally
        // tree-parallel) rather than one silently overriding the other --
        // regression guard for the dispatch-order fix in `choose_action`
        // (root parallelism is checked first, and each of its worker trees'
        // recursive `choose_action` call is what picks up `num_tree_threads`
        // for that individual tree).
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(200)
                .num_threads(2)
                .num_tree_threads(2),
        );

        let mut legal = Vec::new();
        G::generate_actions(&init_state, &mut legal);

        let action = ts.choose_action(&init_state);
        assert!(legal.contains(&action));
    }

    #[test]
    fn test_tree_parallel_transpositions_survive_many_real_time_games() {
        let _guard = parallel_test_guard();
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
        // This didn't show up in the original tree-parallel stress test
        // above because that one budgets by *iteration count*: 2000
        // iterations split across 8 threads on trivially-cheap TicTacToe
        // finishes in microseconds of real wall-clock time, sampling very
        // few actual thread interleavings. Budgeting by *time* instead
        // forces every thread to keep racing for the same real duration
        // regardless of how fast an iteration is, sampling far more
        // interleavings per test-second -- which is what actually caught
        // this originally (on Druid, under a real multi-hundred-ms budget).
        // Playing many full games (not just one `choose_action` call) adds
        // further exposure across many distinct board positions.
        use crate::games::ttt::*;
        type G = TicTacToe;

        type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
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
    fn test_basics() {
        use crate::games::ttt::*;
        type G = TicTacToe;

        // Initial State
        // X O X
        // . O O
        // . X X
        // Turn: O
        //
        // for Move(3), score += 1
        // for Move(6), score += 0
        let init_state = HashedPosition {
            position: Position {
                turn: Piece::O,
                board: [
                    // ..X
                    // .OO
                    // XOX
                    (0, Piece::X),
                    (1, Piece::O),
                    (2, Piece::X),
                    (4, Piece::O),
                    (5, Piece::O),
                    (8, Piece::X),
                ]
                .iter()
                .fold(0, |board, (i, piece)| {
                    let value = match piece {
                        Piece::X => 0b01,
                        Piece::O => 0b10,
                    };
                    board | (value << (i << 1))
                }),
            },
            hashes: [0; 8],
        };

        // Configure new MCTS
        type TS = mcts::TreeSearch<G, mcts::strategy::Amaf>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(100),
        );

        // Construct new root
        let root_id = ts.reset(0, 0);
        // Helper step function
        let step = |ts: &mut TS| {
            ts.reset_iter();
            let mut ctx = mcts::SearchContext::new(root_id, init_state);
            ts.select(&mut ctx);
            let trial = ts.simulate(&ctx.state);
            println!("trial actions: {:?}", trial.actions);
            println!("trial status: {:?}", trial.status);
            println!("utilites: {:?}", G::compute_utilities(&trial.state));
            println!(
                "relevant utility: {:?}",
                G::compute_utilities(&trial.state)[G::player_to_move(&init_state).to_index()]
            );
            ts.trial = Some(trial);
            ts.backprop();

            ctx.current_id
        };

        // First pass: simulate over root node
        let child_id = step(&mut ts);

        assert_eq!(child_id, root_id);
        assert_eq!(ts.root_stats.num_visits(), 1);

        // Second pass: expand child node
        let child_id = step(&mut ts);

        assert_ne!(child_id, root_id);
        assert_eq!(ts.root_stats.num_visits(), 2);

        // Third pass: expand child node
        let _child_id = step(&mut ts);

        println!("{:#?}", ts.index);
    }
}
