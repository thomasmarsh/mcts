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
    // that the new config field doesn't perturb existing single-rollout
    // behavior.
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

#[test]
fn test_root_report_flags_the_proven_winning_move() {
    // `Search::root_report` (used by the server's `analyze` endpoint)
    // must surface the same forced win MCTS-Solver
    // finds via `choose_action`, not just the single action it returns:
    // the winning move should come back with the most visits and
    // `is_proven`, and the PV should start with it.
    use crate::games::ttt::*;
    type G = TicTacToe;

    // O to move, exactly one immediate win: row 0 (indices 0,1,2) needs only
    // index 2. Row/column/diagonal lines touching O's existing pieces
    // (indices 0, 1) are otherwise unfinished, so this is a *unique* forced
    // win, not just one of several -- unlike a naively hand-picked board,
    // where a second coincidental line (e.g. a column) can also be one move
    // from completion, making "the chosen move" ambiguous between two
    // equally winning candidates.
    // O O .
    // X . .
    // X . .
    let init_state = HashedPosition {
        position: Position {
            turn: Piece::O,
            board: [
                (0, Piece::O),
                (1, Piece::O),
                (3, Piece::X),
                (6, Piece::X),
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

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .expand_threshold(1)
            .use_mcts_solver(true)
            .max_iterations(200),
    );

    let chosen = ts.choose_action(&init_state);
    assert_eq!(chosen, Move(2), "should find the immediate winning move");

    let report = ts.root_report(&init_state);
    assert!(report.total_visits > 0);
    assert!(!report.actions.is_empty());

    let winning = report
        .actions
        .iter()
        .find(|a| a.action == Move(2))
        .expect("winning move should be an explored root action");
    assert!(winning.is_proven, "winning move should be reported as proven");
    // Not asserting "most visits": MCTS-Solver stops the moment the root is
    // proven (see `choose_action`'s early-break on `Proven::Unproven`), which
    // can fire right after the winning move's *first* visit -- whichever
    // sibling(s) got explored earlier that same run may already have more
    // visits by then. The move `choose_action`/`compute_pv` actually pick is
    // still deterministic (`proven_win_child`'s scan for a proven-win child
    // bypasses visit-based selection entirely), which is what the assertions
    // above and below check instead.
    assert_eq!(
        report.principal_variation.first(),
        Some(&Move(2)),
        "PV should start with the winning move"
    );
}

#[test]
fn test_update_amaf_matches_by_movers_player_not_childs() {
    // `update_amaf` is deciding, for `parent_id`'s sibling actions,
    // whether a `trace` entry `(action, p)` is "the same player replaying
    // the same action later in the simulation". The player who could
    // have played any of `parent_id`'s candidate actions is
    // `parent_id`'s own mover (`index.get(parent_id).player_idx`) -- the
    // *sibling* node's `player_idx` is the mover of the position
    // *after* that action (the opposite player in an alternating game,
    // per the identical class of bug already fixed in `update`'s
    // tree-path push, see the comment there). Comparing against the
    // sibling's `player_idx` instead inverts the check for any
    // alternating 2-player game.
    use crate::games::ttt::*;
    use mcts::backprop::{BackpropStrategy, Classic};
    use mcts::node::{ChildArray, Node, NodeState};

    type G = TicTacToe;

    let index = mcts::search::TreeIndex::<Move>::new();

    // root: O (player 1) to move.
    let root = Node::new_root(1, 2, 0);
    // sibling: reached by playing Move(7) at root -- X (player 0) to
    // move there, the mover *after* root's action.
    let sibling_id = index.insert(Node::new(0, 7));
    // the node actually being processed this call -- a different child
    // of root, irrelevant to the match itself beyond not being root.
    let processed_id = index.insert(Node::new(0, 6));

    let children = ChildArray::new(vec![Move(6), Move(7)], 2);
    children.get_or_create_child(0, || processed_id);
    children.get_or_create_child(1, || sibling_id);
    root.expand(|| NodeState::Expanded(children));
    let root_id = index.insert(root);

    let utilities = [0.25, 0.75];

    // Case 1: O (root's mover, player 1) plays Move(7) later in the
    // simulation -- a genuine AMAF match, should update Move(7)'s edge.
    Classic.update_amaf::<G>(
        Some(root_id),
        &[(Move(7), 1)],
        &index,
        processed_id,
        &utilities,
    );
    let root = index.get(root_id);
    let sibling_idx = root.child_index(sibling_id);
    let sibling_children = root.children();
    assert_eq!(
        sibling_children.amaf(sibling_idx, 0).num_visits,
        1,
        "O replaying Move(7) later should count as an AMAF match"
    );
    assert_eq!(sibling_children.amaf(sibling_idx, 0).score, 0.25);
    assert_eq!(sibling_children.amaf(sibling_idx, 1).score, 0.75);

    // Case 2: X (the *sibling's* mover, not root's) "plays" Move(7)
    // later -- not a valid AMAF match for root's Move(7) option, since X
    // never had the choice to play it from root. Must not update.
    Classic.update_amaf::<G>(
        Some(root_id),
        &[(Move(7), 0)],
        &index,
        processed_id,
        &utilities,
    );
    let root = index.get(root_id);
    let sibling_idx = root.child_index(sibling_id);
    assert_eq!(
        root.children().amaf(sibling_idx, 0).num_visits,
        1,
        "X playing Move(7) is not a valid AMAF match for root's option and must not be counted"
    );
}

// Regression guard for `ChildArray::child_index`'s indexed lookup (an O(n)
// scan before this test was written). Two
// parts: correctness of the id -> idx mapping itself, and the concurrency
// race the indexed version can introduce that a plain scan can't (a thread
// observing a resolved child id before `id_index` has caught up).
#[test]
fn test_child_array_child_index_matches_creation_order() {
    use mcts::node::ChildArray;
    use mcts::search::TreeIndex;
    use mcts::node::Node;

    let index = TreeIndex::<u32>::new();
    let ids: Vec<_> = (0..5).map(|i| index.insert(Node::new(0, i))).collect();

    let children = ChildArray::new(vec![10, 11, 12, 13, 14], 1);
    // Resolve out of creation order to make sure `child_index` isn't
    // secretly relying on id/idx happening to already agree.
    for &idx in &[3usize, 0, 4, 1, 2] {
        let resolved = children.get_or_create_child(idx, || ids[idx]);
        assert_eq!(resolved, ids[idx]);
    }

    for (idx, &id) in ids.iter().enumerate() {
        assert_eq!(
            children.child_index(id),
            idx,
            "child_index should invert get_or_create_child's id -> idx mapping"
        );
        // Re-resolving an already-set slot must return the same id and not
        // disturb the reverse mapping.
        assert_eq!(children.get_or_create_child(idx, || panic!("should not re-create")), id);
        assert_eq!(children.child_index(id), idx);
    }
}

#[test]
fn test_child_array_child_index_survives_concurrent_resolution() {
    use mcts::node::ChildArray;
    use mcts::search::TreeIndex;
    use mcts::node::Node;
    use std::sync::Arc;

    // Regression test for a race introduced (and caught by
    // `test_tree_parallel_transpositions_survive_many_real_time_games` in
    // tests/stress.rs) while adding `ChildArray`'s `id_index`: a naive
    // "check `child_ids`, fall back to `get_or_init`, then update
    // `id_index`" implementation
    // lets one thread observe another thread's freshly-resolved child id
    // *before* that thread's `id_index` insert has run, so `child_index`
    // panics on a lookup miss. A slot only resolves once ever (the
    // underlying `OnceLock`), so the race window only exists for the very
    // first call across all racing threads -- repeat with a fresh
    // `ChildArray` every trial to give many independent shots at hitting it,
    // rather than one shot diluted across many now-uncontended calls.
    for _ in 0..500 {
        let index: Arc<TreeIndex<u32>> = Arc::new(TreeIndex::new());
        let created_id = index.insert(Node::new(0, 0));
        let children = Arc::new(ChildArray::<u32>::new(vec![42], 1));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let children = Arc::clone(&children);
                scope.spawn(move || {
                    let id = children.get_or_create_child(0, || created_id);
                    assert_eq!(children.child_index(id), 0);
                });
            }
        });
    }
}

// Regression guard for the memory-profiling helpers
// (`ChildArray::explored_len`/`heap_bytes_estimate`, `TreeSearch::
// memory_stats`). These have no single "correct" answer to check against on
// a real game (that's what examples/mem_profile.rs is for), but the
// counting/estimation logic itself is a plain deterministic computation that
// should never silently drift -- worth pinning against small,
// hand-verifiable shapes.
#[test]
fn test_child_array_explored_len_and_heap_bytes_estimate() {
    use mcts::node::ChildArray;

    let children = ChildArray::<u32>::new(vec![10, 11, 12, 13], 2);
    assert_eq!(children.explored_len(), 0, "nothing resolved yet");

    children.get_or_create_child(1, crate::strategies::mcts::index::Id::invalid_id);
    children.get_or_create_child(3, crate::strategies::mcts::index::Id::invalid_id);
    assert_eq!(
        children.explored_len(),
        2,
        "explored_len should count only resolved slots, not len()"
    );

    let n = 4usize;
    let explored = 2usize;
    let expected = n * std::mem::size_of::<u32>()
        + n * std::mem::size_of::<std::sync::OnceLock<crate::strategies::mcts::index::Id>>()
        + explored
            * (std::mem::size_of::<crate::strategies::mcts::index::Id>() + std::mem::size_of::<usize>())
        + n * std::mem::size_of::<std::sync::atomic::AtomicU32>()
        + n * std::mem::size_of::<u32>()
        + n * 2 * std::mem::size_of::<mcts::node::PlayerStats>();
    assert_eq!(
        children.heap_bytes_estimate(),
        expected,
        "heap_bytes_estimate should be exactly the sum of each parallel array's element count * element size"
    );
}

#[test]
fn test_memory_stats_matches_hand_walked_arena() {
    use crate::games::ttt::*;
    type G = TicTacToe;

    // A handful of iterations on a fresh board: enough to expand the root
    // and explore a couple of children, small enough to hand-verify by
    // walking the arena the same way `memory_stats` does, independently.
    let mut ts = mcts::TreeSearch::<G, mcts::strategy::Ucb1>::default().config(
        mcts::SearchConfig::default()
            .expand_threshold(1)
            .max_iterations(10),
    );
    let init_state = HashedPosition::new();
    ts.choose_action(&init_state);

    let stats = ts.memory_stats();

    let mut want_total = 0usize;
    let mut want_leaf = 0usize;
    let mut want_terminal = 0usize;
    let mut want_expanded = 0usize;
    let mut want_total_slots = 0usize;
    let mut want_explored_slots = 0usize;
    ts.index.for_each(|node: &mcts::node::Node<Move>| {
        want_total += 1;
        match node.status() {
            None => want_leaf += 1,
            Some(mcts::node::NodeState::Terminal) => want_terminal += 1,
            Some(mcts::node::NodeState::Expanded(children)) => {
                want_expanded += 1;
                want_total_slots += children.len();
                want_explored_slots += children.explored_len();
            }
        }
    });

    assert_eq!(stats.total_nodes, want_total);
    assert_eq!(stats.total_nodes, ts.arena_len());
    assert_eq!(stats.leaf_nodes, want_leaf);
    assert_eq!(stats.terminal_nodes, want_terminal);
    assert_eq!(stats.expanded_nodes, want_expanded);
    assert_eq!(
        want_leaf + want_terminal + want_expanded,
        want_total,
        "every arena entry is exactly one of leaf/terminal/expanded"
    );
    assert_eq!(stats.total_child_slots, want_total_slots);
    assert_eq!(stats.explored_child_slots, want_explored_slots);
    assert!(
        stats.explored_child_slots <= stats.total_child_slots,
        "can't explore more slots than exist"
    );
    assert!(want_expanded > 0, "root should have expanded with expand_threshold(1)");
    assert_eq!(
        stats.node_bytes,
        stats.total_nodes * std::mem::size_of::<mcts::node::Node<Move>>()
    );
    assert_eq!(
        stats.table_entries, 0,
        "transpositions are off by default in this config"
    );
    assert_eq!(stats.table_bytes, 0);
}

#[test]
fn test_nst_bigram_table_populated_by_backprop() {
    use crate::games::ttt::*;
    // Two empty cells (7, 8), O to move, no winner yet -- forces a
    // deterministic zero-tree-descent, exactly-two-ply playout (O plays
    // one of the two remaining cells, X is then forced into the last
    // one, filling the board), so `trial.actions` has exactly the one
    // consecutive pair NST's bigram table needs, with no tree-path
    // segment to reason about.
    let init_state = HashedPosition {
        position: Position {
            turn: Piece::O,
            board: [
                (0, Piece::X),
                (1, Piece::X),
                (2, Piece::O),
                (3, Piece::X),
                (4, Piece::X),
                (5, Piece::O),
                (6, Piece::O),
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

    type G = TicTacToe;
    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1Nst>;
    let mut ts = TS::default().config(mcts::SearchConfig::default().seed(7));

    let root_id = ts.reset(G::player_to_move(&init_state).to_index(), 0);
    ts.reset_iter();
    let mut ctx = mcts::SearchContext::new(root_id, init_state);
    ts.select(&mut ctx);
    let trial = ts.simulate(&ctx.state);
    assert_eq!(
        trial.actions.len(),
        2,
        "exactly two empty cells should force a two-ply playout"
    );
    let (first_action, first_player) = trial.actions[0];
    let (second_action, second_player) = trial.actions[1];
    assert_ne!(first_player, second_player);

    ts.trial = Some(trial);
    ts.backprop();

    let bigram = ts.stats.player_bigram_actions[second_player]
        .read()
        .unwrap();
    let stats = bigram
        .get(&(first_action, second_action))
        .unwrap_or_else(|| {
            panic!(
                "expected bigram entry for ({:?}, {:?}) under player {}",
                first_action, second_action, second_player
            )
        });
    assert_eq!(stats.num_visits, 1);

    // The pair shouldn't have been (mis)attributed to the other player's
    // table.
    let other = 1 - second_player;
    assert!(ts.stats.player_bigram_actions[other]
        .read()
        .unwrap()
        .is_empty());
}

#[test]
fn test_nst_backoff_falls_back_to_unigram_below_threshold() {
    use crate::games::ttt::*;
    use mcts::simulate::{Nst, SimulateStrategy};
    use rand::SeedableRng;

    type G = TicTacToe;
    let state = HashedPosition::new();
    let available = vec![Move(0), Move(1)];
    let prev = Move(2);

    let stats = mcts::search::TreeStats::<G>::default();
    // Unigram strongly favors Move(1): visited, average utility 1.0,
    // versus Move(0)'s average 0.0.
    {
        let mut player_actions = stats.player_actions[0].write().unwrap();
        player_actions.insert(
            Move(0),
            mcts::node::ActionStats {
                num_visits: 10,
                score: 0.,
            },
        );
        player_actions.insert(
            Move(1),
            mcts::node::ActionStats {
                num_visits: 10,
                score: 10.,
            },
        );
    }
    // Bigram (context = `prev`) strongly favors the opposite: Move(0)
    // over Move(1) -- but with too few samples to trust yet.
    {
        let mut bigram = stats.player_bigram_actions[0].write().unwrap();
        bigram.insert(
            (prev, Move(0)),
            mcts::node::ActionStats {
                num_visits: 2,
                score: 2.,
            },
        );
        bigram.insert(
            (prev, Move(1)),
            mcts::node::ActionStats {
                num_visits: 2,
                score: 0.,
            },
        );
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    // Below `backoff_threshold` (default 5, only 2 samples): falls back
    // to the unigram score, which prefers Move(1).
    let mut below = Nst::default();
    assert_eq!(
        *below.select_move(&state, &available, &stats, 0, Some(&prev), &mut rng),
        Move(1)
    );

    // Bump the bigram sample count to meet a lowered threshold: now the
    // bigram score (favoring Move(0)) should win instead.
    {
        let mut bigram = stats.player_bigram_actions[0].write().unwrap();
        bigram.get_mut(&(prev, Move(0))).unwrap().num_visits = 5;
        bigram.get_mut(&(prev, Move(0))).unwrap().score = 5.;
        bigram.get_mut(&(prev, Move(1))).unwrap().num_visits = 5;
    }
    let mut above = Nst::default().backoff_threshold(5);
    assert_eq!(
        *above.select_move(&state, &available, &stats, 0, Some(&prev), &mut rng),
        Move(0)
    );

    // With no previous action at all (e.g. rolling out directly from the
    // tree root), there's no context to look up -- always the unigram
    // score.
    let mut no_context = Nst::default().backoff_threshold(1);
    assert_eq!(
        *no_context.select_move(&state, &available, &stats, 0, None, &mut rng),
        Move(1)
    );
}

// Tree reuse across moves ("re-rooting", search.rs's
// `reuse_or_reset`/`find_reachable`). `reuse_or_reset` is exercised
// directly (rather than only indirectly via two `choose_action` calls)
// so each test can isolate exactly what the reuse mechanism itself did,
// uncontaminated by whatever the *next* call's own fresh iterations
// would separately discover.

#[test]
fn test_reuse_tree_promotes_matching_child_with_inherited_stats() {
    use crate::games::ttt::*;
    type G = TicTacToe;
    let init_state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(200)
            .reuse_tree(true)
            .seed(42),
    );

    let action = ts.choose_action(&init_state);
    let after_own_move = G::apply(init_state, &action);

    // Pick a reply that was actually explored during the search above --
    // `generate_actions` doesn't guarantee every legal reply got visited
    // at only 200 iterations, so read the tree directly for one that
    // was, to deterministically exercise the promote path rather than
    // the fallback-to-reset path.
    let x_child_id = {
        let root = ts.index.get(ts.root_id);
        let children = root.children();
        let idx = (0..children.len())
            .find(|&i| *children.action(i) == action)
            .unwrap();
        children
            .node_id(idx)
            .expect("the played action must have been explored")
    };
    let (reply, expected_id) = {
        let x_child = ts.index.get(x_child_id);
        let children = x_child.children();
        let idx = (0..children.len())
            .find(|&i| children.is_explored(i))
            .expect("some reply should have been explored at 200 iterations");
        (*children.action(idx), children.node_id(idx).unwrap())
    };
    let next_state = G::apply(after_own_move, &reply);

    let hash = G::zobrist_hash(&next_state);
    let player_idx = G::player_to_move(&next_state).to_index();
    let root_id = ts.reuse_or_reset(player_idx, &next_state);

    assert_eq!(
        root_id, expected_id,
        "should have promoted the tree node reached by these two real \
             plies (depth 2 from the original root), not created a new one"
    );
    assert!(
        ts.root_stats.num_visits() > 0,
        "promoted root should inherit its incoming edge's accumulated \
             visits, not start from zero like a fresh reset() would"
    );
    assert_eq!(ts.index.get(root_id).hash, hash);
    assert!(ts.index.get(root_id).is_root());
    // The node this used to be a child of must give up `is_root` --
    // otherwise a transposition landing back on it later would read
    // `root_stats` instead of its own edge's stats (see `reuse_or_reset`'s
    // doc comment in search.rs).
}

#[test]
fn test_reuse_tree_rejects_hash_match_with_wrong_replayed_state() {
    // A genuine 64-bit Zobrist collision can't be constructed on demand,
    // but the safety property it's guarding against -- "a hash match
    // alone isn't proof, verify the actual state" -- can be exercised
    // directly by corrupting `root_state` (the known-good state
    // `find_reachable`'s candidate path gets replayed from) so it no
    // longer matches what's actually in the tree. `find_reachable`
    // still finds the same hash-matching candidate (it only reads tree
    // hashes, not `root_state`), but `try_promote`'s replay-and-compare
    // must now disagree and fall back to `reset()` rather than
    // promoting onto a node that doesn't actually correspond to the
    // caller's real state.
    use crate::games::ttt::*;
    type G = TicTacToe;
    let init_state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(200)
            .reuse_tree(true)
            .seed(42),
    );

    let action = ts.choose_action(&init_state);
    let after_own_move = G::apply(init_state, &action);
    let x_child_id = {
        let root = ts.index.get(ts.root_id);
        let children = root.children();
        let idx = (0..children.len())
            .find(|&i| *children.action(i) == action)
            .unwrap();
        children
            .node_id(idx)
            .expect("the played action must have been explored")
    };
    let reply = {
        let x_child = ts.index.get(x_child_id);
        let children = x_child.children();
        (0..children.len())
            .find(|&i| children.is_explored(i))
            .map(|i| *children.action(i))
            .expect("some reply should have been explored at 200 iterations")
    };
    let next_state = G::apply(after_own_move, &reply);

    // Corrupt the replay starting point to a different, but still real
    // and legally-replayable, one-move position -- one extra piece on a
    // cell neither `action` nor `reply` touch, so replaying those same
    // two actions on top of it stays perfectly legal (no overwriting an
    // occupied cell) while still landing on a provably different
    // (3-piece, not 2-piece) final state than the real `next_state`.
    let used_cells = [action.0, reply.0];
    let safe_cell = (0u8..9).find(|c| !used_cells.contains(c)).unwrap();
    ts.root_state = Some(G::apply(init_state, &Move(safe_cell)));

    let root_id = ts.reuse_or_reset(G::player_to_move(&next_state).to_index(), &next_state);

    assert_eq!(
        ts.root_stats.num_visits(),
        0,
        "a hash match whose replayed state disagrees with the real target \
             state must not be promoted -- should fall back to reset()"
    );
    assert_eq!(ts.index.get(root_id).hash, G::zobrist_hash(&next_state));
    assert_eq!(ts.index.len(), 1);
}

#[test]
fn test_reuse_tree_falls_back_to_reset_when_no_match() {
    // Five real plies past the searched root -- one more than
    // `MAX_REROOT_DEPTH` (4), so this is guaranteed unreachable within
    // the bound regardless of how much of the (tiny) TicTacToe tree 200
    // iterations happened to explore.
    use crate::games::ttt::*;
    type G = TicTacToe;
    let init_state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(200)
            .reuse_tree(true)
            .seed(42),
    );

    let _ = ts.choose_action(&init_state);

    let mut far_state = init_state;
    for m in [0u8, 1, 2, 3, 4] {
        far_state = G::apply(far_state, &Move(m));
    }
    let hash = G::zobrist_hash(&far_state);
    let player_idx = G::player_to_move(&far_state).to_index();
    let root_id = ts.reuse_or_reset(player_idx, &far_state);

    assert_eq!(
        ts.root_stats.num_visits(),
        0,
        "no match found within MAX_REROOT_DEPTH -- should fall back to a fresh reset()"
    );
    assert_eq!(ts.index.get(root_id).hash, hash);
    assert!(ts.index.get(root_id).is_root());
    assert_eq!(
        ts.index.len(),
        1,
        "reset() should have cleared the old tree entirely"
    );
}

#[test]
fn test_reuse_tree_disabled_always_resets() {
    // `reuse_tree` defaults to `false` -- baseline regression guard that
    // the new field doesn't perturb the untouched full-reset behavior
    // when left off, mirroring sessions 7-9's own "field defaults to a
    // no-op" pattern.
    use crate::games::ttt::*;
    type G = TicTacToe;
    let init_state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts =
        TS::default().config(mcts::SearchConfig::default().max_iterations(200).seed(42));

    let action = ts.choose_action(&init_state);
    let next_state = G::apply(init_state, &action);

    let hash = G::zobrist_hash(&next_state);
    let player_idx = G::player_to_move(&next_state).to_index();
    let root_id = ts.reuse_or_reset(player_idx, &next_state);

    assert_eq!(ts.root_stats.num_visits(), 0);
    assert_eq!(ts.index.get(root_id).hash, hash);
    assert_eq!(ts.index.len(), 1);
}

#[test]
fn test_mcts_solver_proof_survives_rerooting() {
    // A `Proven`
    // status is a property of a position, not of the search path that
    // found it, so it should still be readable after re-rooting promotes
    // that position's node to root -- confirmed directly here rather
    // than assumed.
    use crate::games::ttt::*;
    type G = TicTacToe;

    // X to move; the only move that doesn't lose outright is Move(7)
    // (see `must_block_position`'s doc comment in games/ttt.rs).
    let mut state = HashedPosition::new();
    for m in [0u8, 4, 8, 1] {
        state = G::apply(state, &Move(m));
    }

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .expand_threshold(0)
            .max_iterations(5000)
            .q_init(mcts::node::QInit::Loss)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .seed(42),
    );

    let action = ts.choose_action(&state);
    assert_eq!(action, Move(7));
    let iters = ts
        .stats
        .iter_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        iters < 5000,
        "the solver should have fully proven this near-forced position \
             well within budget (used {iters} iterations)"
    );
    state = G::apply(state, &action);

    // Every one of O's replies here is a proven loss for O (that's what
    // made Move(7)'s node provably a win for X above) -- any legal one
    // exercises a real, explored, proven child.
    let mut o_replies = Vec::new();
    G::generate_actions(&state, &mut o_replies);
    let o_move = o_replies[0];
    state = G::apply(state, &o_move);

    let hash = G::zobrist_hash(&state);
    let player_idx = G::player_to_move(&state).to_index();
    let root_id = ts.reuse_or_reset(player_idx, &state);

    assert_eq!(ts.index.get(root_id).hash, hash);
    assert_ne!(
        ts.index.get(root_id).proven(),
        mcts::node::Proven::Unproven,
        "the promoted node's Proven status (set while it was still a \
             non-root descendant during move 1's search) should have \
             survived re-rooting unchanged"
    );
}

#[test]
fn test_reuse_tree_self_play_many_moves_no_panic() {
    // Integration-level smoke test: a full self-play game with reuse
    // enabled, alternating which mover's perspective is at root every
    // ply (matching `util::self_play`'s one-engine-both-sides pattern --
    // `find_reachable`'s target is always exactly 1 ply from the current
    // root here, the shallowest real case). Exercises the promote path
    // repeatedly across a real game rather than just once.
    use crate::games::ttt::*;
    type G = TicTacToe;
    let mut state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(100)
            .reuse_tree(true)
            .seed(7),
    );

    while !G::is_terminal(&state) {
        let action = ts.choose_action(&state);
        state = G::apply(state, &action);
    }
}

#[test]
fn test_reuse_tree_composes_with_tree_parallel_self_play_no_panic() {
    // `reuse_or_reset` runs single-threaded, strictly before
    // `choose_action_tree_parallel` spawns its worker threads, so there's
    // no concurrent access to `self.index`/`self.root_stats` while it
    // mutates `is_root`/`root_stats` -- exercised here across a real
    // multi-move self-play game (not just one call): only sustained
    // real-time multi-move games sample enough interleavings to catch
    // tree-parallel races.
    let _guard = parallel_test_guard();
    use crate::games::ttt::*;
    type G = TicTacToe;
    let mut state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(200)
            .num_tree_threads(4)
            .reuse_tree(true)
            .seed(7),
    );

    while !G::is_terminal(&state) {
        let action = ts.choose_action(&state);
        state = G::apply(state, &action);
    }
}

#[test]
fn test_reuse_tree_composes_with_root_parallel_self_play_no_panic() {
    let _guard = parallel_test_guard();
    use crate::games::ttt::*;
    type G = TicTacToe;
    let mut state = HashedPosition::new();

    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::SearchConfig::default()
            .max_iterations(200)
            .num_threads(4)
            .reuse_tree(true)
            .seed(7),
    );

    while !G::is_terminal(&state) {
        let action = ts.choose_action(&state);
        state = G::apply(state, &action);
    }
}