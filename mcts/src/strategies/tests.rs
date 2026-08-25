// Regression tests for internal MCTS infrastructure that don't need a
// specific game -- they construct raw arena nodes / child arrays /
// transposition tables directly. Tests that exercise the search through a
// real game (Tic-Tac-Toe) live in `mcts-tests/tests/ttt_strategies.rs`.

#[test]
fn test_search_report_termination_boundary_classification() {
    use crate::strategies::mcts::search::search_impl::classify_termination;
    use crate::strategies::SearchTermination;

    assert_eq!(
        classify_termination(Some(10), 10, false, false),
        SearchTermination::Iterations
    );
    assert_eq!(
        classify_termination(None, 3, true, false),
        SearchTermination::Time
    );
    assert_eq!(
        classify_termination(Some(10), 10, true, true),
        SearchTermination::Solved,
        "a proof found on the final allowed iteration is solved evidence"
    );
    assert_eq!(
        classify_termination(Some(10), 3, false, false),
        SearchTermination::Unknown
    );
}

#[test]
fn test_child_array_child_index_matches_creation_order() {
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();
    let ids: Vec<_> = (0..5).map(|i| index.insert(Node::new(0, i))).collect();

    let children = ChildArray::new(vec![10, 11, 12, 13, 14], 1, false);
    for &idx in [3usize, 0, 4, 1, 2].iter() {
        let resolved = children.get_or_create_child(idx, || ids[idx]);
        assert_eq!(resolved, ids[idx]);
    }

    for (idx, &id) in ids.iter().enumerate() {
        assert_eq!(
            children.child_index(id),
            idx,
            "child_index should invert get_or_create_child's id -> idx mapping"
        );
        assert_eq!(
            children.get_or_create_child(idx, || panic!("should not re-create")),
            id
        );
        assert_eq!(children.child_index(id), idx);
    }
}

#[test]
fn test_child_array_child_index_survives_concurrent_resolution() {
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use std::sync::Arc;

    for _ in 0..500 {
        let index: Arc<TreeIndex<u32>> = Arc::new(TreeIndex::new());
        let created_id = index.insert(Node::new(0, 0));
        let children = Arc::new(ChildArray::<u32>::new(vec![42], 1, false));

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

#[test]
fn test_child_array_explored_len_and_heap_bytes_estimate() {
    use crate::strategies::mcts::index::Id;
    use crate::strategies::mcts::node::ActionStats;
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::PlayerStats;

    let children = ChildArray::<u32>::new(vec![10, 11, 12, 13], 2, true);
    assert_eq!(children.explored_len(), 0, "nothing resolved yet");

    children.get_or_create_child(1, Id::invalid_id);
    children.get_or_create_child(3, Id::invalid_id);
    assert_eq!(
        children.explored_len(),
        2,
        "explored_len should count only resolved slots, not len()"
    );

    let n = 4usize;
    let explored = 2usize;
    let expected = n * std::mem::size_of::<u32>()
        + n * std::mem::size_of::<std::sync::OnceLock<Id>>()
        + explored * (std::mem::size_of::<Id>() + std::mem::size_of::<usize>())
        + n * std::mem::size_of::<std::sync::atomic::AtomicU32>()
        + n * std::mem::size_of::<u32>()
        + n * 2 * std::mem::size_of::<PlayerStats>()
        + n * 2 * std::mem::size_of::<ActionStats>();
    assert_eq!(
        children.heap_bytes_estimate(),
        expected,
        "heap_bytes_estimate should be exactly the sum of each parallel array's element count * element size"
    );
}

// A `Strategy` that never sets `Requirements.amaf` must not pay for the
// per-(child, player) AMAF side table at all -- not just leave it logically
// unused. `heap_bytes_estimate` excluding the `ActionStats` term is the
// observable proxy for "the `Vec` was never allocated" from outside
// `node.rs`.
#[test]
fn test_child_array_amaf_side_table_empty_when_has_amaf_false() {
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::PlayerStats;

    let n = 4usize;
    let num_players = 2usize;
    let children = ChildArray::<u32>::new(vec![10, 11, 12, 13], num_players, false);

    let default_amaf = children.amaf(0, 0);
    assert_eq!(
        (default_amaf.num_visits, default_amaf.score),
        (0, 0.0),
        "amaf() should return a harmless default when has_amaf is false, not panic on an empty Vec"
    );
    let snapshot_amaf = children.snapshot(0, 0).amaf;
    assert_eq!((snapshot_amaf.num_visits, snapshot_amaf.score), (0, 0.0));

    let expected_without_amaf = n * std::mem::size_of::<u32>()
        + n * std::mem::size_of::<std::sync::OnceLock<crate::strategies::mcts::index::Id>>()
        + n * std::mem::size_of::<std::sync::atomic::AtomicU32>()
        + n * std::mem::size_of::<u32>()
        + n * num_players * std::mem::size_of::<PlayerStats>();
    assert_eq!(
        children.heap_bytes_estimate(),
        expected_without_amaf,
        "heap_bytes_estimate should exclude the ActionStats side table entirely when has_amaf is false"
    );
}

#[test]
fn test_transposition_table_compact_drops_unmapped_and_remaps_survivors() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::TranspositionTable;
    use rustc_hash::FxHashMap;

    let old_index = TreeIndex::<u32>::new();
    let old_ids: Vec<_> = (0..4).map(|i| old_index.insert(Node::new(0, i))).collect();
    let new_index = TreeIndex::<u32>::new();
    let new_ids: Vec<_> = (0..2).map(|i| new_index.insert(Node::new(0, i))).collect();

    let mut table = TranspositionTable::default();
    table.insert(100, old_ids[0]);
    table.insert(200, old_ids[1]);
    table.insert(300, old_ids[2]);
    table.insert(400, old_ids[3]);

    let mut old_to_new = FxHashMap::default();
    old_to_new.insert(old_ids[0], new_ids[0]);
    old_to_new.insert(old_ids[1], new_ids[1]);

    table.compact(&old_to_new);

    assert_eq!(table.get_const(100).unwrap(), new_ids[0]);
    assert_eq!(table.get_const(200).unwrap(), new_ids[1]);
    assert!(table.get_const(300).is_none());
    assert!(table.get_const(400).is_none());
    assert_eq!(table.len(), 2);
}

#[test]
fn test_transposition_table_trusts_the_hash_first_write_wins() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::TranspositionTable;

    let index = TreeIndex::<u32>::new();
    let ids: Vec<_> = (0..2).map(|i| index.insert(Node::new(0, i))).collect();

    let table = TranspositionTable::default();
    let first = table.get_or_insert(42, || ids[0]);
    let second = table.get_or_insert(42, || ids[1]);

    assert_eq!(first, ids[0]);
    assert_eq!(second, ids[0]);
    assert_eq!(table.len(), 1);
}

#[test]
fn test_graph_table_keeps_equal_hashes_at_distinct_plies_separate() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::{TranspositionKey, TranspositionTable};

    let index = TreeIndex::<u32>::new();
    let shallow = index.insert(Node::new_at_ply(0, 42, 1, 2, false, false));
    let deep = index.insert(Node::new_at_ply(0, 42, 3, 2, false, false));
    let table = TranspositionTable::default();

    assert_eq!(
        table.get_or_insert_graph(
            TranspositionKey::PerPly {
                position_hash: 42,
                ply: 1,
            },
            || shallow,
        ),
        shallow
    );
    assert_eq!(
        table.get_or_insert_graph(
            TranspositionKey::PerPly {
                position_hash: 42,
                ply: 3,
            },
            || deep,
        ),
        deep
    );
    assert_eq!(table.len(), 2);
}

#[test]
fn test_transposition_key_new_selects_the_configured_variant() {
    use crate::strategies::mcts::config::TranspositionKeying;
    use crate::strategies::mcts::table::TranspositionKey;

    assert_eq!(
        TranspositionKey::new(TranspositionKeying::PerPly, 42, 3),
        TranspositionKey::PerPly {
            position_hash: 42,
            ply: 3,
        },
        "PerPly must keep producing exactly today's key shape -- no behavior change"
    );
    assert_eq!(
        TranspositionKey::new(TranspositionKeying::StateOnly, 42, 3),
        TranspositionKey::StateOnly { position_hash: 42 },
        "StateOnly drops ply from the key entirely"
    );
}

#[test]
fn test_graph_table_merges_equal_hashes_at_distinct_plies_under_state_only() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::{TranspositionKey, TranspositionTable};

    let index = TreeIndex::<u32>::new();
    let shallow = index.insert(Node::new_at_ply(0, 42, 1, 2, false, false));
    let table = TranspositionTable::default();

    assert_eq!(
        table.get_or_insert_graph(TranspositionKey::StateOnly { position_hash: 42 }, || {
            shallow
        }),
        shallow
    );
    // A lookup at a different ply for the same position hashes to the same
    // `StateOnly` key, so it returns the node already inserted rather than
    // creating a second one -- the cross-ply merge this keying mode exists
    // for.
    let deep = index.insert(Node::new_at_ply(0, 42, 3, 2, false, false));
    assert_eq!(
        table.get_or_insert_graph(TranspositionKey::StateOnly { position_hash: 42 }, || deep),
        shallow
    );
    assert_eq!(table.len(), 1);
}

#[test]
fn test_node_incoming_edge_count_marks_transpositions() {
    use crate::strategies::mcts::node::Node;

    let node = Node::<u32>::new_at_ply(0, 7, 2, 2, false, false);
    assert!(!node.is_transposition());
    node.add_incoming_edge();
    assert_eq!(node.incoming_edges(), 1);
    assert!(!node.is_transposition());
    node.add_incoming_edge();
    assert_eq!(node.incoming_edges(), 2);
    assert!(node.is_transposition());
}

// The residual correction, wired at the point `select_step` resolves an
// existing child (`shared::mcgs_correction_at_edge`): compares one edge's
// local Q against its shared target node's Q, both for the parent's own
// mover, only once the target has more than one incoming edge.
mod mcgs_correction_at_edge_tests {
    use crate::strategies::mcts::config::{GraphStats, McgsCorrection};
    use crate::strategies::mcts::node::{ChildArray, Node};
    use crate::strategies::mcts::search::shared::mcgs_correction_at_edge;
    use crate::strategies::mcts::search::TreeIndex;

    const RESIDUAL: McgsCorrection = McgsCorrection::Residual { epsilon: 0.1 };

    // A dummy arena Id -- `ChildArray::get_or_create_child` needs one to mark
    // its slot explored, but nothing here ever dereferences it through the
    // arena, so any distinct `Id` will do.
    fn dummy_id() -> crate::strategies::mcts::index::Id {
        let index = TreeIndex::<u32>::new();
        index.insert(Node::new(0, 0))
    }

    // One player-0 mover with a single action, whose edge and target node
    // stats are set to given (score, visits) pairs so a residual can be
    // driven above or below `epsilon` directly, without a real playout.
    fn edge_and_target(edge: (f64, u32), node: (f64, u32)) -> (ChildArray<u32>, Node<u32>) {
        let children = ChildArray::new(vec![0u32], 2, false);
        children.get_or_create_child(0, dummy_id);
        for _ in 0..edge.1 {
            children.update(0, &[edge.0 / edge.1 as f64, 0.0]);
        }
        let target = Node::<u32>::new_at_ply(1, 99, 1, 2, false, false);
        target.add_incoming_edge();
        target.add_incoming_edge();
        for _ in 0..node.1 {
            target.stats.update(&[node.0 / node.1 as f64, 0.0]);
        }
        (children, target)
    }

    #[test]
    fn not_both_mode_never_fires() {
        let (children, target) = edge_and_target((10.0, 10), (-10.0, 10));
        for graph_stats in [None, Some(GraphStats::Edges), Some(GraphStats::Nodes)] {
            assert_eq!(
                mcgs_correction_at_edge(RESIDUAL, graph_stats, 2, &children, 0, 0, &target),
                None
            );
        }
    }

    #[test]
    fn single_incoming_edge_never_fires() {
        // Same disagreeing stats as `fires_and_returns_every_players_node_estimate`,
        // but only one `add_incoming_edge()` -- `is_transposition()` is
        // false, so nothing shared by another parent exists yet to trust
        // over this edge.
        let children = ChildArray::new(vec![0u32], 2, false);
        children.get_or_create_child(0, dummy_id);
        for _ in 0..10 {
            children.update(0, &[1.0, 0.0]);
        }
        let target = Node::<u32>::new_at_ply(1, 99, 1, 2, false, false);
        target.add_incoming_edge();
        for _ in 0..10 {
            target.stats.update(&[-1.0, 0.0]);
        }
        assert_eq!(
            mcgs_correction_at_edge(
                RESIDUAL,
                Some(GraphStats::Both),
                2,
                &children,
                0,
                0,
                &target
            ),
            None
        );
    }

    #[test]
    fn agreeing_estimates_within_epsilon_do_not_fire() {
        let (children, target) = edge_and_target((4.0, 10), (4.05, 10));
        assert_eq!(
            mcgs_correction_at_edge(
                RESIDUAL,
                Some(GraphStats::Both),
                2,
                &children,
                0,
                0,
                &target
            ),
            None
        );
    }

    #[test]
    fn fires_and_returns_every_players_node_estimate() {
        // Edge (player 0's mover Q) says +1.0; the shared node -- informed
        // by its other parent -- says -1.0. Player 1's own node estimate is
        // never read by the residual check itself, but the whole vector it
        // returns comes from the node, not the edge, once the check fires.
        let (children, target) = edge_and_target((10.0, 10), (-10.0, 10));
        target.stats.update(&[0.0, 5.0]); // makes player 1's node score nonzero
        let got = mcgs_correction_at_edge(
            RESIDUAL,
            Some(GraphStats::Both),
            2,
            &children,
            0,
            0,
            &target,
        )
        .expect("large residual should fire");
        assert_eq!(got.len(), 2);
        assert!((got[0] - target.stats.expected_score(0)).abs() < 1e-9);
        assert!((got[1] - target.stats.expected_score(1)).abs() < 1e-9);
    }

    #[test]
    fn disabled_config_never_fires() {
        let (children, target) = edge_and_target((10.0, 10), (-10.0, 10));
        assert_eq!(
            mcgs_correction_at_edge(
                McgsCorrection::Disabled,
                Some(GraphStats::Both),
                2,
                &children,
                0,
                0,
                &target
            ),
            None
        );
    }
}

// Guards `Node::solver`'s "no allocation when the solver is off" storage
// split the same way `test_child_array_amaf_side_table_empty_when_has_amaf_false`
// guards the AMAF side table: a future regression that unconditionally
// allocates `SolverState` again wouldn't be caught by any behavioral test,
// since `try_prove`/`set_pn_dpn`/`set_pn_dpn2` are already no-ops and
// `proven`/`pn`/`dpn`/`pn2`/`dpn2` already return the same sentinels
// whether the block is absent or merely never written to.
#[test]
fn test_node_solver_state_absent_when_has_solver_false() {
    use crate::strategies::mcts::node::{Node, Proven};

    let node = Node::<u32>::new_at_ply(0, 0, 0, 2, false, false);
    assert!(!node.has_solver());

    // Exercise every solver-mutating accessor to confirm they no-op rather
    // than panicking on the absent block, and never move off their
    // solver-off sentinels.
    node.try_prove(Proven::Win(0));
    node.set_pn_dpn(0, 0);
    node.set_pn_dpn2(0, 0);

    assert_eq!(node.proven(), Proven::Unproven);
    assert_eq!(node.pn(), 1);
    assert_eq!(node.dpn(), 1);
    assert_eq!(node.pn2(), 1);
    assert_eq!(node.dpn2(), 1);

    let solver_node = Node::<u32>::new_at_ply(0, 0, 0, 2, false, true);
    assert!(solver_node.has_solver());
}

#[test]
fn test_child_array_remap_child_ids_rewrites_resolved_slots_only() {
    use crate::strategies::mcts::index::Id;
    use crate::strategies::mcts::node::{ChildArray, Node};
    use crate::strategies::mcts::search::TreeIndex;
    use rustc_hash::FxHashMap;

    let old_index = TreeIndex::<u32>::new();
    let old_ids: Vec<Id> = (0..3).map(|i| old_index.insert(Node::new(0, i))).collect();
    let new_index = TreeIndex::<u32>::new();
    let new_ids: Vec<Id> = (0..3).map(|i| new_index.insert(Node::new(0, i))).collect();

    let mut children = ChildArray::<u32>::new(vec![10, 11, 12], 1, false);
    children.get_or_create_child(0, || old_ids[0]);
    children.get_or_create_child(2, || old_ids[2]);

    let mut old_to_new = FxHashMap::default();
    old_to_new.insert(old_ids[0], new_ids[0]);
    old_to_new.insert(old_ids[2], new_ids[2]);

    children.remap_child_ids(&old_to_new);

    assert_eq!(children.node_id(0), Some(new_ids[0]));
    assert_eq!(children.node_id(1), None);
    assert_eq!(children.node_id(2), Some(new_ids[2]));
    assert_eq!(children.child_index(new_ids[0]), 0);
    assert_eq!(children.child_index(new_ids[2]), 2);
}

// A tiny deterministic `Game` used only by tests, whose two action orders
// converge on the same node at the same ply, so `select`/`backprop` (via
// `choose_action`) can be driven through a real, if minimal, search instead
// of relying on random discovery in a full game like tic-tac-toe.
mod converge_game_tests {
    use crate::game::{Game, PlayerIndex};
    use crate::strategies::mcts::node::NodeState;
    use crate::strategies::mcts::strategy::Ucb1;
    use crate::strategies::Search;
    use crate::{GraphSearch, GraphStats, SearchConfig, TranspositionKeying, TreeSearch};

    // 6 distinct "pick" actions; a state is the order-independent set of 4
    // already picked, so two orders that pick the same 4-of-6 subset
    // converge on one shared node -- a wider diamond than tic-tac-toe's
    // incidental transpositions, small enough to enumerate by hand
    // (`C(6, 4) == 15` distinct terminal states, `1 + 6 + 15 + 20 == 42`
    // non-terminal ones), but with enough terminal-state variety (`winner`
    // depends on *which* 4 were picked, not just how many) that `Both`
    // mode's node/edge Q estimates can genuinely diverge instead of
    // trivially agreeing on one constant outcome the way a fully
    // deterministic single-terminal-state game would.
    const NUM_ACTIONS: u8 = 6;
    const PICKS: u32 = 4;

    #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
    struct Player(usize);

    impl PlayerIndex for Player {
        fn to_index(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    struct State {
        mask: u8,
    }

    impl std::fmt::Display for State {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:06b}", self.mask)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize)]
    struct Action(u8);

    #[derive(Clone)]
    struct ConvergeGame;

    impl Game for ConvergeGame {
        type S = State;
        type A = Action;
        type P = Player;

        fn apply(state: Self::S, action: &Self::A) -> Self::S {
            State {
                mask: state.mask | (1 << action.0),
            }
        }

        fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
            if state.mask.count_ones() >= PICKS {
                return;
            }
            for i in 0..NUM_ACTIONS {
                if state.mask & (1 << i) == 0 {
                    actions.push(Action(i));
                }
            }
        }

        fn winner(state: &Self::S) -> Option<Self::P> {
            if state.mask.count_ones() < PICKS {
                return None;
            }
            let sum: u32 = (0..NUM_ACTIONS)
                .filter(|&i| state.mask & (1 << i) != 0)
                .map(u32::from)
                .sum();
            Some(Player((sum % 2) as usize))
        }

        fn player_to_move(state: &Self::S) -> Self::P {
            Player((state.mask.count_ones() % 2) as usize)
        }

        fn zobrist_hash(state: &Self::S) -> u64 {
            state.mask as u64
        }
    }

    type TS = TreeSearch<ConvergeGame, Ucb1>;

    fn legal(state: &State) -> Vec<Action> {
        let mut actions = Vec::new();
        ConvergeGame::generate_actions(state, &mut actions);
        actions
    }

    #[test]
    fn every_graph_stats_mode_chooses_a_legal_action() {
        let state = State::default();
        for graph_search in [
            None,
            Some(GraphStats::Edges),
            Some(GraphStats::Nodes),
            Some(GraphStats::Both),
        ]
        .map(|s| s.map(GraphSearch::Dag))
        {
            let mut config = SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .seed(7);
            if let Some(gs) = graph_search {
                config = config.graph_search(gs);
            }
            let mut search = TS::default().config(config);
            let action = search.choose_action(&state);
            assert!(legal(&state).contains(&action), "{graph_search:?}");
        }
    }

    #[test]
    fn graph_mode_creates_fewer_arena_nodes_than_plain_tree() {
        let state = State::default();

        let mut tree = TS::default().config(
            SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .seed(7),
        );
        tree.choose_action(&state);

        let mut dag = TS::default().config(
            SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .seed(7)
                .graph_search(GraphSearch::Dag(GraphStats::Both)),
        );
        dag.choose_action(&state);

        assert!(
            dag.arena_len() < tree.arena_len(),
            "merging equal-depth transpositions should need fewer arena nodes \
             than a plain tree for the same iteration budget: dag={} tree={}",
            dag.arena_len(),
            tree.arena_len()
        );
    }

    #[test]
    fn only_both_mode_lets_a_shared_nodes_edges_disagree_with_it() {
        for graph_stats in [GraphStats::Edges, GraphStats::Nodes, GraphStats::Both] {
            let state = State::default();
            let mut search = TS::default().config(
                SearchConfig::default()
                    .max_iterations(500)
                    .expand_threshold(0)
                    .seed(7)
                    .graph_search(GraphSearch::Dag(graph_stats)),
            );
            search.choose_action(&state);

            let mut found_shared_edge = false;
            let mut found_divergence = false;
            search.index.for_each(|node| {
                let Some(NodeState::Expanded(children)) = node.status() else {
                    return;
                };
                for idx in 0..children.len() {
                    let Some(child_id) = children.node_id(idx) else {
                        continue;
                    };
                    let child = search.index.get(child_id);
                    if !child.is_transposition() {
                        continue;
                    }
                    found_shared_edge = true;
                    if graph_stats == GraphStats::Both {
                        let edge_q = children.expected_score(idx, 0);
                        let node_q = child.stats.expected_score(0);
                        if (edge_q - node_q).abs() > 1e-9 {
                            found_divergence = true;
                        }
                    }
                }
            });

            assert!(
                found_shared_edge,
                "{graph_stats:?}: expected at least one transposition edge over 500 iterations"
            );
            assert_eq!(
                found_divergence,
                graph_stats == GraphStats::Both,
                "{graph_stats:?}: edge/shared-node Q divergence should only appear in Both mode"
            );
        }
    }

    #[test]
    fn per_ply_keying_config_produces_the_same_search_as_leaving_it_unset() {
        let state = State::default();

        let mut default_keying = TS::default().config(
            SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .seed(7)
                .graph_search(GraphSearch::Dag(GraphStats::Both)),
        );
        let default_action = default_keying.choose_action(&state);

        let mut explicit_per_ply = TS::default().config(
            SearchConfig::default()
                .max_iterations(300)
                .expand_threshold(0)
                .seed(7)
                .graph_search(GraphSearch::Dag(GraphStats::Both))
                .transposition_keying(TranspositionKeying::PerPly),
        );
        let explicit_action = explicit_per_ply.choose_action(&state);

        assert_eq!(
            explicit_per_ply.arena_len(),
            default_keying.arena_len(),
            "explicitly requesting the default PerPly keying must not change how many \
             arena nodes a search creates"
        );
        assert_eq!(
            explicit_action, default_action,
            "explicitly requesting the default PerPly keying must not change the chosen action"
        );
    }

    #[test]
    fn state_only_keying_is_rejected_with_reuse_tree_or_unbounded_playout_depth() {
        let base = SearchConfig::<ConvergeGame, Ucb1>::default()
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .transposition_keying(TranspositionKeying::StateOnly);

        assert!(
            base.clone().max_playout_depth(50).validate().is_ok(),
            "StateOnly with a bounded max_playout_depth and reuse_tree off must validate"
        );
        assert!(
            base.clone()
                .max_playout_depth(50)
                .reuse_tree(true)
                .validate()
                .is_err(),
            "StateOnly is not yet supported together with reuse_tree"
        );
        assert!(
            base.clone().validate().is_err(),
            "StateOnly requires a finite max_playout_depth to bound descent under cycles"
        );
        assert!(
            SearchConfig::<ConvergeGame, Ucb1>::default()
                .graph_search(GraphSearch::Dag(GraphStats::Both))
                .transposition_keying(TranspositionKeying::PerPly)
                .validate()
                .is_ok(),
            "PerPly keeps today's behavior: neither new restriction applies to it"
        );
    }
}

mod cycle_game_tests {
    use crate::game::{Game, PlayerIndex};
    use crate::strategies::mcts::strategy::Ucb1;
    use crate::strategies::Search;
    use crate::{GraphSearch, GraphStats, SearchConfig, TranspositionKeying, TreeSearch};

    #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
    struct Player(usize);

    impl PlayerIndex for Player {
        fn to_index(&self) -> usize {
            self.0
        }
    }

    // The root state (`kind == 0`) leads into a two-node cycle between `A`
    // (`kind == 1`) and `B` (`kind == 2`) that never returns to the root --
    // under `TranspositionKeying::StateOnly` these merge into a single
    // two-node cycle in the graph, since ply is dropped from the key (see
    // `TranspositionKeying`'s doc comment). The root only ever occupies the
    // first stack entry this way (a search's root node is otherwise
    // unreachable from any non-root node, an invariant `backprop` relies
    // on), so the only thing exercised is whether `select_step`'s descent
    // guard bounds the A/B cycle -- ply's strict increase, which `PerPly`
    // gets for free, can't do it, since `StateOnly` deliberately drops ply
    // from the key.
    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    struct State {
        kind: u8,
    }

    impl std::fmt::Display for State {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.kind)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize)]
    struct Flip;

    #[derive(Clone)]
    struct CycleGame;

    impl Game for CycleGame {
        type S = State;
        type A = Flip;
        type P = Player;

        fn apply(state: Self::S, _action: &Self::A) -> Self::S {
            let kind = match state.kind {
                0 => 1,
                1 => 2,
                _ => 1,
            };
            State { kind }
        }

        fn generate_actions(_state: &Self::S, actions: &mut Vec<Self::A>) {
            actions.push(Flip);
        }

        fn winner(_state: &Self::S) -> Option<Self::P> {
            None
        }

        fn player_to_move(_state: &Self::S) -> Self::P {
            Player(0)
        }

        fn num_players() -> usize {
            1
        }

        fn zobrist_hash(state: &Self::S) -> u64 {
            state.kind as u64
        }
    }

    #[test]
    fn state_only_keying_descent_guard_terminates_on_a_two_cycle() {
        let state = State::default();
        let mut search = TreeSearch::<CycleGame, Ucb1>::default().config(
            SearchConfig::default()
                .max_iterations(50)
                .expand_threshold(0)
                .max_playout_depth(8)
                .seed(7)
                .graph_search(GraphSearch::Dag(GraphStats::Both))
                .transposition_keying(TranspositionKeying::StateOnly),
        );

        // Without the depth guard, `select_step` would either descend the
        // A<->B cycle forever or overflow `stack` -- this call returning at
        // all, with a legal action, is the regression test.
        let action = search.choose_action(&state);

        assert_eq!(action, Flip, "the only legal action must be the one chosen");
    }
}

// PN-MCTS (Kowalski et al. 2023): `derive_pn_dpn`'s negamax recurrence,
// hand-verified on a tiny 3-child arena rather than through a real game --
// small enough to compute the expected pn/dpn by hand, which a purely
// behavioral (does-it-pick-the-right-move) test wouldn't necessarily
// exercise: `pn(root) = min` over children of the child's *dpn*, and
// `dpn(root) = sum` over children of the child's *pn* -- easy to get
// backwards, since it's the opposite of the naive "sum pn, min dpn" a
// classic (non-negamax) OR-node formula would suggest.
//
// Root has three child slots:
//   - idx 0: an unexplored slot (no tree node at all) -- PNS's "unknown
//     leaf" case, contributes (pn=1, dpn=1).
//   - idx 1: a tree node proven `Win` for its own mover -- contributes
//     (pn=0, dpn=MAX) via `Node::pn`/`dpn`'s `Proven` short-circuit.
//   - idx 2: an unvisited tree node (never `expand()`ed, `Unproven`) --
//     also (pn=1, dpn=1), but read from the *stored* atomic default this
//     time rather than the unexplored-slot fallback, exercising the other
//     code path that produces the same value.
//
// Expected: pn(root) = min(dpn_0=1, dpn_1=MAX, dpn_2=1) = 1.
//           dpn(root) = pn_0=1 + pn_1=0 + pn_2=1 = 2.
#[test]
fn test_derive_pn_dpn_negamax_recurrence_hand_verified() {
    use crate::strategies::mcts::backprop::derive_pn_dpn;
    use crate::strategies::mcts::node::{ChildArray, Node, NodeState, Proven};
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();

    let proven_win_child = Node::new(1, 0);
    proven_win_child.try_prove(Proven::Win(1));
    let proven_win_id = index.insert(proven_win_child);

    let unvisited_child = Node::new(1, 0);
    let unvisited_id = index.insert(unvisited_child);

    let children = ChildArray::<u32>::new(vec![10, 11, 12], 2, false);
    children.get_or_create_child(1, || proven_win_id);
    children.get_or_create_child(2, || unvisited_id);
    // idx 0 deliberately left unresolved (no `get_or_create_child` call).

    assert_eq!(index.get(proven_win_id).pn(), 0);
    assert_eq!(index.get(proven_win_id).dpn(), u32::MAX);
    assert_eq!(index.get(unvisited_id).pn(), 1);
    assert_eq!(index.get(unvisited_id).dpn(), 1);

    let root = Node::<u32>::new(0, 0);
    root.expand(|| NodeState::Expanded(children));

    derive_pn_dpn(&root, &index);

    assert_eq!(root.pn(), 1, "pn(root) = min(dpn) over children");
    assert_eq!(root.dpn(), 2, "dpn(root) = sum(pn) over children");
}

// Double-layer PN-MCTS (Kowalski et al. 2023, Section VII): `derive_pn_dpn2`
// runs the identical negamax recurrence as `derive_pn_dpn` above, but for
// the "not lost" goal instead of "won" -- this is the divergence that lets
// PN-MCTS handle games with draws. Same 3-child shape as the test above,
// with idx 1 a proven *Draw* instead of a proven Win, to demonstrate exactly
// where the two layers disagree:
//
//   - First layer (goal "won"): a Draw is just as much a non-win as a loss,
//     so it counts as a *disproof* -- `pn=MAX, dpn=0` -- collapsing "the
//     opponent forced a draw" and "the opponent forced a loss" into the same
//     bookkeeping.
//   - Second layer (goal "not lost"): a Draw satisfies the goal, so it
//     counts as a *proof* -- `pn2=0, dpn2=MAX` -- the opposite magnitudes.
//
// Expected: pn(root) = min(dpn_0=1, dpn_1=0, dpn_2=1) = 0.
//           dpn(root) = pn_0=1 + pn_1=MAX + pn_2=1, saturating to MAX.
//           pn2(root) = min(dpn2_0=1, dpn2_1=MAX, dpn2_2=1) = 1.
//           dpn2(root) = pn2_0=1 + pn2_1=0 + pn2_2=1 = 2.
//
// The first layer alone reads root as "already disproven" (pn=0, as cheap to
// disprove as an outright loss); the second layer correctly reads it as
// still wide open (pn2=1, dpn2=2, same as the fully-unresolved baseline
// case) -- precisely the ambiguity Section VII exists to resolve.
#[test]
fn test_derive_pn_dpn2_not_lost_goal_diverges_from_first_layer_on_a_draw() {
    use crate::strategies::mcts::backprop::{derive_pn_dpn, derive_pn_dpn2};
    use crate::strategies::mcts::node::{ChildArray, Node, NodeState, Proven};
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();

    let proven_draw_child = Node::new(1, 0);
    proven_draw_child.try_prove(Proven::Draw);
    let proven_draw_id = index.insert(proven_draw_child);

    let unvisited_child = Node::new(1, 0);
    let unvisited_id = index.insert(unvisited_child);

    let children = ChildArray::<u32>::new(vec![10, 11, 12], 2, false);
    children.get_or_create_child(1, || proven_draw_id);
    children.get_or_create_child(2, || unvisited_id);
    // idx 0 deliberately left unresolved (no `get_or_create_child` call).

    assert_eq!(index.get(proven_draw_id).pn(), u32::MAX);
    assert_eq!(index.get(proven_draw_id).dpn(), 0);
    assert_eq!(index.get(proven_draw_id).pn2(), 0);
    assert_eq!(index.get(proven_draw_id).dpn2(), u32::MAX);

    let root = Node::<u32>::new(0, 0);
    root.expand(|| NodeState::Expanded(children));

    derive_pn_dpn(&root, &index);
    derive_pn_dpn2(&root, &index);

    assert_eq!(
        root.pn(),
        0,
        "first layer: a drawn child already disproves the win goal"
    );
    assert_eq!(root.dpn(), u32::MAX);
    assert_eq!(
        root.pn2(),
        1,
        "second layer: still wide open, unlike the first layer's pn=0"
    );
    assert_eq!(root.dpn2(), 2);
}

// MCTS-MB-n (Baier & Winands): `derive_minimax_value`'s backward-induction
// backup, hand-verified on a tiny 3-child root -- small enough to compute
// the expected overwrite by hand, which a purely behavioral test wouldn't
// necessarily exercise (in particular, that the *non-mover's* row is
// overwritten from the mover's chosen child, not from the non-mover's own
// independent max over children -- easy to get backwards).
//
// Root (player_idx = 0, i.e. player 0 to move) has three child slots:
//   - idx 0: an explored edge with one real playout's utilities [0.2, -0.2]
//     -- expected_score(0, player 0) = 0.2.
//   - idx 1: an explored edge with one real playout's utilities [0.8, -0.8]
//     -- expected_score(1, player 0) = 0.8, the mover's best.
//   - idx 2: an unresolved slot (no tree node at all) -- contributes
//     nothing, the same "unknown leaf, skip it" treatment
//     `derive_pn_dpn` gives an unresolved child.
//
// Player 0 (the mover) picks idx 1 (0.8 > 0.2), so root's value is
// overwritten to idx 1's own row for *every* player: player 0 -> 0.8,
// player 1 -> -0.8 (not player 1's own max, which would incorrectly read
// idx 0's -0.2 as "better for player 1").
#[test]
fn test_derive_minimax_value_backup_hand_verified() {
    use crate::strategies::mcts::backprop::{derive_minimax_value, PosteriorSlot};
    use crate::strategies::mcts::node::{ChildArray, Node, NodeState};
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();

    let child0_id = index.insert(Node::new(1, 0));
    let child1_id = index.insert(Node::new(1, 0));

    let children = ChildArray::<u32>::new(vec![10, 11, 12], 2, false);
    children.get_or_create_child(0, || child0_id);
    children.update(0, &[0.2, -0.2]);
    children.get_or_create_child(1, || child1_id);
    children.update(1, &[0.8, -0.8]);
    // idx 2 deliberately left unresolved (no `get_or_create_child` call).

    let root = Node::<u32>::new(0, 0);
    root.expand(|| NodeState::Expanded(children));
    root.stats.update(&[0.0, 0.0]);

    let slot = PosteriorSlot::Root(&root.stats);
    derive_minimax_value(&root, &slot, 2);

    assert_eq!(
        root.stats.score(0),
        0.8,
        "mover's row takes its own best child's value"
    );
    assert_eq!(
        root.stats.score(1),
        -0.8,
        "non-mover's row follows the *mover's* chosen child, not its own independent max"
    );
}
