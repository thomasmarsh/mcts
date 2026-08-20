// `TreeSearch::reuse_or_reset_graph`'s (`mcts/src/strategies/mcts/search/
// reroot.rs`) DAG re-rooting, exercised against Othello rather than
// tic-tac-toe: Othello's `canonical_representation` picks a genuinely
// non-identity `Transform` far more often than tic-tac-toe's, so this is
// what actually proves a promoted node's `ChildArray` gets translated back
// to the literal board correctly (not masked by the transform happening to
// be the identity most of the time).

#[test]
fn test_othello_graph_reroot_self_play_keeps_ply_and_legal_moves() {
    use game_othello::*;
    use mcts::game::Game;
    use mcts::strategies::Search;
    use mcts::{GraphSearch, GraphStats};

    for stats in [GraphStats::Edges, GraphStats::Nodes, GraphStats::Both] {
        type TS = mcts::TreeSearch<Othello, mcts::strategy::Ucb1>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(100)
                .expand_threshold(0)
                .reuse_tree(true)
                .graph_search(GraphSearch::Dag(stats))
                .seed(11),
        );

        let mut state = State::default();
        for _ in 0..8 {
            if Othello::is_terminal(&state) {
                break;
            }
            let mut legal = Vec::new();
            Othello::generate_actions(&state, &mut legal);
            let action = ts.choose_action(&state);
            assert!(legal.contains(&action), "{stats:?} chose a legal move");

            assert!(ts.index.get(ts.root_id).is_root());
            assert_eq!(
                ts.index.get(ts.root_id).ply,
                0,
                "{stats:?}: root ply must stay rebased to 0 across every re-root"
            );
            assert_eq!(ts.root_state.as_ref(), Some(&state));

            state = Othello::apply(state, &action);
        }
    }
}
