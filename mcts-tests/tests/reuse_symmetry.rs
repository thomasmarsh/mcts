// Smoke coverage for tree reuse (`mcts::algorithms::mcts::search::reuse`)
// composed with a symmetric game's canonicalization
// (`Game::canonical_representation`/`apply_to_action`/`invert_action`).
// `reuse_or_reset`'s promote path has to translate a promoted node's own
// `ChildArray` actions out of that node's canonical orientation before it
// becomes root -- getting this wrong applies a canonical-space action
// straight to the literal board, which `game-traffic-lights`'s own
// `Position::apply` catches loudly (it asserts a cell was never already
// cycled all the way to `Piece::G`), unlike `game-ttt`'s reuse tests, whose
// `Position::apply` only asserts a cell is empty and so tolerates a
// mistranslated action landing on another empty cell without crashing.
// This doesn't reliably reproduce any specific past bug (that needed a
// particular search strategy/config to build a large enough tree), but it
// exercises the same real_action/retranslate_actions code paths against a
// game that fails loudly rather than silently.
use mcts::game::Game;
use mcts::algorithms::Search;

#[test]
fn test_reuse_tree_self_play_traffic_lights_many_seeds_no_panic() {
    use game_traffic_lights::*;

    type G = TrafficLights;
    type TS = mcts::TreeSearch<G, mcts::strategy::Ucb1>;

    for seed in 0..10 {
        let mut state = HashedPosition::new();
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .max_iterations(500)
                .reuse_tree(true)
                .use_transpositions(true)
                .seed(seed),
        );

        while !G::is_terminal(&state) {
            let action = ts.choose_action(&state);
            state = G::apply(state, &action);
        }
    }
}
