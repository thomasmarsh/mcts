use game_druid::{Druid, HashedState, Size};
use mcts::algorithms::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;

fn main() {
    let state = HashedState::new(Size { w: 5, h: 5 });
    let mut search: TreeSearch<Druid, strategy::Ucb1> = TreeSearch::new().config(
        SearchConfig::new()
            .name("debug")
            .expand_threshold(1)
            .use_transpositions(true)
            .q_init(QInit::Infinity)
            .max_iterations(175_000)
            .seed(42)
            .select(select::Ucb1::with_c(1.414)),
    );
    println!("Before choose_action...");
    let action = search.choose_action(&state);
    println!("action={:?}", action);
}
