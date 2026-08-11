use game_druid::{Druid, HashedState, Size};
use mcts::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

fn main() {
    let state = HashedState::new(Size { w: 5, h: 5 });
    // Use expand_threshold=0 so all nodes are fully expanded immediately
    let mut search: TreeSearch<Druid, strategy::Ucb1> = TreeSearch::new().config(
        SearchConfig::new()
            .name("debug")
            .expand_threshold(0)
            .use_transpositions(true)
            .q_init(QInit::Infinity)
            .max_iterations(5000)
            .seed(42)
            .select(select::Ucb1::with_c(1.414)),
    );
    let action = search.choose_action(&state);
    println!("action={:?}", action);
    println!("All good!");
}
