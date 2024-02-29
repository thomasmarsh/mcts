use mcts::game::Game;
use mcts::strategies::human::HumanAgent;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::battle_royale;

fn play<G: Game>()
where
    G::S: std::fmt::Display,
{
    let mut ts: TreeSearch<G, strategy::Ucb1Grave> = TreeSearch::new()
        .config(
            SearchConfig::new()
                .select(
                    select::Ucb1Grave::new()
                        .exploration_constant(1.32)
                        .threshold(700)
                        .bias(430.),
                )
                .max_iterations(300000)
                // .max_time(Duration::from_secs(10))
                .expand_threshold(1),
        )
        .verbose(true);

    let mut human = HumanAgent::new();

    _ = battle_royale(&mut ts, &mut human);
}

fn main() {
    use mcts::games::gonnect::Gonnect;

    play::<Gonnect<8>>();
}
