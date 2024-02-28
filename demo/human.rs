use mcts::strategies::human::HumanAgent;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::battle_royale;

fn main() {
    use mcts::games::gonnect::Gonnect;

    type TS = TreeSearch<Gonnect<8>, strategy::Ucb1Grave>;
    let mut ts = TS::default()
        .config(
            SearchConfig::default()
                .select(select::Ucb1Grave {
                    exploration_constant: 1.32,
                    threshold: 700,
                    bias: 430.,
                    current_ref_id: None,
                })
                .max_iterations(300000)
                // .max_time(Duration::from_secs(10))
                .expand_threshold(1),
        )
        .verbose(true);

    let mut human = HumanAgent::new();

    _ = battle_royale(&mut ts, &mut human);
}
