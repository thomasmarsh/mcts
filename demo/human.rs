use std::time::Duration;

use mcts::game::Game;
use mcts::games::gonnect::Gonnect;
use mcts::strategies::human::HumanAgent;
use mcts::strategies::mcts::node::QInit;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::battle_royale;

fn play<G: Game>()
where
    G::S: std::fmt::Display,
{
    let mut ts: TreeSearch<G, strategy::RaveMastDm> = TreeSearch::new().config(
        SearchConfig::new()
            .name("mcts[rave]+mast+ucd")
            .expand_threshold(1)
            .max_time(Duration::from_secs(10))
            .q_init(QInit::Infinity)
            .select(
                select::Rave::default()
                    .exploration_constant(0.305949)
                    .threshold(600)
                    .schedule(select::RaveSchedule::MinMSE { bias: 4.313335 }),
            )
            .simulate(
                simulate::DecisiveMove::new().inner(simulate::EpsilonGreedy::with_epsilon(0.29739)),
            ),
    );

    let mut human = HumanAgent::new();

    _ = battle_royale(&mut ts, &mut human);
}

fn main() {
    play::<Gonnect<8>>();
}
