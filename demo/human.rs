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
    // hash=ba2047 cost=0.43333333333333335 dict={'epsilon': 0.6096443583623276, 'final-action': 'robust_child', 'q-init': 'Draw', 'rave-ucb': 'none', 'schedule': 'min_mse', 'threshold': 910, 'bias': 0.6937972231158172}
    // hash=840351 cost=0.025000000000000022 dict={'epsilon': 0.7775134909898043, 'final-action': 'robust_child', 'q-init': 'Infinity', 'rave-ucb': 'tuned', 'schedule': 'min_mse', 'threshold': 204, 'bias': 5.286671416833997, 'c': 0.28941824845969677}
    let mut ts: TreeSearch<G, strategy::RaveMastDm> = TreeSearch::new().config(
        SearchConfig::new()
            .name("mcts[rave]+mast+ucd+dm")
            .verbose(true)
            .expand_threshold(1)
            .max_time(Duration::from_secs(10))
            .q_init(QInit::Draw)
            .select(
                select::Rave::default()
                    .ucb(select::RaveUcb::Ucb1Tuned {
                        exploration_constant: 0.2894182,
                    })
                    .threshold(204)
                    .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
            )
            .simulate(
                simulate::DecisiveMove::new().inner(simulate::EpsilonGreedy::with_epsilon(0.7775)),
            ),
    );

    let mut human = HumanAgent::new();

    _ = battle_royale(&mut ts, &mut human);
}

fn main() {
    play::<Gonnect<8>>();
}
