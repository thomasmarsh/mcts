//! Equal-budget Connect Four Gumbel fidelity comparison.
//!
//! cargo run --release -p mcts-tests --example gumbel_connect4_fidelity -- \
//!   <zero.bin> <zero.policy.bin> <trained.bin> <trained.policy.bin> [games] [sims]

use std::time::Instant;

use game_connect4::policynet::NTuplePolicyNet;
use game_connect4::valuenet::NTupleValueNet;
use game_connect4::{Move, Standard, State};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::{GumbelCompletedQ, Ucb1};
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::game::{Game, PlayerIndex, TerminalStatus};

type Eval = EvaluatedCutoff<Standard, NTupleValueNet>;
type UcbProfile = Mcts<Ucb1, Eval>;
type CompletedProfile = Mcts<GumbelCompletedQ, Eval>;

enum Player {
    Ucb {
        search: TreeSearch<Standard, UcbProfile>,
        net: NTupleValueNet,
        cfg: GumbelConfig,
    },
    Completed {
        search: TreeSearch<Standard, CompletedProfile>,
        net: NTupleValueNet,
        cfg: GumbelConfig,
    },
}

impl Player {
    fn ucb(net: NTupleValueNet, policy: NTuplePolicyNet, cfg: GumbelConfig, seed: u64) -> Self {
        Self::Ucb {
            search: TreeSearch::default().config(
                SearchConfig::default()
                    .expand_threshold(1)
                    .max_playout_depth(0)
                    .q_init(QInit::Loss)
                    .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                    .with_policy_logits(policy)
                    .seed(seed),
            ),
            net,
            cfg,
        }
    }

    fn completed(
        net: NTupleValueNet,
        policy: NTuplePolicyNet,
        cfg: GumbelConfig,
        seed: u64,
    ) -> Self {
        Self::Completed {
            search: TreeSearch::default().config(
                SearchConfig::default()
                    .expand_threshold(1)
                    .max_playout_depth(0)
                    .q_init(QInit::Loss)
                    .select(GumbelCompletedQ::with_config(cfg))
                    .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                    .with_policy_logits(policy)
                    .seed(seed),
            ),
            net,
            cfg,
        }
    }

    fn choose(&mut self, state: &State<6, 7>) -> (Move, usize) {
        match self {
            Self::Ucb { search, net, cfg } => {
                let action =
                    gumbel_search_with_root_value(search, state, cfg, net.value(state) as f64)
                        .action;
                (action, search.arena_len())
            }
            Self::Completed { search, net, cfg } => {
                let action =
                    gumbel_search_with_root_value(search, state, cfg, net.value(state) as f64)
                        .action;
                (action, search.arena_len())
            }
        }
    }
}

fn play(first: &mut Player, second: &mut Player) -> (Option<usize>, usize) {
    let mut state = State::default();
    let mut nodes = 0;
    loop {
        let (action, added) = if Standard::player_to_move(&state).to_index() == 0 {
            first.choose(&state)
        } else {
            second.choose(&state)
        };
        nodes += added;
        state = Standard::apply(state, &action);
        match Standard::terminal_status(&state) {
            TerminalStatus::NotTerminal => {}
            TerminalStatus::Draw => return (None, nodes),
            TerminalStatus::Winner(player) => return (Some(player.to_index()), nodes),
        }
    }
}

fn compare(
    name: &str,
    games: usize,
    make_candidate: impl Fn(u64) -> Player,
    make_base: impl Fn(u64) -> Player,
) {
    let start = Instant::now();
    let (mut wins, mut draws, mut losses, mut nodes) = (0, 0, 0, 0usize);
    for game in 0..games {
        let candidate_first = game % 2 == 0;
        let seed = (game as u64) + 1;
        let mut candidate = make_candidate(seed);
        let mut base = make_base(seed);
        let (winner, game_nodes) = if candidate_first {
            play(&mut candidate, &mut base)
        } else {
            play(&mut base, &mut candidate)
        };
        nodes += game_nodes;
        match winner {
            None => draws += 1,
            Some(0) if candidate_first => wins += 1,
            Some(1) if !candidate_first => wins += 1,
            _ => losses += 1,
        }
    }
    let share = (wins as f64 + draws as f64 * 0.5) / games as f64;
    println!(
        "{name}: {wins}-{draws}-{losses} W-D-L, score share {share:.3}, wall {:.2}s, nodes {nodes}",
        start.elapsed().as_secs_f64()
    );
}

fn load_value(path: &str) -> NTupleValueNet {
    NTupleValueNet::load(path).expect("value weights")
}
fn load_policy(path: &str) -> NTuplePolicyNet {
    NTuplePolicyNet::load(path).expect("policy weights")
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        panic!("usage: gumbel_connect4_fidelity <zero.bin> <zero.policy.bin> <trained.bin> <trained.policy.bin> [games] [sims]");
    }
    let games = args.get(5).map_or(100, |s| s.parse().expect("games"));
    let sims = args.get(6).map_or(32, |s| s.parse().expect("sims"));
    let zero = load_value(&args[1]);
    let zero_policy = load_policy(&args[2]);
    let trained = load_value(&args[3]);
    let trained_policy = load_policy(&args[4]);
    let old_root = GumbelConfig {
        sims,
        use_completed_q: false,
        ..GumbelConfig::default()
    };
    let full_root = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };
    compare(
        "1 uniform policy + old root Q + UCB1 interior",
        games,
        |seed| Player::ucb(zero.clone(), zero_policy.clone(), old_root, seed),
        |seed| Player::ucb(zero.clone(), zero_policy.clone(), old_root, seed),
    );
    compare(
        "2 learned policy + old root Q + UCB1 interior",
        games,
        |seed| Player::ucb(trained.clone(), trained_policy.clone(), old_root, seed),
        |seed| Player::ucb(zero.clone(), zero_policy.clone(), old_root, seed),
    );
    compare(
        "3 learned policy + completed-Q root + UCB1 interior",
        games,
        |seed| Player::ucb(trained.clone(), trained_policy.clone(), full_root, seed),
        |seed| Player::ucb(zero.clone(), zero_policy.clone(), old_root, seed),
    );
    compare(
        "4 learned policy + completed-Q root + completed-Q interior",
        games,
        |seed| Player::completed(trained.clone(), trained_policy.clone(), full_root, seed),
        |seed| Player::ucb(zero.clone(), zero_policy.clone(), old_root, seed),
    );
}
