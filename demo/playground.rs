use std::time::Duration;

use mcts::game::Game;
use mcts::games::nim;
use mcts::games::ttt;
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::node::UnvisitedValueEstimate;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::random::Random;
use mcts::strategies::Search;
use mcts::util::battle_royale;
use mcts::util::self_play;
use mcts::util::AnySearch;

use mcts::games::nim::*;
use mcts::games::ttt::*;
use mcts::util::round_robin_multiple;

type TttFlatMC = FlatMonteCarloStrategy<TicTacToe>;
type NimFlatMC = FlatMonteCarloStrategy<Nim>;

type NimMCTS = TreeSearch<Nim, strategy::Ucb1>;
type TttMCTS = TreeSearch<TicTacToe, strategy::Ucb1>;

fn ucd() {
    use mcts::games::traffic_lights::TrafficLights;

    type Uct = TreeSearch<TrafficLights, strategy::Ucb1>;
    let uct = Uct::default()
        .config(
            SearchConfig::default()
                .max_iterations(10_000)
                .q_init(mcts::strategies::mcts::node::UnvisitedValueEstimate::Parent)
                .expand_threshold(1)
                .q_init(UnvisitedValueEstimate::Infinity)
                .select(select::Ucb1 {
                    exploration_constant: 0.01f64.sqrt(),
                }),
        )
        .verbose(false);

    type Ucd = TreeSearch<TrafficLights, strategy::Ucb1>;
    let mut ucd = Ucd::default()
        .config(
            SearchConfig::default()
                .max_iterations(10_000)
                // .q_init(mcts::strategies::mcts::node::UnvisitedValueEstimate::Parent)
                .expand_threshold(1)
                .use_transpositions(true)
                .q_init(UnvisitedValueEstimate::Infinity)
                .select(select::Ucb1 {
                    exploration_constant: 0.01f64.sqrt(),
                }),
        )
        .verbose(false);
    ucd.set_friendly_name("mcts[ucb1]+ucd");

    type UcdDm = TreeSearch<TrafficLights, strategy::Ucb1DM>;
    let mut ucd_dm = UcdDm::default()
        .config(
            SearchConfig::default()
                .max_iterations(10_000)
                // .q_init(mcts::strategies::mcts::node::UnvisitedValueEstimate::Parent)
                .expand_threshold(1)
                .use_transpositions(true)
                .q_init(UnvisitedValueEstimate::Infinity)
                .select(select::Ucb1 {
                    exploration_constant: 0.01f64.sqrt(),
                }),
        )
        .verbose(false);
    ucd_dm.set_friendly_name("mcts[ucb1]+ucd+dm");

    let mast: TreeSearch<TrafficLights, strategy::Ucb1Mast> = TreeSearch::default()
        .config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_iterations(10_000)
                .select(select::Ucb1 {
                    exploration_constant: 1.86169408634305,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.10750788170844316)),
        )
        .verbose(false);

    let mut mast_ucd: TreeSearch<TrafficLights, strategy::Ucb1Mast> = TreeSearch::default()
        .config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_iterations(10_000)
                .use_transpositions(true)
                .q_init(UnvisitedValueEstimate::Infinity)
                .select(select::Ucb1 {
                    exploration_constant: 0.01,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.10750788170844316)),
        )
        .verbose(false);
    mast_ucd.set_friendly_name("mcts[ucb1_mast]+ucd");

    let tuned: TreeSearch<TrafficLights, strategy::Ucb1Tuned> = TreeSearch::default().config(
        SearchConfig::default()
            .expand_threshold(1)
            .max_iterations(10_000)
            .select(select::Ucb1Tuned {
                exploration_constant: 1.8617,
            }),
    );

    let mut tuned_ucd: TreeSearch<TrafficLights, strategy::Ucb1Tuned> = TreeSearch::default()
        .config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_iterations(10_000)
                .use_transpositions(true)
                .select(select::Ucb1Tuned {
                    exploration_constant: 1.8617,
                }),
        );
    tuned_ucd.set_friendly_name("mcts[ucb1_tuned]+ucd");

    let mut strats = vec![
        AnySearch::new(uct),
        AnySearch::new(ucd),
        AnySearch::new(ucd_dm),
        // AnySearch::new(mast),
        // AnySearch::new(mast_ucd),
        // AnySearch::new(tuned),
        // AnySearch::new(tuned_ucd),
    ];

    _ = round_robin_multiple::<TrafficLights, AnySearch<_>>(
        &mut strats,
        1000,
        &Default::default(),
        mcts::util::Verbosity::Verbose,
    );
}

fn traffic_lights() {
    use mcts::games::traffic_lights::TrafficLights;

    type TS = TreeSearch<TrafficLights, strategy::Ucb1>;
    let ts = TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(10_000)
                .q_init(mcts::strategies::mcts::node::UnvisitedValueEstimate::Parent)
                .expand_threshold(0)
                .use_transpositions(true)
                .q_init(UnvisitedValueEstimate::Infinity)
                .select(select::Ucb1 {
                    exploration_constant: 0.001,
                }),
        )
        .verbose(true);

    self_play(ts);
}

fn knightthrough() {
    use mcts::games::knightthrough::Knightthrough;

    type TS = TreeSearch<Knightthrough<8, 8>, strategy::Ucb1GraveMast>;
    let ts = TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(20000)
                .select(select::Ucb1Grave {
                    exploration_constant: 2.12652,
                    threshold: 131,
                    bias: 68.65,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.12)),
        )
        .verbose(true);

    self_play(ts);
}

fn breakthrough() {
    use mcts::games::breakthrough::Breakthrough;

    type TS = TreeSearch<Breakthrough<6, 4>, strategy::Ucb1GraveMast>;
    let ts = TS::default()
        .config(
            SearchConfig::default()
                .max_time(Duration::from_secs(10))
                .select(select::Ucb1Grave {
                    exploration_constant: 1.32562,
                    threshold: 720,
                    bias: 430.36,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.98)),
        )
        .verbose(true);

    self_play(ts);
}

fn atarigo() {
    use mcts::games::atarigo::AtariGo;

    type TS = TreeSearch<AtariGo<5>, strategy::Ucb1GraveMast>;
    let ts = TS::default()
        .config(
            SearchConfig::default()
                .max_time(Duration::from_secs(10))
                .select(select::Ucb1Grave {
                    exploration_constant: 1.32562,
                    threshold: 720,
                    bias: 430.36,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.98)),
        )
        .verbose(true);

    self_play(ts);
}

fn gonnect() {
    use mcts::games::gonnect::Gonnect;

    type TS = TreeSearch<Gonnect<7>, strategy::Ucb1Grave>;
    let ts = TS::default()
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

    self_play(ts);
}

fn expansion_test() {
    use mcts::games::bid_ttt as ttt;

    type TS = TreeSearch<ttt::BiddingTicTacToe, strategy::Ucb1>;

    let expand5 = TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(10000)
                .expand_threshold(5),
        )
        .name("expand5");

    let expand0 = TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(10000)
                .expand_threshold(0),
        )
        .name("expand0");

    let mut strats = vec![AnySearch::new(expand5), AnySearch::new(expand0)];

    _ = round_robin_multiple::<ttt::BiddingTicTacToe, AnySearch<_>>(
        &mut strats,
        1000,
        &ttt::BiddingTicTacToe::new(),
        mcts::util::Verbosity::Verbose,
    );
}

fn ucb_test() {
    let mut flat = NimFlatMC::default();
    let mut ucb1 = NimFlatMC::default();
    flat.samples_per_move = 5000;
    ucb1.samples_per_move = 5000;
    ucb1.ucb1 = Some(100f64.sqrt());

    flat.set_friendly_name("classic");
    ucb1.set_friendly_name("ucb1");

    let mut strats = vec![AnySearch::new(flat), AnySearch::new(ucb1)];

    _ = round_robin_multiple::<Nim, AnySearch<_>>(
        &mut strats,
        5,
        &NimState::new(),
        mcts::util::Verbosity::Verbose,
    );
}

fn summarize(label_a: &str, label_b: &str, results: Vec<Option<usize>>) {
    let (win_a, win_b, draw) = results.iter().fold((0, 0, 0), |(a, b, c), x| match x {
        Some(0) => (a + 1, b, c),
        Some(1) => (a, b + 1, c),
        None => (a, b, c + 1),
        _ => (a, b, c),
    });
    let total = (win_a + win_b + draw) as f32;
    let pct_a = win_a as f32 / total * 100.;
    let pct_b = win_b as f32 / total * 100.;
    println!("{label_a} / {label_b}: {win_a} ({pct_a:.2}%) / {win_b} ({pct_b:.2}%) [{draw} draws]");
}

struct BattleConfig {
    num_samples: usize, // number of games to play
    samples_per_move: Vec<usize>,
}

fn battle_nim_mcts(config: &BattleConfig) {
    for samples in &config.samples_per_move {
        println!("samples per move: {}", samples);
        let mut mcts = NimMCTS::default().config(SearchConfig::default().max_iterations(*samples));
        let mut flat_mc = NimFlatMC::new().set_samples_per_move(*samples as u32);

        let mut fst = Vec::with_capacity(100);
        let mut snd = Vec::with_capacity(100);
        for _ in 0..config.num_samples {
            let result = battle_royale(&mut mcts, &mut flat_mc);
            fst.push(result);
            let result = battle_royale(&mut flat_mc, &mut mcts);
            snd.push(result);
        }
        summarize(
            mcts.friendly_name().as_str(),
            flat_mc.friendly_name().as_str(),
            fst,
        );
        summarize(
            flat_mc.friendly_name().as_str(),
            mcts.friendly_name().as_str(),
            snd,
        );
    }
}

fn battle_ttt(config: &BattleConfig) {
    for samples in &config.samples_per_move {
        println!("samples per move: {}", samples);
        let mut flat_mc = TttFlatMC::new().set_samples_per_move(*samples as u32);
        let mut mcts = TttMCTS::default();
        mcts.config.max_iterations = *samples;

        let mut fst = Vec::with_capacity(100);
        let mut snd = Vec::with_capacity(100);
        for _ in 0..config.num_samples {
            let result = battle_royale(&mut flat_mc, &mut mcts);
            fst.push(result);
            let result = battle_royale(&mut mcts, &mut flat_mc);
            snd.push(result);
        }
        summarize(
            flat_mc.friendly_name().as_str(),
            mcts.friendly_name().as_str(),
            fst,
        );
        summarize(
            mcts.friendly_name().as_str(),
            flat_mc.friendly_name().as_str(),
            snd,
        );
    }
}

fn demo_mcts() {
    let mut mcts = TttMCTS::default();
    mcts.verbose = true;
    mcts.config.max_time = Duration::from_secs(5);
    mcts.config.max_iterations = usize::MAX;
    let mut random: Random<TicTacToe> = Random::new();
    let mut player = ttt::Piece::X;

    let mut state = HashedPosition::new();
    println!("Initial state:\n{}", state);
    while !TicTacToe::is_terminal(&state) {
        if player == ttt::Piece::X {
            let m = mcts.choose_action(&state);
            println!("MCTS player move...");
            state = TicTacToe::apply(state, &m);
        } else {
            let m = random.choose_action(&state);
            println!("Random player move...");
            state = TicTacToe::apply(state, &m);
        }
        println!("State:\n{}", state);

        player = player.next();
    }
    println!("DONE");
}

fn demo_nim() {
    let mut mcts = NimMCTS::default();
    mcts.verbose = true;
    mcts.config.max_time = Duration::from_secs(5);
    mcts.config.max_iterations = usize::MAX;
    let mut random: Random<Nim> = Random::new();
    let mut player = nim::Player::Black;

    let mut state = NimState::new();
    println!("Initial state:\n{:?}", state);
    while !Nim::is_terminal(&state) {
        if player == nim::Player::Black {
            let m = mcts.choose_action(&state);
            println!("{} player move...", mcts.friendly_name());
            state = Nim::apply(state, &m);
        } else {
            let m = random.choose_action(&state);
            println!("{} player move...", random.friendly_name());
            state = Nim::apply(state, &m);
        }
        println!("State:\n{:?}", state);

        player = player.next();
    }
    println!("winner: {:?}", Nim::winner(&state));
}

fn battle_nim(config: &BattleConfig) {
    for samples in &config.samples_per_move {
        println!("samples per move: {}", samples);
        let mut flat_mc = NimFlatMC::new().set_samples_per_move(*samples as u32);
        let mut random = Random::new();

        let mut fst = Vec::with_capacity(100);
        let mut snd = Vec::with_capacity(100);
        for _ in 0..config.num_samples {
            let result = battle_royale(&mut flat_mc, &mut random);
            fst.push(result);
            let result = battle_royale(&mut random, &mut flat_mc);
            snd.push(result);
        }
        summarize(
            flat_mc.friendly_name().as_str(),
            random.friendly_name().as_str(),
            fst,
        );
        summarize(
            random.friendly_name().as_str(),
            flat_mc.friendly_name().as_str(),
            snd,
        );
    }
}

fn _demo_flat_mc() {
    let mut strategy = TttFlatMC::new().verbose();

    let mut state = HashedPosition::new();
    while !TicTacToe::is_terminal(&state) {
        println!("State:\n{}", state);

        let m = strategy.choose_action(&state);
        state = TicTacToe::apply(state, &m);
    }
    println!("State:\n{}", state);
}

fn main() {
    color_backtrace::install();
    pretty_env_logger::init();

    traffic_lights();
    ucd();
    knightthrough();
    breakthrough();
    gonnect();
    atarigo();
    expansion_test();
    ucb_test();

    demo_mcts();
    demo_nim();

    println!("\nTicTacToe");
    println!("--------------------------");
    battle_ttt(&BattleConfig {
        num_samples: 1000,
        samples_per_move: vec![10, 100, 1000, 2000],
    });

    println!("\nNim MCTS");
    println!("--------------------------");
    battle_nim_mcts(&BattleConfig {
        num_samples: 1000,
        samples_per_move: vec![10, 100, 1000, 2000],
    });

    println!("\nNim");
    println!("--------------------------");
    battle_nim(&BattleConfig {
        num_samples: 1000,
        samples_per_move: vec![1, 5, 10, 100, 200],
    });
}
