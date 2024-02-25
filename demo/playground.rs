use std::time::Duration;

use mcts::game::Game;
use mcts::games::nim;
use mcts::games::ttt;
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::util;
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

type NimMCTS = TreeSearch<Nim, util::Ucb1>;
type TttMCTS = TreeSearch<TicTacToe, util::Ucb1>;

fn traffic_lights() {
    use mcts::games::ttt_traffic_lights::TttTrafficLights;

    type TS = TreeSearch<TttTrafficLights, util::Ucb1GraveMast>;
    let ts = TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(10_000_000)
                .q_init(mcts::strategies::mcts::node::UnvisitedValueEstimate::Parent)
                .expand_threshold(0)
                .select(select::Ucb1Grave {
                    exploration_constant: 2.0f64.sqrt(),
                    threshold: 1000,
                    bias: 764.,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1865)),
        )
        .verbose(true);

    self_play(ts);
}

fn knightthrough() {
    use mcts::games::knightthrough::Knightthrough;

    type TS = TreeSearch<Knightthrough<8, 8>, util::Ucb1GraveMast>;
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

    type TS = TreeSearch<Breakthrough<6, 4>, util::Ucb1GraveMast>;
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

    type TS = TreeSearch<AtariGo<5>, util::Ucb1GraveMast>;
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

    type TS = TreeSearch<Gonnect<7>, util::Ucb1Grave>;
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

    type TS = TreeSearch<ttt::BiddingTicTacToe, util::Ucb1>;

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
    println!("Initial state:\n{}", state.position);
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
        println!("State:\n{}", state.position);

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
        println!("State:\n{}", state.position);

        let m = strategy.choose_action(&state);
        state = TicTacToe::apply(state, &m);
    }
    println!("State:\n{}", state.position);
}

fn main() {
    color_backtrace::install();
    pretty_env_logger::init();

    traffic_lights();
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
