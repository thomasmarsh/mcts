use std::time::Duration;

use mcts::game::Game;
use mcts::games::nim;
use mcts::games::ttt;
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::random::Random;
use mcts::strategies::Search;
use mcts::util::battle_royale;

use mcts::games::nim::*;
use mcts::games::ttt::*;

type TttFlatMC = FlatMonteCarloStrategy<TicTacToe>;
type NimFlatMC = FlatMonteCarloStrategy<Nim>;

type NimMCTS = TreeSearch<Nim, util::Ucb1>;
type TttMCTS = TreeSearch<TicTacToe, util::Ucb1>;

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
        let mut mcts = NimMCTS::default();
        mcts.strategy.max_iterations = *samples;
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
        mcts.strategy.max_iterations = *samples;

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
    mcts.strategy.max_time = Duration::from_secs(5);
    mcts.strategy.max_iterations = usize::MAX;
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
    mcts.strategy.max_time = Duration::from_secs(5);
    mcts.strategy.max_iterations = usize::MAX;
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
