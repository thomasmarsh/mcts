use mcts::game::Game;
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::{MCTSOptions, MonteCarloTreeSearch};
use mcts::strategies::random::Random;
use mcts::strategies::Strategy;
use mcts::util::battle_royale;

use mcts::games::nim::*;
use mcts::games::ttt::*;

type TttFlatMC = FlatMonteCarloStrategy<TicTacToe>;
type NimFlatMC = FlatMonteCarloStrategy<Nim>;
type TttMCTS = MonteCarloTreeSearch<TicTacToe>;

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
    num_samples: u16, // number of games to play
    samples_per_move: Vec<u32>,
}

fn battle_ttt(config: &BattleConfig) {
    for samples in &config.samples_per_move {
        println!("samples per move: {}", samples);
        let mut flat_mc = TttFlatMC::new().set_samples_per_move(*samples);
        let mut random = Random::new();

        let mut fst = Vec::with_capacity(100);
        let mut snd = Vec::with_capacity(100);
        for _ in 0..config.num_samples {
            let result = battle_royale(&mut flat_mc, &mut random);
            fst.push(result);
            let result = battle_royale(&mut random, &mut flat_mc);
            snd.push(result);
        }
        summarize("FlatMC", "Random", fst);
        summarize("Random", "FlatMC", snd);
    }
}

fn battle_nim(config: &BattleConfig) {
    for samples in &config.samples_per_move {
        println!("samples per move: {}", samples);
        let mut flat_mc = NimFlatMC::new().set_samples_per_move(*samples);
        let mut random = Random::new();

        let mut fst = Vec::with_capacity(100);
        let mut snd = Vec::with_capacity(100);
        for _ in 0..config.num_samples {
            let result = battle_royale(&mut flat_mc, &mut random);
            fst.push(result);
            let result = battle_royale(&mut random, &mut flat_mc);
            snd.push(result);
        }
        summarize("FlatMC", "Random", fst);
        summarize("Random", "FlatMC", snd);
    }
}

fn _demo_flat_mc() {
    let mut strategy = TttFlatMC::new().verbose();

    let mut state = HashedPosition::new();
    while TicTacToe::get_winner(&state).is_none() {
        println!("State:\n{}", state.position);

        if let Some(m) = TttFlatMC::choose_move(&mut strategy, &state) {
            if let Some(new_state) = TicTacToe::apply(&mut state, m) {
                state = new_state;
            }
        }
    }
    println!("State:\n{}", state.position);
}

fn _demo_ttt() {
    // TODO: these declarations are out of control
    let mut strategy = TttMCTS::new(MCTSOptions {
        verbose: true,
        max_rollout_depth: 100,
        rollouts_before_expanding: 5,
    });

    let mut state = HashedPosition::new();
    while TicTacToe::get_winner(&state).is_none() {
        if let Some(m) = strategy.choose_move(&state) {
            if let Some(new_state) = TicTacToe::apply(&mut state, m) {
                state = new_state;
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }
}

fn main() {
    println!("\nTicTacToe");
    println!("--------------------------");
    battle_ttt(&BattleConfig {
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
