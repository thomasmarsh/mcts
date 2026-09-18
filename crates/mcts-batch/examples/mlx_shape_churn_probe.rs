//! Diagnostic probe, no MCTS/search involved: calls
//! `game_othello::convnet::mlx::evaluate_batch` directly, either at a fixed
//! batch size (repeated many times) or cycling through many distinct batch
//! sizes, to isolate whether MLX's memory growth is caused by shape churn
//! itself rather than anything in `mcts-batch`'s search/tree code.
//!
//!   cargo run --release -p mcts-batch --example mlx_shape_churn_probe -- fixed <n> <calls>
//!   cargo run --release -p mcts-batch --example mlx_shape_churn_probe -- churn <max_n> <calls>

use game_othello::{Othello, State};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// A live game plus its move history so `evaluate_batch` sees real
/// (non-terminal, legal-move-bearing) Othello positions rather than a fixed
/// starting position repeated -- shouldn't matter for a shape/memory
/// question, but keeps this probe honest about what it's testing.
fn random_state(rng: &mut SmallRng, plies: usize) -> State {
    let mut s = State::default();
    for _ in 0..plies {
        if Othello::is_terminal(&s) {
            break;
        }
        let mut actions = Vec::new();
        Othello::generate_actions(&s, &mut actions);
        let a = actions[rng.gen_range(0..actions.len())];
        s = Othello::apply(s, &a);
    }
    s
}

fn report(call: usize, n: usize) {
    let (active, cache, peak) = game_othello::convnet::mlx::memory_stats();
    println!(
        "call {call}: n={n} mlx_active={:.1}MB mlx_cache={:.1}MB mlx_peak={:.1}MB",
        active as f64 / 1e6,
        cache as f64 / 1e6,
        peak as f64 / 1e6,
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "fixed".to_string());
    let n_arg: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let calls: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);

    let net = game_othello::convnet::CnnValueNet::default();
    let mut rng = SmallRng::seed_from_u64(42);

    match mode.as_str() {
        "fixed" => {
            // Same batch size every call -- the allocator cache should stay
            // flat for as long as the shape doesn't change.
            let states: Vec<State> = (0..n_arg).map(|_| random_state(&mut rng, 20)).collect();
            for call in 1..=calls {
                let _ = game_othello::convnet::mlx::evaluate_batch(&net, &states);
                report(call, n_arg);
            }
        }
        "churn" => {
            // A distinct, never-repeated batch size every call -- isolates
            // whether shape churn ALONE (no tree search, no self-play loop)
            // reproduces the growth this session's instrumented self-play
            // run showed only at the point a new shape first appeared.
            let clear_each_call = std::env::var("CLEAR_MLX_CACHE_EACH_CALL").is_ok();
            for call in 1..=calls {
                let n = 1 + (call * 7919) % n_arg; // pseudo-random distinct-ish sizes, no repeats needed
                let states: Vec<State> = (0..n).map(|_| random_state(&mut rng, 20)).collect();
                let _ = game_othello::convnet::mlx::evaluate_batch(&net, &states);
                if clear_each_call {
                    game_othello::convnet::mlx::clear_cache();
                }
                report(call, n);
            }
        }
        other => panic!("unknown mode {other}, expected fixed|churn"),
    }
}
