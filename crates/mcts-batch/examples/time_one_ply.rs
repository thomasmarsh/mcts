//! Single-`gumbel_explore`-call latency at a given batch size/sims, in
//! isolation from a full multi-ply self-play loop -- useful for isolating
//! whether a throughput regression lives in one simulation round itself or
//! in the surrounding self-play loop (`bench_othello_selfplay.rs`'s own
//! full-game comparison couples both).
//!
//!   cargo run --release -p mcts-batch --example time_one_ply [games] [sims]
use std::time::Instant;

use game_othello::convnet::CnnValueNet;
use game_othello::State;
use mcts_batch::othello::OthelloOracle;
use mcts_batch::{gumbel_explore, Config};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    let oracle = OthelloOracle::new(CnnValueNet::default());
    let cfg = Config { num_simulations: sims as usize, num_considered_actions: 8, ..Config::default() };
    let mut rng = SmallRng::seed_from_u64(1);
    let envs = vec![State::default(); games];

    // warmup
    let _ = gumbel_explore(&cfg, &oracle, &envs, &mut rng);

    let start = Instant::now();
    let iters = 5;
    for _ in 0..iters {
        let _ = gumbel_explore(&cfg, &oracle, &envs, &mut rng);
    }
    let elapsed = start.elapsed();
    println!(
        "{games} games x {sims} sims: {:.1} ms/call ({:.3} ms/game-sim-unit)",
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
        elapsed.as_secs_f64() * 1000.0 / (iters * games) as f64
    );
}
