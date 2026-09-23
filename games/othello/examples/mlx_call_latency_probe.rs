//! Isolates one MLX forward-call's wall-clock cost from any tree-search or
//! threading overhead, at a few `states.len()` batch sizes, so the
//! architecture-audit prototype (`mlx_tree_batch_bench.rs`) can tell whether
//! its throughput is bounded by per-call MLX/weight-upload overhead or by its
//! own queue/thread machinery.
//!
//! cargo run --release -p game-othello --features mlx --example mlx_call_latency_probe

use std::time::Instant;

use game_othello::convnet::mlx::{all_policy_logits, evaluate_batch, value};
use game_othello::convnet::CnnValueNet;
use game_othello::State;

fn main() {
    let net = CnnValueNet::default();
    let state = State::default();

    let reps = 200;
    let start = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(value(&net, &state));
    }
    println!("value() alone: {:.4}ms/call", start.elapsed().as_secs_f64() * 1000.0 / reps as f64);

    let start = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(all_policy_logits(&net, &state));
    }
    println!("all_policy_logits() alone: {:.4}ms/call", start.elapsed().as_secs_f64() * 1000.0 / reps as f64);

    for &batch in &[1usize, 8, 32, 128, 800] {
        let states: Vec<State> = (0..batch).map(|_| state).collect();
        let warmup = 5;
        for _ in 0..warmup {
            std::hint::black_box(evaluate_batch(&net, &states, 64));
        }
        let reps = if batch >= 128 { 10 } else { 50 };
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(evaluate_batch(&net, &states, 64));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!(
            "evaluate_batch(batch={batch}): {:.4}ms/call total, {:.4}ms/item",
            elapsed,
            elapsed / batch as f64
        );
    }
}
