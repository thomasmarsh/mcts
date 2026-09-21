//! Forward-pass cost of a 7x7 128-channel 6-block net at several batch sizes (weights are random,
//! which does not matter for cost).
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example bench_forward -p grid-cnn
//! ```

use grid_cnn::{Geometry, Net, Weights};
use std::time::Instant;

fn main() {
    let geometry = Geometry {
        size: 7,
        in_planes: 7,
        channels: 128,
        blocks: 6,
        policy_planes: 4,
        policy_out: 51,
        value_planes: 2,
        value_hidden: 64,
    };
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let data = (0..geometry.n_weights()).map(|_| 0.05 * next()).collect();
    let net = Net::new(&Weights { geometry, data });
    println!("{} weights", geometry.n_weights());
    for n in [1usize, 8, 32, 64, 128, 256, 512] {
        let planes: Vec<f32> = (0..n * geometry.cells() * geometry.in_planes)
            .map(|_| (next() > 0.2) as u8 as f32)
            .collect();
        for _ in 0..5 {
            net.forward(&planes, n);
        }
        let reps = (2048 / n).max(20);
        let t = Instant::now();
        for _ in 0..reps {
            net.forward(&planes, n);
        }
        let per_call = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "batch {n:>4}: {:8.3} ms per call, {:7.4} ms per position",
            per_call * 1e3,
            per_call * 1e3 / n as f64
        );
    }
}
