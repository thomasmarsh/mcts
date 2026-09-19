//! Env steps/sec for `othello-env`, serial vs rayon.
//!
//! `cargo run --release -p othello-env --example throughput -- [n_envs] [rounds] [max_depth]`
//! (defaults 4096 envs, 300 rounds, reset depth U[0, 50]; RAYON_NUM_THREADS sets the pool size).
//!
//! One round is observe -> pick a uniformly random legal action per env -> step -> reset the envs
//! that finished. The random pick is a serial stand-in for the policy and is timed separately.

use othello_env::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

struct Timings {
    observe: Duration,
    pick: Duration,
    step: Duration,
    reset: Duration,
    steps: u64,
    episodes: u64,
}

fn run(n: usize, rounds: usize, max_depth: u32, parallel: bool) -> Timings {
    let mut states = vec![[0; STATE_WORDS]; n];
    reset_random(&mut states, max_depth, 1, parallel);
    let (mut obs, mut mask) = (vec![0f32; n * OBS_LEN], vec![0u8; n * NUM_ACTIONS]);
    let (mut rewards, mut done) = (vec![0f32; n], vec![0u8; n]);
    let mut actions = vec![0u8; n];
    let mut rng = SmallRng::seed_from_u64(2);
    let mut t = Timings { observe: Duration::ZERO, pick: Duration::ZERO, step: Duration::ZERO, reset: Duration::ZERO, steps: 0, episodes: 0 };
    let mut legal = Vec::with_capacity(NUM_ACTIONS);
    for round in 0..rounds {
        let start = Instant::now();
        observe(&states, &mut obs, &mut mask, parallel);
        t.observe += start.elapsed();

        let start = Instant::now();
        for (row, action) in mask.chunks(NUM_ACTIONS).zip(actions.iter_mut()) {
            legal.clear();
            legal.extend((0..NUM_ACTIONS as u8).filter(|&a| row[a as usize] == 1));
            *action = legal[rng.gen_range(0..legal.len())];
        }
        t.pick += start.elapsed();

        let start = Instant::now();
        let rejected = step(&mut states, &actions, &mut rewards, &mut done, parallel);
        t.step += start.elapsed();
        assert_eq!(rejected, 0);
        t.steps += n as u64;

        let finished: Vec<usize> = (0..n).filter(|&i| done[i] == 1).collect();
        t.episodes += finished.len() as u64;
        let start = Instant::now();
        let mut fresh = vec![[0; STATE_WORDS]; finished.len()];
        reset_random(&mut fresh, max_depth, 1000 + round as u64, parallel);
        for (&i, s) in finished.iter().zip(fresh) {
            states[i] = s;
        }
        t.reset += start.elapsed();
    }
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: usize| args.get(i).map_or(default, |s| s.parse().unwrap());
    let (n, rounds, max_depth) = (arg(1, 4096), arg(2, 300), arg(3, 50) as u32);
    println!("n_envs {n}, rounds {rounds}, reset depth U[0,{max_depth}], rayon threads {}", rayon::current_num_threads());
    run(n, 20, max_depth, true); // warm up the pool
    for parallel in [false, true] {
        let t = run(n, rounds, max_depth, parallel);
        let total = t.observe + t.pick + t.step + t.reset;
        let per_sec = |d: Duration| t.steps as f64 / d.as_secs_f64();
        println!(
            "{:>8}: {:>10.0} steps/s end-to-end | step {:>10.0}/s | observe {:>10.0}/s | pick {:>10.0}/s | reset {:>4.1}% of wall | {} episodes",
            if parallel { "rayon" } else { "serial" },
            per_sec(total),
            per_sec(t.step),
            per_sec(t.observe),
            per_sec(t.pick),
            100.0 * t.reset.as_secs_f64() / total.as_secs_f64(),
            t.episodes,
        );
    }
}
