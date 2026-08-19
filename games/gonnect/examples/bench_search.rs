// `games/gonnect/src/lib.rs`'s `generate_actions` recomputes Go-style
// capture/suicide legality from scratch for every empty cell on every call,
// which `SimulateStrategy::playout` (mcts/src/strategies/mcts/simulate.rs)
// then calls on every ply of every random rollout. This measures the actual
// cost of that: how much it drags down search throughput, and what fraction
// of rollout time it eats -- the reference numbers to compare against once
// `generate_actions`'s legality check gets cheaper.
//
// Two things are reported per board size:
//
// - Search throughput: a fixed-iteration, single-threaded "strong"-preset
//   search from the empty board, timed end to end -> iterations/sec. This is
//   the number that should move once `generate_actions` stops re-flooding
//   every empty cell on every call.
// - A rough breakdown of where a rollout's time goes: `SimulateStrategy::
//   playout` (mcts/src/strategies/mcts/simulate.rs) calls `generate_actions`
//   on every ply of every random rollout, so this replays that same
//   uniform-random-playout shape directly (seeded, terminating games only --
//   see `seeded_random_play` in games/gonnect/src/lib.rs's tests for the
//   same pattern) and reports what fraction of total wall time was spent
//   inside `generate_actions` itself vs. everything else (RNG, `apply`,
//   Vec allocation).
//
// Usage: cargo run --release --example bench_search [iterations]
// `iterations` (default 200) is the fixed iteration count for the search
// throughput measurement -- kept modest by default since Gonnect's current
// unincremental legality check is exactly what's expected to make 19x19
// slow here.
use std::time::{Duration, Instant};

use game_gonnect::{Gonnect, State};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use mcts_tune::SearchBudget;
use rand::{rngs::SmallRng, Rng, SeedableRng};

const ROLLOUT_GAMES: usize = 20;
const ROLLOUT_SEED: u64 = 0;

fn presets() -> PresetTable {
    PresetTable::load(include_str!("../presets.json"), None)
        .expect("games/gonnect/presets.json must parse")
}

/// Fixed-iteration, single-threaded "strong" search from the empty board.
/// Returns (elapsed, iterations/sec).
fn bench_search<const N: usize, const WORDS: usize, const CELLS: usize>(
    iterations: usize,
) -> (Duration, f64) {
    let mut search = presets()
        .build_with::<Gonnect<N, WORDS, CELLS>>("strong", 0, |b: &mut SearchBudget| {
            b.threads = 1;
            b.max_iterations = Some(iterations);
            b.max_time = None;
        })
        .expect("\"strong\" preset must be buildable");
    let state = State::<N, WORDS, CELLS>::default();

    let t0 = Instant::now();
    let _ = search.choose_action(&state);
    let elapsed = t0.elapsed();
    let rate = iterations as f64 / elapsed.as_secs_f64();
    (elapsed, rate)
}

/// Plays `ROLLOUT_GAMES` seeded uniform-random games to completion (same
/// shape as `SimulateStrategy::playout`'s rollouts), timing total wall time
/// against time spent purely inside `generate_actions`. Returns
/// (total_elapsed, generate_actions_elapsed, total_plies).
fn bench_rollouts<const N: usize, const WORDS: usize, const CELLS: usize>(
) -> (Duration, Duration, usize) {
    let mut rng = SmallRng::seed_from_u64(ROLLOUT_SEED);
    let max_plies = N * N * 8 + 32;
    let mut total_plies = 0usize;
    let mut gen_actions_time = Duration::ZERO;

    let t0 = Instant::now();
    for _ in 0..ROLLOUT_GAMES {
        let mut state = State::<N, WORDS, CELLS>::default();
        for _ in 0..max_plies {
            if Gonnect::<N, WORDS, CELLS>::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            let ga0 = Instant::now();
            Gonnect::<N, WORDS, CELLS>::generate_actions(&state, &mut actions);
            gen_actions_time += ga0.elapsed();
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::<N, WORDS, CELLS>::apply(state, &action);
            total_plies += 1;
        }
    }
    let total_elapsed = t0.elapsed();
    (total_elapsed, gen_actions_time, total_plies)
}

fn run_size<const N: usize, const WORDS: usize, const CELLS: usize>(iterations: usize) {
    let (search_elapsed, rate) = bench_search::<N, WORDS, CELLS>(iterations);
    let (rollout_total, rollout_gen_actions, total_plies) = bench_rollouts::<N, WORDS, CELLS>();
    let gen_actions_pct = 100.0 * rollout_gen_actions.as_secs_f64() / rollout_total.as_secs_f64();

    println!("=== {N}x{N} ===");
    println!(
        "  search: {iterations} iterations in {:.3}s -> {:.1} iters/sec",
        search_elapsed.as_secs_f64(),
        rate
    );
    println!(
        "  rollouts: {ROLLOUT_GAMES} games, {total_plies} plies in {:.3}s ({:.1} plies/sec); \
         generate_actions: {:.3}s ({gen_actions_pct:.1}% of rollout time)",
        rollout_total.as_secs_f64(),
        total_plies as f64 / rollout_total.as_secs_f64(),
        rollout_gen_actions.as_secs_f64(),
    );
}

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("iterations must be a positive integer"))
        .unwrap_or(200);

    println!("=== bench_search: Gonnect generate_actions cost baseline ===");
    println!("Fixed-iteration search count: {iterations} (single-threaded, \"strong\" preset).");
    println!("Rollouts: {ROLLOUT_GAMES} seeded uniform-random self-play games per size.");
    println!();

    run_size::<9, 2, 81>(iterations);
    run_size::<13, 3, 169>(iterations);
    run_size::<19, 6, 361>(iterations);
}
