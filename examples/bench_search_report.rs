//! Measures final-report construction after retaining small and larger MCTS
//! trees. Run with `cargo run --release --example bench_search_report`; an
//! optional iteration cap keeps ad-hoc runs bounded:
//! `cargo run --release --example bench_search_report -- 25000`.

use std::time::Instant;

use game_druid::{Druid, HashedState};
use mcts::algorithms::mcts::{profile, SearchConfig, TreeSearch};
use mcts::algorithms::Search;

const DEFAULT_LARGE_ITERATIONS: usize = 20_000;
const MAX_ITERATIONS: usize = 100_000;

fn iteration_cap() -> usize {
    let Some(argument) = std::env::args().nth(1) else {
        return DEFAULT_LARGE_ITERATIONS;
    };
    let iterations = argument
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("iteration cap must be a positive integer"));
    assert!(
        (1..=MAX_ITERATIONS).contains(&iterations),
        "iteration cap must be between 1 and {MAX_ITERATIONS}"
    );
    iterations
}

fn measure(label: &str, iterations: usize, seed: u64) {
    let state = HashedState::default();
    let mut search: TreeSearch<Druid, profile::Mcts> = TreeSearch::new().config(
        SearchConfig::new()
            .max_iterations(iterations)
            .expand_threshold(0)
            .seed(seed),
    );
    let search_started = Instant::now();
    let selected = search.choose_action(&state);
    let search_elapsed = search_started.elapsed();
    let report_started = Instant::now();
    let report = search.search_report(&state, &selected);
    let report_elapsed = report_started.elapsed();
    println!(
        "{label:>6}: {iterations:>6} iterations, search {:>8.3} ms, report {:>8.3} ms, {} retained nodes, {} actions",
        search_elapsed.as_secs_f64() * 1_000.0,
        report_elapsed.as_secs_f64() * 1_000.0,
        report.tree_nodes,
        report.actions.len(),
    );
}

fn main() {
    let large = iteration_cap();
    let small = large.min(2_000);
    println!("final-search report construction (Druid 5x5, fixed UCB1 seeds)");
    measure("small", small, 0x5eed);
    measure("large", large, 0x5eee);
}
