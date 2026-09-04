// Iterations/sec benchmark for Akron's "strong" preset (Ucb1 + mcgs, i.e.
// `GraphSearch::Dag(GraphStats::Both)` + `use_transpositions(true)`) across
// board sizes -- mirrors `games/margo/examples/bench_mcgs.rs` exactly (same
// config-derivation comment applies: byte-for-byte what `mcgs: true` in
// `games/akron/presets.json`'s "strong" preset resolves to via
// `mcts_tune::presets::build_custom`), used to track Akron's own move-gen/
// connectivity cost as board size grows toward `MAX_N` (10).
//
// Usage: cargo run --release --example bench_mcgs -p game-akron -- [n ...]
// (defaults to 4 5 6 7)

use std::time::{Duration, Instant};

use game_akron::{Akron, State};
use mcts::algorithms::mcts::{
    node::QInit, profile, select, GraphSearch, GraphStats, SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;

fn strong_config() -> TreeSearch<Akron, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/akron-strong")
            .use_transpositions(false)
            .reuse_tree(false)
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .q_init(QInit::Loss)
            .max_playout_depth(200)
            .max_time(Duration::from_secs(3))
            .select(select::Ucb1::with_c(std::f64::consts::SQRT_2)),
    )
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("board size must be a positive integer"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![4, 5, 6, 7]
    } else {
        sizes
    };

    for n in sizes {
        let state = State::new(n);
        let mut search = strong_config();
        let t0 = Instant::now();
        let _action = search.choose_action(&state);
        let elapsed = t0.elapsed();
        let iters = search
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let iters_per_sec = iters as f64 / elapsed.as_secs_f64();
        println!("n={n}: {iters} iters in {elapsed:.2?} ({iters_per_sec:.0} iters/s)");
    }
}
