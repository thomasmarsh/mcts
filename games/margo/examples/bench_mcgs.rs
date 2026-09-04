// Iterations/sec benchmark for Margo's "strong" preset (Ucb1 + mcgs, i.e.
// `GraphSearch::Dag(GraphStats::Both)` + `use_transpositions(true)`) across
// board sizes -- the config route reported to bottleneck badly at n=6/n=7
// despite `random_action` rejection sampling and `cells_key`'s allocation
// fix. Single-threaded and time-budgeted so `search.stats.iter_count` is a
// clean throughput number to compare across board sizes and future changes,
// same methodology as `examples/bench_move_splitting.rs`.
//
// Usage: cargo run --release --example bench_mcgs -- [n ...]
// (defaults to 4 5 6 7)

use std::time::{Duration, Instant};

use game_margo::{Margo, State};
use mcts::algorithms::mcts::{
    node::QInit, profile, select, GraphSearch, GraphStats, SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;

/// Byte-for-byte the same axes `mcts-tune::resolve_graph_search` derives
/// from `mcgs: true` -- `use_transpositions` forced off in favor of the
/// explicit `graph_search: Dag(Both)` mode, `reuse_tree` off too -- so this
/// matches what the "strong" preset (`games/margo/presets.json`) actually
/// builds via `mcts_tune::presets::build_custom`.
fn strong_config() -> TreeSearch<Margo, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/margo-strong")
            .use_transpositions(false)
            .reuse_tree(false)
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .q_init(QInit::Loss)
            // Matches `mcts_tune::search`/`presets`' `PLAYOUT_DEPTH` -- the
            // cap production actually builds `strong`'s search with. Without
            // it, a uniform rollout that hits a repeated-position cycle
            // (Margo only enforces single-position ko, not full positional
            // superko -- see `random_play_smoke_test`'s `#[ignore]` reason)
            // never terminates, since `max_time` is only checked between
            // whole iterations, not mid-playout.
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
