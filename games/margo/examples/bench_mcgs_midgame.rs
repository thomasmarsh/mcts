// Same "strong" preset / methodology as `bench_mcgs.rs`, but starting each
// search from a mid-game position instead of the empty board.
// `candidate_is_legal`'s cost scales with group/board occupancy, which is
// smallest at move 1 (bench_mcgs.rs's only sample point) and largest well
// into the game -- so profiling only the opening move is the least
// informative point on the curve for questions about that function's cost.
//
// Usage: cargo run --release --example bench_mcgs_midgame -- [n ...]
// (defaults to 4 5 6 7 8 9 10)
//
// Each size plays a fixed fraction of `total_cells` random legal plies from
// the empty board (same `Margo::random_action` rejection sampling the real
// engine's rollouts use) to reach a reproducible mid-game position, then
// times `strong_config()`'s search from there -- same clean `iter_count /
// elapsed` throughput number `bench_mcgs.rs` reports, just from a different
// starting position.

use std::time::{Duration, Instant};

use game_margo::{Margo, State};
use mcts::game::Game;
use mcts::algorithms::mcts::{
    node::QInit, select, strategy, GraphSearch, GraphStats, SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;
use rand::{rngs::SmallRng, SeedableRng};

/// Byte-for-byte the same config `bench_mcgs.rs::strong_config` builds --
/// kept as a separate copy rather than a shared helper since the two
/// examples are meant to stay independently runnable/readable, matching
/// this crate's existing `examples/*.rs` convention.
fn strong_config() -> TreeSearch<Margo, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/margo-strong-midgame")
            .use_transpositions(false)
            .reuse_tree(false)
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .q_init(QInit::Loss)
            .max_playout_depth(200)
            .max_time(Duration::from_secs(3))
            .select(select::Ucb1::with_c(std::f64::consts::SQRT_2)),
    )
}

/// Plays random legal plies from the empty board until either `target_plies`
/// moves have been made or the game reaches a terminal state first (a small
/// board can run out of legal moves before hitting the target fraction).
/// Seeded for reproducibility across repeated runs/profiling sessions.
fn mid_game_state(n: usize, target_plies: usize, seed: u64) -> State {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut state = State::new(n);
    for _ in 0..target_plies {
        if Margo::is_terminal(&state) {
            break;
        }
        match Margo::random_action(&state, &mut rng) {
            Some(action) => state = Margo::apply(state, &action),
            None => break,
        }
    }
    state
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("board size must be a positive integer"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![4, 5, 6, 7, 8, 9, 10]
    } else {
        sizes
    };

    // ~40% of the board filled -- comfortably past the opening (where group
    // sizes are still tiny) without running into the sparse, capture-heavy
    // endgame where legal-move scarcity itself dominates the cost.
    const FILL_FRACTION: f64 = 0.4;
    const REPS: usize = 3;

    for n in sizes {
        let total_cells = State::new(n).total_cells();
        let target_plies = (total_cells as f64 * FILL_FRACTION) as usize;

        let mut total_iters = 0u64;
        let mut total_elapsed = Duration::ZERO;
        for rep in 0..REPS {
            let seed = rep as u64;
            let state = mid_game_state(n, target_plies, seed);
            let occupied = state.occupied_indices().len();

            let mut search = strong_config();
            let t0 = Instant::now();
            let _action = search.choose_action(&state);
            let elapsed = t0.elapsed();
            let iters = search
                .stats
                .iter_count
                .load(std::sync::atomic::Ordering::Relaxed);
            let iters_per_sec = iters as f64 / elapsed.as_secs_f64();
            println!(
                "n={n} rep={rep} (occupied={occupied}/{total_cells}): {iters} iters in \
                 {elapsed:.2?} ({iters_per_sec:.0} iters/s)"
            );
            total_iters += iters as u64;
            total_elapsed += elapsed;
        }
        let avg_iters_per_sec = total_iters as f64 / total_elapsed.as_secs_f64();
        println!("n={n} average across {REPS} reps: {avg_iters_per_sec:.0} iters/s");
    }
}
