// Playouts/sec benchmark for the linearized (move-splitting) Druid
// representation. Measures iterations/sec at 5x5 and 9x9, comparing against
// the previous commit (flat `Move(Piece, u8)` before the linearization).
//
// Methodology:
//   - Build the benchmark from the parent commit (flat moves)
//   - Build the same benchmark from this commit (linearized moves)
//   - Run each on 5x5 and 9x9, 3 runs of 5s each, report avg iters/s
//
// Usage: cargo run --release --example bench_move_splitting
use std::time::{Duration, Instant};

use mcts::game::Game;
use mcts::games::druid::{Druid, HashedState, Size};
use mcts::strategies::mcts::{
    node::QInit, select, simulate, strategy, SearchConfig, Strategy, TreeSearch,
};
use mcts::strategies::Search;

/// Shipped Strong preset strategy shape: Ucb1 select + DecisiveMove wrapping
/// EpsilonGreedy wrapping NST. The struct lives in server/adapters/druid.rs
/// (not the lib crate), so we define a local equivalent.
type Ucb1DmNstLocal = strategy::Compose<select::Ucb1, simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>>;

/// Shipped Strong config with the given time budget and tree thread count.
fn strong_config(budget: Duration, tree_threads: usize) -> TreeSearch<Druid, Ucb1DmNstLocal> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/strong")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .num_tree_threads(tree_threads)
            .select(select::Ucb1::with_c(1.414))
            .simulate(simulate::DecisiveMove::new().inner(
                simulate::EpsilonGreedy::default()
                    .epsilon(0.3)
                    .inner(simulate::Nst::new().backoff_threshold(5)),
            )),
    )
}

/// Plain Ucb1-only config (no NST/DecisiveMove), for direct comparison of
/// raw iteration throughput without simulate overhead.
fn plain_ucb1_config(budget: Duration) -> TreeSearch<Druid, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/ucb1")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(false)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn benchmark<G, S>(label: &str, board_label: &str, search: &mut TreeSearch<G, S>, state: &G::S)
where
    G: Game,
    S: Strategy<G>,
    <G as Game>::A: std::fmt::Debug,
    G::S: std::fmt::Debug + Clone,
{
    const RUNS: usize = 3;

    let mut total_iters = 0usize;
    let mut total_time = Duration::ZERO;

    for run in 0..RUNS {
        let state_clone = state.clone();
        let t0 = Instant::now();
        let _action = search.choose_action(&state_clone);
        let elapsed = t0.elapsed();
        let iters = search.stats.iter_count.load(std::sync::atomic::Ordering::Relaxed);
        total_iters += iters;
        total_time += elapsed;
        println!(
            "  run {}: {} iters in {:.2}s ({:.0} iters/s)",
            run + 1,
            iters,
            elapsed.as_secs_f64(),
            iters as f64 / elapsed.as_secs_f64()
        );
    }

    let avg_iters_per_s = total_iters as f64 / total_time.as_secs_f64();
    println!(
        "  [{label}/{board_label}] avg {:.0} iters/s across {RUNS} runs (total {total_iters} iters in {:.2}s)",
        avg_iters_per_s,
        total_time.as_secs_f64(),
    );
}

fn main() {
    println!("=== bench_move_splitting: playouts/sec with linearized move representation ===");
    println!();
    println!("Configs tested:");
    println!("  ucb1:       Ucb1 select only, 5s budget");
    println!("  strong:     shipped Strong preset (Ucb1 + DecisiveMove<EpsilonGreedy<Nst>>), 5s budget, 1 thread");
    println!("  strong/par: same but tree-parallel across all cores");
    println!();

    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // --- 5x5 ---
    let board_5x5 = Size { w: 5, h: 5 };
    let state_5x5 = HashedState::new(board_5x5);

    println!("--- 5x5 ---");
    let mut ucb1_5x5 = plain_ucb1_config(Duration::from_secs_f64(5.0));
    benchmark("ucb1", "5x5", &mut ucb1_5x5, &state_5x5);

    let mut strong_5x5_st = strong_config(Duration::from_secs_f64(5.0), 1);
    benchmark("strong", "5x5", &mut strong_5x5_st, &state_5x5);

    let mut strong_5x5_par = strong_config(Duration::from_secs_f64(5.0), thread_count);
    benchmark("strong/par", "5x5", &mut strong_5x5_par, &state_5x5);

    // --- 9x9 ---
    let board_9x9 = Size { w: 9, h: 9 };
    let state_9x9 = HashedState::new(board_9x9);

    println!("--- 9x9 ---");
    let mut ucb1_9x9 = plain_ucb1_config(Duration::from_secs_f64(5.0));
    benchmark("ucb1", "9x9", &mut ucb1_9x9, &state_9x9);

    let mut strong_9x9_st = strong_config(Duration::from_secs_f64(5.0), 1);
    benchmark("strong", "9x9", &mut strong_9x9_st, &state_9x9);

    let mut strong_9x9_par = strong_config(Duration::from_secs_f64(5.0), thread_count);
    benchmark("strong/par", "9x9", &mut strong_9x9_par, &state_9x9);

    println!();
    println!("=== done ===");
}