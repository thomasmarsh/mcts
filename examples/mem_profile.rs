// Memory profiling: where does a real Druid game's arena/table footprint
// actually go, by byte category? Reports node/child-slot counts and an
// estimated byte breakdown (`Node<A>` fixed cost, `ChildArray` heap,
// transposition table) every 10 plies through a real self-play game, so
// memory-reduction work (progressive widening, bounded pruning, ...) can be
// ranked against real data instead of a structural guess, and so the same
// measurement can be re-run before/after such a change to check whether it
// actually reduced what it targeted.
//
// Replicates server/main.rs's real `Strong` preset config byte-for-byte
// (RAVE-tuned select, DecisiveMove/EpsilonGreedy/DruidHeuristic simulate,
// transpositions + solver on, tree-parallel across all cores, 3s/move,
// reuse_tree on -- the actual shipped config) since the server binary isn't
// a lib target `examples/` can import from, same reason
// `strength_reuse_tree.rs`/`strength_solver.rs` duplicate it too.
//
// Usage: cargo run --release --example mem_profile
use std::time::Duration;

use game_druid::{
    Druid, DruidHeuristic, DruidHeuristicWeights, HashedState, RaveDecisiveHeuristic,
};
use mcts::strategies::mcts::{node::QInit, select, simulate, MemoryStats, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::game::Game;

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// Byte-for-byte the same config server/main.rs's `build_ai(Strong)` builds
// -- see strength_reuse_tree.rs, which duplicates this same config for the
// same reason.
fn strong_config() -> TreeSearch<Druid, RaveDecisiveHeuristic> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("strong/mem-profile")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(3))
            .num_tree_threads(ai_thread_count())
            .select(
                select::Rave::default()
                    .ucb(select::RaveUcb::Ucb1Tuned {
                        exploration_constant: 0.2894182,
                    })
                    .threshold(204)
                    .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
            )
            .simulate(simulate::DecisiveMove::new().inner(
                simulate::EpsilonGreedy::default().epsilon(0.5).inner(
                    DruidHeuristic::new(DruidHeuristicWeights {
                        block_threat: 1.0,
                        defend_fork: 1.0,
                        threaten_connection: 1.0,
                    }),
                ),
            )),
    )
}

fn report(stats: &MemoryStats, ply: usize) {
    let total_bytes = stats.node_bytes + stats.child_array_heap_bytes + stats.table_bytes;
    let pct = |b: usize| {
        if total_bytes == 0 {
            0.0
        } else {
            100.0 * b as f64 / total_bytes as f64
        }
    };

    println!("--- memory_stats after ply {ply} ---");
    println!(
        "nodes: {} total ({} leaf, {} terminal, {} expanded)",
        stats.total_nodes, stats.leaf_nodes, stats.terminal_nodes, stats.expanded_nodes
    );
    println!(
        "child slots: {} total, {} explored ({:.1}% -- the gap is what progressive widening would avoid allocating)",
        stats.total_child_slots,
        stats.explored_child_slots,
        if stats.total_child_slots == 0 {
            0.0
        } else {
            100.0 * stats.explored_child_slots as f64 / stats.total_child_slots as f64
        }
    );
    println!(
        "{:<24} {:>14} {:>10}",
        "category", "bytes (est.)", "% of total"
    );
    println!(
        "{:<24} {:>14} {:>9.1}%",
        "Node<A> (fixed)",
        stats.node_bytes,
        pct(stats.node_bytes)
    );
    println!(
        "{:<24} {:>14} {:>9.1}%",
        "ChildArray heap",
        stats.child_array_heap_bytes,
        pct(stats.child_array_heap_bytes)
    );
    println!(
        "{:<24} {:>14} {:>9.1}% ({} entries)",
        "transposition table",
        stats.table_bytes,
        pct(stats.table_bytes),
        stats.table_entries
    );
    println!("{:<24} {:>14}", "total (est.)", total_bytes);
    println!();
}

fn main() {
    let mut search = strong_config();
    let mut state = HashedState::default(); // 5x5, same as server's fresh page load
    let mut ply = 0;

    println!("=== mem_profile: real Strong-preset self-play, memory breakdown ===");
    println!(
        "Board: 5x5 default. 3s/move, tree-parallel across {} cores, reuse_tree on.",
        ai_thread_count()
    );
    println!("This is a real production-budget game -- expect several minutes.");
    println!();

    while !Druid::is_terminal(&state) {
        let action = search.choose_action(&state);
        state = Druid::apply(state, &action);
        ply += 1;

        // Report every 10 plies so growth-over-time is visible, not just the
        // final snapshot.
        if ply % 10 == 0 {
            report(&search.memory_stats(), ply);
        }
    }

    println!("=== final ({ply} plies) ===");
    report(&search.memory_stats(), ply);
    println!("arena_len(): {}", search.arena_len());
}
