// Storage quantization payoff: how much does gating AMAF/solver storage on
// `Requirements`/`use_mcts_solver` actually save, for a fixed search budget?
// `ChildArrayData`/`NodeStatsData` only allocate their AMAF side table when
// `Requirements.amaf` is set, and `Node` only allocates its solver side
// block when `SearchConfig::use_mcts_solver` is on -- this reports the real
// byte numbers for both, not just "it compiles".
//
// Each axis (AMAF, solver) is measured independently, feature off vs on,
// same fixed iteration budget, single-threaded (default `num_tree_threads`)
// for reproducibility -- a time budget would let thread scheduling and
// iteration count vary between runs, which is not what this script is
// measuring. `TreeSearch::memory_stats` is the same accessor-mediated
// introspection `mem_profile.rs` uses; `solver_bytes` (added alongside this
// script) is its per-node analogue of `child_array_heap_bytes`'s AMAF term.
//
// Usage: cargo run --release --example mem_quantization
use game_druid::{Druid, HashedState};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

const ITERATIONS: usize = 20_000;

fn report(label: &str, stats: &mcts::strategies::mcts::MemoryStats) {
    let child_bytes = stats.child_array_heap_bytes;
    println!(
        "{label:<20} nodes={:<7} node_bytes={:<10} child_array_heap_bytes={:<10} solver_bytes={:<9} total={}",
        stats.total_nodes,
        stats.node_bytes,
        child_bytes,
        stats.solver_bytes,
        stats.node_bytes + child_bytes + stats.solver_bytes
    );
}

fn main() {
    println!("=== mem_quantization: AMAF/solver storage-gating payoff ===");
    println!("Board: 5x5 default. {ITERATIONS} iterations/move, single-threaded, fixed budget.");
    println!();

    println!("--- AMAF axis (select::Amaf vs select::Ucb1, both solver off) ---");
    let mut amaf_off = TreeSearch::<Druid, strategy::Ucb1>::new().config(
        SearchConfig::new()
            .name("amaf-off")
            .expand_threshold(1)
            .q_init(QInit::Infinity)
            .max_iterations(ITERATIONS),
    );
    let mut amaf_on = TreeSearch::<Druid, strategy::Amaf>::new().config(
        SearchConfig::new()
            .name("amaf-on")
            .expand_threshold(1)
            .q_init(QInit::Infinity)
            .max_iterations(ITERATIONS),
    );
    let state = HashedState::default();
    amaf_off.choose_action(&state);
    amaf_on.choose_action(&state);
    report("amaf=off", &amaf_off.memory_stats());
    report("amaf=on", &amaf_on.memory_stats());
    println!();

    println!("--- Solver axis (use_mcts_solver off vs on, both plain Ucb1 select) ---");
    let mut solver_off = TreeSearch::<Druid, strategy::Ucb1>::new().config(
        SearchConfig::new()
            .name("solver-off")
            .expand_threshold(1)
            .q_init(QInit::Infinity)
            .use_mcts_solver(false)
            .max_iterations(ITERATIONS),
    );
    let mut solver_on = TreeSearch::<Druid, strategy::Ucb1>::new().config(
        SearchConfig::new()
            .name("solver-on")
            .expand_threshold(1)
            .q_init(QInit::Infinity)
            .use_mcts_solver(true)
            .max_iterations(ITERATIONS),
    );
    solver_off.choose_action(&state);
    solver_on.choose_action(&state);
    report("solver=off", &solver_off.memory_stats());
    report("solver=on", &solver_on.memory_stats());
    println!();

    println!("=== Interpretation ===");
    println!("child_array_heap_bytes should be lower for amaf=off than amaf=on (the");
    println!("Vec<ActionStats> side table on ChildArrayData/NodeStatsData is Vec::new()");
    println!("when unused). solver_bytes should be 0 for solver=off and > 0 for solver=on");
    println!("(the Option<Box<SolverState>> is None when unused). node_bytes is the same");
    println!("across all four runs -- it's Node<A>'s fixed type size, not a per-run");
    println!("allocation, so it doesn't move with either axis.");
}
