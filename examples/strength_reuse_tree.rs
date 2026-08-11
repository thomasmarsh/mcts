// Background strength/timing comparison: does `reuse_tree` actually help
// once it's wired to something real (AppState persistence across moves)?
// Before that wiring landed, `reuse_tree` had no effect on any code path a
// comparison could meaningfully exercise.
//
// Replicates server/main.rs's real `Strong` preset config byte-for-byte
// (RAVE-tuned select + DecisiveMove/EpsilonGreedy/DruidHeuristic simulate,
// transpositions on, MCTS-Solver on, tree-parallel across all cores, 3s/move)
// since the server binary isn't a lib target `examples/` can import from --
// same reason `strength_solver.rs` duplicated Easy/Medium's
// configs rather than importing `build_ai`. Only `reuse_tree` toggles
// between the two strategies under comparison.
//
// Sequential execution (no rayon fan-out across games) so each tree-parallel
// search gets the whole machine to itself -- same rationale as this repo's
// other background strength jobs. This is a long-running job (each move is
// a real 3s search, tree-parallel across every core) -- run in the
// background via `nohup`, not synchronously: a synchronous attempt at this
// kind of comparison previously got n=4, too few for a meaningful CI.
//
// Usage: cargo run --release --example strength_reuse_tree
use std::time::Duration;

use game_druid::{
    Druid, DruidHeuristic, DruidHeuristicWeights, RaveDecisiveHeuristic,
};
use mcts::strategies::mcts::{node::QInit, select, simulate, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::bench::tournament::{round_robin_multiple, Result as GameResult};
use mcts::util::{AnySearch, Verbosity};

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// Byte-for-byte the same config server/main.rs's `build_ai(Strong)` builds,
// with `reuse_tree` as the one deliberate variable under test.
fn strong_config(reuse_tree: bool, name: &str) -> TreeSearch<Druid, RaveDecisiveHeuristic> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(reuse_tree)
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

fn fmt_result(r: &GameResult) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson)",
        r.wins,
        r.losses,
        r.draws,
        r.total(),
        point * 100.0,
        lo * 100.0,
        hi * 100.0
    )
}

fn main() {
    println!("=== reuse_tree strength comparison (background job) ===");
    println!(
        "Board: 5x5 default. Real Strong preset config (3s/move, tree-parallel across {} cores).",
        ai_thread_count()
    );
    println!("reuse_tree previously had no wired caller to compare against.");
    println!("Sequential games, n>=30, per this repo's established background-job pattern.");
    println!();

    let rounds = 15; // 30 games total, sequential, alternating who moves first
    println!(
        "--- Strong (3s, tree-parallel): reuse off vs reuse on, {} rounds ({} games) ---",
        rounds,
        rounds * 2
    );
    let mut strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(strong_config(false, "strong/reuse-off")),
        AnySearch::new(strong_config(true, "strong/reuse-on")),
    ];
    let results = round_robin_multiple::<Druid, _>(
        &mut strategies,
        rounds,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );

    println!();
    println!("=== Summary ===");
    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
    println!("Interpretation: reuse_tree carries forward accumulated search across a game's");
    println!("moves instead of discarding the tree every move -- expect reuse-on to be >=");
    println!("reuse-off, since it's strictly more effective search per move at the same");
    println!("wall-clock budget, not a different algorithm. Small n still gives wide CI;");
    println!("larger n is just more wall-clock. Ran as a background process, per this");
    println!("repo's established pattern for long-running strength comparisons.");
}
