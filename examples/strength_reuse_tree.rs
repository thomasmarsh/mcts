// Background strength/timing comparison for PLAN-WORK.md Session 14: does
// `reuse_tree` (Session 13) actually help once it's wired to something real
// (this session's own AppState persistence)? Deferred from Session 13 for
// exactly this reason -- before Session 14 landed, `reuse_tree` had no
// effect on any code path a comparison could meaningfully exercise.
//
// Replicates server/main.rs's real `Strong` preset config byte-for-byte
// (RAVE-tuned select + DecisiveMove/EpsilonGreedy/DruidHeuristic simulate,
// transpositions on, MCTS-Solver on, tree-parallel across all cores, 3s/move)
// since the server binary isn't a lib target `examples/` can import from --
// same reason Session 4's `strength_solver.rs` duplicated Easy/Medium's
// configs rather than importing `build_ai`. Only `reuse_tree` toggles
// between the two strategies under comparison.
//
// Sequential execution (no rayon fan-out across games) so each tree-parallel
// search gets the whole machine to itself -- same rationale as Sessions
// 10/11/(DRUID)4's background jobs. This is a long-running job (each move is
// a real 3s search, tree-parallel across every core) -- run in the
// background via `nohup`, not synchronously in-session, per Session 11's own
// lesson (a synchronous attempt there got n=4, too few for a meaningful CI).
//
// Usage: cargo run --release --example strength_reuse_tree
use std::time::Duration;

use mcts::games::druid::{
    Druid, DruidHeuristic, DruidHeuristicWeights, HashedState, RaveDecisiveHeuristic,
};
use mcts::strategies::mcts::{node::QInit, select, simulate, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::{round_robin_multiple, AnySearch, Result as GameResult, Verbosity};

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
    let init = HashedState::default(); // 5x5, same as server's fresh page load
    println!("=== Session 14: reuse_tree strength comparison (background job) ===");
    println!(
        "Board: 5x5 default. Real Strong preset config (3s/move, tree-parallel across {} cores).",
        ai_thread_count()
    );
    println!("This is what Session 13 deferred: reuse_tree had no wired caller to compare then.");
    println!("Sequential games, n>=30, per Sessions 10-12's established background-job pattern.");
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
    let results = round_robin_multiple::<Druid, AnySearch<Druid>>(
        &mut strategies,
        rounds,
        &init,
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
    println!("larger n is just more wall-clock. Ran as a background process, not blocking");
    println!("the session, per Sessions 10-12/DRUID-4's established pattern.");
}
