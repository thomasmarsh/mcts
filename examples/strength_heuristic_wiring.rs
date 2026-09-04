// Background strength comparison for the production wiring of the `Strong`
// preset: the newly-wired `Strong` preset (RAVE select/backprop unchanged,
// `DruidHeuristic`-guided playouts at the grid sweep's chosen epsilon=0.5 /
// equal(1,1,1) weights) vs. the previously-shipped config it replaced
// (`Mast`-guided playouts at epsilon=0.7775134, tuner tuned). Both configs
// run with `use_mcts_solver(true)` (matching what's actually shipped both
// before and after this change), so the comparison isolates the playout
// policy swap alone.
//
// Real production budget (3s/move, tree-parallel across all cores), so each
// move already saturates every core -- sequential game execution, same
// rationale as `strength_solver_strong.rs`.
//
// This is a long-running job (each move takes up to the full ~3s budget).
// Run as a background process, not synchronously in-session.
//
// Usage: cargo run --release --example strength_heuristic_wiring
use std::time::Duration;

use game_druid::{Druid, DruidHeuristic, DruidHeuristicWeights, RaveDecisiveHeuristic};
use mcts::algorithms::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn tuned_select() -> select::Rave {
    select::Rave::default()
        .ucb(select::RaveUcb::Ucb1Tuned {
            exploration_constant: 0.2894182,
        })
        .threshold(204)
        .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 })
}

// Previously-shipped Strong config: RaveMastDm (Mast-guided playouts).
fn old_config(
    name: &str,
) -> TreeSearch<
    Druid,
    strategy::Compose<
        select::Rave,
        simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Mast>>,
    >,
> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(3))
            .num_tree_threads(ai_thread_count())
            .select(tuned_select())
            .simulate(
                simulate::DecisiveMove::new()
                    .inner(simulate::EpsilonGreedy::with_epsilon(0.7775134)),
            ),
    )
}

// Newly-wired Strong config: RaveDecisiveHeuristic (DruidHeuristic-guided
// playouts), epsilon=0.5 / equal(1,1,1) weights per the grid sweep.
fn new_config(name: &str) -> TreeSearch<Druid, RaveDecisiveHeuristic> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(3))
            .num_tree_threads(ai_thread_count())
            .select(tuned_select())
            .simulate(
                simulate::DecisiveMove::new().inner(
                    simulate::EpsilonGreedy::default()
                        .epsilon(0.5)
                        .inner(DruidHeuristic::new(DruidHeuristicWeights {
                            block_threat: 1.0,
                            defend_fork: 1.0,
                            threaten_connection: 1.0,
                        })),
                ),
            ),
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
    println!("=== Strong preset: newly-wired DruidHeuristic config vs. previously-shipped Mast config ===");
    println!("Board: 5x5 default (same as server fresh state)");
    println!(
        "Budget: 3s/move, tree-parallel across {} threads (matches build_ai(Strong)), solver on for both",
        ai_thread_count()
    );
    println!("Sequential games, n>=30 games so CI is meaningful.");
    println!();

    let rounds = 15; // 30 games total, sequential
    println!(
        "--- Strong (3s, tree-parallel) : old (Mast) vs new (DruidHeuristic), {} rounds ({} games) ---",
        rounds,
        rounds * 2
    );
    let mut strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(old_config("strong/old-mast")),
        AnySearch::new(new_config("strong/new-heuristic")),
    ];
    let results = round_robin_multiple::<Druid, _>(
        &mut strategies,
        rounds,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );
    for (i, r) in results.iter().enumerate() {
        println!(
            "[Strong] {} : {}",
            strategies[i].friendly_name(),
            fmt_result(r)
        );
    }

    println!();
    println!("=== Summary ===");
    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
    println!("Interpretation: expect the new config to be >= the old one, per the grid");
    println!(
        "sweep finding that DruidHeuristic-guided playouts beat a uniform baseline decisively"
    );
    println!("at this epsilon/weights combo. Small n still gives wide CI; larger n is just more wall-clock.");
}
