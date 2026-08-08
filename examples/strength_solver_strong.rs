// Background strength comparison for the tree-parallel `Strong` preset:
// MCTS-Solver off vs on, at the real production time budget (3s/move,
// `ai_thread_count()` worker threads per move -- same config as
// `server/main.rs`'s `build_ai(Strong)`). Sequential game execution: each
// move already saturates every core via tree parallelism, so there's nothing
// to gain (and real risk of oversubscription) from also parallelizing across
// games.
//
// This is a long-running job (each move takes the full ~3s budget regardless
// of thread count, so wall-clock scales with moves-per-game x games, not
// with core count). Run as a background process, not synchronously in-session.
//
// Usage: cargo run --release --example strength_solver_strong
use std::time::Duration;

use mcts::games::druid::Druid;
use mcts::strategies::mcts::{select, simulate, strategy, node::QInit, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::bench::tournament::{round_robin_multiple, Result as GameResult};
use mcts::util::{AnySearch, Verbosity};

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn strong_config(use_solver: bool, name: &str) -> TreeSearch<Druid, strategy::RaveMastDm<Druid>> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(use_solver)
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
            .simulate(
                simulate::DecisiveMove::new()
                    .inner(simulate::EpsilonGreedy::with_epsilon(0.7775134)),
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
    println!("=== Strong preset MCTS-Solver strength comparison (background job) ===");
    println!("Board: 5x5 default (same as server fresh state)");
    println!(
        "Budget: 3s/move, tree-parallel across {} threads (matches build_ai(Strong))",
        ai_thread_count()
    );
    println!("Sequential games, n>=20 games so CI is meaningful.");
    println!();

    let rounds = 10; // 20 games total, sequential
    println!(
        "--- Strong (3s, tree-parallel) : without solver vs with solver, {} rounds ({} games) ---",
        rounds,
        rounds * 2
    );
    let mut strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(strong_config(false, "strong/no-solver")),
        AnySearch::new(strong_config(true, "strong/solver")),
    ];
    let results = round_robin_multiple::<Druid, _>(
        &mut strategies,
        rounds,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );
    for (i, r) in results.iter().enumerate() {
        println!("[Strong] {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }

    println!();
    println!("=== Summary ===");
    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
    println!("Interpretation: expect solver to be >= baseline, especially on tactical");
    println!("positions. Small n still gives wide CI; larger n is just more wall-clock.");
}
