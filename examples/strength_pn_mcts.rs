// Background strength comparison: MCTS-Solver alone vs MCTS-Solver + UCT-PN
// (Kowalski, Doe, Winands, Górski & Soemers, "Proof Number Based
// Monte-Carlo Tree Search", 2023) at the real production time budgets
// (1s/2s). Sequential execution so each single-threaded search gets the
// whole machine -- same rationale as this repo's other strength_* scripts.
//
// This isolates UCT-PN's own marginal contribution on top of the solver:
// "Final move selection" and "Solving subtrees" from the paper are already
// unconditionally on whenever `use_mcts_solver` is set here (see
// `search/core.rs`'s `proven_win_child` call sites), unlike the paper's
// three independent flags, so both configs below already include those --
// the only difference is `select::Ucb1` vs `select::UctPn`.
//
// This is intentionally a long-running job (tens of minutes to hours for
// n=30). Run as a background process, not synchronously -- see
// strength_solver.rs's doc comment for why.
//
// Usage: cargo run --release --example strength_pn_mcts
use std::time::Duration;

use game_druid::Druid;
use mcts::algorithms::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

fn easy_config(
    c_pn: Option<f64>,
    budget: Duration,
    name: &str,
) -> TreeSearch<Druid, strategy::Ucb1Pn> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .select(select::UctPn::with_c(1.414, c_pn.unwrap_or(0.0))),
    )
}

fn medium_config(
    c_pn: Option<f64>,
    budget: Duration,
    name: &str,
) -> TreeSearch<Druid, strategy::Ucb1PnMast> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .select(select::UctPn::with_c(1.625, c_pn.unwrap_or(0.0)))
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
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
    println!("=== UCT-PN strength comparison (background job) ===");
    println!("Board: 5x5 default (same as server fresh state)");
    println!("Both sides run with use_mcts_solver=true; c_pn=0.0 (left) vs c_pn=1.0 (right)");
    println!("isolates UCT-PN's own contribution -- see strength_solver.rs for solver-vs-off.");
    println!();

    // Easy: 1s budget, Ucb1Pn, c_pn off vs on
    let easy_rounds = 15; // 30 games total, sequential
    println!(
        "--- Easy (1s) : c_pn=0.0 vs c_pn=1.0, {} rounds ({} games) ---",
        easy_rounds,
        easy_rounds * 2
    );
    let mut easy_strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(easy_config(None, Duration::from_secs(1), "easy/c_pn=0")),
        AnySearch::new(easy_config(
            Some(1.0),
            Duration::from_secs(1),
            "easy/c_pn=1",
        )),
    ];
    let easy_results = round_robin_multiple::<Druid, _>(
        &mut easy_strategies,
        easy_rounds,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );
    for (i, r) in easy_results.iter().enumerate() {
        println!(
            "[Easy] {} : {}",
            easy_strategies[i].friendly_name(),
            fmt_result(r)
        );
    }
    println!();

    // Medium: 2s budget, Ucb1PnMast
    let medium_rounds = 10; // 20 games, each 2s/move => ~hours, still meaningful
    println!(
        "--- Medium (2s) : c_pn=0.0 vs c_pn=1.0, {} rounds ({} games) ---",
        medium_rounds,
        medium_rounds * 2
    );
    let mut medium_strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(medium_config(None, Duration::from_secs(2), "medium/c_pn=0")),
        AnySearch::new(medium_config(
            Some(1.0),
            Duration::from_secs(2),
            "medium/c_pn=1",
        )),
    ];
    let medium_results = round_robin_multiple::<Druid, _>(
        &mut medium_strategies,
        medium_rounds,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );
    for (i, r) in medium_results.iter().enumerate() {
        println!(
            "[Medium] {} : {}",
            medium_strategies[i].friendly_name(),
            fmt_result(r)
        );
    }

    println!();
    println!("=== Summary ===");
    println!("Easy comparison (1s):");
    for (i, r) in easy_results.iter().enumerate() {
        println!(
            "  {} : {}",
            easy_strategies[i].friendly_name(),
            fmt_result(r)
        );
    }
    println!("Medium comparison (2s):");
    for (i, r) in medium_results.iter().enumerate() {
        println!(
            "  {} : {}",
            medium_strategies[i].friendly_name(),
            fmt_result(r)
        );
    }
    println!();
    println!("Interpretation: c_pn=0.0 degenerates UctPn's rank bonus to a no-op, so this");
    println!("compares apples-to-apples against plain solver+UCB1 without needing a second");
    println!("strategy type. Small n still gives wide CI; larger n is just more wall-clock.");
    println!("This job ran as a background process, not blocking synchronously.");
}
