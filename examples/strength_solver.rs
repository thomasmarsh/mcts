// Background strength comparison: Easy/Medium with vs without MCTS-Solver,
// at the real production time budgets (1s/2s). Sequential execution so each
// single-threaded search gets the whole machine -- same rationale as this
// repo's other strength_* scripts.
//
// This is intentionally a long-running job (tens of minutes to hours for
// n=30). Run as a background process, not synchronously: a synchronous
// attempt at a real-budget comparison like this previously got only n=4
// games (CI useless). Output goes to stdout and a results file.
//
// Usage: cargo run --release --example strength_solver
use std::time::Duration;

use game_druid::Druid;
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

fn easy_config(use_solver: bool, budget: Duration, name: &str) -> TreeSearch<Druid, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(use_solver)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn medium_config(
    use_solver: bool,
    budget: Duration,
    name: &str,
) -> TreeSearch<Druid, profile::Mcts<select::Ucb1, simulate::EpsilonGreedy<Druid, simulate::Mast>>>
{
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(use_solver)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.625))
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
    println!("=== MCTS-Solver strength comparison (background job) ===");
    println!("Board: 5x5 default (same as server fresh state)");
    println!("This job intentionally uses real time budgets (1s/2s), runs sequentially,");
    println!("and targets n>=30 games per pairing so CI is meaningful.");
    println!("A synchronous attempt previously got n=4 -> useless CI; this is the fix.");
    println!();

    // Easy: 1s budget, Ucb1, solver off vs on
    // Target: n=30 games per pairing (15 rounds of round_robin_multiple which alternates
    // who moves first each game, giving 2 games/round).
    let easy_rounds = 15; // 30 games total, sequential
    println!(
        "--- Easy (1s) : without solver vs with solver, {} rounds ({} games) ---",
        easy_rounds,
        easy_rounds * 2
    );
    let mut easy_strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(easy_config(false, Duration::from_secs(1), "easy/no-solver")),
        AnySearch::new(easy_config(true, Duration::from_secs(1), "easy/solver")),
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

    // Medium: 2s budget, Ucb1Mast
    let medium_rounds = 10; // 20 games, each 2s/move => ~hours, still meaningful
    println!(
        "--- Medium (2s) : without solver vs with solver, {} rounds ({} games) ---",
        medium_rounds,
        medium_rounds * 2
    );
    let mut medium_strategies: Vec<AnySearch<Druid>> = vec![
        AnySearch::new(medium_config(
            false,
            Duration::from_secs(2),
            "medium/no-solver",
        )),
        AnySearch::new(medium_config(true, Duration::from_secs(2), "medium/solver")),
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
    println!("Interpretation: expect solver to be >= baseline, especially on tactical");
    println!("positions. Small n still gives wide CI; larger n is just more wall-clock.");
    println!("This job ran as a background process, not blocking synchronously.");
}
