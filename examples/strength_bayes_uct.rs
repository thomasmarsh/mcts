// Background strength comparison: classic UCT (c = sqrt(2)) vs Bayes-UCT2
// (Tesauro, Rajan & Segal 2010, "Bayesian Inference in Monte-Carlo Tree
// Search", UAI), paired with the `BayesGaussian` (Clark's closed-form
// max-of-Gaussians) posterior-propagation backprop -- see
// `mcts/src/strategies/mcts/select/bayes.rs` and `backprop.rs`'s
// `BayesGaussian` for the math. Fixed iteration budget on both sides (not
// time), so the comparison isolates search-quality-per-playout rather than
// raw speed -- Bayesian backprop's extra per-node work (Clark's formula
// folded over children every backup) makes it slower per iteration, which a
// time-based budget would unfairly penalize.
//
// Board: Gonnect's default size (`game_gonnect::DEFAULT_SIZE`, 13x13) --
// board size is a runtime field on `State` rather than a const generic, and
// `round_robin_multiple` always starts games from `G::S::default()`, so this
// always plays on the default size rather than a size this file can pick
// itself.
//
// Usage: cargo run --release --example strength_bayes_uct [c] [rounds]
// `c` (default 1.0) is Bayes-UCT2's exploration constant, matching classic
// UCT's `c=sqrt(2)` on the other side when swept to that value -- lets a
// caller check whether the paper's fixed `c=1` default (an untuned choice
// for this codebase's conventions) is itself costing Bayes-UCT2 games
// against a tuned opponent, independent of the domain-correlation gap.
use game_gonnect::Gonnect;
use mcts::strategies::mcts::{node::QInit, select, strategy::Compose, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type G7 = Gonnect;

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
    let args: Vec<String> = std::env::args().collect();
    let bayes_c: f64 = args
        .get(1)
        .map(|s| s.parse().expect("c must be a float"))
        .unwrap_or(1.0);
    let rounds: usize = args
        .get(2)
        .map(|s| s.parse().expect("rounds must be an int"))
        .unwrap_or(20);
    let iterations: usize = 20_000;

    println!("=== Bayes-UCT2 (c={bayes_c}) vs classic UCT (c=sqrt2) strength comparison ===");
    println!(
        "Board: Gonnect, default size ({}x{})",
        game_gonnect::DEFAULT_SIZE,
        game_gonnect::DEFAULT_SIZE
    );
    println!("Fixed budget: {} iterations/move, both sides", iterations);
    println!("{} rounds ({} games total)", rounds, rounds * 2);
    println!();

    let ucb1 =
        TreeSearch::<G7, Compose<select::Ucb1, mcts::strategies::mcts::simulate::Uniform>>::new()
            .config(
                SearchConfig::new()
                    .name("ucb1/c=sqrt2")
                    .max_iterations(iterations)
                    .q_init(QInit::Infinity)
                    .select(select::Ucb1::with_c(std::f64::consts::SQRT_2)),
            );

    let bayes_uct2 = TreeSearch::<
        G7,
        Compose<
            select::BayesUct2,
            mcts::strategies::mcts::simulate::Uniform,
            mcts::strategies::mcts::backprop::BayesGaussian,
        >,
    >::new()
    .config(
        SearchConfig::new()
            .name(&format!("bayes_uct2/c={bayes_c}"))
            .max_iterations(iterations)
            .q_init(QInit::Infinity)
            .select(select::BayesUct2::with_c(bayes_c))
            .backprop(mcts::strategies::mcts::backprop::BayesGaussian::default()),
    );

    let mut strategies: Vec<AnySearch<G7>> = vec![AnySearch::new(ucb1), AnySearch::new(bayes_uct2)];

    let results = round_robin_multiple::<G7, _>(
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
    println!("Interpretation: both sides run the same fixed iteration budget per move, so");
    println!("any win-rate skew reflects search quality per playout, not wall-clock speed.");
    println!("This job ran as a background process, not blocking synchronously.");
}
