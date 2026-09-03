// Background strength comparison: plain UCT vs Sarsa-UCT(λ) / TD(λ)
// (Vodopivec, Samothrakis, Šter, "On Monte Carlo Tree Search and
// Reinforcement Learning", JAIR 2017), which replaces each ancestor's
// stale cumulative Monte-Carlo average with a truncated λ-return that
// bootstraps from its on-path child's *current* estimate --
// `backprop::TdBackprop`, see `mcts/src/strategies/mcts/backprop.rs`.
//
// The knob is `lambda`: `lambda = 1` is exactly plain UCT (the strategy
// returns `None` from `td_lambda`, so it should tie the baseline within
// noise -- a cheap end-to-end check of that structural no-op), and lower
// values bootstrap more aggressively. Vodopivec's guidance for adversarial
// games: the useful band is [0.8, 1.0]; one low arm is included to show the
// expected degradation. `max_child` switches the bootstrap from the on-path
// child (Sarsa) to `max` over children (MaxMCTS(λ), Khandelwal et al. ICML
// 2016). A `power_uct/p=4` arm (session A1) is included so the round-robin
// compares TD vs power-mean vs baseline directly.
//
// Two games spanning the roster:
//   - Breakthrough 8x8: tactical sudden-death, no draws.
//   - Ingenious (2p): a race-to-fill scoring game.
//
// Real time budget, sequential execution, same rationale as this repo's
// other strength_* scripts: a synchronous small-n attempt is CI-useless, so
// this is kept as re-runnable tooling (AGENTS.md), not a `#[test]`.
//
// Usage: cargo run --release --example strength_sarsa_uct [rounds]
use std::io;
use std::time::Duration;

use game_breakthrough::Breakthrough;
use game_ingenious::Ingenious;
use mcts::game::Game;
use mcts::algorithms::mcts::{
    backprop::{PowerMeanBackprop, TdBackprop},
    select, simulate, strategy,
    strategy::Compose,
    SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Td = Compose<select::Ucb1, simulate::Uniform, TdBackprop>;
type Power = Compose<select::Ucb1, simulate::Uniform, PowerMeanBackprop>;

// lambda = 1.0 is the structural no-op control (should tie the baseline).
const LAMBDAS: [f64; 5] = [1.0, 0.9, 0.8, 0.6, 0.3];
// MaxMCTS(λ): bootstrap from max over children. One mid-band arm.
const MAX_CHILD_LAMBDAS: [f64; 2] = [0.9, 0.7];

fn baseline_config<G: Game>(budget: Duration) -> TreeSearch<G, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("baseline/uct")
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn td_config<G: Game>(budget: Duration, lambda: f64) -> TreeSearch<G, Td> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("sarsa_uct/l={lambda}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414))
            .backprop(TdBackprop::new(lambda, false)),
    )
}

fn td_config_max<G: Game>(budget: Duration, lambda: f64) -> TreeSearch<G, Td> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("maxmcts/l={lambda}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414))
            .backprop(TdBackprop::new(lambda, true)),
    )
}

fn power_config<G: Game>(budget: Duration, p: f64) -> TreeSearch<G, Power> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("power_uct/p={p}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414))
            .backprop(PowerMeanBackprop::new(p, None)),
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

fn run_game<G: Game + Clone>(label: &str, budget: Duration, rounds: usize)
where
    G::S: Sync,
{
    println!(
        "--- {label} ({} rounds, {} games per pair) ---",
        rounds,
        rounds * 2
    );
    let mut strategies: Vec<AnySearch<G>> = vec![AnySearch::new(baseline_config::<G>(budget))];
    for lambda in LAMBDAS {
        strategies.push(AnySearch::new(td_config::<G>(budget, lambda)));
    }
    for lambda in MAX_CHILD_LAMBDAS {
        strategies.push(AnySearch::new(td_config_max::<G>(budget, lambda)));
    }
    strategies.push(AnySearch::new(power_config::<G>(budget, 4.0)));

    let results = round_robin_multiple::<G, _>(
        &mut strategies,
        rounds,
        &mut io::stdout(),
        Verbosity::Silent,
    );

    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("rounds must be an int"))
        .unwrap_or(50);
    let budget = Duration::from_millis(200);

    println!("=== Sarsa-UCT(λ) strength comparison (background job) ===");
    println!("Arms: baseline UCT, Sarsa-UCT l in {LAMBDAS:?}, MaxMCTS l in {MAX_CHILD_LAMBDAS:?}, power_uct/p=4");
    println!("200ms/move, sequential, round-robin so every pair of arms is checked.");
    println!("l=1 is the structural no-op control -- it should tie the baseline within noise.");
    println!();

    run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", budget, rounds);
    run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", budget, rounds);

    println!("=== done ===");
}
