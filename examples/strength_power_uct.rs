// Background strength comparison: plain UCT vs Power-UCT (Dam et al.,
// "Generalized Mean Estimation in Monte-Carlo Tree Search", IJCAI 2020),
// which replaces each ancestor's Monte-Carlo average with the visit-weighted
// Hölder power mean of its children -- `backprop::PowerMeanBackprop`, see
// `mcts/src/strategies/mcts/backprop.rs`'s `derive_power_mean_value`. The one
// knob is the exponent `p`: `p = 1` is exactly plain UCT (the strategy
// disables its own recompute pass, so it should tie the baseline within
// noise -- a cheap end-to-end check of that structural no-op), and larger `p`
// biases the backup toward the max over children.
//
// Two games spanning the roster, per `plan/selection-backup/phase-a.md`:
//   - Breakthrough 8x8: tactical sudden-death, no draws -- the game Baier &
//     Winands' MCTS-minimax-hybrid papers use, so a max-ward backup bias has
//     a clear place to help or hurt.
//   - Ingenious (2p): a race-to-fill scoring game, where the min-of-two-
//     colors objective makes over-optimistic backups a different kind of
//     risk.
//
// Real time budget, sequential execution (each single-threaded search gets
// the whole machine), same rationale as this repo's other strength_*
// scripts: a synchronous small-n attempt is CI-useless (see
// strength_solver.rs's own note), so this is kept as re-runnable tooling
// (AGENTS.md) rather than a `#[test]`.
//
// Usage: cargo run --release --example strength_power_uct [rounds]
use std::io;
use std::time::Duration;

use game_breakthrough::Breakthrough;
use game_ingenious::Ingenious;
use mcts::game::Game;
use mcts::strategies::mcts::{
    backprop::PowerMeanBackprop, select, simulate, strategy, strategy::Compose, SearchConfig,
    TreeSearch,
};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Power = Compose<select::Ucb1, simulate::Uniform, PowerMeanBackprop>;

const P_VALUES: [f64; 4] = [1.0, 2.0, 4.0, 8.0];

fn baseline_config<G: Game>(budget: Duration) -> TreeSearch<G, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("baseline/uct")
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
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
    for p in P_VALUES {
        strategies.push(AnySearch::new(power_config::<G>(budget, p)));
    }

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

    println!("=== Power-UCT strength comparison (background job) ===");
    println!("Arms: baseline UCT, Power-UCT p in {P_VALUES:?}");
    println!("200ms/move, sequential, round-robin so every pair of arms is checked.");
    println!("p=1 is the structural no-op control -- it should tie the baseline within noise.");
    println!();

    run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", budget, rounds);
    run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", budget, rounds);

    println!("=== done ===");
}
