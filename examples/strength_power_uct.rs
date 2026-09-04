// Background strength comparison: plain UCT vs Power-UCT (Dam et al.,
// "Generalized Mean Estimation in Monte-Carlo Tree Search", IJCAI 2020),
// which replaces each ancestor's Monte-Carlo average with the visit-weighted
// Hölder power mean of its children -- `backprop::PowerMeanBackprop`, see
// `mcts/src/strategies/mcts/backprop.rs`'s `derive_power_mean_value`. The one
// knob is the exponent `p`: `p = 1` is exactly plain UCT (the strategy
// disables its own recompute pass, so it should tie the baseline within
// noise -- a cheap end-to-end check of that structural no-op), and larger `p`
// biases the backup toward the max over children. A second knob `alpha`
// blends the power mean with the plain max (`alpha = 1` is the Full-Bellman
// max backup, Asai & Wissow AAAI 2025); the mixed arms sweep its interior.
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
use mcts::algorithms::mcts::{
    backprop::PowerMeanBackprop, profile, profile::Mcts, select, simulate, SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;
use mcts::game::Game;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Power = Mcts<select::Ucb1, simulate::Uniform, PowerMeanBackprop>;

const P_VALUES: [f64; 4] = [1.0, 2.0, 4.0, 8.0];

// (p, alpha) pairs: the pure power-mean sweep above plus the mixed middle
// ground the EVT paper (Asai & Wissow, AAAI 2025 / arXiv 2405.18248 §6)
// points at -- a mean<->max blend inside `derive_power_mean_value`. Pure max
// (alpha = 1) is expected to help only the weaker configs, so the arms sweep
// the interior.
const MIXED_ARMS: [(f64, f64); 3] = [(1.0, 0.25), (1.0, 0.5), (4.0, 0.5)];

fn baseline_config<G: Game>(budget: Duration) -> TreeSearch<G, profile::Mcts> {
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

fn power_config_mixed<G: Game>(budget: Duration, p: f64, alpha: f64) -> TreeSearch<G, Power> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("power_uct/p={p},a={alpha}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414))
            .backprop(PowerMeanBackprop::new_mixed(p, alpha, None)),
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
    for (p, alpha) in MIXED_ARMS {
        strategies.push(AnySearch::new(power_config_mixed::<G>(budget, p, alpha)));
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
    println!("Arms: baseline UCT, Power-UCT p in {P_VALUES:?}, mixed (p,alpha) in {MIXED_ARMS:?}");
    println!("200ms/move, sequential, round-robin so every pair of arms is checked.");
    println!("p=1 is the structural no-op control -- it should tie the baseline within noise.");
    println!();

    run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", budget, rounds);
    run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", budget, rounds);

    println!("=== done ===");
}
