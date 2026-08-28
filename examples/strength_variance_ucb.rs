// Background strength comparison: plain UCT vs the variance-aware bandits
// UCB-V (Audibert, Munos, Szepesvári, "Exploration-exploitation tradeoff
// using variance estimates in multi-armed bandits", TCS 2009) and KL-UCB
// (Garivier & Cappé, "The KL-UCB algorithm for bounded stochastic bandits
// and beyond", COLT 2011) -- both drop-in `Ucb1` replacements in the tree
// policy, reading each child's already-tracked `sum_squared_score` for the
// variance estimate. See `mcts/src/strategies/mcts/select/variance.rs`.
//
// UCB-V scales the exploration term by observed child variance; its knob `c`
// scales the range/bias term (the paper's `3b`). KL-UCB returns the tightest
// index-policy upper confidence bound directly; its knob `c` scales the
// second-order `ln ln N` term -- the paper fixes `c = 3` for the proof and
// notes `c = 0` works well in practice, so the sweep brackets both.
//
// Two games spanning the roster, same as this repo's other strength_*
// scripts:
//   - Breakthrough 8x8: tactical sudden-death, no draws.
//   - Ingenious (2p): a race-to-fill scoring game.
//
// Real time budget, sequential execution, same rationale as the other
// strength_* scripts: a synchronous small-n attempt is CI-useless, so this
// is kept as re-runnable tooling (AGENTS.md), not a `#[test]`.
//
// Usage: cargo run --release --example strength_variance_ucb [rounds]
use std::io;
use std::time::Duration;

use game_breakthrough::Breakthrough;
use game_ingenious::Ingenious;
use mcts::game::Game;
use mcts::strategies::mcts::{
    backprop::Classic, select, simulate, strategy, strategy::Compose, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type V = Compose<select::UcbV, simulate::Uniform, Classic>;
type K = Compose<select::KlUcb, simulate::Uniform, Classic>;

const UCB_V_C: [f64; 3] = [0.5, 1.0, 2.0];
const KL_UCB_C: [f64; 3] = [0.0, 1.0, 3.0];

fn baseline_config<G: Game>(budget: Duration) -> TreeSearch<G, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("baseline/uct")
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn ucb_v_config<G: Game>(budget: Duration, c: f64) -> TreeSearch<G, V> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("ucb_v/c={c}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::UcbV::with_c(c)),
    )
}

fn kl_ucb_config<G: Game>(budget: Duration, c: f64) -> TreeSearch<G, K> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("kl_ucb/c={c}"))
            .use_transpositions(true)
            .max_time(budget)
            .select(select::KlUcb::with_c(c)),
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
    for c in UCB_V_C {
        strategies.push(AnySearch::new(ucb_v_config::<G>(budget, c)));
    }
    for c in KL_UCB_C {
        strategies.push(AnySearch::new(kl_ucb_config::<G>(budget, c)));
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

    println!("=== Variance-aware bandit strength comparison (background job) ===");
    println!("Arms: baseline UCT, UCB-V c in {UCB_V_C:?}, KL-UCB c in {KL_UCB_C:?}");
    println!("200ms/move, sequential, round-robin so every pair of arms is checked.");
    println!();

    run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", budget, rounds);
    run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", budget, rounds);

    println!("=== done ===");
}
