// Background strength comparison: plain UCT vs MENTS (Xiao, Huang, Weinman,
// Müller, "Maximum Entropy Monte-Carlo Planning", NeurIPS 2019) -- the E2W
// stochastic tree policy (`select::Ments`) paired with the mellowmax soft
// value backup (`backprop::SoftmaxBackprop`, Asadi & Littman 2017's bounded
// form of the paper's log-sum-exp). See
// `mcts/src/strategies/mcts/select/regularized.rs`.
//
// The literature claims MENTS wins *at low simulation counts*, so this
// sweeps iteration budget rather than just wall-clock: a small budget (the
// regime MENTS should win) and a large one (where it should converge to
// parity or lose). `tau` is swept low-to-moderate; `epsilon` fixed at 0.1.
//
// Two games spanning the roster, same as this repo's other strength_*
// scripts:
//   - Breakthrough 8x8: tactical sudden-death, no draws.
//   - Ingenious (2p): a race-to-fill scoring game.
//
// Sequential execution, kept as re-runnable tooling (AGENTS.md), not a
// `#[test]`.
//
// Usage: cargo run --release --example strength_ments [rounds]
use std::io;

use game_breakthrough::Breakthrough;
use game_ingenious::Ingenious;
use mcts::game::Game;
use mcts::algorithms::mcts::{
    backprop::SoftmaxBackprop, select, simulate, strategy, strategy::Compose, SearchConfig,
    TreeSearch,
};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type M = Compose<select::Ments, simulate::Uniform, SoftmaxBackprop>;

const MENTS_TAU: [f64; 3] = [0.1, 0.3, 1.0];
const MENTS_EPSILON: f64 = 0.1;
const BUDGETS: [usize; 2] = [100, 2000];

fn baseline_config<G: Game>(iters: usize) -> TreeSearch<G, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("baseline/uct")
            .use_transpositions(true)
            .max_iterations(iters)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn ments_config<G: Game>(iters: usize, tau: f64) -> TreeSearch<G, M> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("ments/tau={tau}"))
            .use_transpositions(true)
            .max_iterations(iters)
            .select(select::Ments::new(tau, MENTS_EPSILON))
            .backprop(SoftmaxBackprop::new(tau)),
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

fn run_game<G: Game + Clone>(label: &str, iters: usize, rounds: usize)
where
    G::S: Sync,
{
    println!(
        "--- {label} @ {iters} iters/move ({rounds} rounds, {} games per pair) ---",
        rounds * 2
    );
    let mut strategies: Vec<AnySearch<G>> = vec![AnySearch::new(baseline_config::<G>(iters))];
    for tau in MENTS_TAU {
        strategies.push(AnySearch::new(ments_config::<G>(iters, tau)));
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

    println!("=== MENTS strength comparison (background job) ===");
    println!("Arms: baseline UCT, MENTS tau in {MENTS_TAU:?} (epsilon = {MENTS_EPSILON})");
    println!("Sequential, round-robin, sweeping iteration budget {BUDGETS:?}.");
    println!();

    for iters in BUDGETS {
        run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", iters, rounds);
        run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", iters, rounds);
    }

    println!("=== done ===");
}
