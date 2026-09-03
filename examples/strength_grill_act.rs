// Background strength comparison: plain UCT vs Grill et al. ("MCTS as
// Regularized Policy Optimization", ICML 2020) closed-form acting policy
// `pi_bar` used as the tree-descent selector (`select::GrillAct`). See
// `mcts/src/strategies/mcts/select/regularized.rs`.
//
// The literature claims the regularised-policy selector wins *at low
// simulation counts*, so this sweeps iteration budget: a small budget (the
// regime it should win) and a large one (where it should converge to parity
// or lose). `c` scales lambda_N = c*sqrt(N)/(N+|A|).
//
// TODO: the phase plan also wants a run with an `EvaluatorPrior` vs the
// uniform-prior fallback, to quantify how much the prior matters. `GrillAct`
// is uniform-prior-only for now (an explicit per-action pi_prior term is
// deferred, shared with MENTS), so there is no prior arm to run yet.
//
// Two games spanning the roster, same as this repo's other strength_*
// scripts:
//   - Breakthrough 8x8: tactical sudden-death, no draws.
//   - Ingenious (2p): a race-to-fill scoring game.
//
// Sequential execution, kept as re-runnable tooling (AGENTS.md), not a
// `#[test]`.
//
// Usage: cargo run --release --example strength_grill_act [rounds]
use std::io;

use game_breakthrough::Breakthrough;
use game_ingenious::Ingenious;
use mcts::game::Game;
use mcts::algorithms::mcts::{
    backprop::Classic, backprop::SoftmaxBackprop, select, simulate, strategy, strategy::Compose,
    SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Grill = Compose<select::GrillAct, simulate::Uniform, Classic>;
type Ments = Compose<select::Ments, simulate::Uniform, SoftmaxBackprop>;

const GRILL_C: [f64; 3] = [0.5, 1.0, 2.0];
const MENTS_TAU: f64 = 0.3;
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

fn grill_config<G: Game>(iters: usize, c: f64) -> TreeSearch<G, Grill> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(&format!("grill_act/c={c}"))
            .use_transpositions(true)
            .max_iterations(iters)
            .select(select::GrillAct::with_c(c)),
    )
}

fn ments_config<G: Game>(iters: usize) -> TreeSearch<G, Ments> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("ments/tau=0.3")
            .use_transpositions(true)
            .max_iterations(iters)
            .select(select::Ments::new(MENTS_TAU, MENTS_EPSILON))
            .backprop(SoftmaxBackprop::new(MENTS_TAU)),
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
    let mut strategies: Vec<AnySearch<G>> = vec![
        AnySearch::new(baseline_config::<G>(iters)),
        AnySearch::new(ments_config::<G>(iters)),
    ];
    for c in GRILL_C {
        strategies.push(AnySearch::new(grill_config::<G>(iters, c)));
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

    println!("=== GrillAct strength comparison (background job) ===");
    println!("Arms: baseline UCT, MENTS tau={MENTS_TAU}, GrillAct c in {GRILL_C:?}");
    println!("Sequential, round-robin, sweeping iteration budget {BUDGETS:?}.");
    println!();

    for iters in BUDGETS {
        run_game::<Breakthrough<8, 8>>("Breakthrough 8x8 (tactical, sudden-death)", iters, rounds);
        run_game::<Ingenious<2>>("Ingenious 2p (scoring race)", iters, rounds);
    }

    println!("=== done ===");
}
