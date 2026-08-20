// Background strength comparison: plain UCT vs. two MCTS-minimax hybrids
// (MCTS-MR-n minimax rollouts, MCTS-IC-E informed cutoffs), on Breakthrough
// -- the game Baier & Winands' MCTS-minimax hybrid papers use as their own
// test game, so results are checkable against the papers' own findings, not
// just internal A/B noise.
//
// `simulate::MinimaxRollout` and `simulate::EvaluatedCutoff` otherwise only
// have hand-solvable-position unit tests (tic-tac-toe, "does it prefer the
// provably winning move") checking correctness, not game strength against a
// baseline -- this fills that gap.
//
// Real time budgets, sequential execution (each single-threaded search gets
// the whole machine), same rationale as this repo's other strength_*
// scripts -- a synchronous small-n attempt is CI-useless (see
// strength_solver.rs's own note on this).
//
// Usage: cargo run --release --example strength_breakthrough_hybrid
use std::time::Duration;

use game_breakthrough::{Breakthrough, Heuristic};
use mcts::strategies::mcts::{select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Board = Breakthrough<8, 8>;

// Baseline: vanilla UCT, uniform-random rollouts played to a real terminal
// state (Breakthrough always terminates -- forward-only pawn moves bound
// game length, no draws), no depth cutoff. This is what every hybrid below
// is trying to beat.
type Baseline = strategy::Ucb1;

// MCTS-MR-n: uniform-random rollout except the last `n` plies before the
// depth cutoff, which are resolved by exact bounded negamax instead
// (`MinimaxRollout::n`). Needs a finite `max_playout_depth` to ever trigger
// -- see `MinimaxRollout`'s doc comment: "last n plies before the playout's
// depth cutoff", not "last n plies of the game".
type MrN = strategy::Compose<select::Ucb1, simulate::MinimaxRollout<Board, Heuristic>>;

// MCTS-IC-E: uniform-random rollout, but a playout that hits the depth
// cutoff without reaching a real terminal state gets scored by `Heuristic`
// instead of falling through to `Game::compute_utilities`'s draw default.
type IcE = strategy::Compose<select::Ucb1, simulate::EvaluatedCutoff<Board, Heuristic>>;

// Plies before a still-unresolved rollout gets cut off and scored via
// `Heuristic` (`IcE`) or handed to bounded negamax for its last few plies
// (`MrN`). Baseline plays to a genuine terminal state instead (no cutoff),
// matching Baier & Winands' own comparison shape -- the hybrids' whole
// point is trading rollout length for an informed leaf value.
const MAX_PLAYOUT_DEPTH: usize = 60;

// MR-n's exact-search window: the last 3 plies of a (cut-off) rollout are
// resolved by bounded negamax instead of uniform-random play.
const MR_N: u32 = 3;

fn baseline_config(budget: Duration) -> TreeSearch<Board, Baseline> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("baseline/uct")
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414)),
    )
}

fn mr_n_config(budget: Duration) -> TreeSearch<Board, MrN> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("hybrid/mr-n")
            .use_transpositions(true)
            .max_time(budget)
            .max_playout_depth(MAX_PLAYOUT_DEPTH)
            .select(select::Ucb1::with_c(1.414))
            .simulate(simulate::MinimaxRollout::new().n(MR_N)),
    )
}

fn ic_e_config(budget: Duration) -> TreeSearch<Board, IcE> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("hybrid/ic-e")
            .use_transpositions(true)
            .max_time(budget)
            .max_playout_depth(MAX_PLAYOUT_DEPTH)
            .select(select::Ucb1::with_c(1.414))
            .simulate(simulate::EvaluatedCutoff::new()),
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
    println!("=== MCTS-minimax hybrid strength comparison (background job) ===");
    println!("Game: Breakthrough 8x8 (Baier & Winands' own test game)");
    println!("Arms: baseline UCT, MCTS-MR-n (n={MR_N}), MCTS-IC-E");
    println!(
        "Hybrid arms cut rollouts at {MAX_PLAYOUT_DEPTH} plies; baseline plays to a real terminal state."
    );
    println!("1s/move, sequential, round-robin so every pair of arms is checked.");
    println!();

    let budget = Duration::from_secs(1);
    let rounds = 15; // 30 games per pair (round-robin alternates who moves first)
    println!(
        "--- {} rounds ({} games per pair, {} games total) ---",
        rounds,
        rounds * 2,
        rounds * 2 * 3
    );

    let mut strategies: Vec<AnySearch<Board>> = vec![
        AnySearch::new(baseline_config(budget)),
        AnySearch::new(mr_n_config(budget)),
        AnySearch::new(ic_e_config(budget)),
    ];
    let results = round_robin_multiple::<Board, _>(
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
    println!("Interpretation: full-rollout epsilon-greedy hybrids (MCTS-IR-E/IR-M) are");
    println!("more expensive than either arm here and are only worth building if MR-n");
    println!("or IC-E show real wins over the baseline. `round_robin_multiple` plays");
    println!("every pair (baseline-vs-MR-n, baseline-vs-IC-E, MR-n-vs-IC-E), so all");
    println!("three comparisons come out of this one run.");
}
