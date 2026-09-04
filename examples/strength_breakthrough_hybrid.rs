// Background strength comparison: plain UCT vs. three MCTS-minimax hybrids
// (MCTS-MR-n minimax rollouts, MCTS-IC-E informed cutoffs, MCTS-MS-2-Visit-0
// minimax-informed expansion priors), on Breakthrough -- the game Baier &
// Winands' MCTS-minimax hybrid papers use as their own test game, so results
// are checkable against the papers' own findings, not just internal A/B
// noise. Of the three, MS is the literature's own strongest performer on
// Breakthrough (62.2% vs. an MCTS-Solver baseline, 2015 paper's MS-2-Visit-2);
// MS-2-Visit-0 here is the closest checkable proxy this codebase currently
// builds (`Visit-v>0` isn't implemented).
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
use mcts::algorithms::mcts::{profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type Board = Breakthrough<8, 8>;

// Baseline: vanilla UCT, uniform-random rollouts played to a real terminal
// state (Breakthrough always terminates -- forward-only pawn moves bound
// game length, no draws), no depth cutoff. This is what every hybrid below
// is trying to beat.
type Baseline = profile::Mcts;

// MCTS-MR-n: uniform-random rollout except the last `n` plies before the
// depth cutoff, which are resolved by exact bounded negamax instead
// (`MinimaxRollout::n`). Needs a finite `max_playout_depth` to ever trigger
// -- see `MinimaxRollout`'s doc comment: "last n plies before the playout's
// depth cutoff", not "last n plies of the game".
type MrN = profile::Mcts<select::Ucb1, simulate::MinimaxRollout<Board, Heuristic>>;

// MCTS-IC-E: uniform-random rollout, but a playout that hits the depth
// cutoff without reaching a real terminal state gets scored by `Heuristic`
// instead of falling through to `Game::compute_utilities`'s draw default.
type IcE = profile::Mcts<select::Ucb1, simulate::EvaluatedCutoff<Board, Heuristic>>;

// MCTS-MS-2-Visit-0: no rollout-side change at all -- `Baseline`'s own
// select/simulate strategies, plus a `NegamaxPrior` that seeds every
// freshly-expanded node's children with a depth-2 bounded-negamax prior
// before any of them has a real playout (`SearchConfig::with_prior`). Reuses
// `Baseline`'s type since the prior lives in `SearchConfig`, not in
// `PolicyProfile<G>`'s associated types -- see `mcts::prior`'s module doc comment.
type MsPrior = Baseline;

// MS-2's own search depth, matching the literature's own best-performing
// depth on Breakthrough (Baier & Winands 2015).
const MS_DEPTH: u32 = 2;
// How many fictitious visits each seeded prior is worth -- see
// `mcts::prior::PriorPolicy::pseudo_visits`'s doc comment.
const MS_PSEUDO_VISITS: u32 = 4;

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

fn ms_prior_config(budget: Duration) -> TreeSearch<Board, MsPrior> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("hybrid/ms-2-visit-0")
            .use_transpositions(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(1.414))
            .with_prior(
                mcts::algorithms::mcts::prior::NegamaxPrior::<Board, Heuristic>::new()
                    .depth(MS_DEPTH)
                    .pseudo_visits(MS_PSEUDO_VISITS),
            ),
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
    println!("Arms: baseline UCT, MCTS-MR-n (n={MR_N}), MCTS-IC-E, MCTS-MS-2-Visit-0");
    println!(
        "Hybrid arms cut rollouts at {MAX_PLAYOUT_DEPTH} plies; baseline plays to a real terminal state."
    );
    println!("MS-2-Visit-0 doesn't cut rollouts -- it seeds expansion-time priors instead.");
    println!("1s/move, sequential, round-robin so every pair of arms is checked.");
    println!();

    let budget = Duration::from_secs(1);
    let rounds = 15; // 30 games per pair (round-robin alternates who moves first)
    let num_arms = 4;
    println!(
        "--- {} rounds ({} games per pair, {} games total) ---",
        rounds,
        rounds * 2,
        rounds * 2 * (num_arms * (num_arms - 1) / 2)
    );

    let mut strategies: Vec<AnySearch<Board>> = vec![
        AnySearch::new(baseline_config(budget)),
        AnySearch::new(mr_n_config(budget)),
        AnySearch::new(ic_e_config(budget)),
        AnySearch::new(ms_prior_config(budget)),
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
    println!("Interpretation: published results (Baier & Winands 2015) predict MR-n loses");
    println!("to the baseline at this real time budget (its per-node cost isn't recouped),");
    println!("while MS is the strongest reported Breakthrough hybrid -- MS-2-Visit-0 losing");
    println!("here too would be evidence worth investigating, not an expected result the way");
    println!("an MR-n loss is. `round_robin_multiple` plays every pair, so all six");
    println!("comparisons among the four arms come out of this one run.");
}
