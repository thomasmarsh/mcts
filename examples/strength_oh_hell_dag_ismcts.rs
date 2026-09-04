// Background strength comparison: single-tree ISMCTS with and without
// explicit-DAG node merging (E4), on 2-player Oh Hell.
//
// Oh Hell is this roadmap's least-confounded E4 candidate --
// `examples/transposition_density.rs` measured real, non-trivial
// transposition density in Oh Hell's midgame (double digits to a few
// hundred x depending on depth), unlike Ingenious (~1.0x, no measurable
// density) or Phantom (confounded by its small 16-cell board reconverging
// on literal full states, not just information sets, at this sample size).
//
// - "cheating" runs one ordinary tree search directly against the literal
//   state, so it can see the opponent's true hand the whole game.
// - "ismcts" runs plain `IsmctsMode::SingleTree`, `GraphSearch::Tree` (no
//   merging) -- every iteration walks its own `OhHell::determinize`d sample
//   from the root, one shared tree, no node sharing across different
//   histories that reach the same information set.
// - "dag_ismcts" additionally sets `GraphSearch::Dag(GraphStats::Both)` and
//   `McgsCorrection::RaveBlend` -- the RAVE-blended alternative to
//   `McgsCorrection::Residual` (see `config::McgsCorrection`'s doc comment).
//   Nodes are keyed by `Game::info_set_hash` (`OhHell::public_hash`) instead
//   of the literal `zobrist_hash`, so different real histories/
//   determinizations that reach the same information set now share one
//   node's visit/score/availability statistics, pooled via `GraphStats::
//   Both` and blended into each merged edge's own selection score via a
//   `select::RaveSchedule`-style decay.
//
// An earlier version of this benchmark ran `McgsCorrection::Residual` here
// instead and landed at noise-level parity with plain "ismcts" (36.7% vs.
// 35.8%) -- but a dedicated soundness test (`dag_ismcts_error_shrinks_with_
// budget_like_plain_ismcts_does`, `mcts/src/strategies/tests.rs`) later confirmed
// `Residual` is genuinely biased under `ismcts_mode`, not just unmeasured:
// it always corrects a diverging edge toward the merged node's pooled
// estimate (sound for the perfect-information paper it's borrowed from,
// arXiv:2012.11045, but strategy fusion's own definition under hidden
// information) and, because the check gates descent itself, permanently
// freezes the corrected edge out of further direct sampling once it starts
// firing. `RaveBlend` was designed to fix both problems at once -- see
// `config::McgsCorrection::RaveBlend`'s doc comment -- and the same
// soundness test confirms it actually converges to the true value as the
// iteration budget grows, unlike `Residual`. This run is the natural
// follow-up the soundness test's own design called for: does removing the
// confirmed bias also produce a measurable strength win over plain "ismcts",
// the question the original Residual-based run's noise-parity result left
// open.
//
// Total node budget is matched across all three (`ITERATIONS` each, all
// single-threaded). Sequential execution (no rayon fan-out across games) so
// nothing competes with anything else for the machine. This is a
// long-running job -- run as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_oh_hell_dag_ismcts
use game_oh_hell::OhHell;
use mcts::algorithms::mcts::{
    profile, select, GraphSearch, GraphStats, IsmctsMode, McgsCorrection, SearchConfig, TreeSearch,
};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type OhHell2 = OhHell<2, 7>;

const ITERATIONS: usize = 8000;
const ROUNDS: usize = 15;

fn cheating_config(name: &str) -> TreeSearch<OhHell2, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .seed(1),
    )
}

fn ismcts_config(name: &str) -> TreeSearch<OhHell2, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(2),
    )
}

fn dag_ismcts_config(name: &str) -> TreeSearch<OhHell2, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .mcgs_correction(McgsCorrection::RaveBlend {
                schedule: select::RaveSchedule::default(),
            })
            .seed(3),
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
    println!(
        "=== Oh Hell ISMCTS explicit-DAG merging (E4) strength comparison (background job) ==="
    );
    println!(
        "Matched node budget: {ITERATIONS} simulations/move per tree. 3 strategies, {ROUNDS} \
         rounds, {} games (6 ordered pairs/round).",
        ROUNDS * 6
    );
    println!();

    let mut strategies: Vec<AnySearch<OhHell2>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(ismcts_config("ismcts")),
        AnySearch::new(dag_ismcts_config("dag_ismcts")),
    ];
    let results = round_robin_multiple::<OhHell2, _>(
        &mut strategies,
        ROUNDS,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );

    println!();
    println!("=== Summary ===");
    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
    println!(
        "Interpretation: \"cheating\" isolates what hiding the hand costs at matched node \
         budget, same as `strength_oh_hell_redeterminize.rs` -- not a claim \"ismcts\"/\
         \"dag_ismcts\" should beat it. \"dag_ismcts\" vs \"ismcts\" asks whether explicit-DAG \
         information-set merging helps single-tree ISMCTS, now gated behind \
         `McgsCorrection::RaveBlend` instead of the earlier `Residual` run -- a dedicated \
         soundness test (mcts/src/strategies/tests.rs's dag_ismcts_error_shrinks_with_budget_\
         like_plain_ismcts_does) already confirmed RaveBlend converges to the true info-set \
         value as the budget grows, unlike Residual, so a win/loss here is a cleaner read on \
         whether DAG merging itself helps ISMCTS than the earlier Residual-based run was."
    );
}
