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
//   `McgsCorrection::Residual` -- the only `SearchConfig` pairing
//   `validate()` accepts alongside `ismcts_mode`. Nodes are keyed by
//   `Game::info_set_hash` (`OhHell::public_hash`) instead of the literal
//   `zobrist_hash`, so different real histories/determinizations that reach
//   the same information set now share one node's visit/score/availability
//   statistics, pooled via `GraphStats::Both` and checked against the
//   residual correction at each merged edge.
//
// This is *not* a clean test of whether DAG merging helps ISMCTS in
// isolation. `McgsCorrection::Residual` (`correction::residual_correction`)
// is borrowed from a perfect-information Monte-Carlo graph search paper
// (arXiv:2012.11045, chess/crazyhouse): when a merged node's pooled Q
// diverges from the traversing edge's own local Q by more than `epsilon`,
// it always corrects toward the pooled node estimate. That's sound when a
// merge is exact (perfect information, so divergence is just undersampled
// noise), but under hidden information a divergence can instead reflect a
// real, persistent difference in correct play depending on which hidden
// cards an opponent holds -- forcing every such edge to trust the
// pooled/averaged value is strategy fusion's own definition, applied
// automatically whenever the check fires. A "dag_ismcts" win here could
// come from DAG merging's larger effective sample density outweighing that
// cost at this budget, not from the correction mechanism working as
// intended; a loss could come from the correction making fusion worse, not
// from "DAG merging doesn't help ISMCTS" in general. Keep this in mind
// before drawing conclusions from the numbers below.
//
// Total node budget is matched across all three (`ITERATIONS` each, all
// single-threaded). Sequential execution (no rayon fan-out across games) so
// nothing competes with anything else for the machine. This is a
// long-running job -- run as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_oh_hell_dag_ismcts
use game_oh_hell::OhHell;
use mcts::strategies::mcts::{
    strategy, GraphSearch, GraphStats, IsmctsMode, McgsCorrection, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type OhHell2 = OhHell<2, 7>;

const ITERATIONS: usize = 8000;
const ROUNDS: usize = 15;

fn cheating_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .seed(1),
    )
}

fn ismcts_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(2),
    )
}

fn dag_ismcts_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .graph_search(GraphSearch::Dag(GraphStats::Both))
            .mcgs_correction(McgsCorrection::Residual { epsilon: 0.1 })
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
         information-set merging helps single-tree ISMCTS, but the `McgsCorrection::Residual` \
         gate this pairing requires is borrowed from a perfect-information MCGS paper and \
         always corrects a diverging edge toward the merged node's pooled estimate -- sound \
         when a merge is exact, but pushing toward strategy fusion when the divergence instead \
         reflects a real difference in correct play under different hidden information. So a \
         win or loss here doesn't cleanly isolate whether DAG merging itself helps ISMCTS \
         versus whether this correction's polarity is helping or hurting."
    );
}
