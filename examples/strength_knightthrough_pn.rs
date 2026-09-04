// Reproduces the PN-MCTS paper's (Kowalski, Doe, Winands, Górski & Soemers,
// "Proof Number Based Monte-Carlo Tree Search", 2023) Knightthrough result
// under matching conditions: 8x8 board, 1 second per turn, C=sqrt(2) on
// both sides, C_pn=1, random (uniform) playouts, plain UCT baseline vs. the
// paper's best-performing "FSU" variant (final move selection + solving
// subtrees + the UCT-PN formula). In this engine F and S are already
// bundled as one `use_mcts_solver` flag rather than the paper's three
// independent flags (see `select::UctPn`'s doc comment), so the FSU
// equivalent here is `use_mcts_solver(true)` + `select::UctPn` -- the only
// difference from the baseline config below.
//
// Both configs enable `reuse_tree`: Ludii's `MCTS` base class -- which the
// paper's own baseline `createUCT()` inherits without overriding, per
// `~/git/Ludii/AI/src/search/mcts/MCTS.java:206` -- defaults `treeReuse` to
// `true`. This engine defaults `reuse_tree` to `false`, and an earlier run
// of this script left it at that default on both sides, which the paper's
// text never mentions turning off. That's a much bigger miss for the FSU
// side specifically: a solved subtree (or an accumulated pn/dpn count) is
// only worth anything across the many searches of one full game if it
// survives from one move to the next -- `reuse.rs`'s `try_promote` confirms
// the promoted node's `proven`/`pn`/`dpn` fields are left untouched, so nothing
// else needs to change for this to take effect.
//
// The paper reports 66.8% (+-2.9pp, 1000 games) for FSU vs. basic UCT on
// Knightthrough at this time setting (Table II) -- one of their larger,
// more reliable margins (unlike Awari, where PN-MCTS is closer to a wash).
//
// This is intentionally a long-running job. Run as a background process,
// not synchronously -- see strength_solver.rs's doc comment for why.
//
// Usage: cargo run --release --example strength_knightthrough_pn
use std::time::Duration;

use game_knightthrough::Knightthrough;
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type KT = Knightthrough<8, 8>;

const C: f64 = std::f64::consts::SQRT_2;
const C_PN: f64 = 1.0;

fn baseline(budget: Duration, name: &str) -> TreeSearch<KT, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .q_init(QInit::default())
            .reuse_tree(true)
            .max_time(budget)
            .select(select::Ucb1::with_c(C)),
    )
}

fn fsu(
    budget: Duration,
    name: &str,
) -> TreeSearch<KT, profile::Mcts<select::UctPn, simulate::Uniform>> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .expand_threshold(1)
            .q_init(QInit::default())
            .use_mcts_solver(true)
            .reuse_tree(true)
            .max_time(budget)
            .select(select::UctPn::with_c(C, C_PN)),
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
    println!("=== Knightthrough 8x8: FSU (PN-MCTS) vs basic UCT ===");
    println!("C={C:.4} (sqrt(2)), C_pn={C_PN:.1}, 1s/turn, uniform playouts, no transpositions.");
    println!("reuse_tree=true on both sides (matches Ludii's MCTS.treeReuse default).");
    println!("Paper's reported result at this setting: FSU wins 66.8% +-2.9pp (n=1000).");
    println!();

    let rounds = 20; // 40 games total, sequential
    let budget = Duration::from_secs(1);
    println!("--- {} rounds ({} games) ---", rounds, rounds * 2);

    let mut strategies: Vec<AnySearch<KT>> = vec![
        AnySearch::new(baseline(budget, "knightthrough/ucb1")),
        AnySearch::new(fsu(budget, "knightthrough/fsu")),
    ];
    let results = round_robin_multiple::<KT, _>(
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
    println!("Interpretation: compare knightthrough/fsu's win_rate against the paper's");
    println!("66.8% (n=1000). n=40 here gives a much wider CI -- this checks whether the");
    println!("improvement is in the same direction and rough ballpark, not a precise replication.");
}
