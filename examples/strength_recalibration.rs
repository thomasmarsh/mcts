// Validates a recalibration of Druid's Strong preset (after the MAST/GRAVE/
// AMAF correctness fixes) at real production settings. Cheap fixed-iteration,
// single-threaded exploration (scripts not kept) found:
//   - Rave/GRAVE select doesn't help Druid at all, even after re-tuning
//     every hyperparameter -- plain Ucb1 select beats every Rave variant
//     tried.
//   - DecisiveMove<EpsilonGreedy<Nst>> is a strong simulate policy on its
//     own, clearly ahead of Mast and of the then-shipped DruidHeuristic, at
//     epsilon=0.3, backoff_threshold=5.
// This compares the resulting recommendation -- Ucb1 select +
// DecisiveMove<EpsilonGreedy<Nst>> simulate, otherwise matching every other
// shipped Strong knob (transpositions, solver, tree-parallel thread count,
// 3s/move) -- against the then-currently-shipped Strong preset
// (RaveDecisiveHeuristic), at the real budget, not the cheap proxy. Result
// (81.2% vs. 18.8%, n=16) confirmed the recommendation; it's since been
// wired into server/main.rs's build_ai. Kept as the reproducible record of
// that validation, per this repo's `strength_*.rs` convention.
//
// Real 3s/move tree-parallel budget means each individual game already
// saturates every core, so games run strictly sequentially (same rationale
// as every prior real-budget strength script in this repo --
// strength_solver_strong.rs, strength_heuristic_wiring.rs, etc.) rather than
// fanning out via rayon the way the cheap-budget phases did.
//
// This is a long-running background job. Progress is printed after every
// game so it can be checked in on without waiting for the final summary.
//
// Usage: cargo run --release --example strength_recalibration
use std::time::Duration;

use mcts::games::druid::{Druid, DruidHeuristic, DruidHeuristicWeights, RaveDecisiveHeuristic};
use mcts::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::bench::tournament::wilson_interval;
use mcts::util::battle_royale;

const ROUNDS: usize = 8; // 16 games, alternating who moves first
const BUDGET_SECS: u64 = 3;

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

type Ucb1DmNst = strategy::Compose<select::Ucb1, simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>>;

fn new_config() -> TreeSearch<Druid, Ucb1DmNst> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("recalibrated/ucb1+dm+nst")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(BUDGET_SECS))
            .num_tree_threads(ai_thread_count())
            .simulate(simulate::DecisiveMove::new().inner(
                simulate::EpsilonGreedy::default()
                    .epsilon(0.3)
                    .inner(simulate::Nst::new().backoff_threshold(5)),
            )),
    )
}

fn tuned_select() -> select::Rave {
    select::Rave::default()
        .ucb(select::RaveUcb::Ucb1Tuned {
            exploration_constant: 0.2894182,
        })
        .threshold(204)
        .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 })
}

// Byte-for-byte the currently-shipped Strong preset (server/main.rs).
fn shipped_config() -> TreeSearch<Druid, RaveDecisiveHeuristic> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("shipped/rave+heuristic")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(BUDGET_SECS))
            .num_tree_threads(ai_thread_count())
            .select(tuned_select())
            .simulate(simulate::DecisiveMove::new().inner(
                simulate::EpsilonGreedy::default().epsilon(0.5).inner(
                    DruidHeuristic::new(DruidHeuristicWeights {
                        block_threat: 1.0,
                        defend_fork: 1.0,
                        threaten_connection: 1.0,
                    }),
                ),
            )),
    )
}

fn fmt(w: usize, l: usize, d: usize) -> String {
    let total = w + l + d;
    let score = w as f64 + 0.5 * d as f64;
    let (lo, hi) = wilson_interval(score, total, 1.96);
    let pct = if total > 0 {
        100.0 * score / total as f64
    } else {
        0.0
    };
    format!(
        "W={w} L={l} D={d} total={total} win_rate={pct:.1}% [{:.1}%, {:.1}%] (95% Wilson)",
        lo * 100.0,
        hi * 100.0
    )
}

fn main() {
    println!("=== recalibrated config vs. currently-shipped Strong, real budget ===");
    println!(
        "Budget: {}s/move, tree-parallel across {} threads, 5x5 board, {} rounds ({} games)",
        BUDGET_SECS,
        ai_thread_count(),
        ROUNDS,
        ROUNDS * 2
    );
    println!();

    let (mut new_w, mut new_l, mut new_d) = (0usize, 0usize, 0usize);

    for round in 0..ROUNDS {
        // Alternate who moves first each round.
        let new_first = round % 2 == 0;
        let mut new_engine = new_config();
        let mut old_engine = shipped_config();

        let outcome = if new_first {
            battle_royale::<Druid, _, _>(&mut new_engine, &mut old_engine)
        } else {
            battle_royale::<Druid, _, _>(&mut old_engine, &mut new_engine).map(|w| 1 - w)
        };
        // outcome: Some(0) => "new" (recalibrated) won, Some(1) => shipped won, None => draw
        match outcome {
            Some(0) => new_w += 1,
            Some(1) => new_l += 1,
            _ => new_d += 1,
        }

        println!(
            "[game {}/{}] new_first={} outcome={:?} -- running: {}",
            round + 1,
            ROUNDS,
            new_first,
            outcome,
            fmt(new_w, new_l, new_d)
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    println!();
    println!("=== Final: recalibrated (Ucb1+DM+Nst) vs. shipped (Rave+DruidHeuristic) ===");
    println!("recalibrated: {}", fmt(new_w, new_l, new_d));
    println!("shipped:      {}", fmt(new_l, new_w, new_d));
}
