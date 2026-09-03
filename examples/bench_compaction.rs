// Bounded pruning after re-rooting (`SearchConfig::max_arena_len`,
// `search/compact.rs`'s `TreeSearch::compact`): what does a compaction event
// actually cost in wall-clock time, at realistic subtree sizes? The unit
// tests (`algorithms::tests::test_compact_*`) pin the *correctness* of
// compaction on small hand-verifiable shapes; this measures the *cost* on a
// real game, which needs a real arena at real size and can't be a
// deterministic `cargo test --lib` assertion (see AGENTS.md's "keep
// `cargo test --lib` fast" note -- timing numbers belong in `examples/`,
// same reason `mem_profile.rs` does).
//
// `reuse_or_reset` (and therefore `compact`) runs *before* `self.timer.start`
// in `choose_action` (search/search_impl.rs) -- compaction's cost is pure
// added latency on top of the per-move search budget, not time stolen from
// it, so it shows up directly as this ply's `choose_action` call taking
// noticeably longer than a normal budget-bound move.
//
// Usage: cargo run --release --example bench_compaction
use std::time::{Duration, Instant};

use game_druid::{
    Druid, DruidHeuristic, DruidHeuristicWeights, HashedState, RaveDecisiveHeuristic,
};
use mcts::game::Game;
use mcts::algorithms::mcts::{node::QInit, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// Byte-for-byte the same config server/main.rs's `build_ai(Strong)` builds,
// plus a low `max_arena_len` to force several compaction events over the
// course of one game instead of the zero a production-sized threshold would
// realistically hit -- see mem_profile.rs/strength_reuse_tree.rs for why this
// is duplicated here rather than imported (the server binary isn't a lib
// target `examples/` can pull `build_ai` from).
fn strong_config(max_arena_len: usize) -> TreeSearch<Druid, RaveDecisiveHeuristic> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("strong/bench-compaction")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .max_arena_len(Some(max_arena_len))
            .q_init(QInit::Infinity)
            .max_time(Duration::from_secs(1))
            .num_tree_threads(ai_thread_count())
            .select(
                select::Rave::default()
                    .ucb(select::RaveUcb::Ucb1Tuned {
                        exploration_constant: 0.2894182,
                    })
                    .threshold(204)
                    .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
            )
            .simulate(
                simulate::DecisiveMove::new().inner(
                    simulate::EpsilonGreedy::default()
                        .epsilon(0.5)
                        .inner(DruidHeuristic::new(DruidHeuristicWeights {
                            block_threat: 1.0,
                            defend_fork: 1.0,
                            threaten_connection: 1.0,
                        })),
                ),
            ),
    )
}

fn main() {
    // A 1s/move budget (vs. mem_profile.rs's real 3s) so a full game
    // finishes in a reasonable time for a script that's meant to be re-run
    // by hand -- this measures compaction's added latency on top of the
    // budget, which doesn't depend on how long the budget itself is.
    let max_arena_len = 50_000;
    let mut search = strong_config(max_arena_len);
    let mut state = HashedState::default(); // 5x5, same as server's fresh page load
    let mut ply = 0;

    println!("=== bench_compaction: real Strong-preset self-play, compaction wall-clock cost ===");
    println!(
        "Board: 5x5 default. 1s/move, tree-parallel across {} cores, reuse_tree on, max_arena_len={}.",
        ai_thread_count(),
        max_arena_len
    );
    println!(
        "{:>4} {:>7} {:>12} {:>12} {:>10}",
        "ply", "elapsed", "arena_before", "arena_after", "compacted?"
    );

    while !Druid::is_terminal(&state) {
        let arena_before = search.arena_len();
        let t0 = Instant::now();
        let action = search.choose_action(&state);
        let elapsed = t0.elapsed();
        let arena_after = search.arena_len();
        ply += 1;

        // Not a precise "compaction ran" signal (this move's own search adds
        // nodes too, offsetting whatever compaction removed) -- only a lower
        // bound: the arena can end a ply smaller than it started only if
        // compaction removed more than this move's iterations added back.
        // Read this alongside `elapsed`: a ply that took noticeably longer
        // than the ~1s budget is the real tell that compaction ran (and
        // roughly how much it cost), regardless of whether growth happened
        // to mask the net size change.
        let likely_compacted = arena_after < arena_before;
        println!(
            "{:>4} {:>6.2}s {:>12} {:>12} {:>10}",
            ply,
            elapsed.as_secs_f64(),
            arena_before,
            arena_after,
            if likely_compacted { "yes" } else { "" }
        );

        state = Druid::apply(state, &action);
    }

    println!(
        "=== final ({ply} plies), arena_len(): {} ===",
        search.arena_len()
    );
}
