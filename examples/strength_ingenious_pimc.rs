// Background strength comparison: PIMC (Perfect Information Monte Carlo)
// ensemble search and single-tree ISMCTS vs. a single ordinary tree search,
// on 2-player Ingenious.
//
// `Ingenious::State` always stores every player's rack in full -- that's
// just how the game's `apply`/`generate_actions` are implemented, and it
// never changes. The three strategies compared here differ only in what a
// search is *allowed to look at* before it moves:
//
// - "cheating" runs one ordinary tree search directly against the literal
//   state, so it can see the opponent's true rack the whole game.
// - "pimc" runs `SearchConfig::determinize_root` with `num_threads > 1`:
//   each of the `PIMC_WORKERS` trees first calls `Ingenious::determinize`,
//   which keeps the mover's own rack and redraws every other rack from the
//   pool of tiles consistent with public information, then searches that
//   sampled state independently. The ensemble's chosen action is a
//   plurality vote across those trees' visit counts.
// - "ismcts" runs `SearchConfig::use_ismcts` (single-threaded, single
//   shared tree, one `Ingenious::determinize` call per *iteration* rather
//   than once per worker): every iteration walks its own determinized
//   sample from the root, and per-child availability counts (not raw visit
//   counts) drive `Ucb1`. Note this is single-tree SO-ISMCTS as landed --
//   it re-determinizes once at the root of each iteration and then walks
//   `G::apply` for the rest of that iteration's descent, so an opponent's
//   node several plies deep still sees the exact rack sampled at the root,
//   not a fresh guess of its own (see `plan/ingenious/ismcts.md` Part E1).
//   This run's "ismcts" numbers are exactly what E1 would fix, or not --
//   the point of running this before building E1 is finding out whether
//   that gap costs anything measurable in Ingenious specifically before
//   investing in the fix.
//
// Total node budget is matched across all three: "cheating" and "ismcts"
// each get one tree with `max_iterations = PIMC_WORKERS *
// PER_WORKER_ITERATIONS`, "pimc" gets `PIMC_WORKERS` trees each with
// `max_iterations = PER_WORKER_ITERATIONS`, so a win for "pimc"/"ismcts"
// reflects the value of restricting the search to realistic information
// rather than simply spending more compute.
//
// Sequential execution (no rayon fan-out across games) so each root-parallel
// search gets the whole machine to itself. This is a long-running job --
// run as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_ingenious_pimc
use game_ingenious::Ingenious2;
use mcts::strategies::mcts::{strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

const PIMC_WORKERS: usize = 4;
const PER_WORKER_ITERATIONS: usize = 2000;
const ROUNDS: usize = 15;

fn cheating_config(name: &str) -> TreeSearch<Ingenious2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PIMC_WORKERS * PER_WORKER_ITERATIONS)
            .seed(1),
    )
}

fn pimc_config(name: &str) -> TreeSearch<Ingenious2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PER_WORKER_ITERATIONS)
            .num_threads(PIMC_WORKERS)
            .determinize_root(true)
            .seed(2),
    )
}

fn ismcts_config(name: &str) -> TreeSearch<Ingenious2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PIMC_WORKERS * PER_WORKER_ITERATIONS)
            .use_ismcts(true)
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
    println!("=== Ingenious PIMC/ISMCTS strength comparison (background job) ===");
    println!(
        "Matched node budget: {} total simulations/move (1 cheating tree, {} PIMC trees x \
         {}, 1 ismcts tree). 3 strategies, {} rounds, {} games (6 ordered pairs/round).",
        PIMC_WORKERS * PER_WORKER_ITERATIONS,
        PIMC_WORKERS,
        PER_WORKER_ITERATIONS,
        ROUNDS,
        ROUNDS * 6
    );
    println!();

    let mut strategies: Vec<AnySearch<Ingenious2>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(pimc_config("pimc")),
        AnySearch::new(ismcts_config("ismcts")),
    ];
    let results = round_robin_multiple::<Ingenious2, _>(
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
        "Interpretation: \"pimc\" and \"ismcts\" both search only realistic determinizations \
         against \"cheating\"'s full view of the opponent's rack, at matched total node \
         budget, isolating what hiding the rack costs a search that can no longer see it -- \
         not a claim that either should beat cheating, since cheating has strictly more \
         information available to it. The \"pimc\" vs \"ismcts\" head-to-head result is the \
         one that matters for plan/ingenious/ismcts.md Part E: if ismcts doesn't clearly beat \
         pimc despite the root-only-determinization leakage in Part E1, that leakage may not \
         be worth fixing yet for this game."
    );
}
