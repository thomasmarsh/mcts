// Background strength comparison: PIMC (Perfect Information Monte Carlo)
// ensemble search vs. a single ordinary tree search, on 2-player Ingenious.
//
// `Ingenious::State` always stores every player's rack in full -- that's
// just how the game's `apply`/`generate_actions` are implemented, and it
// never changes. The two strategies compared here differ only in what a
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
//
// Total node budget is matched between the two: "cheating" gets one tree
// with `max_iterations = PIMC_WORKERS * PER_WORKER_ITERATIONS`, "pimc" gets
// `PIMC_WORKERS` trees each with `max_iterations = PER_WORKER_ITERATIONS`,
// so a win for "pimc" reflects the value of restricting each tree to
// realistic information rather than simply spending more compute.
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
    println!("=== Ingenious PIMC strength comparison (background job) ===");
    println!(
        "Matched node budget: {} total simulations/move ({} cheating tree vs. {} PIMC \
         trees x {}). {} rounds ({} games).",
        PIMC_WORKERS * PER_WORKER_ITERATIONS,
        PIMC_WORKERS * PER_WORKER_ITERATIONS,
        PIMC_WORKERS,
        PER_WORKER_ITERATIONS,
        ROUNDS,
        ROUNDS * 2
    );
    println!();

    let mut strategies: Vec<AnySearch<Ingenious2>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(pimc_config("pimc")),
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
        "Interpretation: \"pimc\" searching only realistic determinizations against \
         \"cheating\"'s full view of the opponent's rack, at matched total node budget, \
         isolates what hiding the rack costs a search that can no longer see it -- not \
         a claim that pimc should beat cheating, since cheating has strictly more \
         information available to it."
    );
}
