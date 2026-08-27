// Background strength comparison: PIMC (Perfect Information Monte Carlo)
// ensemble search, single-tree ISMCTS (SO-ISMCTS), and multi-tree ISMCTS
// (MO-ISMCTS) vs. a single ordinary tree search, on 2-player Ingenious.
//
// `Ingenious::State` always stores every player's rack in full -- that's
// just how the game's `apply`/`generate_actions` are implemented, and it
// never changes. The four strategies compared here differ only in what a
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
// - "ismcts" runs `IsmctsMode::SingleTree` (single shared tree, one
//   `Ingenious::determinize` call per *iteration* rather than once per
//   worker): every iteration walks its own determinized sample from the
//   root, and per-child availability counts (not raw visit counts) drive
//   `Ucb1`. This is SO-ISMCTS -- it re-determinizes once at the root of each
//   iteration and then walks `G::apply` for the rest of that iteration's
//   descent, so an opponent's node several plies deep still sees the exact
//   rack sampled at the root, not a fresh guess of its own.
// - "mo_ismcts" runs `IsmctsMode::MultiTree`: one tree per player, all
//   descended together every iteration, each node only ever selected from
//   via the tree belonging to whoever is about to move there (see
//   `SearchConfig::ismcts_mode`'s doc comment). This is MO-ISMCTS.
//
// Total node budget is matched across all four: "cheating"/"ismcts"/
// "mo_ismcts" each get one tree with `max_iterations = PIMC_WORKERS *
// PER_WORKER_ITERATIONS`, "pimc" gets `PIMC_WORKERS` trees each with
// `max_iterations = PER_WORKER_ITERATIONS`, so a win for "pimc"/"ismcts"/
// "mo_ismcts" reflects the value of restricting the search to realistic
// information rather than simply spending more compute. Ingenious's own
// rack mechanic (mostly hidden-until-played, one reveal per tile pair,
// monotonic board) is closest in flavor to the paper's Dou Di Zhu domain,
// where neither strategy fusion nor leakage dominates -- see
// `examples/strength_phantom_mo_ismcts.rs` for the sharper, published-
// ranking correctness gate this workspace checks MO-ISMCTS against first.
//
// Sequential execution (no rayon fan-out across games) so each root-parallel
// search gets the whole machine to itself. This is a long-running job --
// run as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_ingenious_pimc
use game_ingenious::Ingenious2;
use mcts::strategies::mcts::{strategy, IsmctsMode, SearchConfig, TreeSearch};
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
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(3),
    )
}

fn mo_ismcts_config(name: &str) -> TreeSearch<Ingenious2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PIMC_WORKERS * PER_WORKER_ITERATIONS)
            .ismcts_mode(IsmctsMode::MultiTree)
            .seed(4),
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
    println!("=== Ingenious PIMC/SO-ISMCTS/MO-ISMCTS strength comparison (background job) ===");
    println!(
        "Matched node budget: {} total simulations/move (1 cheating tree, {} PIMC trees x \
         {}, 1 ismcts tree, 1 mo_ismcts tree-per-player set). 4 strategies, {} rounds, {} games \
         (12 ordered pairs/round).",
        PIMC_WORKERS * PER_WORKER_ITERATIONS,
        PIMC_WORKERS,
        PER_WORKER_ITERATIONS,
        ROUNDS,
        ROUNDS * 12
    );
    println!();

    let mut strategies: Vec<AnySearch<Ingenious2>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(pimc_config("pimc")),
        AnySearch::new(ismcts_config("ismcts")),
        AnySearch::new(mo_ismcts_config("mo_ismcts")),
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
        "Interpretation: \"pimc\", \"ismcts\", and \"mo_ismcts\" all search only realistic \
         determinizations against \"cheating\"'s full view of the opponent's rack, at matched \
         total node budget, isolating what hiding the rack costs a search that can no longer \
         see it -- not a claim that any of them should beat cheating, since cheating has \
         strictly more information available to it. The \"mo_ismcts\" vs \"pimc\" vs \"ismcts\" \
         three-way result is the one worth checking here: on Phantom (4, 4, 4) mo_ismcts clearly \
         beats pimc and ismcts (the paper's own published ranking); if that ordering doesn't \
         hold here too, Ingenious's rack mechanic may simply sit in the same \"neither strategy \
         fusion nor leakage matters much\" bucket the paper's Dou Di Zhu domain does, not that \
         this implementation disagrees with the paper."
    );
}
