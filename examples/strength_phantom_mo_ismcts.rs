// Background strength comparison: cheating UCT, PIMC, single-tree ISMCTS
// (SO-ISMCTS), and multi-tree ISMCTS (MO-ISMCTS), on Phantom (4, 4, 4).
//
// This is a correctness gate worth clearing before trusting MO-ISMCTS's
// numbers on any other game: Cowling, Powley & Whitehouse's own Phantom
// (4, 4, 4) experiment (IEEE ToCIAIG 2012, Section VI) ranks six algorithms,
// best to worst, at 95% confidence --
//
//   1. cheating ensemble UCT
//   2. cheating (single-tree) UCT
//   3. MO-ISMCTS
//   4. determinized UCT (PIMC)
//   5. SO-ISMCTS
//   6. SO-ISMCTS+POM
//
// -- i.e. plain SO-ISMCTS (what `IsmctsMode::SingleTree` already was before
// this file's `IsmctsMode::MultiTree` landed) ranks *below* PIMC in the
// paper's own headline strategy-fusion benchmark, and only MO-ISMCTS
// reliably beats PIMC. The four strategies below are chosen to check that
// qualitative ordering (mo_ismcts > pimc; ismcts does *not* reliably beat
// pimc) against this implementation, using the paper's own board size --
// if these numbers come out qualitatively different, that's a strong signal
// of an implementation bug, not a research finding.
//
// `Position` always carries the ground-truth board -- that's just how
// `Phantom::apply`/`generate_actions`/`winner` are implemented (referee
// bookkeeping needs the real board regardless of what any player has
// personally discovered), and it never changes. The four strategies compared
// here differ only in what a search is *allowed to look at* before it moves:
//
// - "cheating" runs one ordinary tree search directly against the literal
//   board, so it always knows exactly which cells the opponent occupies.
// - "pimc" runs `SearchConfig::determinize_root` with `num_threads > 1`:
//   each of the `PIMC_WORKERS` trees calls `Phantom::determinize` once (a
//   guess at the opponent's marks consistent with what this mover has
//   personally discovered via rejected placements), then searches that
//   sampled board independently. The ensemble's chosen action is a
//   plurality vote across those trees' visit counts.
// - "ismcts" runs `IsmctsMode::SingleTree`: one shared tree, one
//   `Phantom::determinize` call per *iteration* rather than once per worker,
//   with per-child availability counts (not raw visit counts) driving
//   `Ucb1`. This is SO-ISMCTS.
// - "mo_ismcts" runs `IsmctsMode::MultiTree`: one tree per player, all
//   descended together every iteration, each node only ever selected from
//   via the tree belonging to whoever is about to move there (see
//   `SearchConfig::ismcts_mode`'s doc comment). This is MO-ISMCTS.
//
// Total node budget is matched across all four: "cheating"/"ismcts"/
// "mo_ismcts" each get one tree with `max_iterations = PIMC_WORKERS *
// PER_WORKER_ITERATIONS`, "pimc" gets `PIMC_WORKERS` trees each with
// `max_iterations = PER_WORKER_ITERATIONS` -- "mo_ismcts" spends
// `num_players` times the memory of the other three for the same iteration
// budget (one tree per player instead of one shared tree), but not more
// simulations.
//
// Sequential execution (no rayon fan-out across games) so each root-parallel
// search gets the whole machine to itself. This is a long-running job --
// run as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_phantom_mo_ismcts
use game_phantom::Phantom;
use mcts::algorithms::mcts::{strategy, IsmctsMode, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

const PIMC_WORKERS: usize = 4;
const PER_WORKER_ITERATIONS: usize = 2000;
const ROUNDS: usize = 15;

fn cheating_config(name: &str) -> TreeSearch<Phantom, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PIMC_WORKERS * PER_WORKER_ITERATIONS)
            .seed(1),
    )
}

fn pimc_config(name: &str) -> TreeSearch<Phantom, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PER_WORKER_ITERATIONS)
            .num_threads(PIMC_WORKERS)
            .determinize_root(true)
            .seed(2),
    )
}

fn ismcts_config(name: &str) -> TreeSearch<Phantom, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PIMC_WORKERS * PER_WORKER_ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(3),
    )
}

fn mo_ismcts_config(name: &str) -> TreeSearch<Phantom, strategy::Ucb1> {
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
    println!(
        "=== Phantom (4, 4, 4) PIMC/SO-ISMCTS/MO-ISMCTS strength comparison (background job) ==="
    );
    println!(
        "Matched node budget: {} total simulations/move (1 cheating tree, {} PIMC trees x {}, \
         1 ismcts tree, 1 mo_ismcts tree-per-player set). 4 strategies, {} rounds, {} games (12 \
         ordered pairs/round).",
        PIMC_WORKERS * PER_WORKER_ITERATIONS,
        PIMC_WORKERS,
        PER_WORKER_ITERATIONS,
        ROUNDS,
        ROUNDS * 12
    );
    println!();

    let mut strategies: Vec<AnySearch<Phantom>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(pimc_config("pimc")),
        AnySearch::new(ismcts_config("ismcts")),
        AnySearch::new(mo_ismcts_config("mo_ismcts")),
    ];
    let results = round_robin_multiple::<Phantom, _>(
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
        "Interpretation: the paper's own Phantom (4, 4, 4) ranking puts mo_ismcts above pimc, \
         above ismcts, above cheating (Section VI's headline strategy-fusion result) -- \
         cheating loses here specifically because knowing the true board doesn't help against \
         an opponent whose moves you still can't see, unlike the racks/hands in Ingenious/Oh \
         Hell where the mover's own information is the scarce resource. Check mo_ismcts vs pimc \
         and ismcts vs pimc against that ordering before trusting mo_ismcts's numbers on any \
         other game: a qualitatively different result here points at an implementation bug in \
         this workspace's ISMCTS/MO-ISMCTS engine code, not a new research finding."
    );
}
