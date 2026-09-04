// Background strength comparison: single-tree SO-ISMCTS/MO-ISMCTS against
// their root-parallel counterparts (`SearchConfig::num_threads > 1` composed
// with `ismcts_mode`), on Phantom (4, 4, 4).
//
// Sephton, Cowling, Powley & Slaven (IEEE CEC 2014) found root
// parallelization to be the empirically strongest parallelization strategy
// for ISMCTS -- unlike tree parallelism (concurrent descent into one shared,
// growable tree), which `SearchConfig::validate()` still rejects for
// `ismcts_mode` (see its own doc comment), root parallelism needs no new
// engine machinery: each worker already runs its own independent tree, so
// `choose_action_root_parallel` runs an ordinary `ismcts_mode` search
// unmodified in every worker and merges the resulting trees' root visit
// counts exactly as it would for any other mode. This file checks whether
// that recovers a real strength gain here, not just a mechanical one.
//
// Four strategies, total node budget matched across all of them:
//
// - "ismcts"/"mo_ismcts": one tree (or one tree-per-player set), `
//   max_iterations = WORKERS * PER_WORKER_ITERATIONS`.
// - "root_parallel_ismcts"/"root_parallel_mo_ismcts": `WORKERS` independent
//   trees (or tree-per-player sets), each with `max_iterations =
//   PER_WORKER_ITERATIONS`, merged by summing root visit counts.
//
// If root parallelism is genuinely a net win here (as Sephton et al. found),
// the root-parallel variants should outperform their single-tree
// counterparts despite splitting the same total simulation budget N ways --
// each independent tree gets a real vote instead of only ever refining one
// shared estimate, and (per Sephton et al.) that variance reduction tends to
// outweigh the loss of sharing statistics within a single larger tree.
//
// Sequential execution (no rayon fan-out across games) so each root-parallel
// search gets the whole machine to itself. This is a long-running job -- run
// as a background process, not synchronously.
//
// Usage: cargo run --release --example strength_phantom_root_parallel_ismcts
use game_phantom::Phantom;
use mcts::algorithms::mcts::{profile, IsmctsMode, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

const WORKERS: usize = 4;
const PER_WORKER_ITERATIONS: usize = 2000;
const ROUNDS: usize = 15;

fn ismcts_config(name: &str) -> TreeSearch<Phantom, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(WORKERS * PER_WORKER_ITERATIONS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(1),
    )
}

fn root_parallel_ismcts_config(name: &str) -> TreeSearch<Phantom, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PER_WORKER_ITERATIONS)
            .num_threads(WORKERS)
            .ismcts_mode(IsmctsMode::SingleTree)
            .seed(2),
    )
}

fn mo_ismcts_config(name: &str) -> TreeSearch<Phantom, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(WORKERS * PER_WORKER_ITERATIONS)
            .ismcts_mode(IsmctsMode::MultiTree)
            .seed(3),
    )
}

fn root_parallel_mo_ismcts_config(name: &str) -> TreeSearch<Phantom, profile::Mcts> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(PER_WORKER_ITERATIONS)
            .num_threads(WORKERS)
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
        "=== Phantom (4, 4, 4) root-parallel ISMCTS/MO-ISMCTS strength comparison (background \
         job) ==="
    );
    println!(
        "Matched node budget: {} total simulations/move (1 ismcts tree, {} root-parallel ismcts \
         trees x {}, 1 mo_ismcts tree-per-player set, {} root-parallel mo_ismcts tree-per-player \
         sets x {}). 4 strategies, {} rounds, {} games (12 ordered pairs/round).",
        WORKERS * PER_WORKER_ITERATIONS,
        WORKERS,
        PER_WORKER_ITERATIONS,
        WORKERS,
        PER_WORKER_ITERATIONS,
        ROUNDS,
        ROUNDS * 12
    );
    println!();

    let mut strategies: Vec<AnySearch<Phantom>> = vec![
        AnySearch::new(ismcts_config("ismcts")),
        AnySearch::new(root_parallel_ismcts_config("root_parallel_ismcts")),
        AnySearch::new(mo_ismcts_config("mo_ismcts")),
        AnySearch::new(root_parallel_mo_ismcts_config("root_parallel_mo_ismcts")),
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
        "Interpretation: Sephton et al. (IEEE CEC 2014) found root parallelization to be the \
         strongest parallelization strategy for ISMCTS -- root_parallel_ismcts should beat \
         ismcts, and root_parallel_mo_ismcts should beat mo_ismcts, despite an identical total \
         simulation budget split across independent trees instead of spent on one larger tree. \
         A root-parallel variant losing to its single-tree counterpart here would point at a \
         bug in how ismcts_mode composes with num_threads (SearchConfig::validate()'s doc \
         comment lays out why the two are expected to compose cleanly), not a new research \
         finding."
    );
}
