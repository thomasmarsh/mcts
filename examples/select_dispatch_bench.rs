//! Compares a `select`-axis search with `select` fully monomorphized into
//! `TreeSearch`'s type parameter against the same search with `select`
//! erased through `mcts_tune::config_ir::DynSelect<G>`
//! (`Box<dyn ErasedSelectStrategy<G>>` under a `SelectStrategy<G>` newtype) --
//! the latter is what `mcts_tune::config_ir::build_search` actually builds.
//! `simulate`/`final_action` are erased in both builds via `DynSimulate<G>`/
//! `DynFinalAction<G>`, matching what `build_search` does regardless of how
//! `select` is handled, so those two axes contribute no difference between
//! the two columns below -- only `select`'s own dispatch mechanism varies.
//! `backprop` is fixed to `Classic` in both builds for the same reason.
//!
//! Reports iterations/sec, mean and standard deviation over several seeds
//! per (game, select family) pair, single-threaded, at a fixed iteration
//! budget per game. Run with `cargo run --release --example
//! select_dispatch_bench`; an optional argument overrides every game's
//! default iteration budget: `cargo run --release --example
//! select_dispatch_bench -- 2000`.

use std::time::Instant;

use game_margo::{Margo, State as MargoState};
use game_nim::{Nim, NimState};
use game_othello::{Othello, State as OthelloState};
use mcts::backprop;
use mcts::game::Game;
use mcts::select::{Rave, SelectStrategy, Ucb1};
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{SearchConfig, TreeSearch};
use mcts_tune::config_ir::{resolve_select, DynFinalAction, DynSimulate, SelectSpec};

/// Enough seeds to see run-to-run noise without the benchmark itself taking
/// too long at the expensive end (`margo`).
const SEEDS: [u64; 5] = [0x5eed0, 0x5eed1, 0x5eed2, 0x5eed3, 0x5eed4];

/// Bounds a rollout's length so a uniform playout that lands in a repeated
/// position (Margo enforces only single-position ko, not full positional
/// superko) can't run unbounded -- see `games/margo/examples/bench_mcgs.rs`'s
/// `strong_config` for the same guard.
const MAX_PLAYOUT_DEPTH: usize = 200;

/// Builds and runs one fixed-iteration, single-threaded search, returning
/// iterations/sec. Generic over `S1` so the same function drives both the
/// statically monomorphized path (`S1` a concrete `SelectStrategy`) and the
/// erased path (`S1 = DynSelect<G>`) -- `simulate`/`final_action`/`backprop`
/// are identical in both calls, isolating `select` as the only variable.
fn measure<G, S1>(select: S1, state: &G::S, iterations: usize, seed: u64) -> f64
where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
{
    type Strat<G, S1> = Compose<S1, DynSimulate<G>, backprop::Classic, DynFinalAction<G>>;
    let config = SearchConfig::<G, Strat<G, S1>>::new()
        .max_iterations(iterations)
        .max_playout_depth(MAX_PLAYOUT_DEPTH)
        .num_tree_threads(1)
        .seed(seed)
        .select(select)
        .simulate(DynSimulate::default())
        .backprop(backprop::Classic)
        .final_action(DynFinalAction::default());
    let mut search = TreeSearch::<G, Strat<G, S1>>::new().config(config);
    let started = Instant::now();
    let _ = search.choose_action(state);
    iterations as f64 / started.elapsed().as_secs_f64()
}

fn mean_stddev(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    (mean, variance.sqrt())
}

/// Runs both the static and erased builds of one (game, select family) pair
/// across `SEEDS` and prints the comparison. `make_static` constructs a
/// fresh concrete `S1` per seed; `spec` names the same family/parameters for
/// `resolve_select` to erase into a `DynSelect<G>` per seed.
fn compare<G, S1>(
    game: &str,
    select_name: &str,
    state: &G::S,
    iterations: usize,
    make_static: impl Fn() -> S1,
    spec: &SelectSpec,
) where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
{
    let static_rates: Vec<f64> = SEEDS
        .iter()
        .map(|&seed| measure::<G, _>(make_static(), state, iterations, seed))
        .collect();
    let erased_rates: Vec<f64> = SEEDS
        .iter()
        .map(|&seed| measure::<G, _>(resolve_select::<G>(spec), state, iterations, seed))
        .collect();

    let (static_mean, static_sd) = mean_stddev(&static_rates);
    let (erased_mean, erased_sd) = mean_stddev(&erased_rates);
    let delta_pct = 100.0 * (erased_mean - static_mean) / static_mean;

    println!(
        "{game:<8} {select_name:<6} static {static_mean:>10.0} +/- {static_sd:>7.0} it/s   \
         erased {erased_mean:>10.0} +/- {erased_sd:>7.0} it/s   delta {delta_pct:>+6.2}%"
    );
}

/// `Ucb1::default()`'s `select::Ucb1` and the `SelectSpec::Ucb1` that
/// `resolve_select` would erase it from must agree field-for-field, or the
/// two columns would be measuring different searches rather than the same
/// search built two ways.
fn ucb1_spec() -> SelectSpec {
    SelectSpec::Ucb1 {
        c: Ucb1::default().exploration_constant,
    }
}

/// Same agreement requirement as `ucb1_spec`, against `Rave::default()`.
fn rave_spec() -> SelectSpec {
    let default = Rave::default();
    SelectSpec::Rave {
        threshold: default.threshold,
        schedule: default.schedule,
        ucb: default.ucb,
    }
}

fn run_game<G>(name: &str, state: G::S, iterations: usize)
where
    G: Game + 'static,
    G::S: std::fmt::Display,
{
    compare::<G, Ucb1>(name, "ucb1", &state, iterations, Ucb1::default, &ucb1_spec());
    compare::<G, Rave>(name, "rave", &state, iterations, Rave::default, &rave_spec());
}

fn main() {
    let iterations_override: Option<usize> = std::env::args().nth(1).map(|s| {
        s.parse()
            .expect("iteration count argument must be a positive integer")
    });

    println!(
        "select-axis dyn-dispatch benchmark: monomorphized select vs DynSelect<G> \
         ({} seeds/pair, single-threaded, simulate/final_action erased in both columns)\n",
        SEEDS.len()
    );

    run_game::<Nim>("nim", NimState::default(), iterations_override.unwrap_or(50_000));
    run_game::<Othello>(
        "othello",
        OthelloState::default(),
        iterations_override.unwrap_or(8_000),
    );
    run_game::<Margo>(
        "margo",
        MargoState::default(),
        iterations_override.unwrap_or(1_500),
    );
}
