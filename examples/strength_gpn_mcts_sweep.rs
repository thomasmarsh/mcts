// C_pn / bias sweep for Generalized Proof-Number MCTS (Kowalski, Soemers,
// Kosakowski & Winands, arXiv:2506.13249). The single-point run in
// `strength_gpn_mcts.rs` (C_pn = 1.0, bias = Max) showed GPN swinging from a
// dominant win at 2-player Focus to significantly worse than the plain-solver
// baseline at 4-player Focus and 3-player Ingenious. The paper reports the
// best `C_pn` is strongly game- *and* formula-dependent, so this script
// sweeps `C_pn` in {0.1, 0.5, 1.0, 2.0, 5.0} against each `GpnBias`
// {Max, Sum, Rank} for every game, to find whether a per-game setting
// recovers the many-player cases (a near-zero `C_pn` effectively turns the
// bias off in wide fields, the obvious first hypothesis).
//
// Each (game, C_pn, bias) cell plays one GPN seat vs `P-1` identical
// plain-solver seats, GPN seat rotated over all seats. Null win rate is
// `1/P`. Wilson 95% CI via `mcts_bench::tournament::Result`.
//
// Long-running background job -- run detached. This is O(games * cells);
// with the defaults (5 C_pn * 3 bias * 4 games, N=10 rounds/seat, 200ms/move)
// that is 60 cells and ~1750 games. Every game is independent, so the whole
// sweep fans out across a rayon pool (`--workers N`, default = all cores).
// Per-search tree threads would change strength rather than throughput (each
// move is wall-clock-budgeted), so game-level fan-out is the right speedup.
// Within a game the GPN and plain seats still alternate under identical
// contention, so the head-to-head stays fair; only exact run-to-run
// reproducibility is lost (already true of any `max_time` search -- drop to
// `max_iterations` in the configs if a deterministic sweep is needed).
// Narrow further with filter args.
//
// Usage:
//   cargo run --release --example strength_gpn_mcts_sweep
//   cargo run --release --example strength_gpn_mcts_sweep -- --workers 8 --rounds 4 --budget-ms 100
//   cargo run --release --example strength_gpn_mcts_sweep -- focus4 sum   # substring filters on the cell label
use std::time::Duration;

use game_focus::{Focus, State as FocusState};
use game_ingenious::{Ingenious, State as IngeniousState};
use mcts::game::{Game, PlayerIndex};
use mcts::algorithms::mcts::select::GpnBias;
use mcts::algorithms::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const C: f64 = 1.414;
const MAX_PLIES: usize = 2000;
const DEFAULT_ROUNDS_PER_SEAT: usize = 10;
const DEFAULT_BUDGET_MS: u64 = 200;

const C_PN_GRID: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 5.0];
const BIAS_GRID: [GpnBias; 3] = [GpnBias::Max, GpnBias::Sum, GpnBias::Rank];

struct Opts {
    rounds_per_seat: usize,
    budget: Duration,
    workers: usize,
    filters: Vec<String>,
}

fn parse_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        eprintln!("{flag} needs a positive integer");
        std::process::exit(2);
    })
}

fn parse_opts() -> Opts {
    let mut rounds_per_seat = DEFAULT_ROUNDS_PER_SEAT;
    let mut budget_ms = DEFAULT_BUDGET_MS;
    let mut workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut filters = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rounds" => rounds_per_seat = parse_usize(&mut args, "--rounds"),
            "--budget-ms" => budget_ms = parse_usize(&mut args, "--budget-ms") as u64,
            "--workers" => workers = parse_usize(&mut args, "--workers").max(1),
            other => filters.push(other.to_lowercase()),
        }
    }
    Opts {
        rounds_per_seat,
        budget: Duration::from_millis(budget_ms),
        workers,
        filters,
    }
}

fn plain_config<G: Game>(seed: u64, budget: Duration) -> SearchConfig<G, strategy::Ucb1> {
    SearchConfig::new()
        .expand_threshold(1)
        .use_mcts_solver(true)
        .q_init(QInit::Loss)
        .max_time(budget)
        .seed(seed)
        .select(select::Ucb1::with_c(C))
}

fn gpn_config<G: Game>(
    seed: u64,
    budget: Duration,
    c_pn: f64,
    bias: GpnBias,
) -> SearchConfig<G, strategy::Ucb1Gpn> {
    SearchConfig::new()
        .expand_threshold(1)
        .use_mcts_solver(true)
        .q_init(QInit::Loss)
        .max_time(budget)
        .seed(seed)
        .select(select::GpnUct::with_c(C, c_pn).bias(bias))
}

/// Plays one game with `gpn_seat` running GPN-MCTS at `(c_pn, bias)` and every
/// other seat the identical plain solver. Returns the winning seat, ply
/// count, and whether the ply cap was hit.
fn play_one_game<G, F>(
    gpn_seat: usize,
    seed: u64,
    num_seats: usize,
    budget: Duration,
    c_pn: f64,
    bias: GpnBias,
    initial: F,
) -> (Option<usize>, usize, bool)
where
    G: Game,
    F: Fn() -> G::S,
{
    let mut strategies: Vec<AnySearch<G>> = (0..num_seats)
        .map(|seat| {
            let s = seed * 100 + seat as u64;
            if seat == gpn_seat {
                AnySearch::new(
                    TreeSearch::<G, strategy::Ucb1Gpn>::new()
                        .config(gpn_config(s, budget, c_pn, bias)),
                )
            } else {
                AnySearch::new(
                    TreeSearch::<G, strategy::Ucb1>::new().config(plain_config(s, budget)),
                )
            }
        })
        .collect();

    let mut state = initial();
    for ply in 0..MAX_PLIES {
        if G::is_terminal(&state) {
            return (G::winner(&state).map(|w| w.to_index()), ply, false);
        }
        let mover = G::player_to_move(&state).to_index();
        let action = strategies[mover].choose_action(&state);
        state = G::apply(state, &action);
    }
    (None, MAX_PLIES, true)
}

/// A monomorphized single-game runner for one `GameCase`: given the GPN
/// seat, seed, time budget and `(c_pn, bias)`, play a game and report the
/// winning seat and whether the ply cap was hit. Boxed `Send + Sync` so the
/// sweep can fan games across a rayon pool.
type PlayFn =
    Box<dyn Fn(usize, u64, Duration, f64, GpnBias) -> (Option<usize>, bool) + Send + Sync>;

fn play_fn<G, F>(num_seats: usize, initial: F) -> PlayFn
where
    G: Game,
    F: Fn(u64) -> G::S + Send + Sync + 'static,
{
    Box::new(move |gpn_seat, seed, budget, c_pn, bias| {
        let (winner, _plies, capped) =
            play_one_game::<G, _>(gpn_seat, seed, num_seats, budget, c_pn, bias, || {
                initial(seed)
            });
        (winner, capped)
    })
}

fn fmt_result(r: &GameResult, null: f64, capped: usize) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    let verdict = if lo > null {
        "GPN gain"
    } else if hi < null {
        "GPN worse"
    } else {
        "spans null"
    };
    format!(
        "W={:>3} L={:>3} D={:>3}  win={:>5.1}% [{:>5.1}, {:>5.1}]  null={:>4.1}%  {}{}",
        r.wins,
        r.losses,
        r.draws,
        point * 100.0,
        lo * 100.0,
        hi * 100.0,
        null * 100.0,
        verdict,
        if capped > 0 {
            format!("  ({capped} PLY CAP)")
        } else {
            String::new()
        }
    )
}

/// One game in the sweep: its label, seat count, and a monomorphized
/// single-game runner.
struct GameCase {
    label: &'static str,
    num_seats: usize,
    play: PlayFn,
}

/// One `(game, c_pn, bias)` cell -- the unit a result row is reported for.
struct Cell {
    game_idx: usize,
    label: String,
    num_seats: usize,
    null: f64,
    c_pn: f64,
    bias: GpnBias,
}

fn main() {
    let opts = parse_opts();

    let games: Vec<GameCase> = vec![
        GameCase {
            label: "focus2",
            num_seats: 2,
            play: play_fn::<Focus<2>, _>(2, |_| FocusState::<2>::default()),
        },
        GameCase {
            label: "focus3",
            num_seats: 3,
            play: play_fn::<Focus<3>, _>(3, |_| FocusState::<3>::default()),
        },
        GameCase {
            label: "focus4",
            num_seats: 4,
            play: play_fn::<Focus<4>, _>(4, |_| FocusState::<4>::default()),
        },
        GameCase {
            label: "ingenious3",
            num_seats: 3,
            play: play_fn::<Ingenious<3>, _>(3, IngeniousState::<3>::new),
        },
    ];

    let cells: Vec<Cell> = games
        .iter()
        .enumerate()
        .flat_map(|(game_idx, g)| {
            BIAS_GRID.into_iter().flat_map(move |bias| {
                C_PN_GRID.into_iter().map(move |c_pn| Cell {
                    game_idx,
                    label: format!("{} c_pn={c_pn} {bias:?}", g.label).to_lowercase(),
                    num_seats: g.num_seats,
                    null: 1.0 / g.num_seats as f64,
                    c_pn,
                    bias,
                })
            })
        })
        .filter(|cell| {
            opts.filters.is_empty() || opts.filters.iter().all(|f| cell.label.contains(f.as_str()))
        })
        .collect();

    println!("=== GPN-MCTS C_pn / bias sweep (background job) ===");
    println!(
        "budget={:?}  rounds/seat={}  workers={}  C={C}  C_pn grid={C_PN_GRID:?}  bias grid={BIAS_GRID:?}",
        opts.budget, opts.rounds_per_seat, opts.workers
    );
    if !opts.filters.is_empty() {
        println!("cell-label filters: {:?}", opts.filters);
    }
    let total_games: usize = cells
        .iter()
        .map(|c| opts.rounds_per_seat * c.num_seats)
        .sum();
    println!("{} cells, {total_games} games total", cells.len());
    println!();

    // Flatten every cell into its individual games, run the lot across a
    // rayon pool, then fold the per-game outcomes back per cell. Games are
    // independent; the fold below is order-insensitive so the reported
    // tallies don't depend on task completion order.
    let tasks: Vec<(usize, usize, u64)> = cells
        .iter()
        .enumerate()
        .flat_map(|(cell_idx, cell)| {
            (0..opts.rounds_per_seat).flat_map(move |round| {
                (0..cell.num_seats).map(move |gpn_seat| {
                    (
                        cell_idx,
                        gpn_seat,
                        (round * cell.num_seats + gpn_seat) as u64,
                    )
                })
            })
        })
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.workers)
        .build()
        .expect("build rayon pool");

    let outcomes: Vec<(usize, usize, Option<usize>, bool)> = pool.install(|| {
        use rayon::prelude::*;
        tasks
            .par_iter()
            .map(|&(cell_idx, gpn_seat, seed)| {
                let cell = &cells[cell_idx];
                let (winner, capped) =
                    (games[cell.game_idx].play)(gpn_seat, seed, opts.budget, cell.c_pn, cell.bias);
                (cell_idx, gpn_seat, winner, capped)
            })
            .collect()
    });

    let mut results = vec![(GameResult::default(), 0usize); cells.len()];
    for (cell_idx, gpn_seat, winner, capped) in outcomes {
        let (r, cap) = &mut results[cell_idx];
        if capped {
            *cap += 1;
        }
        match winner {
            Some(w) if w == gpn_seat => r.wins += 1,
            Some(_) => r.losses += 1,
            None => r.draws += 1,
        }
    }

    let mut current_game = usize::MAX;
    for (cell, (result, capped)) in cells.iter().zip(&results) {
        if cell.game_idx != current_game {
            current_game = cell.game_idx;
            let g = &games[cell.game_idx];
            println!(
                "--- {} ({} seats, {} games/cell) ---",
                g.label,
                g.num_seats,
                opts.rounds_per_seat * g.num_seats
            );
        }
        println!(
            "  c_pn={:<3} {:<4?}  {}",
            cell.c_pn,
            cell.bias,
            fmt_result(result, cell.null, *capped)
        );
    }
}
