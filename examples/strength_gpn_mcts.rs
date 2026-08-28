// Background strength comparison: Generalized Proof-Number MCTS (Kowalski,
// Soemers, Kosakowski & Winands, "Generalized Proof-Number Monte-Carlo Tree
// Search", arXiv:2506.13249, 2025) vs a plain MCTS-Solver baseline.
//
// GPN-MCTS's contribution over `Ucb1Pn` (the 2023 PN-MCTS): proof numbers
// are tracked *per player* (`Node::player_pn`, `backprop::derive_player_pn`)
// rather than as a single per-mover negamax pair, and selection is biased by
// the simpler PNMax / PNSum formulas instead of a per-update sibling sort.
// The per-player framing is what makes the technique sound beyond two
// players, so this script exercises it on 3- and 4-player Focus (sudden
// death, capture-quota rule) and 3-player Ingenious (race to completion), as
// well as 2-player Focus as a sanity check against the paper's mostly
// two-player headline.
//
// Both seats run `use_mcts_solver(true)` with identical UCB1 exploration and
// time budget; the only difference is `select::GpnUct` vs `select::Ucb1`.
// Which seat is GPN rotates across games to cancel positional bias, folded
// into the same `mcts_bench::tournament::Result` / Wilson-CI machinery the
// other strength_* scripts use. The no-effect null is `1/P` (one GPN seat in
// a field of P), not 50%.
//
// Long-running background job -- run detached. Tune `ROUNDS_PER_SEAT` /
// `MOVE_BUDGET` / `C_PN` / `BIAS` for the machine and game; the paper found
// the best `C_pn` strongly game- and formula-dependent.
//
// Usage: cargo run --release --example strength_gpn_mcts
use std::time::Duration;

use game_focus::{Focus, State as FocusState};
use game_ingenious::{Ingenious, State as IngeniousState};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::select::GpnBias;
use mcts::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::AnySearch;
use mcts_bench::tournament::Result as GameResult;

const MOVE_BUDGET: Duration = Duration::from_millis(200);
const ROUNDS_PER_SEAT: usize = 10;
const MAX_PLIES: usize = 2000;
const C: f64 = 1.414;
const C_PN: f64 = 1.0;
const BIAS: GpnBias = GpnBias::Max;

fn plain_config<G: Game>(seed: u64) -> SearchConfig<G, strategy::Ucb1> {
    SearchConfig::new()
        .expand_threshold(1)
        .use_mcts_solver(true)
        .q_init(QInit::Loss)
        .max_time(MOVE_BUDGET)
        .seed(seed)
        .select(select::Ucb1::with_c(C))
}

fn gpn_config<G: Game>(seed: u64) -> SearchConfig<G, strategy::Ucb1Gpn> {
    SearchConfig::new()
        .expand_threshold(1)
        .use_mcts_solver(true)
        .q_init(QInit::Loss)
        .max_time(MOVE_BUDGET)
        .seed(seed)
        .select(select::GpnUct::with_c(C, C_PN).bias(BIAS))
}

/// Plays one game with `gpn_seat` running GPN-MCTS and every other seat the
/// identical solver without it. Returns the winning seat, ply count, and
/// whether the ply cap was hit.
fn play_one_game<G, F>(
    gpn_seat: usize,
    seed: u64,
    num_seats: usize,
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
                AnySearch::new(TreeSearch::<G, strategy::Ucb1Gpn>::new().config(gpn_config(s)))
            } else {
                AnySearch::new(TreeSearch::<G, strategy::Ucb1>::new().config(plain_config(s)))
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

fn run<G, F>(label: &str, num_seats: usize, initial: F) -> GameResult
where
    G: Game,
    F: Fn(u64) -> G::S,
{
    println!("--- {label} ({} games) ---", ROUNDS_PER_SEAT * num_seats);
    let mut result = GameResult::default();
    for round in 0..ROUNDS_PER_SEAT {
        for gpn_seat in 0..num_seats {
            let seed = (round * num_seats + gpn_seat) as u64;
            let (winner, plies, capped) =
                play_one_game::<G, _>(gpn_seat, seed, num_seats, || initial(seed));
            let tag = match winner {
                Some(w) if w == gpn_seat => {
                    result.wins += 1;
                    "gpn"
                }
                Some(_) => {
                    result.losses += 1;
                    "plain"
                }
                None => {
                    result.draws += 1;
                    "draw"
                }
            };
            println!(
                "  round {round} gpn_seat {gpn_seat}: {tag} in {plies} plies{}",
                if capped { " (PLY CAP HIT)" } else { "" }
            );
        }
    }
    result
}

fn fmt_result(r: &GameResult, null: f64) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "gpn W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson), null={:.1}%",
        r.wins,
        r.losses,
        r.draws,
        r.total(),
        point * 100.0,
        lo * 100.0,
        hi * 100.0,
        null * 100.0,
    )
}

fn main() {
    println!("=== GPN-MCTS strength comparison (background job) ===");
    println!(
        "Move budget: {MOVE_BUDGET:?}, {ROUNDS_PER_SEAT} rounds/seat, C_pn={C_PN}, bias={BIAS:?}"
    );
    println!();

    let focus2 = run::<Focus<2>, _>("2-player Focus", 2, |_| FocusState::<2>::default());
    println!("{}\n", fmt_result(&focus2, 1.0 / 2.0));

    let focus3 = run::<Focus<3>, _>("3-player Focus", 3, |_| FocusState::<3>::default());
    println!("{}\n", fmt_result(&focus3, 1.0 / 3.0));

    let focus4 = run::<Focus<4>, _>("4-player Focus", 4, |_| FocusState::<4>::default());
    println!("{}\n", fmt_result(&focus4, 1.0 / 4.0));

    let ing3 = run::<Ingenious<3>, _>("3-player Ingenious", 3, IngeniousState::<3>::new);
    println!("{}\n", fmt_result(&ing3, 1.0 / 3.0));
}
