//! Real self-play throughput: batched-GPU self-play (this crate's
//! `gumbel_explore`, one call per ply covering every still-live game in
//! the batch, routed through `MlxOthelloOracle`'s single stacked GPU call
//! per round) vs. the existing per-node engine (`crates/mcts`'s
//! `TreeSearch`/`GumbelCompletedQ`/`EvaluatedCutoff`, one game played
//! start-to-finish before the next begins -- the same shape
//! `games/othello/examples/mlx_selfplay_bench.rs` already benchmarks).
//! Both sides use `MlxCnnValueNet` (GPU, all-zero weights -- correctness
//! of the output doesn't matter for a throughput measurement, only
//! latency): once an MLX/GPU evaluator is validated for a game, it is the
//! default everywhere for that game, not just the call site that first
//! motivated it, and CPU (`CnnValueNet`/`OthelloOracle`) is opt-in only --
//! see `crates/mcts-batch/src/othello.rs`'s `MlxOthelloOracle` for why this
//! needs its own oracle rather than just swapping the evaluator type
//! parameter (spreading per-state GPU calls across `rayon`'s CPU threads,
//! the way `OthelloOracle<E>`'s generic `map_init` path does for a CPU
//! evaluator, would serialize on the one physical GPU instead of helping).
//!
//! Move selection in both loops is the simplest thing that produces a real
//! finished game (argmax of the improved policy / completed-Q, no
//! temperature sampling) -- this measures search throughput, not the
//! self-play *data* a real training loop would record; that wiring
//! (visit-proportional sampling, `RecordV2` output) is a separate,
//! later step once this gate itself passes.
//!
//!   cargo run --release -p mcts-batch --example bench_othello_selfplay [games] [sims]

use std::time::Instant;

use game_othello::convnet::mlx::MlxCnnValueNet;
use game_othello::{Move, Othello, State};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::evaluator::{Evaluator, EVAL_MAGNITUDE_LIMIT};
use mcts::game::Game;
use mcts_batch::othello::MlxOthelloOracle;
use mcts_batch::{gumbel_explore, Config};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// The existing per-node engine, one game at a time -- identical shape to
/// `games/othello/examples/mlx_selfplay_bench.rs::play_games`.
fn play_games_per_node(games: usize, sims: u32) -> (usize, f64) {
    type Profile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Othello, MlxCnnValueNet>>;
    let gcfg = GumbelConfig { sims, max_considered: 8, ..GumbelConfig::default() };
    let net = MlxCnnValueNet::default();

    let mut total_plies = 0usize;
    let start = Instant::now();
    for g in 0..games {
        let mut search: TreeSearch<Othello, Profile> = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .select(GumbelCompletedQ::with_config(gcfg))
                .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                .with_policy_logits(net.clone())
                .seed(1000 + g as u64 * 100_000),
        );
        let mut state = State::default();
        while !Othello::is_terminal(&state) {
            let root_value = net.evaluate(&state) as f64 / EVAL_MAGNITUDE_LIMIT as f64;
            let outcome = gumbel_search_with_root_value(&mut search, &state, &gcfg, root_value);
            state = Othello::apply(state, &outcome.action);
            total_plies += 1;
        }
    }
    (total_plies, start.elapsed().as_secs_f64())
}

/// This crate's batched engine: every still-live game advances one ply per
/// `gumbel_explore` call, so that one call's oracle batching covers the
/// whole live set at once instead of one leaf at a time.
fn play_games_batched(games: usize, sims: u32) -> (usize, f64) {
    let oracle = MlxOthelloOracle::new(MlxCnnValueNet::default());
    let cfg = Config { num_simulations: sims as usize, num_considered_actions: 8, ..Config::default() };
    let mut rng = SmallRng::seed_from_u64(1000);

    let mut live: Vec<State> = vec![State::default(); games];
    let mut orig_idx: Vec<usize> = (0..games).collect();
    let mut plies = vec![0usize; games];

    let start = Instant::now();
    let mut ply_num = 0usize;
    while !live.is_empty() {
        let ply_start = Instant::now();
        let tree = gumbel_explore(&cfg, &oracle, &live, &mut rng);
        ply_num += 1;
        eprintln!(
            "ply {ply_num}: live={} gumbel_explore={:.1}ms cumulative={:.1}s",
            live.len(),
            ply_start.elapsed().as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64()
        );
        let mut next_live = Vec::with_capacity(live.len());
        let mut next_idx = Vec::with_capacity(orig_idx.len());
        for (i, &oi) in orig_idx.iter().enumerate() {
            let qs = tree.completed_qvalues(i, tree.root());
            let aid = qs
                .iter()
                .enumerate()
                .filter(|(_, &q)| q.is_finite())
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(aid, _)| aid)
                .expect("a non-terminal state always has at least one legal action");
            let next_state = Othello::apply(live[i], &Move(aid as u8));
            plies[oi] += 1;
            if !Othello::is_terminal(&next_state) {
                next_live.push(next_state);
                next_idx.push(oi);
            }
        }
        live = next_live;
        orig_idx = next_idx;
    }
    (plies.iter().sum(), start.elapsed().as_secs_f64())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    // Per-node plays one game at a time start-to-finish (inherently
    // sequential, ~11s/game at sims=16), so a large `games` count for the
    // batched side (where batching over many simultaneous games is the
    // whole point) makes the per-node comparison impractically slow to
    // re-run every time. `--skip-per-node` runs only the batched engine;
    // compare its own ms/ply against a per-node number from a prior run at
    // the same `sims` (ms/ply is a per-move rate, independent of `games`).
    let skip_per_node = args.next().as_deref() == Some("--skip-per-node");

    println!("running {games} games at {sims} sims/move (all-zero weights, GPU/MLX evaluator both sides)...");

    let pn = if skip_per_node {
        None
    } else {
        let (pn_plies, pn_wall) = play_games_per_node(games, sims);
        println!(
            "per-node:  {pn_plies} plies in {pn_wall:.3}s -- {:.1} ms/ply, {:.2} games/sec",
            pn_wall * 1000.0 / pn_plies as f64,
            games as f64 / pn_wall
        );
        Some((pn_plies, pn_wall))
    };

    let (b_plies, b_wall) = play_games_batched(games, sims);
    println!(
        "batched:   {b_plies} plies in {b_wall:.3}s -- {:.1} ms/ply, {:.2} games/sec",
        b_wall * 1000.0 / b_plies as f64,
        games as f64 / b_wall
    );

    if let Some((pn_plies, pn_wall)) = pn {
        println!(
            "speedup: {:.2}x wall-clock ({:.2}x ms/ply) -- gate is >= 4x",
            pn_wall / b_wall,
            (pn_wall / pn_plies as f64) / (b_wall / b_plies as f64)
        );
    }
}
