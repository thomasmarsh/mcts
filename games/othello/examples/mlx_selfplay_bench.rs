//! Real self-play throughput, CPU (`CnnValueNet`) vs. MLX
//! (`convnet::mlx::MlxCnnValueNet`), at the coordinator's actual Gumbel
//! settings -- a synthetic single-call latency win only matters if it
//! survives contact with a real self-play loop (tree-search bookkeeping,
//! move generation, D4 orientation construction all still run on the CPU
//! regardless of which evaluator is plugged in). Both evaluators go through
//! the identical
//! `GumbelCompletedQ`/`EvaluatedCutoff`/`TreeSearch` pipeline
//! (`crate::selfplay::CnnGumbelPlayer`'s exact shape) -- only the per-leaf
//! evaluator differs, so this is an apples-to-apples comparison of the one
//! thing that changed, not two different pipelines.
//!
//! Requires the `mlx` feature:
//!   cargo run --release -p game-othello --features mlx --example mlx_selfplay_bench [games] [sims]
use std::time::Instant;

use game_othello::convnet::mlx::MlxCnnValueNet;
use game_othello::convnet::CnnValueNet;
use game_othello::{Othello, State};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::evaluator::Evaluator;
use mcts::game::Game;

fn play_games<E>(net: E, games: usize, sims: u32) -> (usize, f64)
where
    E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static,
{
    type Profile<E> = Mcts<GumbelCompletedQ, EvaluatedCutoff<Othello, E>>;
    let gcfg = GumbelConfig {
        sims,
        max_considered: 8,
        ..GumbelConfig::default()
    };

    let mut total_plies = 0usize;
    let start = Instant::now();
    for g in 0..games {
        let mut search: TreeSearch<Othello, Profile<E>> = TreeSearch::default().config(
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
            let root_value = net.evaluate(&state) as f64 / mcts::evaluator::EVAL_MAGNITUDE_LIMIT as f64;
            let outcome = gumbel_search_with_root_value(&mut search, &state, &gcfg, root_value);
            state = Othello::apply(state, &outcome.action);
            total_plies += 1;
        }
    }
    (total_plies, start.elapsed().as_secs_f64())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("running {games} games at {sims} sims/move (all-zero weights on both sides)...");

    let (cpu_plies, cpu_wall) = play_games(CnnValueNet::default(), games, sims);
    println!(
        "CPU:  {cpu_plies} plies in {cpu_wall:.3}s -- {:.1} ms/ply, {:.2} games/sec",
        cpu_wall * 1000.0 / cpu_plies as f64,
        games as f64 / cpu_wall
    );

    let (mlx_plies, mlx_wall) = play_games(MlxCnnValueNet::default(), games, sims);
    println!(
        "MLX:  {mlx_plies} plies in {mlx_wall:.3}s -- {:.1} ms/ply, {:.2} games/sec",
        mlx_wall * 1000.0 / mlx_plies as f64,
        games as f64 / mlx_wall
    );

    println!(
        "speedup: {:.2}x wall-clock ({:.2}x ms/ply)",
        cpu_wall / mlx_wall,
        (cpu_wall / cpu_plies as f64) / (mlx_wall / mlx_plies as f64)
    );
}
