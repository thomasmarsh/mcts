//! Places [`ClassicalMctsPlayer`] (plain-UCT MCTS over a heuristic-cutoff
//! playout, fixed thread count and playout-cutoff depth, variable per-move
//! time budget) on the same fixed-depth negamax ladder used by
//! `connect4_negamax_gate`: `ClassicalMctsPlayer(max_time)` vs
//! `Connect4NegamaxPlayer(depth)`, alternating colours, reporting W-D-L and
//! score share.
//!
//! This is the ladder rung, not the sweep -- keep `games` small. Run the
//! grid (time budgets x depths) from an orchestrated background job, not
//! here.

use std::process::ExitCode;
use std::time::Duration;

use game_connect4::{
    classical_mcts_player::ClassicalMctsPlayer, negamax_player::Connect4NegamaxPlayer, Standard,
};
use mcts::util::battle_royale;

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: connect4_classical_negamax_gate <max_time_ms> <depth> [games]");
        return ExitCode::FAILURE;
    }
    let max_time_ms: u64 = args[1].parse().expect("max_time_ms");
    let depth: u32 = args[2].parse().expect("depth");
    let games: usize = args.get(3).map_or(40, |s| s.parse().expect("games"));

    let max_time = Duration::from_millis(max_time_ms);
    let mut classical = ClassicalMctsPlayer::new(max_time);
    let mut ladder = Connect4NegamaxPlayer::new(depth);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game in 0..games {
        let (result, classical_first) = if game % 2 == 0 {
            (
                battle_royale::<Standard, _, _>(&mut classical, &mut ladder),
                true,
            )
        } else {
            (
                battle_royale::<Standard, _, _>(&mut ladder, &mut classical),
                false,
            )
        };
        match result {
            None => draws += 1,
            Some(0) if classical_first => wins += 1,
            Some(1) if !classical_first => wins += 1,
            _ => losses += 1,
        }
    }
    let share = (wins as f64 + 0.5 * draws as f64) / games as f64;
    println!(
        "classical-mcts({max_time_ms}ms) vs negamax-d{depth}: {wins}-{draws}-{losses} (W-D-L), score share {share:.3}, games={games}"
    );
    ExitCode::SUCCESS
}
