//! Places a trained compact-CNN Connect Four net on the fixed-depth negamax
//! ladder: `CnnGumbelPlayer(candidate, sims)` vs `Connect4NegamaxPlayer(depth)`,
//! alternating colours, reporting W-D-L and score share.
//!
//! This is the ladder rung, not the sweep -- keep `games` small. Run the grid
//! (nets x sims x depths) from an orchestrated background job, not here.

use std::process::ExitCode;

use game_connect4::{
    convnet::CnnValuePolicyNet, negamax_player::Connect4NegamaxPlayer, selfplay::CnnGumbelPlayer,
    Standard,
};
use mcts::{algorithms::mcts::gumbel::GumbelConfig, util::battle_royale};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: connect4_negamax_gate <candidate.c4cnn> <depth> [games] [sims]");
        return ExitCode::FAILURE;
    }
    let candidate = match CnnValuePolicyNet::load(&args[1]) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{}: {error}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let depth: u32 = args[2].parse().expect("depth");
    let games: usize = args.get(3).map_or(40, |s| s.parse().expect("games"));
    let sims: u32 = args.get(4).map_or(32, |s| s.parse().expect("sims"));

    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };
    let mut net = CnnGumbelPlayer::new(candidate, cfg, 7);
    let mut ladder = Connect4NegamaxPlayer::new(depth);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game in 0..games {
        let (result, net_first) = if game % 2 == 0 {
            (battle_royale::<Standard, _, _>(&mut net, &mut ladder), true)
        } else {
            (battle_royale::<Standard, _, _>(&mut ladder, &mut net), false)
        };
        match result {
            None => draws += 1,
            Some(0) if net_first => wins += 1,
            Some(1) if !net_first => wins += 1,
            _ => losses += 1,
        }
    }
    let share = (wins as f64 + 0.5 * draws as f64) / games as f64;
    println!(
        "CNN candidate vs negamax-d{depth}: {wins}-{draws}-{losses} (W-D-L), score share {share:.3}, games={games}, sims={sims}"
    );
    ExitCode::SUCCESS
}
