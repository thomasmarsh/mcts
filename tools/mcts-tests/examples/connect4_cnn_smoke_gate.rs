//! Equal-budget compact-CNN smoke comparison against the all-zero CNN.

use std::process::ExitCode;

use game_connect4::{convnet::CnnValuePolicyNet, selfplay::CnnGumbelPlayer, Standard};
use mcts::{algorithms::mcts::gumbel::GumbelConfig, util::battle_royale};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: connect4_cnn_smoke_gate <candidate.c4cnn> [games] [sims]");
        return ExitCode::FAILURE;
    }
    let candidate = match CnnValuePolicyNet::load(&args[1]) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{}: {error}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let games: usize = args.get(2).map_or(40, |s| s.parse().expect("games"));
    let sims: u32 = args.get(3).map_or(32, |s| s.parse().expect("sims"));
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };
    let mut trained = CnnGumbelPlayer::new(candidate, cfg, 7);
    let mut zero = CnnGumbelPlayer::new(CnnValuePolicyNet::default(), cfg, 11);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game in 0..games {
        let (result, trained_first) = if game % 2 == 0 {
            (
                battle_royale::<Standard, _, _>(&mut trained, &mut zero),
                true,
            )
        } else {
            (
                battle_royale::<Standard, _, _>(&mut zero, &mut trained),
                false,
            )
        };
        match result {
            None => draws += 1,
            Some(0) if trained_first => wins += 1,
            Some(1) if !trained_first => wins += 1,
            _ => losses += 1,
        }
    }
    let share = (wins as f64 + 0.5 * draws as f64) / games as f64;
    println!("CNN candidate vs zero, equal budget: {wins}-{draws}-{losses} (W-D-L), score share {share:.3}, games={games}, sims={sims}");
    ExitCode::SUCCESS
}
