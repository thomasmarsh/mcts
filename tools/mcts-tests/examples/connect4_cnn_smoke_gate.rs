//! Equal-budget compact-CNN smoke comparison against the all-zero CNN, or
//! against a second trained CNN head via `--opponent <weights.c4cnn>`.

use std::process::ExitCode;

use game_connect4::{convnet::CnnValuePolicyNet, selfplay::CnnGumbelPlayer, Standard};
use mcts::{algorithms::mcts::gumbel::GumbelConfig, util::battle_royale};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: connect4_cnn_smoke_gate <candidate.c4cnn> [games] [sims] [--opponent <weights.c4cnn>]");
        return ExitCode::FAILURE;
    }
    let load = |path: &str| match CnnValuePolicyNet::load(path) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{path}: {error}");
            std::process::exit(1);
        }
    };
    let candidate = load(&args[1]);
    let positional: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let games: usize = positional.first().map_or(40, |s| s.parse().expect("games"));
    let sims: u32 = positional.get(1).map_or(32, |s| s.parse().expect("sims"));
    let opponent = args
        .windows(2)
        .find(|w| w[0] == "--opponent")
        .map_or_else(CnnValuePolicyNet::default, |w| load(&w[1]));
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };
    let mut trained = CnnGumbelPlayer::new(candidate, cfg, 7);
    let mut zero = CnnGumbelPlayer::new(opponent, cfg, 11);
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
    println!("CNN candidate vs opponent, equal budget: {wins}-{draws}-{losses} (W-D-L), score share {share:.3}, games={games}, sims={sims}");
    ExitCode::SUCCESS
}
