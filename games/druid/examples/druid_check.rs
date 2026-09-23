//! Plumbing checks against a Druid CNN checkpoint, printed as one JSON line.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example druid_check -p game-druid -- \
//!     forward --weights gen1.bin --shard shards/gen0.bin [--count 64]
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example druid_check -p game-druid -- \
//!     self-match --config games/druid/cnn/az-druid-5x5.toml --weights gen1.bin \
//!     [--openings 20] [--opening-plies 4] [--seed 1] [--max-plies 300]
//! ```
//!
//! `forward` evaluates the first positions of a shard in one MLX call, so the torch forward on
//! the same positions can be compared. `self-match` plays one checkpoint against itself from
//! seeded random openings, each opening from both seats; the agent is a deterministic function of
//! the position, so every pair is an exact mirror and the score is exactly one half.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use game_druid::cnn::agent::{CnnAgent, Kind};
use game_druid::cnn::config::load;
use game_druid::cnn::encode::{legal_moves, planes};
use game_druid::cnn::shard::read_shard;
use game_druid::{DruidSplit, HashedState, Player, Size};
use grid_cnn::{Net, Weights};
use mcts::algorithms::Search;
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Default)]
struct Args {
    config: String,
    weights: String,
    shard: String,
    count: usize,
    openings: usize,
    opening_plies: usize,
    seed: u64,
    max_plies: usize,
}

fn parse() -> (String, Args) {
    let mut it = std::env::args().skip(1);
    let mode = it.next().expect("first argument: forward | self-match");
    let mut a = Args { count: 64, openings: 20, opening_plies: 4, seed: 1, max_plies: 300, ..Args::default() };
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => a.config = val(),
            "--weights" => a.weights = val(),
            "--shard" => a.shard = val(),
            "--count" => a.count = val().parse().expect("--count takes an integer"),
            "--openings" => a.openings = val().parse().expect("--openings takes an integer"),
            "--opening-plies" => a.opening_plies = val().parse().expect("--opening-plies takes an integer"),
            "--seed" => a.seed = val().parse().expect("--seed takes an integer"),
            "--max-plies" => a.max_plies = val().parse().expect("--max-plies takes an integer"),
            other => panic!("unknown argument {other}"),
        }
    }
    (mode, a)
}

fn forward(a: &Args) {
    let w = Weights::load(&a.weights).unwrap_or_else(|e| panic!("{}: {e}", a.weights));
    let (size, records) = read_shard(Path::new(&a.shard)).unwrap_or_else(|e| panic!("{}: {e}", a.shard));
    assert_eq!(size, w.geometry.size);
    let records = &records[..a.count.min(records.len())];
    let planes: Vec<f32> = records.iter().flat_map(|r| planes(&r.fields.to_state(size))).collect();
    let out = Net::new(&w).forward(&planes, records.len());
    println!("{}", serde_json::json!({ "values": out.values, "logits": out.logits }));
}

/// Plays one game from `opening`; `black`/`white` are the two seats' agents. `None` is a draw
/// (or a game still running at `max_plies`).
fn play<const N: usize>(
    opening: &HashedState,
    black: &mut CnnAgent<N>,
    white: &mut CnnAgent<N>,
    max_plies: usize,
) -> Option<Player> {
    let mut s = opening.clone();
    for _ in 0..max_plies {
        if DruidSplit::is_terminal(&s) {
            break;
        }
        let agent = if DruidSplit::player_to_move(&s) == Player::Black { &mut *black } else { &mut *white };
        let mv = agent.choose_action(&s);
        s = DruidSplit::apply(s, &mv);
    }
    DruidSplit::winner(&s)
}

fn self_match<const N: usize>(a: &Args, cfg: &game_druid::cnn::config::Config, w: Weights) {
    let w = Arc::new(w);
    let mk = |name: &str| {
        CnnAgent::<N>::new(
            name,
            &w,
            cfg.play.search.batch_config(),
            Kind::Gumbel,
            cfg.play.chunk_size,
            Duration::default(),
            0x51,
        )
    };
    let (mut x, mut y) = (mk("cnn"), mk("cnn-b"));
    let mut rng = SmallRng::seed_from_u64(a.seed);
    let (mut a_wins, mut b_wins, mut draws) = (0u32, 0u32, 0u32);
    for _ in 0..a.openings {
        let mut opening = HashedState::new(Size { w: N as u8, h: N as u8 });
        for _ in 0..a.opening_plies {
            if DruidSplit::is_terminal(&opening) {
                break;
            }
            let (moves, _) = legal_moves(&opening);
            opening = DruidSplit::apply(opening, &moves[rng.gen_range(0..moves.len())]);
        }
        for a_is_black in [true, false] {
            let winner = if a_is_black {
                play(&opening, &mut x, &mut y, a.max_plies)
            } else {
                play(&opening, &mut y, &mut x, a.max_plies)
            };
            match winner {
                None => draws += 1,
                Some(p) if (p == Player::Black) == a_is_black => a_wins += 1,
                Some(_) => b_wins += 1,
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({ "games": a_wins + b_wins + draws, "a_wins": a_wins, "b_wins": b_wins, "draws": draws })
    );
}

fn main() {
    let (mode, a) = parse();
    match mode.as_str() {
        "forward" => forward(&a),
        "self-match" => {
            let cfg = load(&a.config);
            let w = Weights::load(&a.weights).unwrap_or_else(|e| panic!("{}: {e}", a.weights));
            assert_eq!(w.geometry, cfg.net.geometry(), "{} does not match [net]", a.weights);
            game_druid::with_board_size!(
                cfg.net.size,
                N => self_match::<N>(&a, &cfg, w),
                n => panic!("size {n} is not compiled in ({:?})", game_druid::cnn::SUPPORTED_SIZES),
            );
        }
        other => panic!("unknown mode {other}"),
    }
}
