//! `game-othello dump --label gumbel --head cnn`'s batched-GPU counterpart:
//! writes the identical `RecordV2` shard format via `crate::selfplay::
//! dump_gumbel_games_batched` (this crate's `gumbel_explore`/
//! `MlxOthelloOracle`, not the per-node `CnnGumbelPlayer`), so a coordinator
//! can point at either binary for a given generation's self-play shard
//! without the trainer/gate downstream of it caring which one wrote it.
//!
//! Run under a memory watchdog (poll `vm_stat`/RSS and kill on a low-memory
//! floor) at any large net geometry / live-batch combination -- a single
//! unchunked-enough MLX call's transient working set can exceed this
//! machine's memory well before the net itself looks unreasonably large;
//! see `game_othello::convnet::mlx::evaluate_batch`'s own docs and this
//! binary's `--chunk-size` flag, which bounds exactly that.
//!
//!   cargo run --release -p mcts-batch --example dump_gumbel_batched -- \
//!     --out shard.bin [--games N] [--seed N] [--sims N] [--max-considered N] \
//!     [--temp-moves N] [--forced-opening-plies N] [--chunk-size N] [--cnn-weights <path>]

use game_othello::convnet::mlx::MlxCnnValueNet;
use mcts_batch::selfplay::dump_gumbel_games_batched;
use mcts_batch::Config;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() {
    let mut out: Option<PathBuf> = None;
    let mut games = 200u64;
    let mut seed = 0u64;
    let mut sims = 32u32;
    let mut max_considered = 8usize;
    let mut temp_moves = 6u8;
    let mut forced_opening_plies = 0u32;
    let mut chunk_size = 128usize;
    let mut cnn_weights: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--games" => games = val().parse().expect("--games must be an integer"),
            "--seed" => seed = val().parse().expect("--seed must be an integer"),
            "--sims" => sims = val().parse().expect("--sims must be an integer"),
            "--max-considered" => {
                max_considered = val().parse().expect("--max-considered must be an integer")
            }
            "--temp-moves" => temp_moves = val().parse().expect("--temp-moves must be an integer"),
            "--forced-opening-plies" => {
                forced_opening_plies = val().parse().expect("--forced-opening-plies must be an integer")
            }
            "--chunk-size" => chunk_size = val().parse().expect("--chunk-size must be an integer"),
            "--cnn-weights" => cnn_weights = Some(PathBuf::from(val())),
            "-h" | "--help" => {
                eprintln!(
                    "usage: dump_gumbel_batched --out <path> [--games N] [--seed N] [--sims N] \
                     [--max-considered N] [--temp-moves N] [--forced-opening-plies N] \
                     [--chunk-size N] [--cnn-weights <path>]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    let out = out.expect("--out is required");
    assert!(sims >= 1, "--sims must be positive");
    assert!(max_considered >= 1, "--max-considered must be positive");
    assert!(chunk_size > 0, "--chunk-size must be positive");

    let net = match &cnn_weights {
        Some(p) => {
            MlxCnnValueNet::load(p).unwrap_or_else(|e| panic!("cannot load CNN weights {}: {e}", p.display()))
        }
        None => MlxCnnValueNet::default(),
    };
    let cfg = Config { num_simulations: sims as usize, num_considered_actions: max_considered, ..Config::default() };

    eprintln!(
        "batched gumbel self-play: games={games} sims={sims} max_considered={max_considered} \
         temp_moves={temp_moves} forced_opening_plies={forced_opening_plies} chunk_size={chunk_size} \
         cnn_weights={:?}",
        cnn_weights
    );
    let records =
        dump_gumbel_games_batched(net, &cfg, chunk_size, games, seed, temp_moves, forced_opening_plies);

    let mut buf = Vec::new();
    for r in &records {
        r.encode(&mut buf);
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).expect("cannot create --out parent directory");
    }
    let mut w = BufWriter::new(File::create(&out).expect("cannot create --out file"));
    w.write_all(&buf).expect("write failed");
    w.flush().expect("flush failed");

    eprintln!(
        "wrote {} v2 records ({} bytes) from {} batched gumbel games to {}",
        records.len(),
        buf.len(),
        games,
        out.display()
    );
}
