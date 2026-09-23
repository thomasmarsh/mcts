//! One generation of Gumbel self-play with a CNN checkpoint, written as a shard.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example druid_selfplay -p game-druid -- \
//!     --config games/druid/cnn/az-druid-5x5.toml [--weights gen0.bin] --out shard.bin [--seed N] [--games N]
//! ```
//!
//! Without `--weights` the generation is played with the all-zero net (a flat prior and value 0).
//!
//! Prints one JSON line of statistics and writes it next to the shard as `<out>.stats.json`.

use std::path::Path;

use game_druid::cnn::config::load;
use game_druid::cnn::oracle::DruidOracle;
use game_druid::cnn::selfplay::play_games;
use game_druid::cnn::shard::write_shard;
use grid_cnn::{Net, Weights};

fn main() {
    let (mut config, mut weights, mut out, mut seed, mut games) =
        (String::new(), String::new(), String::new(), 1u64, None);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => config = val(),
            "--weights" => weights = val(),
            "--out" => out = val(),
            "--seed" => seed = val().parse().expect("--seed takes an integer"),
            "--games" => games = Some(val().parse::<usize>().expect("--games takes an integer")),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(
        !config.is_empty() && !out.is_empty(),
        "--config and --out are required"
    );
    let mut cfg = load(&config);
    if let Some(g) = games {
        cfg.selfplay.games = g;
    }
    let w = if weights.is_empty() {
        Weights::zeros(cfg.net.geometry())
    } else {
        Weights::load(&weights).unwrap_or_else(|e| panic!("{weights}: {e}"))
    };
    assert_eq!(
        w.geometry,
        cfg.net.geometry(),
        "{weights} does not match [net] in {config}"
    );
    game_druid::with_board_size!(
        cfg.net.size,
        N => {
            let oracle = DruidOracle::<N>::new(
                Net::new(&w),
                cfg.selfplay.chunk_size,
            );
            let (records, stats) = play_games(&oracle, &cfg.selfplay, seed);
            write_shard(Path::new(&out), N, &records).unwrap_or_else(|e| panic!("{out}: {e}"));
            let line = serde_json::to_string(&stats).unwrap();
            std::fs::write(format!("{out}.stats.json"), &line).unwrap();
            println!("{line}");
        },
        n => panic!("size {n} is not compiled in ({:?})", game_druid::cnn::SUPPORTED_SIZES),
    );
}
