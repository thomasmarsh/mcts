//! One f32 value per shard position, from that position's own mover's point of view (the same
//! convention as `shard::Record::value`): the net's raw value (`--sims 0`), or the root value of
//! a Gumbel search of `--sims N` simulations at the `[play]` search settings. Positions are
//! independent roots, so they batch together through `mcts-batch` (`cnn::selfplay::root_values`),
//! `--workers` threads at a time.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example druid_value_label -p game-druid -- \
//!     --config games/druid/cnn/az-druid-9x9.toml --weights games/druid/cnn/weights/9x9-H-gen200.bin \
//!     --shard shard.bin --out labels.f32 [--sims 200] [--workers 8] [--limit 200]
//! ```
//!
//! Writes `--out` as raw little-endian f32, one per position in shard order, and `<out>.json`
//! alongside it with `{weights, sims, count, seconds}`.

use std::path::Path;
use std::time::Instant;

use game_druid::cnn::config::load;
use game_druid::cnn::oracle::DruidOracle;
use game_druid::cnn::selfplay::root_values;
use game_druid::cnn::shard::read_shard;
use grid_cnn::{Net, Weights};

fn main() {
    let (mut config, mut weights_path, mut shard_path, mut out) =
        (String::new(), String::new(), String::new(), String::new());
    let (mut sims, mut workers, mut limit) = (0usize, 1usize, None);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => config = val(),
            "--weights" => weights_path = val(),
            "--shard" => shard_path = val(),
            "--out" => out = val(),
            "--sims" => sims = val().parse().expect("--sims takes an integer"),
            "--workers" => workers = val().parse().expect("--workers takes an integer"),
            "--limit" => limit = Some(val().parse::<usize>().expect("--limit takes an integer")),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(
        !config.is_empty() && !weights_path.is_empty() && !shard_path.is_empty() && !out.is_empty(),
        "--config, --weights, --shard and --out are required"
    );

    let cfg = load(&config);
    let w = Weights::load(&weights_path).unwrap_or_else(|e| panic!("{weights_path}: {e}"));
    assert_eq!(
        w.geometry,
        cfg.net.geometry(),
        "{weights_path} does not match [net] in {config}"
    );

    let (size, records) =
        read_shard(Path::new(&shard_path)).unwrap_or_else(|e| panic!("{shard_path}: {e}"));
    assert_eq!(
        size, cfg.net.size,
        "{shard_path} is a {size}x{size} shard, [net] in {config} is {}x{}",
        cfg.net.size, cfg.net.size
    );
    let count = limit.unwrap_or(records.len()).min(records.len());
    let records = &records[..count];

    game_druid::with_board_size!(
        size,
        N => {
            let oracle = DruidOracle::<N>::new(Net::new(&w), cfg.play.chunk_size);
            let states: Vec<_> = records.iter().map(|r| r.fields.to_state(N)).collect();

            let started = Instant::now();
            let values = root_values(&oracle, &states, &cfg.play.search, sims, workers, 1);
            let seconds = started.elapsed().as_secs_f64();

            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("{out}: {e}"));
            let sidecar = serde_json::json!({
                "weights": weights_path,
                "sims": sims,
                "count": values.len(),
                "seconds": seconds,
            });
            std::fs::write(format!("{out}.json"), serde_json::to_string(&sidecar).unwrap())
                .unwrap_or_else(|e| panic!("{out}.json: {e}"));
            println!(
                "{} positions, {:.1} positions/s ({:.1}s)",
                values.len(),
                values.len() as f64 / seconds.max(1e-9),
                seconds,
            );
        },
        n => panic!("size {n} is not compiled in ({:?})", game_druid::cnn::SUPPORTED_SIZES),
    );
}
