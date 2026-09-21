//! Evaluate the first positions of a shard with a checkpoint (identity orientation, one MLX
//! call) and print the values and logits as JSON, so the torch forward on the same positions can
//! be compared against the Rust/MLX forward.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example gonnect_check -p game-gonnect -- \
//!     --weights gen1.bin --shard shards/gen0.bin [--count 64]
//! ```

use std::path::Path;

use game_gonnect::cnn::shard::read_shard;
use grid_cnn::{Net, Weights};

fn main() {
    let (mut weights, mut shard, mut count) = (String::new(), String::new(), 64usize);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--weights" => weights = val(),
            "--shard" => shard = val(),
            "--count" => count = val().parse().expect("--count takes an integer"),
            other => panic!("unknown argument {other}"),
        }
    }
    let w = Weights::load(&weights).unwrap_or_else(|e| panic!("{weights}: {e}"));
    let (size, records) = read_shard(Path::new(&shard)).unwrap_or_else(|e| panic!("{shard}: {e}"));
    assert_eq!(size, w.geometry.size);
    let records = &records[..count.min(records.len())];
    let planes: Vec<f32> = records.iter().flat_map(|r| r.fields.planes(size)).collect();
    let out = Net::new(&w).forward(&planes, records.len());
    println!(
        "{}",
        serde_json::json!({ "values": out.values, "logits": out.logits })
    );
}
