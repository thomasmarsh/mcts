//! Cross-language compatibility check for an `OTCNN001` checkpoint: read a
//! `game-othello dump --label outcome` v1 record file, run
//! `CnnValueNet::value` on every position, and write the predictions as raw
//! little-endian `f32`s (same order as the input records) so a Python
//! script can `np.fromfile` them and compare directly against its own
//! (numpy or torch) predictions on the same positions -- the actual
//! Rust/Python compatibility question, not just "does Python think it's
//! fine". Kept as reusable tooling: any checkpoint fitted by a new trainer
//! or technique on the Python side needs the same check before it's trusted
//! to feed the Rust inference hot path.
//!
//! ```text
//! cargo run --release -p game-othello --example predict_dump -- \
//!     <positions.bin> <weights.bin> <out.bin> [--limit N]
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;

use game_othello::convnet::CnnValueNet;
use game_othello::dump::{Record, RECORD_BYTES};
use game_othello::{Player, State, BB};

fn state_from_record(record: &Record) -> State {
    State {
        black: BB::from_bits(record.black),
        white: BB::from_bits(record.white),
        turn: if record.side == 0 { Player::Black } else { Player::White },
        last_pass: false,
        hashes: [0u64; 8],
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    if positional.len() != 3 {
        eprintln!("usage: predict_dump <positions.bin> <weights.bin> <out.bin> [--limit N]");
        std::process::exit(2);
    }
    let positions_path = PathBuf::from(positional[0]);
    let weights_path = PathBuf::from(positional[1]);
    let out_path = PathBuf::from(positional[2]);
    let limit: Option<usize> = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse().expect("--limit expects an integer"));

    let raw = fs::read(&positions_path).unwrap_or_else(|e| panic!("reading {positions_path:?}: {e}"));
    assert_eq!(raw.len() % RECORD_BYTES, 0, "{positions_path:?}: not a whole number of records");
    let mut records: Vec<Record> = raw
        .chunks_exact(RECORD_BYTES)
        .map(|chunk| Record::decode(chunk.try_into().unwrap()))
        .collect();
    if let Some(limit) = limit {
        records.truncate(limit);
    }

    let net = CnnValueNet::load(&weights_path).unwrap_or_else(|e| panic!("loading {weights_path:?}: {e}"));

    let mut out = Vec::with_capacity(records.len() * 4);
    for record in &records {
        let state = state_from_record(record);
        let value = net.value(&state);
        out.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&out_path, &out).unwrap_or_else(|e| panic!("writing {out_path:?}: {e}"));
    println!("wrote {} predictions to {out_path:?}", records.len());
}
