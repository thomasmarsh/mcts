//! Offline searched-value scoring of Connect Four self-play replay positions.
//!
//!     cargo run --release -p mcts-tests --example connect4_replay_searched_value -- \
//!         --positions gen0.bin,gen1.bin --out searched.f32
//!
//! For every record in the given v2-connect4 replay shards (concatenated in
//! file order), reconstruct the non-terminal state and label it with the
//! single-threaded, deterministic `MaterialBlind` bounded-depth negamax
//! schedule from `game_connect4::reference_diagnostic`, then map the label to a
//! soft scalar in `[-1, 1]` from the side-to-move perspective (proven win/loss
//! -> +/-1, exact draw and unresolved cutoff -> 0).
//!
//! The output is a bare little-endian `f32` array, one value per input record
//! in order, so a consumer can index it with the same row mask it applies to
//! the decoded positions. A `.meta.json` sidecar records counts, label and ply
//! histograms, wall time, and peak RSS. This is diagnostic tooling: the array
//! is never a training target on its own and is not consumed by `az-train`.

use std::{env, fs, path::PathBuf, process::ExitCode, time::Instant};

use game_connect4::{
    dump::Record,
    reference_diagnostic::{searched_reference_label, searched_value_scalar, ReferenceLabel},
    BitBoard, Player, State,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the supplied rusage structure on success.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let rss = unsafe { usage.assume_init().ru_maxrss } as u64;
    #[cfg(target_os = "macos")]
    {
        rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        rss * 1024
    }
}

fn decode_shard(bytes: &[u8]) -> Vec<Record> {
    let mut out = Vec::new();
    let mut cursor = bytes;
    while let Some((record, consumed)) = Record::decode(cursor) {
        out.push(record);
        cursor = &cursor[consumed..];
    }
    assert!(cursor.is_empty(), "trailing bytes in replay shard");
    out
}

fn main() -> ExitCode {
    let mut positions: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--positions" => {
                positions = args
                    .next()
                    .expect("--positions needs a comma-separated list")
                    .split(',')
                    .map(PathBuf::from)
                    .collect()
            }
            "--out" => out = Some(PathBuf::from(args.next().expect("--out needs a path"))),
            "--limit" => {
                limit = Some(
                    args.next()
                        .expect("--limit needs a value")
                        .parse()
                        .expect("integer limit"),
                )
            }
            _ => {
                eprintln!(
                    "usage: connect4_replay_searched_value --positions a.bin,b.bin --out <path.f32> [--limit N]"
                );
                return ExitCode::FAILURE;
            }
        }
    }
    let (Some(out), false) = (out, positions.is_empty()) else {
        eprintln!("--positions and --out are both required");
        return ExitCode::FAILURE;
    };

    let started = Instant::now();
    let mut records: Vec<Record> = Vec::new();
    for path in &positions {
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        records.extend(decode_shard(&bytes));
    }
    if let Some(limit) = limit {
        records.truncate(limit);
    }

    // Every position is scored independently by the deterministic negamax
    // schedule, so `par_iter` only changes wall time, not the result: the
    // collected vector stays in record order.
    let scored_progress = std::sync::atomic::AtomicUsize::new(0);
    let scored: Vec<(f32, ReferenceLabel, u8)> = records
        .par_iter()
        .map(|record| {
            let turn = if record.side == 0 {
                Player::Black
            } else {
                Player::White
            };
            let state = State::<6, 7>::from_parts(
                BitBoard::from_bits(record.black),
                BitBoard::from_bits(record.white),
                turn,
                false,
            );
            let (label, _proof, _max) = searched_reference_label(&state, record.ply);
            let done = scored_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done.is_multiple_of(1000) || done == records.len() {
                eprintln!("  scored {}/{} positions", done, records.len());
            }
            (searched_value_scalar(label), label, record.ply)
        })
        .collect();

    let mut values: Vec<f32> = Vec::with_capacity(records.len());
    let mut label_hist = [0usize; 6];
    let mut ply_hist = [0usize; 42];
    for (value, label, ply) in scored {
        values.push(value);
        label_hist[label as usize] += 1;
        ply_hist[ply as usize] += 1;
    }

    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).expect("create --out parent");
    }
    fs::write(&out, &bytes).expect("write searched values");
    let hash = format!("{:x}", Sha256::digest(&bytes));

    let proven = label_hist[ReferenceLabel::ExactWin as usize]
        + label_hist[ReferenceLabel::ExactLoss as usize]
        + label_hist[ReferenceLabel::BoundedWin as usize]
        + label_hist[ReferenceLabel::BoundedLoss as usize];
    let wall_seconds = started.elapsed().as_secs_f64();
    let peak_rss = peak_rss_bytes();
    let meta = format!(
        "{{\n  \"format\": \"f32-le-array\",\n  \"positions\": {},\n  \"inputs\": {:?},\n  \"label_counts\": {label_hist:?},\n  \"proven_positions\": {proven},\n  \"unresolved_positions\": {},\n  \"ply_histogram\": {ply_hist:?},\n  \"sha256\": \"{hash}\",\n  \"wall_seconds\": {wall_seconds:.3},\n  \"peak_rss_bytes\": {peak_rss}\n}}\n",
        values.len(),
        positions.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        label_hist[ReferenceLabel::Unresolved as usize],
    );
    fs::write(out.with_extension("meta.json"), meta).expect("write metadata");
    println!(
        "searched values={} proven={proven} hash={hash} wall_seconds={wall_seconds:.3} peak_rss_bytes={peak_rss}",
        values.len()
    );
    ExitCode::SUCCESS
}
