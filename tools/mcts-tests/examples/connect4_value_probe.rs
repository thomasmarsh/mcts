//! Diagnose a fitted Connect Four value net against dumped held-out records.
//!
//!     cargo run --release -p mcts-tests --example connect4_value_probe -- \
//!         <records.bin> <weights.bin> [sample-count] [depth] [seed]

use std::{env, process::ExitCode};

use game_connect4::valuenet::NTupleValueNet;
use game_connect4::{dump::Record, BitBoard, Player, Standard, State};
use mcts::algorithms::negamax::{MaterialBlind, Negamax, NegamaxOptions};

struct Sample {
    state: State<6, 7>,
    ply: u8,
    value: f64,
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let (mx, my) = (
        x.iter().sum::<f64>() / x.len() as f64,
        y.iter().sum::<f64>() / y.len() as f64,
    );
    let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        xx += (a - mx).powi(2);
        yy += (b - my).powi(2);
        xy += (a - mx) * (b - my);
    }
    if xx == 0.0 || yy == 0.0 {
        0.0
    } else {
        xy / (xx * yy).sqrt()
    }
}

fn report(label: &str, prediction: &[f64], value: &[f64]) {
    if value.is_empty() {
        println!("{label}: no usable records");
        return;
    }
    let mse = prediction
        .iter()
        .zip(value)
        .map(|(p, v)| (p - v).powi(2))
        .sum::<f64>()
        / value.len() as f64;
    let sign = prediction
        .iter()
        .zip(value)
        .filter(|(p, v)| p.signum() == v.signum())
        .count() as f64
        / value.len() as f64;
    println!(
        "{label}: n={} mse={mse:.4} pearson={:.3} sign={sign:.3}",
        value.len(),
        correlation(prediction, value)
    );
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: connect4_value_probe <records.bin> <weights.bin> [sample-count] [depth] [seed]"
        );
        return ExitCode::FAILURE;
    }
    let sample_count = args
        .get(3)
        .map_or(2000, |s| s.parse().expect("sample-count"));
    let depth = args.get(4).map_or(8, |s| s.parse().expect("depth"));
    let seed: u64 = args.get(5).map_or(0, |s| s.parse().expect("seed"));
    let bytes = match std::fs::read(&args[1]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let net = match NTupleValueNet::load(&args[2]) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{}: {e}", args[2]);
            return ExitCode::FAILURE;
        }
    };
    let mut records = Vec::new();
    let mut off = 0;
    while off < bytes.len() {
        let Some((record, used)) = Record::decode(&bytes[off..]) else {
            eprintln!("malformed record at byte {off}");
            return ExitCode::FAILURE;
        };
        off += used;
        if record.side > 1 || record.ply >= 42 || !record.value.is_finite() {
            eprintln!("invalid fields near byte {}", off - used);
            return ExitCode::FAILURE;
        }
        let side = if record.side == 0 {
            Player::Black
        } else {
            Player::White
        };
        records.push(Sample {
            state: State::from_parts(
                BitBoard::from_bits(record.black),
                BitBoard::from_bits(record.white),
                side,
                false,
            ),
            ply: record.ply,
            value: record.value as f64,
        });
    }
    if records.is_empty() {
        eprintln!("no records");
        return ExitCode::FAILURE;
    }
    // Rank within each coarse ply band using a stable seeded hash, so late
    // tactical positions remain represented even when the sample is small.
    let mut selected = Vec::new();
    for band in 0..3 {
        let mut indices: Vec<_> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| (r.ply as usize / 14).min(2) == band)
            .map(|(i, _)| i)
            .collect();
        indices.sort_by_key(|&i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed);
        selected.extend(indices.into_iter().take(sample_count / 3 + 1));
    }
    selected.truncate(sample_count);
    let mut all_p = Vec::new();
    let mut all_y = Vec::new();
    let mut tactical_p = Vec::new();
    let mut tactical_y = Vec::new();
    let mut solver = Negamax::<Standard, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(depth)
            .with_table_bits(18),
    );
    let mut buckets = [(0usize, 0.0f64, 0.0f64); 5];
    let mut by_band = vec![(Vec::new(), Vec::new()); 3];
    let mut by_side = vec![(Vec::new(), Vec::new()); 2];
    for i in selected {
        let r = &records[i];
        let p = net.value(&r.state) as f64;
        all_p.push(p);
        all_y.push(r.value);
        let bucket = ((p + 1.0) * 2.5).floor().clamp(0.0, 4.0) as usize;
        buckets[bucket].0 += 1;
        buckets[bucket].1 += p;
        buckets[bucket].2 += r.value;
        let band = (r.ply as usize / 14).min(2);
        by_band[band].0.push(p);
        by_band[band].1.push(r.value);
        let side = match r.state.turn() {
            Player::Black => 0,
            Player::White => 1,
        };
        by_side[side].0.push(p);
        by_side[side].1.push(r.value);
        let (_, score) = solver.bounded_negamax(&r.state, depth);
        if score != 0 {
            tactical_p.push(p);
            tactical_y.push(score.signum() as f64);
        }
    }
    report("recorded outcome", &all_p, &all_y);
    report("proven tactical sign", &tactical_p, &tactical_y);
    for (i, (p, y)) in by_band.iter().enumerate() {
        report(&format!("ply band {}-{}", i * 14, i * 14 + 13), p, y);
    }
    report("black to move", &by_side[0].0, &by_side[0].1);
    report("white to move", &by_side[1].0, &by_side[1].1);
    println!("tactical proofs: {}/{}", tactical_y.len(), all_y.len());
    println!("calibration buckets: range count mean_prediction mean_outcome");
    for (i, (n, sum_p, sum_y)) in buckets.into_iter().enumerate() {
        let lo = -1.0 + i as f64 * 0.4;
        let hi = lo + 0.4;
        if n == 0 {
            println!("[{lo:.1},{hi:.1}) 0 - -");
        } else {
            println!(
                "[{lo:.1},{hi:.1}) {n} {:.3} {:.3}",
                sum_p / n as f64,
                sum_y / n as f64
            );
        }
    }
    ExitCode::SUCCESS
}
