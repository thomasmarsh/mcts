//! Measure Connect Four n-tuple inference on recorded positions.
//!
//!     cargo run --release -p mcts-tests --example connect4_value_throughput -- \
//!         <records.bin> <weights.bin> [passes]

use std::{env, process::ExitCode, time::Instant};

use game_connect4::valuenet::NTupleValueNet;
use game_connect4::{dump::Record, BitBoard, Player, State};

fn main() -> ExitCode {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: connect4_value_throughput <records.bin> <weights.bin> [passes]");
        return ExitCode::FAILURE;
    }
    let passes: usize = args.get(3).map_or(20, |s| s.parse().expect("passes"));
    let net = match NTupleValueNet::load(&args[2]) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{}: {error}", args[2]);
            return ExitCode::FAILURE;
        }
    };
    let bytes = match std::fs::read(&args[1]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{}: {error}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let mut states = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let Some((record, used)) = Record::decode(&bytes[offset..]) else {
            eprintln!("malformed record at byte {offset}");
            return ExitCode::FAILURE;
        };
        offset += used;
        states.push(State::from_parts(
            BitBoard::from_bits(record.black),
            BitBoard::from_bits(record.white),
            if record.side == 0 {
                Player::Black
            } else {
                Player::White
            },
            false,
        ));
    }
    if states.is_empty() {
        eprintln!("no records");
        return ExitCode::FAILURE;
    }
    let started = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..passes {
        for state in &states {
            checksum += net.value(state) as f64;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let evaluations = states.len() * passes;
    println!(
        "layout_weights={} evaluations={evaluations} seconds={elapsed:.6} evals_per_second={:.0} checksum={checksum:.6}",
        net.weight_count(),
        evaluations as f64 / elapsed,
    );
    ExitCode::SUCCESS
}
