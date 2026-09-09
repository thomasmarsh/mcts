//! Generate a deterministic, replay-incompatible Connect Four reference corpus.
//!
//!     cargo run --release -p mcts-tests --example connect4_reference_corpus -- \
//!         --out local/output/az/connect4/recovery/reference-diagnostic/corpus-v2.c4ref
//!
//! Each source game is a deterministic uniform-random completion of a distinct
//! forced opening prefix. Non-terminal positions are sampled across the opening,
//! middle, and late ply bands and labelled with single-threaded `MaterialBlind`
//! negamax: searched to the end when few plies remain, otherwise run through an
//! increasing bounded-depth schedule that records the first proof depth.

use std::{collections::HashSet, env, fs, path::PathBuf, process::ExitCode, time::Instant};

use game_connect4::reference_diagnostic::{
    canonical_key, encode_v2, ply_band, searched_reference_label, split_for_group, DiagnosticRecord,
    ReferenceLabel, BOUNDED_DEPTH_SCHEDULE, EXACT_SEARCH_REMAINING_CAP,
};
use game_connect4::{Move, Player, Standard, State};
use mcts::game::Game;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sha2::{Digest, Sha256};

/// Disc counts sampled from every source game. Both parities appear in each of
/// the three ply bands so both sides to move are represented per band.
const SAMPLE_PLIES: [u8; 15] = [6, 7, 10, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36];

/// Deterministic three-move forced opening for a source game.
fn opening_prefix(group: u32) -> [u8; 3] {
    [
        (group % 7) as u8,
        (group / 7 % 7) as u8,
        (group / 49 % 7) as u8,
    ]
}

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

fn main() -> ExitCode {
    let mut out = None;
    let mut games = 1200u32;
    let mut seed = 44u64;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(args.next().expect("--out needs a path"))),
            "--games" => {
                games = args
                    .next()
                    .expect("--games needs a value")
                    .parse()
                    .expect("integer games")
            }
            "--seed" => {
                seed = args
                    .next()
                    .expect("--seed needs a value")
                    .parse()
                    .expect("integer seed")
            }
            _ => {
                eprintln!(
                    "usage: connect4_reference_corpus --out <corpus.c4ref> [--games N] [--seed N]"
                );
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(out) = out else {
        eprintln!("--out is required");
        return ExitCode::FAILURE;
    };
    let started = Instant::now();
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    let mut duplicates = 0usize;
    for group in 0..games {
        let split = split_for_group(group, seed);
        let mut rng =
            SmallRng::seed_from_u64(seed ^ (group as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut state = State::<6, 7>::default();
        for col in opening_prefix(group) {
            if Standard::is_terminal(&state) {
                break;
            }
            state = Standard::apply(state, &Move(col));
        }
        let mut game_positions = Vec::new();
        while !Standard::is_terminal(&state) {
            let ply = (state.black().count_ones() + state.white().count_ones()) as u8;
            let mut actions = Vec::new();
            Standard::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }
            if SAMPLE_PLIES.contains(&ply) {
                let side = if state.turn() == Player::Black { 0 } else { 1 };
                let key = canonical_key(state.black().bits(), state.white().bits(), side);
                if seen.insert(key) {
                    let (label, proof_depth, max_depth) = searched_reference_label(&state, ply);
                    game_positions.push(DiagnosticRecord {
                        black: state.black().bits(),
                        white: state.white().bits(),
                        side,
                        ply,
                        group,
                        split,
                        source_outcome: 0.,
                        label,
                        proof_depth,
                        max_depth,
                    });
                } else {
                    duplicates += 1;
                }
            }
            state = Standard::apply(state, &actions[rng.gen_range(0..actions.len())]);
        }
        let winner = if state.has_winner() {
            Some(Standard::winner(&state).unwrap())
        } else {
            None
        };
        for r in &mut game_positions {
            let side = if r.side == 0 {
                Player::Black
            } else {
                Player::White
            };
            r.source_outcome = match winner {
                Some(w) if w == side => 1.,
                Some(_) => -1.,
                None => 0.,
            };
        }
        records.extend(game_positions);
    }
    let bytes = encode_v2(&records);
    fs::write(&out, &bytes).expect("write corpus");
    let hash = format!("{:x}", Sha256::digest(&bytes));

    let mut labels = [0usize; 6];
    let mut train = 0;
    let mut validation = 0;
    let mut proven_validation = [0usize; 2];
    let mut proof_hist = [0usize; 43];
    let mut max_hist = [0usize; 43];
    let mut ply_hist = [0usize; 42];
    let mut side_hist = [0usize; 2];
    let mut band_proven = [[0usize; 2]; 3]; // [band][side]
    let mut band_total = [0usize; 3];
    let mut unresolved = 0usize;
    let mut split_groups = [HashSet::new(), HashSet::new()];
    let mut split_keys = [HashSet::new(), HashSet::new()];
    for r in &records {
        labels[r.label as usize] += 1;
        proof_hist[r.proof_depth as usize] += 1;
        max_hist[r.max_depth as usize] += 1;
        ply_hist[r.ply as usize] += 1;
        side_hist[r.side as usize] += 1;
        let band = ply_band(r.ply) as usize;
        band_total[band] += 1;
        if r.label == ReferenceLabel::Unresolved {
            unresolved += 1;
        } else if r.label.sign() != Some(0.0) {
            band_proven[band][r.side as usize] += 1;
        }
        let split_index = r.split as usize;
        split_groups[split_index].insert(r.group);
        split_keys[split_index].insert(canonical_key(r.black, r.white, r.side));
        if r.split as u8 == 0 {
            train += 1;
        } else {
            validation += 1;
            match r.label.sign() {
                Some(-1.) => proven_validation[0] += 1,
                Some(1.) => proven_validation[1] += 1,
                _ => {}
            }
        }
    }
    let overlap = split_keys[0].intersection(&split_keys[1]).count();
    let unresolved_fraction = if records.is_empty() {
        0.0
    } else {
        unresolved as f64 / records.len() as f64
    };
    let meta = format!(
        "{{\n  \"format\": \"C4REFD02\",\n  \"version\": 2,\n  \"seed\": {seed},\n  \"games_requested\": {games},\n  \"opening_prefix_plies\": 3,\n  \"sample_plies\": {SAMPLE_PLIES:?},\n  \"positions\": {},\n  \"group_counts\": {{\"train\": {}, \"validation\": {}}},\n  \"split_position_counts\": {{\"train\": {train}, \"validation\": {validation}}},\n  \"validation_proven\": {{\"losses\": {}, \"wins\": {}}},\n  \"duplicates_omitted\": {duplicates},\n  \"canonical_cross_split_overlap\": {overlap},\n  \"unresolved_positions\": {unresolved},\n  \"unresolved_fraction\": {unresolved_fraction:.4},\n  \"depth_schedule\": {BOUNDED_DEPTH_SCHEDULE:?},\n  \"exact_search_remaining_cap\": {EXACT_SEARCH_REMAINING_CAP},\n  \"ply_band_totals\": {band_total:?},\n  \"proven_by_band_black_to_move\": [{}, {}, {}],\n  \"proven_by_band_white_to_move\": [{}, {}, {}],\n  \"label_counts\": {:?},\n  \"proof_depth_histogram\": {:?},\n  \"max_depth_histogram\": {:?},\n  \"ply_histogram\": {:?},\n  \"side_histogram\": {:?},\n  \"sha256\": \"{hash}\"\n}}\n",
        records.len(),
        split_groups[0].len(),
        split_groups[1].len(),
        proven_validation[0],
        proven_validation[1],
        band_proven[0][0],
        band_proven[1][0],
        band_proven[2][0],
        band_proven[0][1],
        band_proven[1][1],
        band_proven[2][1],
        labels,
        proof_hist,
        max_hist,
        ply_hist,
        side_hist,
    );
    fs::write(out.with_extension("meta.json"), meta).expect("write metadata");
    let wall_seconds = started.elapsed().as_secs_f64();
    let peak_rss_bytes = peak_rss_bytes();
    fs::write(
        out.with_extension("run.json"),
        format!("{{\n  \"corpus_sha256\": \"{hash}\",\n  \"wall_seconds\": {wall_seconds:.3},\n  \"peak_rss_bytes\": {peak_rss_bytes}\n}}\n"),
    )
    .expect("write run metadata");
    println!(
        "corpus={} positions={} hash={} wall_seconds={wall_seconds:.3} peak_rss_bytes={peak_rss_bytes}",
        out.display(),
        records.len(),
        hash
    );
    ExitCode::SUCCESS
}
