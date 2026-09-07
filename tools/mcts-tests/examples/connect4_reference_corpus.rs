//! Generate a deterministic, replay-incompatible Connect Four reference corpus.
//!
//!     cargo run --release -p mcts-tests --example connect4_reference_corpus -- \
//!         --out local/output/az/connect4/recovery/reference-diagnostic/corpus.c4ref

use std::{collections::HashSet, env, fs, path::PathBuf, process::ExitCode, time::Instant};

use game_connect4::reference_diagnostic::{
    canonical_key, classify_score, encode, split_for_group, DiagnosticRecord, ReferenceLabel,
};
use game_connect4::{Player, Standard, State};
use mcts::{
    algorithms::negamax::{MaterialBlind, Negamax, NegamaxOptions},
    game::Game,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sha2::{Digest, Sha256};

fn label(state: &State<6, 7>, ply: u8) -> (ReferenceLabel, u8, u8) {
    let remaining = 42 - ply as u32;
    if remaining <= 10 {
        let mut solver = Negamax::<Standard, MaterialBlind>::new_with_options(
            MaterialBlind,
            NegamaxOptions::default()
                .with_max_depth(remaining)
                .with_table_bits(18),
        );
        let (_, score) = solver.bounded_negamax(state, remaining.max(1));
        return (
            classify_score(score, true),
            remaining as u8,
            remaining as u8,
        );
    }
    let mut max = 0;
    for depth in [6, 8, 10] {
        let mut solver = Negamax::<Standard, MaterialBlind>::new_with_options(
            MaterialBlind,
            NegamaxOptions::default()
                .with_max_depth(depth)
                .with_table_bits(18),
        );
        let (_, score) = solver.bounded_negamax(state, depth);
        max = depth as u8;
        let label = classify_score(score, false);
        if label != ReferenceLabel::Unresolved {
            return (label, depth as u8, max);
        }
    }
    (ReferenceLabel::Unresolved, 0, max)
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
    let mut games = 800u32;
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
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    let mut duplicates = 0usize;
    for group in 0..games {
        let split = split_for_group(group, seed);
        let mut state = State::<6, 7>::default();
        let mut game_positions = Vec::new();
        while !Standard::is_terminal(&state) {
            let ply = (state.black().count_ones() + state.white().count_ones()) as u8;
            let mut actions = Vec::new();
            Standard::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }
            // Diverse, deterministic opening samples and uniformly completed legal games.
            if [7, 8, 15, 16, 23, 24, 29, 30, 31, 32, 33, 34].contains(&ply) {
                let key = canonical_key(
                    state.black().bits(),
                    state.white().bits(),
                    if state.turn() == Player::Black { 0 } else { 1 },
                );
                if seen.insert(key) {
                    let (label, proof_depth, max_depth) = label(&state, ply);
                    game_positions.push(DiagnosticRecord {
                        black: state.black().bits(),
                        white: state.white().bits(),
                        side: if state.turn() == Player::Black { 0 } else { 1 },
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
    let bytes = encode(&records);
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
    let mut split_groups = [HashSet::new(), HashSet::new()];
    let mut split_keys = [HashSet::new(), HashSet::new()];
    for r in &records {
        labels[r.label as usize] += 1;
        proof_hist[r.proof_depth as usize] += 1;
        max_hist[r.max_depth as usize] += 1;
        ply_hist[r.ply as usize] += 1;
        side_hist[r.side as usize] += 1;
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
    let meta=format!("{{\n  \"format\": \"C4REFD01\",\n  \"version\": 1,\n  \"seed\": {seed},\n  \"games_requested\": {games},\n  \"positions\": {},\n  \"group_counts\": {{\"train\": {}, \"validation\": {}}},\n  \"split_position_counts\": {{\"train\": {train}, \"validation\": {validation}}},\n  \"validation_proven\": {{\"losses\": {}, \"wins\": {}}},\n  \"duplicates_omitted\": {duplicates},\n  \"canonical_cross_split_overlap\": {overlap},\n  \"depth_schedule\": [6, 8, 10],\n  \"exact_remaining_plies\": 10,\n  \"label_counts\": {:?},\n  \"proof_depth_histogram\": {:?},\n  \"max_depth_histogram\": {:?},\n  \"ply_histogram\": {:?},\n  \"side_histogram\": {:?},\n  \"sha256\": \"{hash}\"\n}}\n",records.len(),split_groups[0].len(),split_groups[1].len(),proven_validation[0],proven_validation[1],labels,proof_hist,max_hist,ply_hist,side_hist);
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
