//! Does a sharper recorded improved-policy target point more toward proven
//! optimal play? (Recovery Slice 7.2.)
//!
//! The Mctx-verbatim recorded target (`c_scale = 0.1`, `rescale_q = true`) is
//! near-flat at a 32-sim budget -- `sigma_scale` is only ~6 and the `[0, 1]`
//! rescale compresses the completed-Q spread, so `softmax(logits + sigma*q)`
//! collapses back onto the raw prior. minizero sharpens the target by dropping
//! the rescale and raising `c_scale`. This measures, over the same validation
//! positions the Slice 6.2 oracle harness uses, whether that sharper target's
//! argmax agrees with the proven-optimal move set more often -- and at what
//! cost to target entropy / cross-entropy.
//!
//! The played move, the Sequential-Halving survivor ranking and the returned
//! action are Mctx-verbatim throughout: only `GumbelConfig::target_c_scale` /
//! `target_rescale_q` -- the recording-only overrides -- change between the
//! columns.
//!
//! Usage:
//!   connect4_target_sharpness <corpus.c4ref> <net.c4cnn>...
//!       [--sims N] [--positions N] [--seed N]

use std::process::ExitCode;

use game_connect4::{
    convnet::CnnValuePolicyNet,
    reference_diagnostic::{decode, optimal_move_set, state_from_record, OracleSkip, Split},
    selfplay::CnnGumbelPlayer,
    State,
};
use mcts::algorithms::mcts::gumbel::GumbelConfig;

const SEARCH_SEED: u64 = 7;

/// `(agreement, mean entropy, mean cross-entropy)` for one (net, setting).
type Metrics = (f64, f64, f64);
/// One net's row: its short name and a [`Metrics`] per entry in [`SETTINGS`].
type NetRow = (String, Vec<Metrics>);

/// `(label, target_rescale_q, target_c_scale)`. `None`/`None` is the current
/// Mctx-verbatim recorded target.
const SETTINGS: [(&str, Option<bool>, Option<f64>); 4] = [
    ("rescale=T c=0.1 (base)", None, None),
    ("rescale=F c=0.1", Some(false), Some(0.1)),
    ("rescale=F c=0.5", Some(false), Some(0.5)),
    ("rescale=F c=1.0", Some(false), Some(1.0)),
];

struct SplitMix(u64);
impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn argmax_column(policy: &[(game_connect4::Move, f32)]) -> u8 {
    policy
        .iter()
        .fold(None::<(u8, f32)>, |best, &(m, p)| match best {
            Some((_, bp)) if bp >= p => best,
            _ => Some((m.0, p)),
        })
        .expect("non-terminal position has a policy")
        .0
}

fn entropy(policy: &[(game_connect4::Move, f32)]) -> f64 {
    policy
        .iter()
        .map(|&(_, p)| p as f64)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

/// Cross-entropy of the target against a uniform-over-optimal reference:
/// `-mean_{c in optimal} ln p[c]`. A proxy for "the recorded CE is not wrecked".
fn cross_entropy_vs_optimal(policy: &[(game_connect4::Move, f32)], optimal: &[u8]) -> f64 {
    let prob = |col: u8| {
        policy
            .iter()
            .find(|&&(m, _)| m.0 == col)
            .map(|&(_, p)| p as f64)
            .unwrap_or(0.0)
    };
    let sum: f64 = optimal
        .iter()
        .map(|&c| -prob(c).max(1e-12).ln())
        .sum();
    sum / optimal.len() as f64
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut corpus_path: Option<String> = None;
    let mut net_paths: Vec<String> = Vec::new();
    let mut sims: u32 = 32;
    let mut positions: usize = 300;
    let mut seed: u64 = 20260910;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sims" => {
                i += 1;
                sims = args[i].parse().expect("--sims N");
            }
            "--positions" => {
                i += 1;
                positions = args[i].parse().expect("--positions N");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed N");
            }
            flag if flag.starts_with("--") => {
                eprintln!("unknown flag {flag}");
                return ExitCode::FAILURE;
            }
            _ if corpus_path.is_none() => corpus_path = Some(args[i].clone()),
            _ => net_paths.push(args[i].clone()),
        }
        i += 1;
    }

    let Some(corpus_path) = corpus_path else {
        eprintln!("usage: connect4_target_sharpness <corpus.c4ref> <net.c4cnn>... [--sims N] [--positions N] [--seed N]");
        return ExitCode::FAILURE;
    };
    if net_paths.is_empty() {
        eprintln!("at least one net path is required");
        return ExitCode::FAILURE;
    }

    let bytes = match std::fs::read(&corpus_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{corpus_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let records = match decode(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{corpus_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Resolve the oracle once over the validation split, then take a seeded
    // sub-sample so the run stays well under half an hour.
    let mut eligible: Vec<(State<6, 7>, Vec<u8>)> = Vec::new();
    let mut examined = 0usize;
    let (mut skip_no_child, mut skip_ambiguous) = (0usize, 0usize);
    for record in &records {
        if record.split != Split::Validation {
            continue;
        }
        examined += 1;
        let state = match state_from_record(record) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping malformed record: {e}");
                continue;
            }
        };
        match optimal_move_set(&state) {
            Ok(set) => eligible.push((state, set)),
            Err(OracleSkip::NoChildResolved) => skip_no_child += 1,
            Err(OracleSkip::AmbiguousUnresolvedChild) => skip_ambiguous += 1,
        }
    }

    // Fisher-Yates with a seeded SplitMix, then truncate.
    let mut rng = SplitMix(seed);
    for j in (1..eligible.len()).rev() {
        let k = (rng.next_u64() % (j as u64 + 1)) as usize;
        eligible.swap(j, k);
    }
    let sampled = positions.min(eligible.len());
    eligible.truncate(sampled);

    println!(
        "corpus {corpus_path}\nvalidation split: examined {examined}, eligible {}, skipped {} ({} no-child, {} ambiguous); sampled {} (seed {seed}), sims {sims}\n",
        eligible.len().max(sampled),
        skip_no_child + skip_ambiguous,
        skip_no_child,
        skip_ambiguous,
        sampled,
    );
    if eligible.is_empty() {
        eprintln!("no eligible positions");
        return ExitCode::FAILURE;
    }

    // One search pass per (net, setting); collect all three metrics at once.
    let n = eligible.len() as f64;
    let mut table: Vec<NetRow> = Vec::new();
    for net_path in &net_paths {
        let net = match CnnValuePolicyNet::load(net_path) {
            Ok(nn) => nn,
            Err(e) => {
                eprintln!("{net_path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let short = net_path.rsplit('/').next().unwrap_or(net_path).to_string();
        let mut row = Vec::new();
        for (_, target_rescale_q, target_c_scale) in SETTINGS {
            let cfg = GumbelConfig {
                sims,
                target_rescale_q,
                target_c_scale,
                ..GumbelConfig::default()
            };
            let mut player = CnnGumbelPlayer::new(net.clone(), cfg, SEARCH_SEED);
            let (mut hits, mut ent_sum, mut ce_sum) = (0usize, 0.0f64, 0.0f64);
            for (state, optimal) in &eligible {
                let policy = player.choose(state).improved_policy;
                if optimal.contains(&argmax_column(&policy)) {
                    hits += 1;
                }
                ent_sum += entropy(&policy);
                ce_sum += cross_entropy_vs_optimal(&policy, optimal);
            }
            row.push((hits as f64 / n, ent_sum / n, ce_sum / n));
        }
        table.push((short, row));
    }

    let col_w = 22;
    for (idx, header) in [
        "target-argmax agreement with the proven-optimal set",
        "mean target entropy (nats)",
        "mean cross-entropy vs uniform-over-optimal (nats)",
    ]
    .into_iter()
    .enumerate()
    {
        println!("== {header} ==");
        print!("{:<52}", "net");
        for (label, _, _) in SETTINGS {
            print!("  {label:>col_w$}");
        }
        println!();
        for (short, row) in &table {
            print!("{short:<52}");
            for cell in row {
                let value = match idx {
                    0 => cell.0,
                    1 => cell.1,
                    _ => cell.2,
                };
                print!("  {value:>col_w$.4}");
            }
            println!();
        }
        println!();
    }

    ExitCode::SUCCESS
}
