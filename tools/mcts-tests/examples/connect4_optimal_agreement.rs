//! Proven-optimal-move agreement for the graded Connect Four nets.
//!
//! Connect Four is solved, so for every resolvable corpus position we can
//! compute the set of proven-optimal moves (every move preserving the mover's
//! best proven outcome class, mate distance ignored) and ask how often each
//! candidate lands in it:
//!
//!   * the CNN policy head's raw argmax over legal columns,
//!   * `CnnGumbelPlayer` at a fixed sim budget,
//!   * `Connect4NegamaxPlayer` at depths 8 and 12 as calibration points.
//!
//! This is a diagnostic, not a gate: it says whether the policy head alone is
//! near-optimal and whether low-sim search adds to or subtracts from it.
//!
//! Usage:
//!   connect4_optimal_agreement <corpus.c4ref> <net.c4cnn>...
//!       [--sims N] [--split all|validation] [--max-positions N]

use std::process::ExitCode;

use game_connect4::{
    convnet::CnnValuePolicyNet,
    negamax_player::Connect4NegamaxPlayer,
    reference_diagnostic::{decode, optimal_move_set, state_from_record, OracleSkip, Split},
    selfplay::CnnGumbelPlayer,
    Move, Standard, State,
};
use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::algorithms::Search;
use mcts::game::Game;

const SEARCH_SEED: u64 = 7;
const NEGAMAX_CALIBRATION_DEPTHS: [u32; 2] = [8, 12];

fn argmax_legal_column(net: &CnnValuePolicyNet, state: &State<6, 7>) -> u8 {
    let logits = net.all_logits(state);
    let mut actions = Vec::new();
    Standard::generate_actions(state, &mut actions);
    actions
        .into_iter()
        .fold(None::<Move>, |best, m| match best {
            Some(b) if logits[b.0 as usize] >= logits[m.0 as usize] => Some(b),
            _ => Some(m),
        })
        .expect("non-terminal position has legal moves")
        .0
}

struct Tally {
    hits: usize,
    total: usize,
}
impl Tally {
    fn new() -> Self {
        Self { hits: 0, total: 0 }
    }
    fn record(&mut self, hit: bool) {
        self.total += 1;
        self.hits += usize::from(hit);
    }
    fn rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f64 / self.total as f64
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut corpus_path: Option<String> = None;
    let mut net_paths: Vec<String> = Vec::new();
    let mut sims: u32 = 32;
    let mut split = Split::Validation;
    let mut split_all = false;
    let mut max_positions: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sims" => {
                i += 1;
                sims = args[i].parse().expect("--sims N");
            }
            "--split" => {
                i += 1;
                match args[i].as_str() {
                    "all" => split_all = true,
                    "validation" => split = Split::Validation,
                    other => {
                        eprintln!("--split must be all|validation, got {other}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--max-positions" => {
                i += 1;
                max_positions = Some(args[i].parse().expect("--max-positions N"));
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
        eprintln!("usage: connect4_optimal_agreement <corpus.c4ref> <net.c4cnn>... [--sims N] [--split all|validation] [--max-positions N]");
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

    // Resolve the oracle once; every net is scored on the same position set.
    let mut positions: Vec<(State<6, 7>, Vec<u8>)> = Vec::new();
    let mut examined = 0usize;
    let mut skip_no_child = 0usize;
    let mut skip_ambiguous = 0usize;
    for record in &records {
        if !split_all && record.split != split {
            continue;
        }
        if max_positions.is_some_and(|cap| examined >= cap) {
            break;
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
            Ok(set) => positions.push((state, set)),
            Err(OracleSkip::NoChildResolved) => skip_no_child += 1,
            Err(OracleSkip::AmbiguousUnresolvedChild) => skip_ambiguous += 1,
        }
    }

    let split_label = if split_all { "all" } else { "validation" };
    println!(
        "corpus {corpus_path}\nsplit {split_label}, examined {examined}, eligible {}, skipped {} ({} no-child-resolved, {} ambiguous-unresolved-child), sims {sims}\n",
        positions.len(),
        skip_no_child + skip_ambiguous,
        skip_no_child,
        skip_ambiguous,
    );
    if positions.is_empty() {
        eprintln!("no eligible positions");
        return ExitCode::FAILURE;
    }

    // Negamax calibration is net-independent: compute it once.
    let mut negamax: Vec<(u32, Tally)> = NEGAMAX_CALIBRATION_DEPTHS
        .iter()
        .map(|&d| (d, Tally::new()))
        .collect();
    for (depth, tally) in &mut negamax {
        let mut player = Connect4NegamaxPlayer::new(*depth);
        for (state, optimal) in &positions {
            let choice = player.choose_action(state).0;
            tally.record(optimal.contains(&choice));
        }
    }

    println!(
        "{:<44}  {:>8}  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
        "net", "eligible", "skipped", "policy-argmax", "search@sims", "negamax-d8", "negamax-d12"
    );
    let skipped = skip_no_child + skip_ambiguous;
    for net_path in &net_paths {
        let net = match CnnValuePolicyNet::load(net_path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{net_path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut policy = Tally::new();
        let mut search = Tally::new();
        let mut player = CnnGumbelPlayer::new(
            net.clone(),
            GumbelConfig {
                sims,
                ..GumbelConfig::default()
            },
            SEARCH_SEED,
        );
        for (state, optimal) in &positions {
            policy.record(optimal.contains(&argmax_legal_column(&net, state)));
            search.record(optimal.contains(&player.choose_action(state).0));
        }
        println!(
            "{:<44}  {:>8}  {:>8}  {:>12.3}  {:>12.3}  {:>10.3}  {:>10.3}",
            net_path,
            positions.len(),
            skipped,
            policy.rate(),
            search.rate(),
            negamax[0].1.rate(),
            negamax[1].1.rate(),
        );
    }

    ExitCode::SUCCESS
}
