//! Generation gate for the Gumbel AlphaZero self-play loop on Othello, the
//! Othello analogue of `tools/mcts-tests/examples/gumbel_connect4_gate.rs`.
//!
//! Lives in `games/othello/examples/` rather than `tools/mcts-tests/
//! examples/` (unlike the Connect Four gate) so it can reuse this crate's
//! own `examples/common` Edax subprocess scaffolding (`EdaxPlayer`,
//! `play_series`) for check 3 below -- that scaffolding is not visible to
//! another crate's examples, and duplicating a subprocess driver just to
//! match the Connect Four gate's directory would be the wrapper-for-its-own-
//! sake `AGENTS.md` warns against.
//!
//! ```text
//! cargo run --release -p game-othello --example gumbel_gate -- \
//!     <baseline_dir> <candidate_dir> [games] [sims] \
//!     [--edax-binary PATH] [--edax-data-dir DIR] [--edax-level N]
//! ```
//!
//! `baseline_dir` / `candidate_dir` are `research/az-train`-shaped
//! checkpoint directories (`model.toml` + `weights.bin` + `weights.meta.json`
//! + `policy.bin` + `policy.meta.json`). Three checks:
//!
//!  1. **Head to head.** `games` alternating-colour games, candidate vs
//!     baseline, both playing the Gumbel self-play search. PASS requires the
//!     candidate's score share's Wilson 95% lower bound strictly above 0.5.
//!  2. **Rollout anchor.** The same head-to-head, candidate vs a fixed
//!     net-free reference (the all-zero value+policy heads over the
//!     candidate's own geometry). The reference never changes generation to
//!     generation, so its score share is directly comparable across a run.
//!     PASS requires the same Wilson lower bound criterion.
//!  3. **Edax fixed-depth yardstick** (diagnostic, does not gate the exit
//!     code -- Othello has an external reference Connect Four's gate does
//!     not). If `--edax-binary`/`--edax-data-dir` are given, plays the
//!     candidate against Edax at `--edax-level` and reports the score share
//!     so a run's `log.jsonl` can carry "gen k is worth Edax depth N" every
//!     generation, not just at the end.

use std::path::{Path, PathBuf};

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::util::battle_royale;

use game_othello::ntuple::{ModelGeometry, NTupleModel, NTupleModelEval};
use game_othello::policy::NTuplePolicyNet;
use game_othello::selfplay::GumbelPlayer;
use game_othello::Othello;

mod common;
use common::{play_series, report_row, EdaxPlayer};

fn wilson_lower_bound(successes: f64, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = n as f64;
    let phat = successes / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    (center - margin) / denom
}

/// Candidate's score share (win 1, draw 0.5) and Wilson 95% lower bound over
/// `games` alternating-colour games against a fresh `opponent` built by
/// `make_opponent` for each call, so a persistent search tree never leaks
/// between checks.
fn score_share(
    candidate_model: &NTupleModel,
    candidate_policy: &NTuplePolicyNet,
    cfg: GumbelConfig,
    games: usize,
    make_opponent: impl Fn(u64) -> GumbelPlayer,
) -> (usize, usize, usize, f64, f64) {
    let mut cand = GumbelPlayer::with_policy(
        NTupleModelEval::new(candidate_model.clone()),
        candidate_policy.clone(),
        cfg,
        7,
    );
    let mut opp = make_opponent(11);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    let mut score = 0.0f64;
    for i in 0..games {
        let (result, candidate_first) = if i % 2 == 0 {
            (battle_royale::<Othello, _, _>(&mut cand, &mut opp), true)
        } else {
            (battle_royale::<Othello, _, _>(&mut opp, &mut cand), false)
        };
        match result {
            None => {
                score += 0.5;
                draws += 1;
            }
            Some(0) if candidate_first => {
                score += 1.0;
                wins += 1;
            }
            Some(1) if !candidate_first => {
                score += 1.0;
                wins += 1;
            }
            _ => losses += 1,
        }
    }
    let lb = wilson_lower_bound(score, games, 1.96);
    (wins, draws, losses, score / games as f64, lb)
}

fn load_dir(dir: &Path) -> (NTupleModel, NTuplePolicyNet) {
    let model = NTupleModel::from_dir(dir);
    let geom = model.geometry().clone();
    let policy = NTuplePolicyNet::from_dir(geom, dir);
    (model, policy)
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| &w[1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: gumbel_gate <baseline_dir> <candidate_dir> [games] [sims] \
             [--edax-binary PATH] [--edax-data-dir DIR] [--edax-level N]"
        );
        std::process::exit(1);
    }
    let baseline_dir = PathBuf::from(&args[1]);
    let candidate_dir = PathBuf::from(&args[2]);
    let games: usize = args.get(3).map_or(200, |s| s.parse().expect("games"));
    let sims: u32 = args.get(4).map_or(32, |s| s.parse().expect("sims"));
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };

    let (baseline_model, baseline_policy) = load_dir(&baseline_dir);
    let (candidate_model, candidate_policy) = load_dir(&candidate_dir);

    let (w, d, l, share, lb) = score_share(&candidate_model, &candidate_policy, cfg, games, {
        let baseline_model = baseline_model.clone();
        let baseline_policy = baseline_policy.clone();
        move |seed| {
            GumbelPlayer::with_policy(
                NTupleModelEval::new(baseline_model.clone()),
                baseline_policy.clone(),
                cfg,
                seed,
            )
        }
    });
    let h2h_pass = lb > 0.5;
    println!(
        "head to head vs baseline: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if h2h_pass { "PASS" } else { "FAIL" }
    );

    let zero_geom: ModelGeometry = candidate_model.geometry().clone();
    let (w, d, l, share, lb) = score_share(&candidate_model, &candidate_policy, cfg, games, {
        let zero_geom = zero_geom.clone();
        move |seed| {
            GumbelPlayer::with_policy(
                NTupleModelEval::default(),
                NTuplePolicyNet::zeros(zero_geom.clone()),
                cfg,
                seed,
            )
        }
    });
    let anchor_pass = lb > 0.5;
    println!(
        "rollout anchor: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if anchor_pass { "PASS" } else { "FAIL" }
    );

    if let (Some(binary), Some(data_dir)) =
        (arg_value(&args, "--edax-binary"), arg_value(&args, "--edax-data-dir"))
    {
        let level: u32 = arg_value(&args, "--edax-level")
            .map_or(3, |s| s.parse().expect("--edax-level must be an integer"));
        let mut candidate = GumbelPlayer::with_policy(
            NTupleModelEval::new(candidate_model.clone()),
            candidate_policy.clone(),
            cfg,
            13,
        );
        let mut edax = EdaxPlayer::spawn(binary, data_dir, level);
        let label = format!("candidate v edax-L{level}");
        let t = play_series(&mut candidate, &mut edax, games as u32, 23, &label);
        report_row(&format!("edax yardstick (level {level})"), &t);
    } else {
        println!("edax yardstick: skipped (--edax-binary/--edax-data-dir not given)");
    }

    if h2h_pass && anchor_pass {
        println!("GATE: PASS");
    } else {
        println!("GATE: FAIL");
        std::process::exit(1);
    }
}
