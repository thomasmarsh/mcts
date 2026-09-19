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
//!     <baseline> <candidate> [games] [sims] [--head ntuple|cnn] \
//!     [--evaluator cpu|mlx] [--edax-binary PATH] [--edax-data-dir DIR] [--edax-level N]
//!     [--edax-only [--gate-config PATH] [--set key=value]... [--edax-levels 1,2,3]
//!                  [--opening-plies N] [--out results.jsonl]]
//! ```
//!
//! `--edax-only` (with `--head cnn`) skips the head-to-head and rollout-anchor
//! checks and just places `candidate` on the Edax ladder through the shared
//! `common::run_edax_gate`, the same driver `edax_match` uses: the settings
//! (Edax binary, levels, games per level, balanced-opening file, seed, threads,
//! JSONL `out`) come from `--gate-config` (default `games/othello/edax/match.toml`)
//! with any key overridable by `--set key=value`; `games` (positional)
//! overrides `games_per_level`, and each opening is played from both seats.
//! `--opening-plies N` switches to the pre-XOT protocol (fresh random N-ply
//! openings instead of the file), for comparing against older numbers.
//! `sims` of `0` means no search at all: the candidate plays the argmax legal
//! move of its raw policy logits ([`RawPolicyPlayer`]), so raw and searched
//! ladders come from the same binary. `baseline` is ignored there (pass the
//! candidate path twice).
//!
//! `--head ntuple` (default) treats `baseline`/`candidate` as `research/
//! az-train`-shaped checkpoint directories (`model.toml` + `weights.bin` +
//! `weights.meta.json` + `policy.bin` + `policy.meta.json`) and plays with
//! [`GumbelPlayer`]. `--head cnn` treats them as single `OTCNN001`-layout
//! checkpoint files and plays with [`CnnGumbelPlayer`], whose per-leaf
//! forward-pass backend is picked by `--evaluator`: `mlx`
//! (`crate::convnet::mlx::MlxCnnValueNet::load`, GPU-backed, default, on by
//! default in the `mlx` Cargo feature) or `cpu` (`CnnValueNet::load`,
//! opt-in fallback for a `--no-default-features` build without Homebrew's
//! `mlx`/`mlx-c`) -- ignored for `--head ntuple`. Both players implement the same
//! `mcts::algorithms::Search` trait `mcts::util::battle_royale` is already
//! generic over, so no engine change was needed to support a second head
//! here -- only `score_share`/`load_*`/`main` becoming head-dispatching.
//! Three checks, run identically for either head or evaluator backend:
//!
//!  1. **Head to head.** `games` alternating-colour games, candidate vs
//!     baseline, both playing the Gumbel self-play search. PASS requires the
//!     candidate's score share's Wilson 95% lower bound strictly above 0.5.
//!  2. **Rollout anchor.** The same head-to-head, candidate vs a fixed
//!     net-free reference (the all-zero value+policy heads over the
//!     candidate's own geometry, or the all-zero CNN for `--head cnn`). The
//!     reference never changes generation to generation, so its score share
//!     is directly comparable across a run.  PASS requires the same Wilson
//!     lower bound criterion.
//!  3. **Edax fixed-depth yardstick** (diagnostic, does not gate the exit
//!     code -- Othello has an external reference Connect Four's gate does
//!     not). If `--edax-binary`/`--edax-data-dir` are given, plays the
//!     candidate against Edax at `--edax-level` and reports the score share
//!     so a run's `log.jsonl` can carry "gen k is worth Edax depth N" every
//!     generation, not just at the end.

use std::path::{Path, PathBuf};

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::algorithms::Search;
use mcts::evaluator::Evaluator;
use mcts::util::battle_royale;

use game_othello::convnet::CnnValueNet;
use game_othello::ntuple::{ModelGeometry, NTupleModel, NTupleModelEval};
use game_othello::policy::NTuplePolicyNet;
use game_othello::selfplay::{CnnGumbelPlayer, GumbelPlayer, RawPolicyPlayer};
use game_othello::Othello;

mod common;
use common::{load_toml_config, play_series, report_row, run_edax_gate, EdaxPlayer, GateConfig};

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
/// between checks. Generic over the `Search` implementation so the same
/// scoring logic serves both `GumbelPlayer` (`--head ntuple`) and
/// `CnnGumbelPlayer` (`--head cnn`) -- `battle_royale` itself is already
/// generic over two independent `Search` types, so nothing engine-side had
/// to change to support this.
fn score_share<S: Search<G = Othello>>(
    make_candidate: &impl Fn() -> S,
    make_opponent: &impl Fn(u64) -> S,
    games: usize,
) -> (usize, usize, usize, f64, f64) {
    let mut cand = make_candidate();
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

/// Run all three checks for one head's candidate/baseline pair, printing each
/// result in the same shape regardless of head. Returns whether both gating
/// checks (head-to-head, rollout anchor) passed; the Edax check is always
/// diagnostic-only.
fn run_checks<S: Search<G = Othello>>(
    make_candidate: impl Fn() -> S,
    make_baseline_opponent: impl Fn(u64) -> S,
    make_anchor_opponent: impl Fn(u64) -> S,
    games: usize,
    edax: Option<(String, String, u32)>,
) -> bool {
    let (w, d, l, share, lb) = score_share(&make_candidate, &make_baseline_opponent, games);
    let h2h_pass = lb > 0.5;
    println!(
        "head to head vs baseline: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if h2h_pass { "PASS" } else { "FAIL" }
    );

    let (w, d, l, share, lb) = score_share(&make_candidate, &make_anchor_opponent, games);
    let anchor_pass = lb > 0.5;
    println!(
        "rollout anchor: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if anchor_pass { "PASS" } else { "FAIL" }
    );

    if let Some((binary, data_dir, level)) = edax {
        let mut candidate = make_candidate();
        let mut edax = EdaxPlayer::spawn(&binary, &data_dir, level);
        let label = format!("candidate v edax-L{level}");
        let t = play_series(&mut candidate, &mut edax, games as u32, 23, &label);
        report_row(&format!("edax yardstick (level {level})"), &t);
    } else {
        println!("edax yardstick: skipped (--edax-binary/--edax-data-dir not given)");
    }

    h2h_pass && anchor_pass
}

fn load_ntuple_dir(dir: &Path) -> (NTupleModel, NTuplePolicyNet) {
    let model = NTupleModel::from_dir(dir);
    let geom = model.geometry().clone();
    let policy = NTuplePolicyNet::from_dir(geom, dir);
    (model, policy)
}

fn load_cnn_file(path: &Path) -> CnnValueNet {
    CnnValueNet::load(path)
        .unwrap_or_else(|e| panic!("cannot load OTCNN001 checkpoint {}: {e}", path.display()))
}

/// The `--head cnn` gate, generic over the per-leaf forward-pass backend so
/// the identical head-to-head/rollout-anchor/Edax checks run unchanged
/// against either `CnnValueNet` (CPU) or `crate::convnet::mlx::MlxCnnValueNet`
/// (GPU-backed) -- the same evaluator-genericization `CnnGumbelPlayer<E>`
/// already uses for self-play, applied here to gating instead.
fn run_cnn_head<E>(baseline_net: E, candidate_net: E, cfg: GumbelConfig, games: usize, edax: Option<(String, String, u32)>) -> bool
where
    E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static,
{
    let make_candidate = {
        let net = candidate_net.clone();
        move || CnnGumbelPlayer::new(net.clone(), cfg, 7)
    };
    let make_baseline_opponent = {
        let net = baseline_net.clone();
        move |seed| CnnGumbelPlayer::new(net.clone(), cfg, seed)
    };
    let make_anchor_opponent = move |seed| CnnGumbelPlayer::new(E::default(), cfg, seed);

    run_checks(make_candidate, make_baseline_opponent, make_anchor_opponent, games, edax)
}

/// `--edax-only` for the CNN head: `sims == 0` is the raw policy (budget: one
/// forward pass per move), otherwise a Gumbel search at that budget.
fn run_cnn_edax_only<E>(net: E, sims: u32, cfg: GumbelConfig, gate: &GateConfig, label: &str)
where
    E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static,
{
    if sims == 0 {
        run_edax_gate(gate, label, Some(1), move |_| RawPolicyPlayer::new(net.clone()));
    } else {
        run_edax_gate(gate, label, Some(sims as u64), move |_| {
            CnnGumbelPlayer::new(net.clone(), cfg, 7)
        });
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| &w[1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: gumbel_gate <baseline> <candidate> [games] [sims] [--head ntuple|cnn] \
             [--evaluator cpu|mlx] [--edax-binary PATH] [--edax-data-dir DIR] [--edax-level N]"
        );
        std::process::exit(1);
    }
    let baseline_path = PathBuf::from(&args[1]);
    let candidate_path = PathBuf::from(&args[2]);
    let games: usize = args.get(3).map_or(200, |s| s.parse().expect("games"));
    let sims: u32 = args.get(4).map_or(32, |s| s.parse().expect("sims"));
    let head = arg_value(&args, "--head").map_or("ntuple", |s| s.as_str());
    assert!(
        matches!(head, "ntuple" | "cnn"),
        "unknown --head mode: {head}"
    );
    // Which per-leaf forward-pass backend `--head cnn` plays with -- `mlx`
    // (default, GPU-backed, on by default in the `mlx` Cargo feature) or
    // `cpu` (opt-in fallback for a `--no-default-features` build without
    // Homebrew's `mlx`/`mlx-c`). Ignored for `--head ntuple`.
    let evaluator = arg_value(&args, "--evaluator").map_or("mlx", |s| s.as_str());
    assert!(
        matches!(evaluator, "cpu" | "mlx"),
        "unknown --evaluator: {evaluator} (want cpu | mlx)"
    );
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };

    let edax = match (arg_value(&args, "--edax-binary"), arg_value(&args, "--edax-data-dir")) {
        (Some(binary), Some(data_dir)) => {
            let level: u32 = arg_value(&args, "--edax-level")
                .map_or(3, |s| s.parse().expect("--edax-level must be an integer"));
            Some((binary.clone(), data_dir.clone(), level))
        }
        _ => None,
    };

    if args.iter().any(|a| a == "--edax-only") {
        assert_eq!(head, "cnn", "--edax-only is only implemented for --head cnn");
        let gate_path = arg_value(&args, "--gate-config")
            .map_or("games/othello/edax/match.toml", |s| s.as_str());
        let mut gate: GateConfig = load_toml_config(gate_path, &args);
        if let Some(g) = args.get(3).filter(|a| !a.starts_with("--")) {
            gate.games_per_level = g.parse().expect("games");
        }
        if let Some(l) = arg_value(&args, "--edax-levels") {
            gate.levels = l
                .split(',')
                .map(|l| l.parse().expect("--edax-levels: comma-separated integers"))
                .collect();
        }
        if let Some(b) = arg_value(&args, "--edax-binary") {
            gate.edax_binary = b.clone();
        }
        if let Some(d) = arg_value(&args, "--edax-data-dir") {
            gate.edax_data_dir = d.clone();
        }
        if let Some(o) = arg_value(&args, "--out") {
            gate.out = Some(o.clone());
        }
        if let Some(n) = arg_value(&args, "--opening-plies") {
            gate.opening_plies = n.parse().expect("--opening-plies");
            gate.openings = None;
        }
        let label = format!(
            "{}@{}",
            candidate_path.parent().and_then(|p| p.file_name()).map_or("?".into(), |n| n.to_string_lossy().to_string())
                + "/"
                + &candidate_path.file_stem().map_or("?".into(), |n| n.to_string_lossy().to_string()),
            if sims == 0 { "raw".to_string() } else { format!("s{sims}") }
        );
        match evaluator {
            "cpu" => run_cnn_edax_only(load_cnn_file(&candidate_path), sims, cfg, &gate, &label),
            "mlx" => {
                #[cfg(feature = "mlx")]
                {
                    let net = game_othello::convnet::mlx::MlxCnnValueNet::load(&candidate_path)
                        .unwrap_or_else(|e| panic!("cannot load OTCNN001 checkpoint {}: {e}", candidate_path.display()));
                    run_cnn_edax_only(net, sims, cfg, &gate, &label)
                }
                #[cfg(not(feature = "mlx"))]
                {
                    panic!("--evaluator mlx requires building game-othello with --features mlx");
                }
            }
            other => unreachable!("--evaluator validated above, got {other}"),
        }
        return;
    }
    assert!(sims > 0, "sims = 0 (raw policy) is only meaningful with --edax-only");

    let pass = match head {
        "ntuple" => {
            let (baseline_model, baseline_policy) = load_ntuple_dir(&baseline_path);
            let (candidate_model, candidate_policy) = load_ntuple_dir(&candidate_path);
            let zero_geom: ModelGeometry = candidate_model.geometry().clone();

            let make_candidate = {
                let model = candidate_model.clone();
                let policy = candidate_policy.clone();
                move || GumbelPlayer::with_policy(NTupleModelEval::new(model.clone()), policy.clone(), cfg, 7)
            };
            let make_baseline_opponent = {
                let model = baseline_model.clone();
                let policy = baseline_policy.clone();
                move |seed| GumbelPlayer::with_policy(NTupleModelEval::new(model.clone()), policy.clone(), cfg, seed)
            };
            let make_anchor_opponent = move |seed| {
                GumbelPlayer::with_policy(NTupleModelEval::default(), NTuplePolicyNet::zeros(zero_geom.clone()), cfg, seed)
            };

            run_checks(make_candidate, make_baseline_opponent, make_anchor_opponent, games, edax)
        }
        "cnn" => match evaluator {
            "cpu" => {
                let baseline_net = load_cnn_file(&baseline_path);
                let candidate_net = load_cnn_file(&candidate_path);
                run_cnn_head(baseline_net, candidate_net, cfg, games, edax)
            }
            "mlx" => {
                #[cfg(feature = "mlx")]
                {
                    let baseline_net = game_othello::convnet::mlx::MlxCnnValueNet::load(&baseline_path)
                        .unwrap_or_else(|e| panic!("cannot load OTCNN001 checkpoint {}: {e}", baseline_path.display()));
                    let candidate_net = game_othello::convnet::mlx::MlxCnnValueNet::load(&candidate_path)
                        .unwrap_or_else(|e| panic!("cannot load OTCNN001 checkpoint {}: {e}", candidate_path.display()));
                    run_cnn_head(baseline_net, candidate_net, cfg, games, edax)
                }
                #[cfg(not(feature = "mlx"))]
                {
                    panic!("--evaluator mlx requires building game-othello with --features mlx");
                }
            }
            other => unreachable!("--evaluator validated above, got {other}"),
        },
        other => unreachable!("--head validated above, got {other}"),
    };

    if pass {
        println!("GATE: PASS");
    } else {
        println!("GATE: FAIL");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_geom() -> ModelGeometry {
        let bytes = std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/tests/tiny.toml"),
        )
        .unwrap();
        ModelGeometry::parse(&bytes)
    }

    /// Fast, `--head`-independent regression check that `score_share`/
    /// `run_checks`'s generic-over-`Search` plumbing works at all: a tiny
    /// n-tuple geometry (cheap arithmetic, unlike the CNN) still plays a
    /// real full-length Othello game to terminal through `battle_royale`
    /// and produces a self-consistent W-D-L tally. This exercises the same
    /// code path `--head ntuple` uses in `main`, just without shelling out
    /// to a checkpoint directory.
    #[test]
    fn score_share_generic_plumbing_works_for_the_ntuple_head() {
        let geom = tiny_geom();
        let cfg = GumbelConfig {
            sims: 4,
            max_considered: 2,
            ..GumbelConfig::default()
        };
        let make_candidate = {
            let geom = geom.clone();
            move || GumbelPlayer::with_policy(NTupleModelEval::default(), NTuplePolicyNet::zeros(geom.clone()), cfg, 7)
        };
        let make_opponent = move |seed| {
            GumbelPlayer::with_policy(NTupleModelEval::default(), NTuplePolicyNet::zeros(geom.clone()), cfg, seed)
        };
        let (w, d, l, share, lb) = score_share(&make_candidate, &make_opponent, 2);
        assert_eq!(w + d + l, 2);
        assert!((0.0..=1.0).contains(&share));
        assert!((0.0..=1.0).contains(&lb));
    }

    /// The `--head cnn` analogue of the test above: proves a real
    /// `OTCNN001`-layout file round-trips through `load_cnn_file` and then
    /// through `score_share`'s generic plumbing via `CnnGumbelPlayer`.
    /// `#[ignore]`d because `CnnValueNet`'s hand-rolled numpy-style
    /// convolution is slow under a debug build -- a single full-length
    /// Othello self-play game costs tens of seconds even at the smallest
    /// possible search budget, since game length (not search width) drives
    /// this cost and Othello always plays to a real terminal; `games/othello/
    /// src/dump.rs`'s own `a_cnn_gumbel_selfplay_run_produces_completed_q_
    /// records` test carries the identical `#[ignore]` for the same reason).
    /// Run via `cargo test --release -p game-othello --example gumbel_gate
    /// cnn_head_round_trips_a_real_checkpoint_file -- --ignored`.
    #[test]
    #[ignore]
    fn cnn_head_round_trips_a_real_checkpoint_file() {
        let path = std::env::temp_dir().join(format!(
            "mcts-othello-gumbel-gate-cnn-{}",
            std::process::id()
        ));
        let net = CnnValueNet::default();
        let bytes = net.to_bytes();
        std::fs::write(&path, &bytes).unwrap();

        let loaded = load_cnn_file(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.weights(), net.weights());

        let cfg = GumbelConfig {
            sims: 2,
            max_considered: 1,
            ..GumbelConfig::default()
        };
        let make_candidate = {
            let net = loaded.clone();
            move || CnnGumbelPlayer::new(net.clone(), cfg, 7)
        };
        let make_opponent = move |seed| CnnGumbelPlayer::new(CnnValueNet::default(), cfg, seed);
        let (w, d, l, share, lb) = score_share(&make_candidate, &make_opponent, 1);
        assert_eq!(w + d + l, 1);
        assert!((0.0..=1.0).contains(&share));
        assert!((0.0..=1.0).contains(&lb));
    }
}
