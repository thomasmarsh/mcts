//! Strength check for the n-tuple Othello evaluator: does MCTS with the
//! n-tuple evaluator beat MCTS without it **at equal `max_iterations`**?
//!
//! ## Usage
//!
//! ```text
//! OTHELLO_NTUPLE_WEIGHTS=<trainer output dir> \
//!   cargo run --release --example ntuple_match -p game-othello -- [CONFIG] [MODE]
//! ```
//!
//! `CONFIG` defaults to `games/othello/ntuple/match.toml`. `MODE`:
//!
//! - `gate` (default) -- baseline vs contender at every `(K, D)` in the
//!   config. Passes iff, for at least one `(K, D)`, the contender's Wilson
//!   95% win-rate interval excludes 0.5.
//! - `edax` -- secondary, not gated: play the first `(K, D)` contender
//!   against the Edax ladder from `games/othello/edax/match.toml` and print
//!   the highest Edax level it clearly beats.
//! - `h2h` -- one ordered pair of the signal bake-off round-robin:
//!   the model in `$OTHELLO_NTUPLE_WEIGHTS` vs the one in
//!   `$OTHELLO_NTUPLE_WEIGHTS_B`, at `ks[0]` and every `depths` value.
//!   `games/othello/ntuple/bakeoff.sh` drives the full round-robin by
//!   setting both env vars per invocation.
//!
//! Both engines use the `strong` recipe (UCB1, `q_init = Loss`); the
//! contender additionally swaps its playout for `EvaluatedCutoff` capped at
//! `max_playout_depth = D`. `D = 0` is a pure value-net leaf; `D > 0` is a
//! short rollout ended by the evaluator. `EvaluatedCutoff` only fires on a
//! depth cutoff, so a finite `D` is required or the evaluator never runs.
//!
//! Background job: per-game progress to stderr. The weights come from the
//! Python trainer (`othello-eval-train`); this example does no training.

use game_othello::ntuple::{NTupleEval, NTupleEvalB};
use game_othello::Othello;
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};

mod common;
use common::{play_series, report_row, Boxed, EdaxPlayer};

#[derive(serde::Deserialize)]
struct Config {
    /// `max_iterations` rungs to test (equal node budget for both engines).
    ks: Vec<usize>,
    /// `max_playout_depth` values for the contender.
    depths: Vec<usize>,
    games_per_pairing: u32,
    seed: u64,
}

type BaselineProfile = profile::Mcts<select::Ucb1, simulate::Uniform>;
type ContenderProfile =
    profile::Mcts<select::Ucb1, simulate::EvaluatedCutoff<Othello, NTupleEval, simulate::Uniform>>;
type ContenderProfileB =
    profile::Mcts<select::Ucb1, simulate::EvaluatedCutoff<Othello, NTupleEvalB, simulate::Uniform>>;

/// The `strong` recipe: UCB1, play to terminal, `q_init = Loss`.
fn baseline(k: usize, seed: u64) -> Boxed {
    Boxed(Box::new(
        TreeSearch::<Othello, BaselineProfile>::new().config(
            SearchConfig::new()
                .name("ntuple/baseline")
                .expand_threshold(1)
                .q_init(QInit::Loss)
                .max_iterations(k)
                .seed(seed),
        ),
    ))
}

/// Same budget and recipe, but the playout is capped at depth `d` and the
/// leaf value at the cutoff comes from the n-tuple evaluator.
fn contender(k: usize, d: usize, seed: u64) -> Boxed {
    Boxed(Box::new(
        TreeSearch::<Othello, ContenderProfile>::new().config(
            SearchConfig::new()
                .name("ntuple/contender")
                .expand_threshold(1)
                .q_init(QInit::Loss)
                .max_iterations(k)
                .max_playout_depth(d)
                .simulate(simulate::EvaluatedCutoff::new())
                .seed(seed),
        ),
    ))
}

/// Same as [`contender`] but backed by the second weight slot
/// (`$OTHELLO_NTUPLE_WEIGHTS_B`).
fn contender_b(k: usize, d: usize, seed: u64) -> Boxed {
    Boxed(Box::new(
        TreeSearch::<Othello, ContenderProfileB>::new().config(
            SearchConfig::new()
                .name("ntuple/contender-b")
                .expand_threshold(1)
                .q_init(QInit::Loss)
                .max_iterations(k)
                .max_playout_depth(d)
                .simulate(simulate::EvaluatedCutoff::new())
                .seed(seed),
        ),
    ))
}

/// One ordered pair of the bake-off round-robin: model A
/// (`$OTHELLO_NTUPLE_WEIGHTS`) vs model B (`$OTHELLO_NTUPLE_WEIGHTS_B`), at
/// `ks[0]` and every `depths` value, `games_per_pairing` games with colours
/// alternated. The W-D-L is from A's perspective. The bake-off driver sets
/// both env vars per invocation, so one process only ever holds one pair.
fn run_h2h(cfg: &Config) {
    let k = *cfg.ks.first().expect("config `ks` is empty");
    let a = std::env::var("OTHELLO_NTUPLE_WEIGHTS").unwrap();
    let b = std::env::var("OTHELLO_NTUPLE_WEIGHTS_B")
        .expect("h2h needs OTHELLO_NTUPLE_WEIGHTS_B set to the opponent's weight dir");
    println!("== h2h: A={a}  vs  B={b}  (K={k}) ==");
    let n = cfg.games_per_pairing.max(40);
    for &d in &cfg.depths {
        let mut ha = contender(k, d, cfg.seed);
        let mut hb = contender_b(k, d, cfg.seed ^ 0x5555);
        let label = format!("A vs B  D={d}");
        let seed = cfg.seed ^ ((k as u64) << 20) ^ ((d as u64) << 8) ^ 0xB;
        let t = play_series(&mut ha, &mut hb, n, seed, &label);
        report_row(&label, &t);
        let (p, (lo, hi)) = t.win_rate_ci(1.96);
        println!(
            "  D={d}: A win_rate={p:.3} ci=[{lo:.3}, {hi:.3}]  {}",
            if lo > 0.5 {
                "A ahead (CI excludes 0.5)"
            } else if hi < 0.5 {
                "B ahead (CI excludes 0.5)"
            } else {
                "no separation"
            }
        );
    }
}

fn run_gate(cfg: &Config) {
    println!("== n-tuple contender vs no-evaluator baseline, equal max_iterations ==");
    let n = cfg.games_per_pairing.max(40);
    let mut any_pass = false;
    for &k in &cfg.ks {
        for &d in &cfg.depths {
            let mut c = contender(k, d, cfg.seed);
            let mut b = baseline(k, cfg.seed);
            let label = format!("K={k} D={d}");
            let seed = cfg.seed ^ ((k as u64) << 20) ^ ((d as u64) << 8);
            let t = play_series(&mut c, &mut b, n, seed, &label);
            report_row(&label, &t);
            let (_, (lo, _)) = t.win_rate_ci(1.96);
            let pass = lo > 0.5;
            any_pass |= pass;
            println!("  {label}: {}", if pass { "PASS (CI excludes 0.5)" } else { "no" });
        }
    }
    println!();
    println!(
        "gate: {}",
        if any_pass {
            "PASS -- at least one (K, D) beats the no-evaluator search"
        } else {
            "FAIL -- no (K, D) beats the no-evaluator search at this budget"
        }
    );
}

#[derive(serde::Deserialize)]
struct EdaxConfig {
    edax_binary: String,
    edax_data_dir: String,
    levels: Vec<u32>,
    games_per_level: u32,
    seed: u64,
}

fn run_edax(cfg: &Config) {
    let edax_path = "games/othello/edax/match.toml";
    let ecfg: EdaxConfig = toml::from_str(
        &std::fs::read_to_string(edax_path).unwrap_or_else(|e| panic!("cannot read {edax_path}: {e}")),
    )
    .expect("edax match.toml must parse");
    let k = *cfg.ks.first().expect("config `ks` is empty");
    let d = *cfg.depths.first().expect("config `depths` is empty");
    println!("== Secondary: contender (K={k}, D={d}) vs the Edax ladder ==");
    let mut n_clear = None;
    for &l in &ecfg.levels {
        let mut c = contender(k, d, cfg.seed.wrapping_add(l as u64));
        let mut edax = EdaxPlayer::spawn(&ecfg.edax_binary, &ecfg.edax_data_dir, l);
        let t = play_series(
            &mut c,
            &mut edax,
            ecfg.games_per_level,
            ecfg.seed.wrapping_add((l as u64) << 8),
            &format!("contender v L{l}"),
        );
        report_row(&format!("vs edax-L{l}"), &t);
        let (_, (lo, _)) = t.win_rate_ci(1.96);
        if lo >= 0.5 {
            n_clear = Some(l);
        }
    }
    match n_clear {
        Some(l) => println!("\nsecondary N = {l} (contender clearly beats Edax up to level {l})"),
        None => println!("\nsecondary N = 0 (contender does not clearly beat the lowest level tested)"),
    }
}

fn main() {
    // Fail fast with the trainer pointer if the weights aren't wired up.
    let _ = NTupleEval;
    if std::env::var("OTHELLO_NTUPLE_WEIGHTS").is_err() {
        eprintln!(
            "OTHELLO_NTUPLE_WEIGHTS is unset -- run `othello-eval-train` and point it at the \
             output directory"
        );
        std::process::exit(2);
    }

    let mut args = std::env::args().skip(1);
    let cfg_path = args
        .next()
        .filter(|a| !a.starts_with("--") && a != "gate" && a != "edax" && a != "h2h")
        .unwrap_or_else(|| "games/othello/ntuple/match.toml".to_string());
    let mode = args.next().unwrap_or_else(|| "gate".to_string());

    let cfg: Config = toml::from_str(
        &std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}")),
    )
    .expect("config must parse");

    println!(
        "weights={}  ks={:?}  depths={:?}  games/pairing={}  seed={}",
        std::env::var("OTHELLO_NTUPLE_WEIGHTS").unwrap(),
        cfg.ks,
        cfg.depths,
        cfg.games_per_pairing,
        cfg.seed
    );

    match mode.as_str() {
        "gate" => run_gate(&cfg),
        "edax" => run_edax(&cfg),
        "h2h" => run_h2h(&cfg),
        other => panic!("unknown mode {other:?} (want gate | edax | h2h)"),
    }
}

#[cfg(test)]
mod tests {
    use super::common::{record_game, Tally};

    /// The alternating-colour aggregation this example's gate verdict rests
    /// on -- same instrumentation check `edax_match` makes, re-asserted here
    /// because the pass/fail number is computed from it.
    #[test]
    fn gate_aggregation_matches_hand_computation() {
        let mut t = Tally::default();
        // Contender ("hero") as s1 wins twice, as s2 wins once, loses once.
        record_game(&mut t, true, Some(0));
        record_game(&mut t, true, Some(0));
        record_game(&mut t, false, Some(1));
        record_game(&mut t, true, Some(1));
        assert_eq!((t.wins, t.losses, t.draws), (3, 1, 0));
        let (p, (lo, hi)) = t.win_rate_ci(1.96);
        assert!((p - 0.75).abs() < 1e-9);
        assert!(lo < 0.75 && hi > 0.75);
    }
}
