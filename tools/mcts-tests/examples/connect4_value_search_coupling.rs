//! Isolate the Connect Four value/search coupling failure.
//!
//!     cargo run --release -p mcts-tests --example connect4_value_search_coupling -- \
//!         <value-weights.c4cnn> [games] [sims] [filter] [--out <path>]
//!
//! With the trained value head frozen, sweep the completed-Q value scale
//! (`c_scale`/`c_visit`), min-max rescale on/off, a learned-value gain, a
//! confidence gate, and an exact bounded-negamax leaf control, all against the
//! same equal-budget zero-net Gumbel opponent used by
//! `connect4_cnn_smoke_gate`. Each configuration prints one line and, with
//! `--out`, appends one JSON object so a long sweep checkpoints as it runs.
//!
//! Two focused modes reuse the same harness:
//!
//!   * `--budget-sweep [--budgets 32,64,...]` runs the two anchor configs
//!     (Mctx-verbatim `c_visit=50,c_scale=0.1,rescale=true`; Slice 4.6z best
//!     `c_visit=0,c_scale=0.05,rescale=false`) across a simulation-budget
//!     sweep.
//!   * `--root-only` runs each anchor as Full Gumbel and as root-only Gumbel
//!     (interior PUCT, no completed-Q override) at 32/128/256 sims.
//!   * `--sigma-sweep` runs each visit-evidence-scaled `sigma` mode (smooth,
//!     hard gate `k in {2,3,4}`, realized-only) across `c_scale in {0.05,0.1}`
//!     and `c_visit in {50,0}` at 32/128 sims, plus the NodeFloor control.

use std::io::Write;
use std::process::ExitCode;

use game_connect4::convnet::CnnValuePolicyNet;
use game_connect4::reference_diagnostic::reference_negamax_score;
use game_connect4::{Move, Standard, State};
use mcts::algorithms::mcts::gumbel::{
    gumbel_search_with_root_value, GumbelConfig, GumbelOutcome, RootMoveSelection, SigmaMode,
};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};
use mcts::util::battle_royale;

/// A leaf/root value source with an optional post-transform. The `Default`
/// impl (all-zero net, identity transform) only ever backs
/// `EvaluatedCutoff::new()` before `.evaluator(..)` overrides it.
#[derive(Clone, Default)]
struct CoupledEvaluator {
    net: Option<CnnValuePolicyNet>,
    negamax_depth: Option<u32>,
    /// Multiply the raw value after gating. `1.0` is untouched.
    gain: f32,
    /// Zero out `|value| < gate` so visit counts drive where the net is
    /// unsure. `0.0` disables the gate.
    gate: f32,
}

impl CoupledEvaluator {
    fn zero() -> Self {
        Self {
            net: None,
            negamax_depth: None,
            gain: 1.0,
            gate: 0.0,
        }
    }

    fn adjusted_value(&self, state: &State<6, 7>) -> f32 {
        let raw = if let Some(depth) = self.negamax_depth {
            reference_negamax_score(state, depth).signum() as f32
        } else if let Some(net) = &self.net {
            net.value(state)
        } else {
            0.0
        };
        let gated = if raw.abs() < self.gate { 0.0 } else { raw };
        (gated * self.gain).clamp(-1.0, 1.0)
    }
}

impl Evaluator<Standard> for CoupledEvaluator {
    fn evaluate(&self, state: &State<6, 7>) -> Score {
        (self.adjusted_value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

type Profile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Standard, CoupledEvaluator>>;

struct CoupledPlayer {
    search: TreeSearch<Standard, Profile>,
    eval: CoupledEvaluator,
    cfg: GumbelConfig,
}

impl CoupledPlayer {
    fn new(eval: CoupledEvaluator, policy: CnnValuePolicyNet, cfg: GumbelConfig, seed: u64) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .select(GumbelCompletedQ::with_config(cfg))
                .simulate(EvaluatedCutoff::new().evaluator(eval.clone()))
                .with_policy_logits(policy)
                .seed(seed),
        );
        Self { search, eval, cfg }
    }

    fn choose(&mut self, state: &State<6, 7>) -> GumbelOutcome<Move> {
        gumbel_search_with_root_value(
            &mut self.search,
            state,
            &self.cfg,
            self.eval.adjusted_value(state) as f64,
        )
    }
}

impl Search for CoupledPlayer {
    type G = Standard;
    fn friendly_name(&self) -> String {
        "coupled".into()
    }
    fn set_friendly_name(&mut self, _: &str) {}
    fn choose_action(&mut self, state: &State<6, 7>) -> Move {
        self.choose(state).action
    }
}

struct Config {
    name: &'static str,
    c_scale: f64,
    c_visit: f64,
    rescale_q: bool,
    gain: f32,
    gate: f32,
    negamax_depth: Option<u32>,
    /// `false` selects root-only Gumbel: interior nodes use PUCT, no
    /// completed-Q override. The default Full Gumbel path keeps this `true`.
    interior_completed_q: bool,
    /// Root Sequential-Halving `sigma` scaling rule. `NodeFloor` is the
    /// Mctx-verbatim per-node floor; the other modes make it per-action and
    /// visit-aware.
    sigma_mode: SigmaMode,
}

/// The Mctx-verbatim completed-Q constants (`c_visit=50, c_scale=0.1,
/// rescale=true`) and Slice 4.6z's best hand-tuned constants
/// (`c_visit=0, c_scale=0.05, rescale=false`), the two anchors both the
/// budget sweep and the root-only control run at.
fn anchor(name: &'static str, mctx: bool, interior_completed_q: bool) -> Config {
    Config {
        name,
        c_scale: if mctx { 0.1 } else { 0.05 },
        c_visit: if mctx { 50.0 } else { 0.0 },
        rescale_q: mctx,
        gain: 1.0,
        gate: 0.0,
        negamax_depth: None,
        interior_completed_q,
        sigma_mode: SigmaMode::NodeFloor,
    }
}

/// Visit-evidence-scaled `sigma` sweep. Each visit-aware sigma mode at the
/// swept budgets with `c_scale in {0.05, 0.1}` and `c_visit in {50, 0}`,
/// raw completed-Q (no min-max rescale), plus the current best baseline
/// (`c_visit=0, c_scale=0.05, NodeFloor`) as the control.
fn sigma_sweep_configs() -> Vec<Config> {
    let mut out = vec![Config {
        name: "control_best4.6z",
        c_scale: 0.05,
        c_visit: 0.0,
        rescale_q: false,
        gain: 1.0,
        gate: 0.0,
        negamax_depth: None,
        interior_completed_q: true,
        sigma_mode: SigmaMode::NodeFloor,
    }];
    for (mtag, mode) in [
        ("smooth", SigmaMode::Smooth),
        ("gate2", SigmaMode::HardGate(2)),
        ("gate3", SigmaMode::HardGate(3)),
        ("gate4", SigmaMode::HardGate(4)),
        ("realized", SigmaMode::RealizedOnly),
    ] {
        for &c_scale in &[0.05f64, 0.1] {
            for &c_visit in &[50.0f64, 0.0] {
                let name: &'static str = Box::leak(
                    format!("{mtag}_cs{c_scale}_cv{c_visit}").into_boxed_str(),
                );
                out.push(Config {
                    name,
                    c_scale,
                    c_visit,
                    rescale_q: false,
                    gain: 1.0,
                    gate: 0.0,
                    negamax_depth: None,
                    interior_completed_q: true,
                    sigma_mode: mode,
                });
            }
        }
    }
    out
}

/// Slice 3.1/3.2 budget sweep: the two anchor configs, Full Gumbel, swept
/// over the simulation budgets supplied on the command line.
fn budget_configs() -> Vec<Config> {
    vec![
        anchor("mctx_verbatim", true, true),
        anchor("best_4_6z", false, true),
    ]
}

/// Root-only-Gumbel control: each anchor as Full Gumbel and as root-only
/// (interior PUCT, no completed-Q override).
fn rootonly_configs() -> Vec<Config> {
    vec![
        anchor("mctx_full", true, true),
        anchor("mctx_rootonly", true, false),
        anchor("best4_6z_full", false, true),
        anchor("best4_6z_rootonly", false, false),
    ]
}

fn configs() -> Vec<Config> {
    let mut out = Vec::new();
    let base = |name, c_scale| Config {
        name,
        c_scale,
        c_visit: 50.0,
        rescale_q: true,
        gain: 1.0,
        gate: 0.0,
        negamax_depth: None,
        interior_completed_q: true,
        sigma_mode: SigmaMode::NodeFloor,
    };
    // c_scale sweep, min-max rescale on (the reference default).
    out.push(base("cscale_0.02", 0.02));
    out.push(base("cscale_0.05", 0.05));
    out.push(base("cscale_0.1", 0.1));
    out.push(base("cscale_0.2", 0.2));
    out.push(base("cscale_0.5", 0.5));
    out.push(base("cscale_1.0", 1.0));
    // Raw completed Q (no min-max rescale): value magnitude now reaches the
    // ranking key directly, so c_scale and a gain both bite.
    for (name, c_scale) in [
        ("noresc_cscale_0.1", 0.1),
        ("noresc_cscale_0.5", 0.5),
        ("noresc_cscale_1.0", 1.0),
    ] {
        out.push(Config {
            rescale_q: false,
            ..base(name, c_scale)
        });
    }
    out.push(Config {
        c_visit: 0.0,
        rescale_q: false,
        ..base("noresc_cvisit0_cscale_0.1", 0.1)
    });
    // Squash the learned leaf value toward zero.
    for (name, gain) in [("gain_0.5", 0.5f32), ("gain_0.25", 0.25)] {
        out.push(Config {
            gain,
            rescale_q: false,
            ..base(name, 0.1)
        });
    }
    // Fire the leaf value only on high-confidence positions.
    for (name, gate) in [("gate_0.3", 0.3f32), ("gate_0.5", 0.5), ("gate_0.7", 0.7)] {
        out.push(Config {
            gate,
            ..base(name, 0.1)
        });
    }
    for (name, gate) in [("noresc_gate_0.5", 0.5f32), ("noresc_gate_0.7", 0.7)] {
        out.push(Config {
            gate,
            rescale_q: false,
            ..base(name, 1.0)
        });
    }
    // Fine grid around a small visit offset, where the sigma term stops
    // swamping the Gumbel exploration at a 32-sim budget.
    for &c_visit in &[0.0f64, 0.5, 1.0, 2.0, 5.0] {
        for &c_scale in &[0.05f64, 0.1, 0.25, 0.5] {
            for &rescale_q in &[true, false] {
                let tag: &'static str = Box::leak(
                    format!("grid_cv{c_visit}_cs{c_scale}_{}", if rescale_q { "resc" } else { "raw" })
                        .into_boxed_str(),
                );
                out.push(Config {
                    c_visit,
                    rescale_q,
                    ..base(tag, c_scale)
                });
            }
        }
    }
    // Small visit offset plus a squashed learned value.
    for &gain in &[0.5f32, 0.25] {
        out.push(Config {
            c_visit: 0.0,
            gain,
            rescale_q: false,
            ..base(Box::leak(format!("cv0_gain{gain}_raw").into_boxed_str()), 0.1)
        });
    }
    // Exact bounded-negamax leaf control at the recovered small-offset config.
    for &(c_visit, c_scale, rescale_q) in &[(0.0f64, 0.1f64, false), (0.0, 0.25, true), (1.0, 0.1, false)] {
        out.push(Config {
            negamax_depth: Some(8),
            c_visit,
            rescale_q,
            ..base(
                Box::leak(
                    format!("negamax8_cv{c_visit}_cs{c_scale}_{}", if rescale_q { "resc" } else { "raw" })
                        .into_boxed_str(),
                ),
                c_scale,
            )
        });
    }
    // Exact bounded-negamax leaf control: does ANY informative leaf value help?
    for (name, c_scale, rescale_q) in [
        ("negamax8_cscale_0.1", 0.1, true),
        ("negamax8_cscale_1.0", 1.0, true),
        ("negamax8_noresc_cscale_1.0", 1.0, false),
    ] {
        out.push(Config {
            negamax_depth: Some(8),
            rescale_q,
            ..base(name, c_scale)
        });
    }
    out
}

fn run(
    cfg_row: &Config,
    net: &CnnValuePolicyNet,
    games: usize,
    sims: u32,
    root_move_selection: RootMoveSelection,
) -> (usize, usize, usize) {
    let gumbel = GumbelConfig {
        sims,
        c_scale: cfg_row.c_scale,
        c_visit: cfg_row.c_visit,
        rescale_q: cfg_row.rescale_q,
        interior_completed_q: cfg_row.interior_completed_q,
        sigma_mode: cfg_row.sigma_mode,
        root_move_selection,
        ..GumbelConfig::default()
    };
    let trained_eval = CoupledEvaluator {
        net: cfg_row.negamax_depth.is_none().then(|| net.clone()),
        negamax_depth: cfg_row.negamax_depth,
        gain: cfg_row.gain,
        gate: cfg_row.gate,
    };
    let mut trained = CoupledPlayer::new(trained_eval, net.clone(), gumbel, 7);
    let mut zero = CoupledPlayer::new(
        CoupledEvaluator::zero(),
        CnnValuePolicyNet::default(),
        gumbel,
        11,
    );
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game in 0..games {
        let (result, trained_first) = if game % 2 == 0 {
            (battle_royale::<Standard, _, _>(&mut trained, &mut zero), true)
        } else {
            (battle_royale::<Standard, _, _>(&mut zero, &mut trained), false)
        };
        match result {
            None => draws += 1,
            Some(0) if trained_first => wins += 1,
            Some(1) if !trained_first => wins += 1,
            _ => losses += 1,
        }
    }
    (wins, draws, losses)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: connect4_value_search_coupling <value-weights.c4cnn> [games] [sims] [filter] [--out <path>]"
        );
        return ExitCode::FAILURE;
    }
    let net = match CnnValuePolicyNet::load(&args[1]) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{}: {error}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let positional: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let games: usize = positional.first().map_or(60, |s| s.parse().expect("games"));
    let sims: u32 = positional.get(1).map_or(32, |s| s.parse().expect("sims"));
    let filter = positional.get(2).map(|s| s.as_str());
    let out_path = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| w[1].clone());
    let flag = |name: &str| args.iter().any(|a| a == name);
    // `--root-move visit-count` demotes the completed-Q ranking to a
    // training-only target: search returns `argmax_a N(a)` and interior
    // selection is plain PUCT.
    let root_move_selection = match args
        .windows(2)
        .find(|w| w[0] == "--root-move")
        .map(|w| w[1].as_str())
    {
        None | Some("completed-q") => RootMoveSelection::CompletedQ,
        Some("visit-count") => RootMoveSelection::VisitCount,
        Some(other) => {
            eprintln!("unknown --root-move {other}");
            return ExitCode::FAILURE;
        }
    };
    let budgets: Vec<u32> = args
        .windows(2)
        .find(|w| w[0] == "--budgets")
        .map(|w| w[1].split(',').map(|s| s.parse().expect("budget")).collect())
        .unwrap_or_else(|| vec![32, 64, 128, 256, 512, 1024]);

    // (label for the run header, config table, simulation budgets to sweep).
    let (header, rows, sweep): (&str, Vec<Config>, Vec<u32>) = if flag("--budget-sweep") {
        ("budget sweep (Full Gumbel, two anchor configs)", budget_configs(), budgets)
    } else if flag("--root-only") {
        ("root-only vs Full Gumbel control", rootonly_configs(), vec![32, 128, 256])
    } else if flag("--sigma-sweep") {
        let sweep = if args.iter().any(|a| a == "--budgets") {
            budgets
        } else {
            vec![32, 128]
        };
        ("visit-evidence-scaled sigma sweep", sigma_sweep_configs(), sweep)
    } else {
        ("value/search coupling sweep", configs(), vec![sims])
    };

    println!(
        "{header}: weights={} games={games} root_move={root_move_selection:?}",
        args[1]
    );
    println!("config                              sims c_scale c_visit rescale intCQ sigma_mode      gain gate negamax  W-D-L        share");
    for &s in &sweep {
        for row in &rows {
            if filter.is_some_and(|f| !row.name.contains(f)) {
                continue;
            }
            let start = std::time::Instant::now();
            let (w, d, l) = run(row, &net, games, s, root_move_selection);
            let secs = start.elapsed().as_secs_f64();
            let share = (w as f64 + 0.5 * d as f64) / games as f64;
            let negamax = row
                .negamax_depth
                .map_or_else(|| "-".to_string(), |d| d.to_string());
            let sigma_tag = format!("{:?}", row.sigma_mode);
            println!(
                "{:<35} {s:>4} {:>7} {:>7} {:>7} {:>5} {:>14} {:>4} {:>4} {:>7}  {w:>3}-{d:>3}-{l:<3}  {share:.3}  {secs:.0}s",
                row.name, row.c_scale, row.c_visit, row.rescale_q, row.interior_completed_q, sigma_tag, row.gain, row.gate, negamax
            );
            if let Some(path) = &out_path {
                if let Ok(mut file) =
                    std::fs::OpenOptions::new().create(true).append(true).open(path)
                {
                    let _ = writeln!(
                        file,
                        "{{\"config\":\"{}\",\"c_scale\":{},\"c_visit\":{},\"rescale_q\":{},\"interior_completed_q\":{},\"sigma_mode\":\"{}\",\"root_move\":\"{root_move_selection:?}\",\"gain\":{},\"gate\":{},\"negamax_depth\":{},\"games\":{games},\"sims\":{s},\"wins\":{w},\"draws\":{d},\"losses\":{l},\"share\":{share:.4},\"wall_s\":{secs:.1}}}",
                        row.name, row.c_scale, row.c_visit, row.rescale_q, row.interior_completed_q, sigma_tag, row.gain, row.gate, negamax
                    );
                }
            }
        }
    }
    ExitCode::SUCCESS
}
