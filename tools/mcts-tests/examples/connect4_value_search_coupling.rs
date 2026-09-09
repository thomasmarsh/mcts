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

use std::io::Write;
use std::process::ExitCode;

use game_connect4::convnet::CnnValuePolicyNet;
use game_connect4::reference_diagnostic::reference_negamax_score;
use game_connect4::{Move, Standard, State};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig, GumbelOutcome};
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

fn run(cfg_row: &Config, net: &CnnValuePolicyNet, games: usize, sims: u32) -> (usize, usize, usize) {
    let gumbel = GumbelConfig {
        sims,
        c_scale: cfg_row.c_scale,
        c_visit: cfg_row.c_visit,
        rescale_q: cfg_row.rescale_q,
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

    println!("value/search coupling sweep: weights={} games={games} sims={sims}", args[1]);
    println!("config                         c_scale c_visit rescale gain gate negamax  W-D-L        share");
    for row in configs() {
        if filter.is_some_and(|f| !row.name.contains(f)) {
            continue;
        }
        let (w, d, l) = run(&row, &net, games, sims);
        let share = (w as f64 + 0.5 * d as f64) / games as f64;
        let negamax = row
            .negamax_depth
            .map_or_else(|| "-".to_string(), |d| d.to_string());
        println!(
            "{:<30} {:>7} {:>7} {:>7} {:>4} {:>4} {:>7}  {w:>3}-{d:>3}-{l:<3}  {share:.3}",
            row.name, row.c_scale, row.c_visit, row.rescale_q, row.gain, row.gate, negamax
        );
        if let Some(path) = &out_path {
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(
                    file,
                    "{{\"config\":\"{}\",\"c_scale\":{},\"c_visit\":{},\"rescale_q\":{},\"gain\":{},\"gate\":{},\"negamax_depth\":{},\"games\":{games},\"sims\":{sims},\"wins\":{w},\"draws\":{d},\"losses\":{l},\"share\":{share:.4}}}",
                    row.name, row.c_scale, row.c_visit, row.rescale_q, row.gain, row.gate, negamax
                );
            }
        }
    }
    ExitCode::SUCCESS
}
