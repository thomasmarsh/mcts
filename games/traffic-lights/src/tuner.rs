//! SMAC3-tunable RAVE strategy for Traffic Lights, backing `TlAdapter::tuner`/
//! `tune_eval` in `main.rs`. Ported from the former `examples/hyper.rs`,
//! which embedded the game type and a `mcts-bench` tournament helper at
//! compile time -- both incompatible with the subprocess-per-game
//! architecture. The evaluation loop here is self-contained (no
//! `mcts-bench` dependency) since it only ever plays one trial's `rounds`
//! games sequentially in a single `tune eval` process invocation.

use std::marker::PhantomData;
use std::str::FromStr;

use game_host::{HostError, TunerCondition, TunerInfo, TunerParameter};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::select::{self, RaveSchedule, RaveUcb, SelectStrategy};
use mcts::strategies::mcts::{backprop, node::QInit, simulate, SearchConfig, Strategy, TreeSearch};
use mcts::strategies::Search;
use serde_json::{json, Value};

use game_traffic_lights::TrafficLights;

type G = TrafficLights;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
pub const EVAL_ROUNDS: u32 = 20;

const PLAYOUT_DEPTH: usize = 200;
const MAX_ITER: usize = 10_000;
const EXPAND_THRESHOLD: u32 = 1;

fn base_config<S: Strategy<G>>() -> SearchConfig<G, S> {
    SearchConfig::new()
        .max_iterations(MAX_ITER)
        .max_playout_depth(PLAYOUT_DEPTH)
        .expand_threshold(EXPAND_THRESHOLD)
}

// ---------------------------------------------------------------------------
// Candidate strategy family
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Default)]
struct CandidateStrategy<FinalAction: SelectStrategy<G>>(PhantomData<FinalAction>);

impl<FinalAction: SelectStrategy<G>> Strategy<G> for CandidateStrategy<FinalAction> {
    type Select = select::Rave;
    type Simulate = simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>;
    type Backprop = backprop::Classic;
    type FinalAction = FinalAction;

    fn friendly_name() -> String {
        "candidate".into()
    }

    fn config() -> SearchConfig<G, Self> {
        base_config()
    }
}

/// One trial's candidate parameters, deserialized from the `params` JSON
/// object `tune_eval` receives -- the merged active-parameter set
/// `target.py` builds from `smac3/config/default.yaml`'s search space.
#[derive(Debug, serde::Deserialize)]
struct TrialParams {
    threshold: u32,
    c: Option<f64>,
    epsilon: f64,
    q_init: String,
    final_action: String,
    a: Option<f64>,
    bias: Option<f64>,
    schedule: String,
    k: Option<u32>,
    rave: Option<u32>,
    rave_ucb: String,
}

impl<FinalAction: SelectStrategy<G>> CandidateStrategy<FinalAction> {
    fn config_with_params(p: &TrialParams, seed: u64) -> Result<SearchConfig<G, Self>, HostError> {
        let missing = |field: &str| HostError::bad_request(format!("missing param: {field}"));

        let schedule = match p.schedule.as_str() {
            "hand_selected" => RaveSchedule::HandSelected {
                k: p.k.ok_or_else(|| missing("k"))?,
            },
            "min_mse" => RaveSchedule::MinMSE {
                bias: p.bias.ok_or_else(|| missing("bias"))?,
            },
            "threshold" => RaveSchedule::Threshold {
                rave: p.rave.ok_or_else(|| missing("rave"))?,
            },
            other => return Err(HostError::bad_request(format!("unknown schedule: {other}"))),
        };

        let ucb = match p.rave_ucb.as_str() {
            "none" => RaveUcb::None,
            "ucb1" => RaveUcb::Ucb1 {
                exploration_constant: p.c.ok_or_else(|| missing("c"))?,
            },
            "tuned" => RaveUcb::Ucb1Tuned {
                exploration_constant: p.c.ok_or_else(|| missing("c"))?,
            },
            other => return Err(HostError::bad_request(format!("unknown rave_ucb: {other}"))),
        };

        let q_init = QInit::from_str(&p.q_init)
            .map_err(|_| HostError::bad_request(format!("invalid q_init: {}", p.q_init)))?;

        Ok(Self::config()
            .q_init(q_init)
            .use_transpositions(true)
            .select(select::Rave::new(p.threshold, schedule, ucb))
            .simulate(
                simulate::DecisiveMove::new()
                    .mode(simulate::DecisiveMoveMode::WinLoss)
                    .inner(simulate::EpsilonGreedy::with_epsilon(p.epsilon)),
            )
            .seed(seed))
    }
}

fn make_candidate(p: &TrialParams, seed: u64) -> Result<Box<dyn Search<G = G>>, HostError> {
    match p.final_action.as_str() {
        "max_avg" => {
            let config = CandidateStrategy::<select::SecureChild>::config_with_params(p, seed)?;
            Ok(Box::new(
                TreeSearch::<G, CandidateStrategy<select::SecureChild>>::new().config(config),
            ))
        }
        "secure_child" => {
            let mut config = CandidateStrategy::<select::SecureChild>::config_with_params(p, seed)?;
            config.final_action.a =
                p.a.ok_or_else(|| HostError::bad_request("missing param: a"))?;
            Ok(Box::new(
                TreeSearch::<G, CandidateStrategy<select::SecureChild>>::new().config(config),
            ))
        }
        "robust_child" => {
            let config = CandidateStrategy::<select::RobustChild>::config_with_params(p, seed)?;
            Ok(Box::new(
                TreeSearch::<G, CandidateStrategy<select::RobustChild>>::new().config(config),
            ))
        }
        other => Err(HostError::bad_request(format!(
            "unknown final_action: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Result of one `tune_eval` call: `cost` is what SMAC3 minimizes (the
/// candidate's loss rate against the baseline); `wins`/`losses`/`draws` are
/// from the candidate's perspective, for display.
#[derive(Debug)]
pub struct TuneEvalOutcome {
    pub cost: f64,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// Play `rounds` rounds (each round: one game candidate-first, one game
/// baseline-first) of `params`-built candidate vs `baseline_build()`, and
/// return the aggregate outcome. `baseline_build` is a thunk rather than an
/// already-built search so a fresh, un-treed instance is used for every
/// single game, matching how each preset's `build_*` function is used
/// elsewhere in this crate.
pub fn tune_eval(
    params: &Value,
    rounds: u32,
    seed: Option<u64>,
    baseline_build: impl Fn() -> Box<dyn Search<G = G>>,
) -> Result<TuneEvalOutcome, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    let seed = seed.unwrap_or(0);

    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    for _ in 0..rounds {
        let mut candidate = make_candidate(&trial, seed)?;
        let mut baseline = baseline_build();

        let (c, b, d) = play_one(candidate.as_mut(), baseline.as_mut());
        wins += c;
        losses += b;
        draws += d;

        // Swap move order so the candidate plays second half the time.
        let mut candidate = make_candidate(&trial, seed)?;
        let mut baseline = baseline_build();
        let (b, c, d) = play_one(baseline.as_mut(), candidate.as_mut());
        wins += c;
        losses += b;
        draws += d;
    }

    Ok(TuneEvalOutcome {
        cost: cost_from_losses(losses, rounds),
        wins,
        losses,
        draws,
    })
}

/// `cost = losses / (2*rounds)`: the candidate's loss rate across the
/// `2*rounds` games it plays (moving first and second each round), the
/// quantity SMAC3 minimizes. Draws and wins both count as "not a loss" --
/// this doesn't reward wins over draws, only penalizes outright losses.
fn cost_from_losses(losses: u32, rounds: u32) -> f64 {
    let total = 2.0 * rounds as f64;
    if total > 0.0 {
        losses as f64 / total
    } else {
        0.0
    }
}

/// Play one game between `first` (moves when `player_to_move` is index 0)
/// and `second` (index 1). Returns `(first_result, second_result, draws)`
/// win/loss counts -- each pair is `(1,0)`/`(0,1)`/`(0,0)` with the third
/// element set on a draw.
fn play_one(first: &mut dyn Search<G = G>, second: &mut dyn Search<G = G>) -> (u32, u32, u32) {
    let mut state = <G as Game>::S::default();
    while !G::is_terminal(&state) {
        let mover = G::player_to_move(&state).to_index();
        let action = if mover == 0 {
            first.choose_action(&state)
        } else {
            second.choose_action(&state)
        };
        state = G::apply(state, &action);
    }
    match G::winner(&state) {
        None => (0, 0, 1),
        Some(p) if p.to_index() == 0 => (1, 0, 0),
        Some(_) => (0, 1, 0),
    }
}

// ---------------------------------------------------------------------------
// Tuner metadata
// ---------------------------------------------------------------------------

/// Search-space metadata for the RAVE candidate family above, ported from
/// `smac3/config/default.yaml`'s `parameters`/`conditions` sections (which
/// remain the source of truth SMAC3 itself reads; this is what `tune
/// describe` reports for display/introspection).
pub fn tuner_info() -> TunerInfo {
    TunerInfo {
        id: "rave".into(),
        baseline: "strong".into(),
        eval_rounds: EVAL_ROUNDS,
        parameters: vec![
            param(
                "bias",
                json!({"type": "float", "bounds": [0, 10], "default": 0.00001}),
            ),
            param(
                "c",
                json!({"type": "float", "bounds": [0, 3], "default": std::f64::consts::SQRT_2}),
            ),
            param(
                "epsilon",
                json!({"type": "float", "bounds": [0, 1], "default": 0.1}),
            ),
            param(
                "final_action",
                json!({"type": "constant", "value": "robust_child"}),
            ),
            param(
                "k",
                json!({"type": "int", "bounds": [0, 2000], "default": 1000}),
            ),
            param(
                "q_init",
                json!({"type": "categorical", "choices": ["Draw", "Infinity", "Loss", "Parent", "Win"], "default": "Infinity"}),
            ),
            param(
                "rave",
                json!({"type": "int", "bounds": [0, 2000], "default": 700}),
            ),
            param(
                "schedule",
                json!({"type": "categorical", "choices": ["hand_selected", "min_mse", "threshold"], "default": "threshold"}),
            ),
            param(
                "threshold",
                json!({"type": "int", "bounds": [0, 2000], "default": 700}),
            ),
            param(
                "rave_ucb",
                json!({"type": "categorical", "choices": ["none", "ucb1", "tuned"], "default": "tuned"}),
            ),
        ],
        conditions: vec![
            condition(json!({"schedule": "hand_selected"}), &["k"]),
            condition(json!({"schedule": "min_mse"}), &["bias"]),
            condition(json!({"schedule": "threshold"}), &["rave"]),
            condition(json!({"rave_ucb": ["ucb1", "tuned"]}), &["c"]),
        ],
    }
}

fn param(name: &str, spec: Value) -> TunerParameter {
    TunerParameter {
        name: name.into(),
        spec,
    }
}

fn condition(if_: Value, then: &[&str]) -> TunerCondition {
    TunerCondition {
        if_,
        then: then.iter().map(|s| s.to_string()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::strategies::mcts::strategy;

    #[test]
    fn test_cost_from_losses_hand_verified() {
        // 20 rounds -> 40 games; 15 losses -> cost 0.375.
        assert_eq!(cost_from_losses(15, 20), 0.375);
        assert_eq!(cost_from_losses(0, 20), 0.0);
        assert_eq!(cost_from_losses(40, 20), 1.0);
        // 10 losses out of 4 rounds (8 games) -> 1.25, clamped nowhere --
        // callers are expected to pass a `losses` that's actually <= 2*rounds.
        assert_eq!(cost_from_losses(10, 4), 1.25);
    }

    #[test]
    fn test_cost_from_losses_zero_rounds_is_zero() {
        assert_eq!(cost_from_losses(0, 0), 0.0);
    }

    #[test]
    fn test_tune_eval_rejects_params_missing_required_field() {
        // "schedule": "threshold" requires "rave", which is absent -- this
        // must fail fast during config construction, before any game is
        // played (no real MCTS search runs in this test).
        let params = json!({
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "threshold",
            "rave_ucb": "tuned",
        });
        let err = tune_eval(&params, 1, Some(0), || {
            Box::new(TreeSearch::<G, strategy::Ucb1>::new())
        })
        .expect_err("missing `rave` must be rejected");
        assert!(err.message.contains("rave"));
    }

    #[test]
    fn test_tune_eval_rejects_unknown_schedule() {
        let params = json!({
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "not_a_real_schedule",
            "rave_ucb": "tuned",
        });
        let err = tune_eval(&params, 1, Some(0), || {
            Box::new(TreeSearch::<G, strategy::Ucb1>::new())
        })
        .expect_err("unknown schedule must be rejected");
        assert!(err.message.contains("schedule"));
    }
}
