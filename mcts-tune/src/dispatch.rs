//! Algorithm-native construction of a search configuration.
//!
//! [`to_algorithm_spec`] resolves a params object -- the top-level
//! `algorithm` categorical and, for `mcts`, the four policy-axis categoricals
//! (`select`/`simulate`/`backprop`/`final_action`) plus each variant's own
//! scalar parameters -- into an [`AlgorithmSpec`]: a `config_ir::SearchSpec`
//! for `mcts`, or the parameter set one of the three standalone `Search`
//! impls needs. It is a `match` per axis rather than a row per named
//! composition; `search.rs` and `tuner_info.rs` are the callers.

use std::str::FromStr;

use game_host::HostError;
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::negamax;
use mcts::evaluator::Score;
use mcts::select::{GpnBias, RaveSchedule, RaveUcb};
use mcts::simulate::DecisiveMoveMode;
use serde_json::Value;

use crate::config_ir::codec::{field, field_opt};
use crate::config_ir::{
    BackpropSpec, BaseSelectSpec, BaseSimulateSpec, FinalActionSpec, SearchSpec, SelectSpec,
    SimulateSpec,
};
use crate::search::META_MCTS_INNER_ITERATIONS;

/// A fully resolved algorithm choice: an MCTS composition (routed through
/// `config_ir::build_search`, same as any axis-composed `TreeSearch`), or one
/// of the three standalone `Search` impls that carry their own parameter set
/// and no policy axes at all.
pub(crate) enum AlgorithmSpec {
    Mcts(SearchSpec),
    Bandit {
        budget: u32,
        max_rollout_depth: u32,
        policy: BanditPolicySpec,
    },
    Negamax {
        max_depth: u32,
        table_bits: u32,
        replacement: negamax::Replacement,
        aspiration_window: Option<Score>,
        principal_variation_search: bool,
        history_heuristic: bool,
        singular_extension: bool,
        countermove_heuristic: bool,
    },
    Random,
}

/// A `bandit_policy` categorical resolved to the parameters
/// `direct_search::build_direct` needs to construct the matching
/// `algorithms::bandit::BanditPolicy` impl.
pub(crate) enum BanditPolicySpec {
    Random,
    EpsilonGreedy { epsilon: f64 },
    Ucb1 { c: f64 },
    Thompson,
}

fn missing(name: &str) -> HostError {
    HostError::bad_request(format!("missing param: {name}"))
}

fn req_f64(cfg: &Value, name: &str) -> Result<f64, HostError> {
    field_opt::<f64>(cfg, name)
        .map_err(HostError::bad_request)?
        .ok_or_else(|| missing(name))
}

fn req_u32(cfg: &Value, name: &str) -> Result<u32, HostError> {
    field_opt::<u32>(cfg, name)
        .map_err(HostError::bad_request)?
        .ok_or_else(|| missing(name))
}

fn req_str<'a>(cfg: &'a Value, name: &str) -> Result<&'a str, HostError> {
    cfg.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| missing(name))
}

fn opt_bool(cfg: &Value, name: &str) -> bool {
    cfg.get(name).and_then(Value::as_bool).unwrap_or(false)
}

fn decisive_move_mode(cfg: &Value) -> Result<DecisiveMoveMode, HostError> {
    let raw = req_str(cfg, "decisive_move_mode")?;
    serde_json::from_value(Value::String(raw.to_string()))
        .map_err(|_| HostError::bad_request(format!("unknown decisive_move_mode: {raw}")))
}

/// The base `select` variant a categorical names, before any
/// `select_epsilon_greedy` wrapping.
fn base_select_spec(select: &str, cfg: &Value) -> Result<BaseSelectSpec, HostError> {
    Ok(match select {
        "ucb1" => BaseSelectSpec::Ucb1 { c: req_f64(cfg, "c")? },
        "ucb1_tuned" => BaseSelectSpec::Ucb1Tuned { c: req_f64(cfg, "c")? },
        "ucb_v" => BaseSelectSpec::UcbV { c: req_f64(cfg, "c")? },
        "kl_ucb" => BaseSelectSpec::KlUcb { c: req_f64(cfg, "c")? },
        "grill_act" => BaseSelectSpec::GrillAct { c: req_f64(cfg, "c")? },
        "ments" => BaseSelectSpec::Ments {
            tau: req_f64(cfg, "tau")?,
            epsilon: req_f64(cfg, "epsilon")?,
        },
        "score_bounded_uct" => BaseSelectSpec::ScoreBoundedUct {
            c: req_f64(cfg, "c")?,
            gamma: req_f64(cfg, "gamma")?,
            delta: req_f64(cfg, "delta")?,
        },
        "gpn" => BaseSelectSpec::Gpn {
            c: req_f64(cfg, "c")?,
            c_pn: req_f64(cfg, "c_pn")?,
            bias: match req_str(cfg, "gpn_bias")? {
                "max" => GpnBias::Max,
                "sum" => GpnBias::Sum,
                "rank" => GpnBias::Rank,
                other => {
                    return Err(HostError::bad_request(format!("unknown gpn_bias: {other}")))
                }
            },
        },
        "amaf" => BaseSelectSpec::Amaf {
            alpha: req_f64(cfg, "amaf_alpha")?,
            c: req_f64(cfg, "c")?,
        },
        "uct_pn" => BaseSelectSpec::UctPn {
            c: req_f64(cfg, "c")?,
            c_pn: req_f64(cfg, "c_pn")?,
        },
        "progressive_history" => BaseSelectSpec::ProgressiveHistory {
            c: req_f64(cfg, "c")?,
            ph_weight: req_f64(cfg, "ph_weight")?,
        },
        "bayes_uct1" => BaseSelectSpec::BayesUct1 { c: req_f64(cfg, "c")? },
        "bayes_uct2" => BaseSelectSpec::BayesUct2 { c: req_f64(cfg, "c")? },
        "rave" => BaseSelectSpec::Rave {
            threshold: req_u32(cfg, "threshold")?,
            schedule: match req_str(cfg, "schedule")? {
                "hand_selected" => RaveSchedule::HandSelected {
                    k: req_u32(cfg, "k")?,
                },
                "min_mse" => RaveSchedule::MinMSE {
                    bias: req_f64(cfg, "bias")?,
                },
                "threshold" => RaveSchedule::Threshold {
                    rave: req_u32(cfg, "rave")?,
                },
                other => {
                    return Err(HostError::bad_request(format!("unknown schedule: {other}")))
                }
            },
            ucb: match req_str(cfg, "rave_ucb")? {
                "none" => RaveUcb::None,
                "ucb1" => RaveUcb::Ucb1 {
                    exploration_constant: req_f64(cfg, "c")?,
                },
                "tuned" => RaveUcb::Ucb1Tuned {
                    exploration_constant: req_f64(cfg, "c")?,
                },
                other => {
                    return Err(HostError::bad_request(format!("unknown rave_ucb: {other}")))
                }
            },
        },
        other => return Err(HostError::bad_request(format!("unknown select: {other}"))),
    })
}

/// Resolves the `select` axis: the base variant, optionally wrapped by the
/// `select_epsilon_greedy` toggle.
pub(crate) fn to_select_spec(cfg: &Value) -> Result<SelectSpec, HostError> {
    let select = req_str(cfg, "select")?;
    let base = base_select_spec(select, cfg)?;
    if opt_bool(cfg, "select_epsilon_greedy") {
        return Ok(SelectSpec::EpsilonGreedy {
            epsilon: req_f64(cfg, "epsilon")?,
            inner: base,
        });
    }
    // No `EpsilonGreedy` wrapper: promote the base variant to its `SelectSpec`
    // twin. Round-tripping through the wire tag keeps this in lockstep with
    // `register_select!`'s table without re-matching every variant here.
    let tagged = serde_json::to_value(&base).map_err(|e| HostError::internal(e.to_string()))?;
    serde_json::from_value(tagged).map_err(|e| HostError::internal(e.to_string()))
}

fn base_simulate_spec(simulate: &str, cfg: &Value) -> Result<BaseSimulateSpec, HostError> {
    Ok(match simulate {
        "uniform" => BaseSimulateSpec::Uniform {},
        "mast" => BaseSimulateSpec::Mast {},
        "lgr" => BaseSimulateSpec::Lgr {},
        "lgr2" => BaseSimulateSpec::Lgr2 {},
        "lgr2_mast" => BaseSimulateSpec::Lgr2Mast {},
        "nst" => BaseSimulateSpec::Nst {
            backoff_threshold: req_u32(cfg, "nst_backoff_threshold")?,
        },
        other => return Err(HostError::bad_request(format!("unknown simulate inner: {other}"))),
    })
}

/// Resolves the `simulate` axis from its categorical, the
/// `simulate_epsilon_greedy` toggle, and `decisive_move_mode` (read only by
/// the `decisive_move*` variants).
pub(crate) fn to_simulate_spec(cfg: &Value) -> Result<SimulateSpec, HostError> {
    let simulate = req_str(cfg, "simulate")?;
    let eg = opt_bool(cfg, "simulate_epsilon_greedy");
    Ok(match simulate {
        "meta_mcts" => SimulateSpec::MetaMcts {
            iterations: META_MCTS_INNER_ITERATIONS,
        },
        "decisive_move" => SimulateSpec::DecisiveMove {
            mode: decisive_move_mode(cfg)?,
            inner: BaseSimulateSpec::Uniform {},
        },
        "decisive_move_mast" => SimulateSpec::DecisiveMoveMast {
            mode: decisive_move_mode(cfg)?,
            epsilon: req_f64(cfg, "epsilon")?,
        },
        "decisive_move_nst" => SimulateSpec::DecisiveMoveNst {
            mode: decisive_move_mode(cfg)?,
            epsilon: req_f64(cfg, "epsilon")?,
            nst_backoff_threshold: req_u32(cfg, "nst_backoff_threshold")?,
        },
        base => {
            let inner = base_simulate_spec(base, cfg)?;
            if eg {
                SimulateSpec::EpsilonGreedy {
                    epsilon: req_f64(cfg, "epsilon")?,
                    inner,
                }
            } else {
                let tagged = serde_json::to_value(&inner)
                    .map_err(|e| HostError::internal(e.to_string()))?;
                serde_json::from_value(tagged).map_err(|e| HostError::internal(e.to_string()))?
            }
        }
    })
}

/// Resolves the `backprop` axis, honouring the four hard-wired
/// select<->backprop couplings that travel together regardless of the
/// `backprop` categorical:
///
/// - `select == bayes_uct1` forces `bayes_gaussian`
/// - `select == bayes_uct2` forces `bayes_numeric`
/// - `select == ments` forces `softmax`, sharing the one `tau`
/// - `bayes_gaussian`/`bayes_numeric` require a `bayes_uct*` select
///
/// `power_mean` and `td` stay free to pair with any select.
pub(crate) fn to_backprop_spec(cfg: &Value) -> Result<BackpropSpec, HostError> {
    let select = req_str(cfg, "select")?;
    let choice = field_opt::<String>(cfg, "backprop")
        .map_err(HostError::bad_request)?
        .unwrap_or_else(|| "classic".to_string());

    match select {
        "bayes_uct1" => {
            return Ok(BackpropSpec::BayesGaussian {
                prior_variance: req_f64(cfg, "prior_variance")?,
                obs_variance: req_f64(cfg, "obs_variance")?,
            })
        }
        "bayes_uct2" => {
            return Ok(BackpropSpec::BayesNumeric {
                prior_variance: req_f64(cfg, "prior_variance")?,
                obs_variance: req_f64(cfg, "obs_variance")?,
                value_lo: req_f64(cfg, "value_lo")?,
                value_hi: req_f64(cfg, "value_hi")?,
            })
        }
        "ments" => return Ok(BackpropSpec::Softmax { tau: req_f64(cfg, "tau")? }),
        _ => {}
    }

    match choice.as_str() {
        "classic" => Ok(BackpropSpec::Classic {}),
        "power_mean" => Ok(BackpropSpec::PowerMean {
            p: req_f64(cfg, "p")?,
            alpha: req_f64(cfg, "alpha")?,
            depth: 0,
        }),
        "td" => Ok(BackpropSpec::Td {
            lambda: req_f64(cfg, "lambda")?,
            max_child: req_u32(cfg, "td_max_child")?,
        }),
        "bayes_gaussian" | "bayes_numeric" => Err(HostError::bad_request(format!(
            "backprop {choice:?} requires a bayes_uct* select"
        ))),
        "softmax" => Err(HostError::bad_request(
            "softmax backprop is only reachable via a ments select",
        )),
        other => Err(HostError::bad_request(format!("unknown backprop: {other}"))),
    }
}

pub(crate) fn to_final_action_spec(cfg: &Value) -> Result<FinalActionSpec, HostError> {
    match req_str(cfg, "final_action")? {
        "robust_child" => Ok(FinalActionSpec::RobustChild {}),
        "max_avg" => Ok(FinalActionSpec::MaxAvg {}),
        "max_robust_child" => Ok(FinalActionSpec::MaxRobustChild {}),
        "secure_child" => Ok(FinalActionSpec::SecureChild {
            a: req_f64(cfg, "a")?,
        }),
        other => Err(HostError::bad_request(format!("unknown final_action: {other}"))),
    }
}

/// The full `select`/`simulate`/`backprop`/`final_action` composition for an
/// `algorithm == mcts` configuration.
pub(crate) fn to_search_spec(cfg: &Value) -> Result<SearchSpec, HostError> {
    Ok(SearchSpec {
        select: to_select_spec(cfg)?,
        simulate: to_simulate_spec(cfg)?,
        backprop: to_backprop_spec(cfg)?,
        final_action: to_final_action_spec(cfg)?,
    })
}

/// The two `SearchSettings` knobs the proof-number selects (Kowalski et al.
/// 2023) populate and every other configuration leaves `None` -- read here
/// off the axis config so `search.rs::mcts_settings` can thread them into
/// `config_ir::SearchSettings`.
pub(crate) fn mcts_engine_overrides(
    cfg: &Value,
) -> Result<(Option<u32>, Option<f64>), HostError> {
    let solver_loss_threshold = field_opt::<u32>(cfg, "solver_loss_threshold")
        .map_err(HostError::bad_request)?;
    let contempt_factor = match cfg.get("contempt").and_then(Value::as_str) {
        None | Some("off") => None,
        Some("on") => Some(req_f64(cfg, "contempt_factor")?),
        Some(other) => {
            return Err(HostError::bad_request(format!("unknown contempt: {other}")))
        }
    };
    Ok((solver_loss_threshold, contempt_factor))
}

/// Reads the `q_init` categorical off the axis config -- required for
/// `algorithm == mcts` (every MCTS configuration needs a Q-value
/// initialization rule), read by `search.rs` when it assembles
/// `SearchSettings`.
pub(crate) fn read_q_init(cfg: &Value) -> Result<QInit, HostError> {
    let raw = req_str(cfg, "q_init")?;
    QInit::from_str(raw).map_err(|_| HostError::bad_request(format!("invalid q_init: {raw}")))
}

/// Resolves the top-level `algorithm` categorical.
pub(crate) fn to_algorithm_spec(cfg: &Value) -> Result<AlgorithmSpec, HostError> {
    match req_str(cfg, "algorithm")? {
        "mcts" => Ok(AlgorithmSpec::Mcts(to_search_spec(cfg)?)),
        "random" => Ok(AlgorithmSpec::Random),
        "bandit" => Ok(AlgorithmSpec::Bandit {
            budget: req_u32(cfg, "budget")?,
            max_rollout_depth: req_u32(cfg, "max_rollout_depth")?,
            policy: match req_str(cfg, "bandit_policy")? {
                "random" => BanditPolicySpec::Random,
                "epsilon_greedy" => BanditPolicySpec::EpsilonGreedy {
                    epsilon: req_f64(cfg, "epsilon")?,
                },
                "ucb1" => BanditPolicySpec::Ucb1 { c: req_f64(cfg, "c")? },
                "thompson" => BanditPolicySpec::Thompson,
                other => {
                    return Err(HostError::bad_request(format!(
                        "unknown bandit_policy: {other}"
                    )))
                }
            },
        }),
        "negamax" => Ok(AlgorithmSpec::Negamax {
            max_depth: req_u32(cfg, "max_depth")?,
            table_bits: req_u32(cfg, "table_bits")?,
            replacement: match req_str(cfg, "negamax_replacement")? {
                "always" => negamax::Replacement::Always,
                "depth_preferred" => negamax::Replacement::DepthPreferred,
                "two_tier" => negamax::Replacement::TwoTier,
                other => {
                    return Err(HostError::bad_request(format!(
                        "unknown negamax_replacement: {other}"
                    )))
                }
            },
            aspiration_window: match cfg.get("negamax_aspiration").and_then(Value::as_str) {
                None | Some("off") => None,
                Some("on") => Some(req_u32(cfg, "aspiration_window")? as Score),
                Some(other) => {
                    return Err(HostError::bad_request(format!(
                        "unknown negamax_aspiration: {other}"
                    )))
                }
            },
            principal_variation_search: field(cfg, "principal_variation_search")
                .map_err(HostError::bad_request)?,
            history_heuristic: field(cfg, "history_heuristic").map_err(HostError::bad_request)?,
            singular_extension: field(cfg, "singular_extension").map_err(HostError::bad_request)?,
            countermove_heuristic: field(cfg, "countermove_heuristic")
                .map_err(HostError::bad_request)?,
        }),
        other => Err(HostError::bad_request(format!("unknown algorithm: {other}"))),
    }
}

