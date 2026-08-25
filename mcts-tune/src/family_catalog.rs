//! A `register_field!` table generating `TrialParams`'s per-family-tunable
//! fields and `strategy_tuner_info`'s matching `TunerParameter` entries from
//! one source, instead of the same field list being hand-declared twice.
//!
//! Four fields stay hand-declared on `TrialParams` and hand-reported by
//! `strategy_tuner_info_with_mcgs` instead of living in this table:
//! `family` and `q_init` are unconditionally active regardless of which
//! family a trial names (nothing in `conditions` gates them), and `mcgs`/
//! `state_only_keying` are reported only when a game's own `supports_mcgs`
//! flag is set (`state_only_keying` additionally gated on `mcgs`'s own
//! sampled value via a `conditions` entry, since it's meaningless without
//! graph search on) -- none of the four is "a field some family's
//! `conditions` entry activates", which is what this table exists to cover.

use crate::config_ir::codec::{field, field_opt};
use crate::config_ir::{BackpropSpec, BaseSimulateSpec, FinalActionSpec, SelectSpec, SimulateSpec};
use game_host::{HostError, TunerCondition, TunerParameter};
use mcts::select::{RaveSchedule, RaveUcb};
use mcts::simulate::DecisiveMoveMode;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};

/// Builds one `TunerParameter` from a name and its JSON-schema spec --
/// shared by both this table's generated entries and the hand-declared
/// `family`/`q_init`/`mcgs` ones in `lib.rs`.
pub(crate) fn param(name: &str, spec: Value) -> TunerParameter {
    TunerParameter {
        name: name.into(),
        spec,
    }
}

/// Builds one `TunerCondition` from an `if_` predicate and the field names
/// it activates -- shared by `family_conditions()` below and the
/// hand-written conditions `strategy_tuner_info_with_mcgs` appends for
/// child-value-gated fields (`final_action`'s own `a`, RAVE's
/// schedule/`rave_ucb`-gated fields) that don't fit `register_family!`'s
/// per-family shape.
pub(crate) fn condition(if_: Value, then: &[&str]) -> TunerCondition {
    TunerCondition {
        if_,
        then: then.iter().map(|s| s.to_string()).collect(),
    }
}

/// Generates `TrialParams` (every row becomes one `Option<$ty>` field,
/// plus the hand-declared `family`/`q_init`/`mcgs` fields) and
/// `tunable_field_parameters()` (every row's `TunerParameter`, in
/// declaration order) from one table. `$spec` is evaluated once per row to
/// build that field's `TunerParameter::spec` JSON.
macro_rules! register_field {
    (
        $(
            $(#[$doc:meta])*
            $field:ident : $ty:ty => $spec:expr
        ),+ $(,)?
    ) => {
        /// One trial's candidate parameters, deserialized from the `params`
        /// JSON object `strategy_tune_eval` receives -- the merged
        /// active-parameter set a tuner harness builds from its search-space
        /// YAML. `family` selects which of the fields below are actually
        /// required; everything except `family`/`q_init` is `Option`
        /// because it's only meaningful for a subset of families (validated
        /// per-family in `to_search_spec`, the same way missing required
        /// fields were already rejected before `family` existed).
        ///
        /// `Deserialize` is hand-implemented below (routed through
        /// `serde_json::Value`, via the same `config_ir::codec` helpers
        /// `register_backprop!`/`register_select!`/`register_simulate!` use)
        /// rather than `#[derive]`d, for the same compile-cost reason --
        /// see `config_ir/backprop.rs`'s `BackpropSpec` doc comment. Never
        /// `Serialize` -- nothing in this crate ever serializes a
        /// `TrialParams` back to JSON, only parses one out of a tuner
        /// trial's `params`.
        #[derive(Debug)]
        pub struct TrialParams {
            pub(crate) family: String,
            pub(crate) q_init: String,
            $(
                $(#[$doc])*
                pub(crate) $field: Option<$ty>,
            )+
            pub(crate) mcgs: Option<bool>,
            /// Selects `TranspositionKeying::StateOnly` over the default
            /// `PerPly` when `mcgs` is also `true`; meaningless (and
            /// rejected by `resolve_graph_search`) otherwise. See
            /// `mcts::TranspositionKeying`'s doc comment for the GHI
            /// precondition this asserts about the game's zobrist hash.
            pub(crate) state_only_keying: Option<bool>,
        }

        impl<'de> Deserialize<'de> for TrialParams {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let v = Value::deserialize(deserializer)?;
                Ok(TrialParams {
                    family: field(&v, "family").map_err(D::Error::custom)?,
                    q_init: field(&v, "q_init").map_err(D::Error::custom)?,
                    $(
                        $field: field_opt(&v, stringify!($field)).map_err(D::Error::custom)?,
                    )+
                    mcgs: field_opt(&v, "mcgs").map_err(D::Error::custom)?,
                    state_only_keying: field_opt(&v, "state_only_keying").map_err(D::Error::custom)?,
                })
            }
        }

        /// `strategy_tuner_info`'s `TunerParameter` entries for every row in
        /// this table, in declaration order. `family`/`q_init`/`mcgs`/
        /// `state_only_keying` are appended by hand in
        /// `strategy_tuner_info_with_mcgs` -- see this module's doc comment
        /// for why those four don't belong here.
        pub(crate) fn tunable_field_parameters() -> Vec<TunerParameter> {
            vec![
                $( param(stringify!($field), $spec) ),+
            ]
        }
    };
}

register_field! {
    final_action: String => json!({"type": "categorical", "choices": ["max_avg", "secure_child", "robust_child"], "default": "robust_child"}),
    a: f64 => json!({"type": "float", "bounds": [0, 10], "default": 4.0}),
    c: f64 => json!({"type": "float", "bounds": [0, 3], "default": std::f64::consts::SQRT_2}),
    epsilon: f64 => json!({"type": "float", "bounds": [0, 1], "default": 0.1}),
    amaf_alpha: f64 => json!({"type": "float", "bounds": [0, 1], "default": 1.0}),
    ph_weight: f64 => json!({"type": "float", "bounds": [0, 5], "default": 1.0}),
    nst_backoff_threshold: u32 => json!({"type": "int", "bounds": [0, 100], "default": 5}),
    bias: f64 => json!({"type": "float", "bounds": [0, 10], "default": 0.00001}),
    k: u32 => json!({"type": "int", "bounds": [0, 2000], "default": 1000}),
    rave: u32 => json!({"type": "int", "bounds": [0, 2000], "default": 700}),
    schedule: String => json!({"type": "categorical", "choices": ["hand_selected", "min_mse", "threshold"], "default": "threshold"}),
    threshold: u32 => json!({"type": "int", "bounds": [0, 2000], "default": 700}),
    rave_ucb: String => json!({"type": "categorical", "choices": ["none", "ucb1", "tuned"], "default": "tuned"}),
    // Kowalski et al. 2023 Eq. 4: clustered 1.0-2.0 in the paper's own
    // experiments, domain-dependent.
    c_pn: f64 => json!({"type": "float", "bounds": [0, 3], "default": 1.0}),
    // MCTS-Solver's proven-loss selection threshold `T` (Kowalski et al.
    // 2023 Section III.B); the paper uses T=5 throughout.
    solver_loss_threshold: u32 => json!({"type": "int", "bounds": [0, 50], "default": 5}),
    contempt: String => json!({"type": "categorical", "choices": ["off", "on"], "default": "off"}),
    // Compared against `Node::expected_score`, whose default range
    // (`Game::compute_utilities`'s default) is [-1, 1].
    contempt_factor: f64 => json!({"type": "float", "bounds": [-1, 1], "default": 0.0}),
    // `backprop::BayesGaussian`/`BayesNumeric`'s conjugate-update
    // hyperparameters -- see `backprop.rs`'s `conjugate_leaf_posterior` doc
    // comment.
    prior_variance: f64 => json!({"type": "float", "bounds": [0.0, 10.0], "default": 1.0}),
    obs_variance: f64 => json!({"type": "float", "bounds": [1e-6, 10.0], "default": 1.0}),
    // `backprop::BayesNumeric`'s grid bounds -- must cover the game's real
    // utility range, defaulting to this codebase's symmetric [-1, 1]
    // convention.
    value_lo: f64 => json!({"type": "float", "bounds": [-10.0, 10.0], "default": -1.0}),
    value_hi: f64 => json!({"type": "float", "bounds": [-10.0, 10.0], "default": 1.0}),
}

fn missing(field: &str) -> HostError {
    HostError::bad_request(format!("missing param: {field}"))
}

fn c(p: &TrialParams) -> Result<f64, HostError> {
    p.c.ok_or_else(|| missing("c"))
}

fn epsilon(p: &TrialParams) -> Result<f64, HostError> {
    p.epsilon.ok_or_else(|| missing("epsilon"))
}

fn prior_variance(p: &TrialParams) -> Result<f64, HostError> {
    p.prior_variance.ok_or_else(|| missing("prior_variance"))
}

fn obs_variance(p: &TrialParams) -> Result<f64, HostError> {
    p.obs_variance.ok_or_else(|| missing("obs_variance"))
}

fn value_lo(p: &TrialParams) -> Result<f64, HostError> {
    p.value_lo.ok_or_else(|| missing("value_lo"))
}

fn value_hi(p: &TrialParams) -> Result<f64, HostError> {
    p.value_hi.ok_or_else(|| missing("value_hi"))
}

fn c_pn(p: &TrialParams) -> Result<f64, HostError> {
    p.c_pn.ok_or_else(|| missing("c_pn"))
}

fn solver_loss_threshold(p: &TrialParams) -> Result<u32, HostError> {
    p.solver_loss_threshold
        .ok_or_else(|| missing("solver_loss_threshold"))
}

fn contempt_factor(p: &TrialParams) -> Result<Option<f64>, HostError> {
    match p.contempt.as_deref() {
        Some("off") => Ok(None),
        Some("on") => Ok(Some(
            p.contempt_factor
                .ok_or_else(|| missing("contempt_factor"))?,
        )),
        Some(other) => Err(HostError::bad_request(format!("unknown contempt: {other}"))),
        None => Err(missing("contempt")),
    }
}

/// `final_action` resolution shared by every family whose own named type
/// leaves it configurable -- the `FinalActionSpec` counterpart of the
/// three-way match on `p.final_action` every such family needs.
pub(crate) fn to_final_action_spec(p: &TrialParams) -> Result<FinalActionSpec, HostError> {
    let fa = p
        .final_action
        .as_deref()
        .ok_or_else(|| missing("final_action"))?;
    match fa {
        "max_avg" => Ok(FinalActionSpec::MaxAvg {}),
        "secure_child" => Ok(FinalActionSpec::SecureChild {
            a: p.a.ok_or_else(|| missing("a"))?,
        }),
        "robust_child" => Ok(FinalActionSpec::RobustChild {}),
        other => Err(HostError::bad_request(format!(
            "unknown final_action: {other}"
        ))),
    }
}

/// `to_search_spec`'s per-family output: the `SelectSpec`/`SimulateSpec`/
/// `FinalActionSpec` triple every family builds, plus the two extra
/// `SearchSettings` fields only the PN families (Kowalski et al. 2023)
/// populate -- `None` for every other family. Bundled into one struct rather
/// than a growing tuple so a row that doesn't need the PN-only fields can
/// just leave them `None` instead of every row threading extra positional
/// values through.
pub(crate) struct FamilySpec {
    pub select: SelectSpec,
    pub simulate: SimulateSpec,
    pub final_action: FinalActionSpec,
    /// Every pre-Bayes family sets this to `BackpropSpec::Classic {}` --
    /// `backprop` had no per-family axis at all until `BayesUct1`/
    /// `BayesUct2`'s `bayes_uct1_gaussian`/`bayes_uct2_numeric` rows needed
    /// to name a non-`Classic` backprop (see `config_ir.rs`'s
    /// `needs_posterior` doc comment for why the two have to travel
    /// together).
    pub backprop: BackpropSpec,
    pub solver_loss_threshold: Option<u32>,
    pub contempt_factor: Option<f64>,
}

/// Generates `dispatch_family`, the single source for `to_search_spec`'s
/// per-family construction -- one row per family, each a literal
/// transcription of that family's own `to_search_spec` match arm as a
/// closure over `p: &TrialParams`. A closure (rather than a bare `expr`
/// referencing an outer `p`) keeps every row fully self-contained: macro
/// hygiene gives each row's closure parameter its own binding, so there's no
/// need for a row's `$ctor` to share identifier context with code written in
/// this macro's own definition.
///
/// Each row also names the subset of table 1's fields that family's own
/// `$ctor` actually reads (including `final_action` for every family whose
/// own named type leaves it configurable) -- this is what generates
/// `family_choices()`'s `family` categorical and `family_conditions()`'s
/// per-(family, field) `TunerCondition` rows, replacing the hand-maintained
/// `C_FAMILIES`/`EPSILON_FAMILIES`/`FINAL_ACTION_FAMILIES`/`PN_FAMILIES`
/// grouping constants that used to describe the same thing. Fields gated by
/// another *field's own value* rather than by `family` directly (`a` under
/// `final_action: secure_child`, RAVE's schedule/`rave_ucb`-gated fields,
/// `contempt_factor` under `contempt: on`) are not listed here -- those stay
/// hand-written extra conditions in `strategy_tuner_info_with_mcgs`, since
/// they're a different kind of condition than "this family always needs
/// this field".
///
/// `"random"`/`"flat_mc"` are not rows here -- see `make_candidate`'s own
/// comment on why those two stay permanently outside this table.
macro_rules! register_family {
    (
        $(
            $name:literal => [$($field:ident),* $(,)?] => $ctor:expr
        ),+ $(,)?
    ) => {
        /// Dispatches `family` to the row constructing the matching
        /// `FamilySpec` -- `to_search_spec`'s whole `match p.family.as_str()`
        /// body, generated from this table instead of hand-written.
        pub(crate) fn dispatch_family(
            family: &str,
            p: &TrialParams,
        ) -> Result<FamilySpec, HostError> {
            match family {
                $( $name => ($ctor)(p), )+
                other => Err(HostError::bad_request(format!("unknown family: {other}"))),
            }
        }

        /// The `family` categorical's `choices` list -- every row's name, in
        /// declaration order. Deliberately excludes `"random"`/`"flat_mc"`
        /// (not rows in this table -- see this macro's doc comment).
        pub(crate) fn family_choices() -> Vec<&'static str> {
            vec![$( $name ),+]
        }

        /// One `TunerCondition` per (family, field) pair a row's `$ctor`
        /// reads -- the generated replacement for the hand-written grouped
        /// conditions this macro's doc comment describes.
        pub(crate) fn family_conditions() -> Vec<TunerCondition> {
            let mut conditions = Vec::new();
            $(
                $(
                    conditions.push(condition(json!({"family": $name}), &[stringify!($field)]));
                )*
            )+
            conditions
        }
    };
}

register_family! {
    "ucb1" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_dm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::Win,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    // `mode` is fixed per family (like every other `*_dm*` row below) rather
    // than exposed as its own tunable field, so a tuner search that wants to
    // compare Teytaud & Teytaud 2010's decisive-move-only check against the
    // pricier anti-decisive one (see `simulate::DecisiveMoveMode::AntiDecisive`'s
    // doc comment) needs both named explicitly -- this is `ucb1_dm`'s
    // anti-decisive counterpart.
    "ucb1_adm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::AntiDecisive,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_lgr" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_lgr2" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr2 {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_lgr2_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr2Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Nst {
                backoff_threshold: p
                    .nst_backoff_threshold
                    .ok_or_else(|| missing("nst_backoff_threshold"))?,
            },
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    // Druid's "strong"/"master" presets (`games/druid/src/main.rs`'s
    // `build_ai`) -- plain Ucb1 select with a decisive-move-checking,
    // NST-guided playout policy.
    "ucb1_dm_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMoveNst {
            mode: DecisiveMoveMode::Win,
            epsilon: epsilon(p)?,
            nst_backoff_threshold: p
                .nst_backoff_threshold
                .ok_or_else(|| missing("nst_backoff_threshold"))?,
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    // `ucb1_dm_nst`'s anti-decisive counterpart -- same NST-guided playout,
    // Druid's actual "strong"/"master" shape, but the pricier two-ply block
    // check instead of a same-ply win check.
    "ucb1_adm_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMoveNst {
            mode: DecisiveMoveMode::AntiDecisive,
            epsilon: epsilon(p)?,
            nst_backoff_threshold: p
                .nst_backoff_threshold
                .ok_or_else(|| missing("nst_backoff_threshold"))?,
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_progressive_history" => [c, ph_weight, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::ProgressiveHistory {
            c: c(p)?,
            ph_weight: p.ph_weight.ok_or_else(|| missing("ph_weight"))?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "amaf" => [amaf_alpha, c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Amaf {
            alpha: p.amaf_alpha.ok_or_else(|| missing("amaf_alpha"))?,
            c: c(p)?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "amaf_mast" => [amaf_alpha, c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Amaf {
            alpha: p.amaf_alpha.ok_or_else(|| missing("amaf_alpha"))?,
            c: c(p)?,
        },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_tuned" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_tuned_mast" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::Mast {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_tuned_dm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::Win,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "ucb1_tuned_dm_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::DecisiveMoveMast {
            mode: DecisiveMoveMode::Win,
            epsilon: epsilon(p)?,
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "rave" => [threshold, schedule, rave_ucb, epsilon, final_action] => |p: &TrialParams| {
        let schedule = match p.schedule.as_deref().ok_or_else(|| missing("schedule"))? {
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
        let ucb = match p.rave_ucb.as_deref().ok_or_else(|| missing("rave_ucb"))? {
            "none" => RaveUcb::None,
            "ucb1" => RaveUcb::Ucb1 {
                exploration_constant: c(p)?,
            },
            "tuned" => RaveUcb::Ucb1Tuned {
                exploration_constant: c(p)?,
            },
            other => return Err(HostError::bad_request(format!("unknown rave_ucb: {other}"))),
        };
        Ok(FamilySpec {
            select: SelectSpec::Rave {
                threshold: p.threshold.ok_or_else(|| missing("threshold"))?,
                schedule,
                ucb,
            },
            simulate: SimulateSpec::DecisiveMoveMast {
                mode: DecisiveMoveMode::WinLoss,
                epsilon: epsilon(p)?,
            },
            final_action: to_final_action_spec(p)?,
            backprop: BackpropSpec::Classic {},
            solver_loss_threshold: None,
            contempt_factor: None,
        })
    },
    "ucb1_pn" => [c, c_pn, solver_loss_threshold, contempt, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::UctPn {
            c: c(p)?,
            c_pn: c_pn(p)?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: Some(solver_loss_threshold(p)?),
        contempt_factor: contempt_factor(p)?,
    }),
    "ucb1_pn_mast" => [c, c_pn, epsilon, solver_loss_threshold, contempt, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::UctPn {
            c: c(p)?,
            c_pn: c_pn(p)?,
        },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: Some(solver_loss_threshold(p)?),
        contempt_factor: contempt_factor(p)?,
    }),
    "ucb1_max_robust" => [c] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: FinalActionSpec::MaxRobustChild {},
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "meta_mcts" => [c] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::MetaMcts {
            iterations: crate::META_MCTS_INNER_ITERATIONS,
        },
        final_action: FinalActionSpec::MaxAvg {},
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    // Tesauro/Rajan/Segal 2010's Bayesian MCTS: `select`/`backprop` have to
    // travel together (`config_ir.rs`'s `needs_posterior`), so these two
    // families each pin one concrete pairing for tuner to tune rather than
    // leaving the select<->backprop choice free (only `build_custom`'s
    // Custom-UI path composes those two axes independently).
    "bayes_uct1_gaussian" => [c, prior_variance, obs_variance, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::BayesUct1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::BayesGaussian {
            prior_variance: prior_variance(p)?,
            obs_variance: obs_variance(p)?,
        },
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
    "bayes_uct2_numeric" => [c, prior_variance, obs_variance, value_lo, value_hi, final_action] => |p: &TrialParams| Ok(FamilySpec {
        select: SelectSpec::BayesUct2 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::BayesNumeric {
            prior_variance: prior_variance(p)?,
            obs_variance: obs_variance(p)?,
            value_lo: value_lo(p)?,
            value_hi: value_hi(p)?,
        },
        solver_loss_threshold: None,
        contempt_factor: None,
    }),
}
