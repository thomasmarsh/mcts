//! A `register_field!` table generating `TrialParams`'s per-family-tunable
//! fields and `strategy_tuner_info`'s matching `TunerParameter` entries from
//! one source, instead of the same field list being hand-declared twice.
//!
//! Four fields stay hand-declared on `TrialParams` and hand-reported by
//! `strategy_tuner_info_with_mcgs` instead of living in this table: `family`
//! is unconditionally active regardless of which family a trial names
//! (nothing in `conditions` gates it); `q_init` is meaningless to a `Direct`
//! family (no `select`/backprop Q-values for it to initialize -- see
//! `DirectFamily`'s doc comment), so it's gated on `family` naming a
//! `Compose` row via a `conditions` entry built from `direct_family_names()`,
//! the same way `mcgs`/`state_only_keying` are reported only when a game's
//! own `supports_mcgs` flag is set (`state_only_keying` additionally gated
//! on `mcgs`'s own sampled value, since it's meaningless without graph
//! search on) -- none of the four is "a field some family's `conditions`
//! entry activates", which is what this table exists to cover.

use crate::config_ir::codec::{field, field_opt};
use crate::config_ir::{BackpropSpec, BaseSimulateSpec, FinalActionSpec, SelectSpec, SimulateSpec};
use game_host::{HostError, TunerCondition, TunerParameter};
use mcts::evaluator::Score;
use mcts::select::{RaveSchedule, RaveUcb};
use mcts::simulate::DecisiveMoveMode;
use mcts::strategies::negamax;
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
        /// required; everything except `family` is `Option` because it's
        /// only meaningful for a subset of families (validated per-family in
        /// `compose_settings`, the same way missing required fields were
        /// already rejected before `family` existed) -- `q_init` included:
        /// every `Compose` family requires it, but a `Direct` family
        /// (`direct_search::build_direct`) never reads it at all.
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
            pub(crate) q_init: Option<String>,
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
                    q_init: field_opt(&v, "q_init").map_err(D::Error::custom)?,
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
    // `flat_mc::FlatMonteCarloStrategy`'s per-move rollout count and
    // per-rollout depth cap.
    samples_per_move: u32 => json!({"type": "int", "bounds": [1, 10000], "default": 100}),
    max_rollout_depth: u32 => json!({"type": "int", "bounds": [1, 1000], "default": 100}),
    // Chooses between `flat_mc`'s two move-selection rules: plain win-rate
    // comparison, or a UCB1 bandit over the same per-move samples (reusing
    // the `c` field every other UCB1-flavored family already has, rather
    // than adding a near-duplicate exploration-constant field).
    flat_mc_selection: String => json!({"type": "categorical", "choices": ["win_rate", "ucb1"], "default": "win_rate"}),
    // `negamax::NegamaxOptions`'s iterative-deepening ceiling and
    // transposition table size (`1 << table_bits` slots, `0` disables it).
    max_depth: u32 => json!({"type": "int", "bounds": [1, 64], "default": 8}),
    table_bits: u32 => json!({"type": "int", "bounds": [0, 24], "default": 20}),
    // `negamax::Replacement`'s three policies, named rather than passed as
    // the real enum -- see this module's doc comment on why `TrialParams`
    // never depends on the concrete type.
    negamax_replacement: String => json!({"type": "categorical", "choices": ["always", "depth_preferred", "two_tier"], "default": "depth_preferred"}),
    principal_variation_search: bool => json!({"type": "bool", "default": true}),
    history_heuristic: bool => json!({"type": "bool", "default": true}),
    singular_extension: bool => json!({"type": "bool", "default": true}),
    countermove_heuristic: bool => json!({"type": "bool", "default": true}),
    // Gates `aspiration_window` on/off, the same on/off-plus-conditioned-
    // value shape `contempt`/`contempt_factor` already uses -- Optuna has
    // no native "sometimes absent" numeric type for
    // `NegamaxOptions::aspiration_window: Option<Score>` to map onto
    // directly.
    negamax_aspiration: String => json!({"type": "categorical", "choices": ["off", "on"], "default": "off"}),
    aspiration_window: u32 => json!({"type": "int", "bounds": [1, mcts::evaluator::EVAL_MAGNITUDE_LIMIT as u32], "default": 50}),
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

/// `compose_settings`'s per-family output: the `SelectSpec`/`SimulateSpec`/
/// `FinalActionSpec` triple every MCTS family builds, plus the two extra
/// `SearchSettings` fields only the PN families (Kowalski et al. 2023)
/// populate -- `None` for every other family. Bundled into one struct rather
/// than a growing tuple so a row that doesn't need the PN-only fields can
/// just leave them `None` instead of every row threading extra positional
/// values through.
pub(crate) struct ComposeSpec {
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

/// `dispatch_family`'s result: either a pre-composed MCTS family (`Compose`,
/// resolved through `config_ir::build_search`, same as every axis-composed
/// `TreeSearch`) or a standalone `Search` impl with no `config_ir::SearchSpec`
/// representation at all (`Direct`, resolved through
/// `direct_search::build_direct`). A family can't mix the two: it either
/// names a point in the four-axis `select`/`simulate`/`backprop`/
/// `final_action` space, or it's a different algorithm entirely.
pub(crate) enum FamilySpec {
    Compose(ComposeSpec),
    Direct(DirectFamily),
}

/// One `FamilySpec::Direct` payload per standalone `Search` impl in the
/// catalog -- the parameters `direct_search::build_direct` needs to
/// construct it, already resolved out of `TrialParams`'s `Option` fields.
pub(crate) enum DirectFamily {
    Random,
    FlatMc {
        samples_per_move: u32,
        max_rollout_depth: u32,
        /// `Some(exploration_constant)` selects `flat_mc`'s UCB1 move rule;
        /// `None` is plain win-rate comparison.
        ucb1: Option<f64>,
    },
    Negamax {
        max_depth: u32,
        table_bits: u32,
        replacement: negamax::Replacement,
        /// `Some(window)` primes the transposition table with a narrow
        /// aspiration pass before each depth's definitive search; `None`
        /// disables it. See `NegamaxOptions::aspiration_window`.
        aspiration_window: Option<Score>,
        principal_variation_search: bool,
        history_heuristic: bool,
        singular_extension: bool,
        countermove_heuristic: bool,
    },
}

/// Resolves `flat_mc_selection` (plus `c` when it names the UCB1 branch)
/// into `flat_mc::FlatMonteCarloStrategy`'s own `ucb1: Option<f64>` field.
fn flat_mc_ucb1(p: &TrialParams) -> Result<Option<f64>, HostError> {
    match p.flat_mc_selection.as_deref() {
        Some("win_rate") => Ok(None),
        Some("ucb1") => Ok(Some(c(p)?)),
        Some(other) => Err(HostError::bad_request(format!(
            "unknown flat_mc_selection: {other}"
        ))),
        None => Err(missing("flat_mc_selection")),
    }
}

/// Matches `negamax_replacement`'s wire name to `negamax::Replacement`'s
/// variants by hand, the same way `to_final_action_spec` matches
/// `final_action` -- `TrialParams` carries the name rather than the real
/// enum so it never needs `negamax::Replacement: Deserialize`.
fn negamax_replacement(p: &TrialParams) -> Result<negamax::Replacement, HostError> {
    match p.negamax_replacement.as_deref() {
        Some("always") => Ok(negamax::Replacement::Always),
        Some("depth_preferred") => Ok(negamax::Replacement::DepthPreferred),
        Some("two_tier") => Ok(negamax::Replacement::TwoTier),
        Some(other) => Err(HostError::bad_request(format!(
            "unknown negamax_replacement: {other}"
        ))),
        None => Err(missing("negamax_replacement")),
    }
}

/// Resolves `negamax_aspiration` (plus `aspiration_window` when it's `"on"`)
/// into `NegamaxOptions::aspiration_window`'s own `Option<Score>` field --
/// the same on/off-plus-conditioned-value shape `contempt_factor` above
/// uses for the same reason (no native "sometimes absent" numeric type on
/// the tuner side).
fn negamax_aspiration_window(p: &TrialParams) -> Result<Option<Score>, HostError> {
    match p.negamax_aspiration.as_deref() {
        Some("off") => Ok(None),
        Some("on") => Ok(Some(
            p.aspiration_window
                .ok_or_else(|| missing("aspiration_window"))? as Score,
        )),
        Some(other) => Err(HostError::bad_request(format!(
            "unknown negamax_aspiration: {other}"
        ))),
        None => Err(missing("negamax_aspiration")),
    }
}

/// Generates `dispatch_family`, the single source of per-family
/// construction `compose_settings`/`make_candidate` both dispatch through --
/// one row per family, each a closure over `p: &TrialParams`. A closure
/// (rather than a bare `expr`
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
/// A row's `$ctor` returns `FamilySpec`, not `ComposeSpec` directly -- most
/// rows wrap their result in `FamilySpec::Compose(...)`, but a family with
/// no `config_ir::SearchSpec` representation (`"random"`) instead returns
/// `FamilySpec::Direct(...)`, built by `direct_search::build_direct` rather
/// than `config_ir::build_search`.
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
        /// declaration order.
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
    "ucb1" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_dm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::Win,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    // `mode` is fixed per family (like every other `*_dm*` row below) rather
    // than exposed as its own tunable field, so a tuner search that wants to
    // compare Teytaud & Teytaud 2010's decisive-move-only check against the
    // pricier anti-decisive one (see `simulate::DecisiveMoveMode::AntiDecisive`'s
    // doc comment) needs both named explicitly -- this is `ucb1_dm`'s
    // anti-decisive counterpart.
    "ucb1_adm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::AntiDecisive,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_lgr" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_lgr2" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr2 {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_lgr2_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::EpsilonGreedy {
            epsilon: epsilon(p)?,
            inner: BaseSimulateSpec::Lgr2Mast {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    // Druid's "strong"/"master" presets (`games/druid/src/main.rs`'s
    // `build_ai`) -- plain Ucb1 select with a decisive-move-checking,
    // NST-guided playout policy.
    "ucb1_dm_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    // `ucb1_dm_nst`'s anti-decisive counterpart -- same NST-guided playout,
    // Druid's actual "strong"/"master" shape, but the pricier two-ply block
    // check instead of a same-ply win check.
    "ucb1_adm_nst" => [c, epsilon, nst_backoff_threshold, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    "ucb1_progressive_history" => [c, ph_weight, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::ProgressiveHistory {
            c: c(p)?,
            ph_weight: p.ph_weight.ok_or_else(|| missing("ph_weight"))?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "amaf" => [amaf_alpha, c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Amaf {
            alpha: p.amaf_alpha.ok_or_else(|| missing("amaf_alpha"))?,
            c: c(p)?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "amaf_mast" => [amaf_alpha, c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    "ucb1_tuned" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_tuned_mast" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::Mast {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_tuned_dm" => [c, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::DecisiveMove {
            mode: DecisiveMoveMode::Win,
            inner: BaseSimulateSpec::Uniform {},
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "ucb1_tuned_dm_mast" => [c, epsilon, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1Tuned { c: c(p)? },
        simulate: SimulateSpec::DecisiveMoveMast {
            mode: DecisiveMoveMode::Win,
            epsilon: epsilon(p)?,
        },
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
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
        Ok(FamilySpec::Compose(ComposeSpec {
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
        }))
    },
    "ucb1_pn" => [c, c_pn, solver_loss_threshold, contempt, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::UctPn {
            c: c(p)?,
            c_pn: c_pn(p)?,
        },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: Some(solver_loss_threshold(p)?),
        contempt_factor: contempt_factor(p)?,
    })),
    "ucb1_pn_mast" => [c, c_pn, epsilon, solver_loss_threshold, contempt, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    "ucb1_max_robust" => [c] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: FinalActionSpec::MaxRobustChild {},
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "meta_mcts" => [c] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::Ucb1 { c: c(p)? },
        simulate: SimulateSpec::MetaMcts {
            iterations: crate::META_MCTS_INNER_ITERATIONS,
        },
        final_action: FinalActionSpec::MaxAvg {},
        backprop: BackpropSpec::Classic {},
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    // Tesauro/Rajan/Segal 2010's Bayesian MCTS: `select`/`backprop` have to
    // travel together (`config_ir.rs`'s `needs_posterior`), so these two
    // families each pin one concrete pairing for tuner to tune rather than
    // leaving the select<->backprop choice free (only `build_custom`'s
    // Custom-UI path composes those two axes independently).
    "bayes_uct1_gaussian" => [c, prior_variance, obs_variance, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
        select: SelectSpec::BayesUct1 { c: c(p)? },
        simulate: SimulateSpec::Uniform {},
        final_action: to_final_action_spec(p)?,
        backprop: BackpropSpec::BayesGaussian {
            prior_variance: prior_variance(p)?,
            obs_variance: obs_variance(p)?,
        },
        solver_loss_threshold: None,
        contempt_factor: None,
    })),
    "bayes_uct2_numeric" => [c, prior_variance, obs_variance, value_lo, value_hi, final_action] => |p: &TrialParams| Ok(FamilySpec::Compose(ComposeSpec {
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
    })),
    // Uniform-random move choice -- no hyperparameters of its own, so its
    // field list is empty and its `TrialParams` reads nothing beyond
    // `family` itself.
    "random" => [] => |_p: &TrialParams| Ok(FamilySpec::Direct(DirectFamily::Random)),
    // Flat Monte Carlo: one independent batch of uniform rollouts per legal
    // move, no tree at all -- `flat_mc_selection` picks how those rollouts'
    // outcomes turn into a move (plain win-rate vs. a UCB1 bandit over the
    // same samples).
    "flat_mc" => [samples_per_move, max_rollout_depth, flat_mc_selection] => |p: &TrialParams| Ok(FamilySpec::Direct(DirectFamily::FlatMc {
        samples_per_move: p.samples_per_move.ok_or_else(|| missing("samples_per_move"))?,
        max_rollout_depth: p.max_rollout_depth.ok_or_else(|| missing("max_rollout_depth"))?,
        ucb1: flat_mc_ucb1(p)?,
    })),
    // Iterative-deepening alpha-beta search -- only sound for a
    // deterministic, perfect-information, two-player alternating-move game
    // (see `negamax::supports`); a game outside that shape still compiles
    // against this row but silently searches only the state in front of it.
    "negamax" => [
        max_depth, table_bits, negamax_replacement, principal_variation_search,
        history_heuristic, singular_extension, countermove_heuristic, negamax_aspiration,
    ] => |p: &TrialParams| Ok(FamilySpec::Direct(DirectFamily::Negamax {
        max_depth: p.max_depth.ok_or_else(|| missing("max_depth"))?,
        table_bits: p.table_bits.ok_or_else(|| missing("table_bits"))?,
        replacement: negamax_replacement(p)?,
        aspiration_window: negamax_aspiration_window(p)?,
        principal_variation_search: p
            .principal_variation_search
            .ok_or_else(|| missing("principal_variation_search"))?,
        history_heuristic: p.history_heuristic.ok_or_else(|| missing("history_heuristic"))?,
        singular_extension: p.singular_extension.ok_or_else(|| missing("singular_extension"))?,
        countermove_heuristic: p
            .countermove_heuristic
            .ok_or_else(|| missing("countermove_heuristic"))?,
    })),
}

/// Family names whose `register_family!` row resolves to `FamilySpec::Direct`
/// -- one entry per `DirectFamily` variant. Hand-maintained rather than
/// generated: unlike a row's field list (read directly off the macro table
/// by `family_conditions()`), whether a row's `$ctor` returns `Compose` or
/// `Direct` is a runtime fact about its closure body, not something the
/// macro can inspect to generate this list itself. Used by
/// `strategy_tuner_info_with_mcgs` to gate `q_init`'s activation on `family`
/// naming a `Compose` row -- see this module's doc comment.
pub(crate) fn direct_family_names() -> &'static [&'static str] {
    &["random", "flat_mc", "negamax"]
}
