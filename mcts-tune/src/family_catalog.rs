//! A `register_field!` table generating `TrialParams`'s per-family-tunable
//! fields and `strategy_tuner_info`'s matching `TunerParameter` entries from
//! one source, instead of the same field list being hand-declared twice.
//!
//! Three fields stay hand-declared on `TrialParams` and hand-reported by
//! `strategy_tuner_info_with_mcgs` instead of living in this table:
//! `family` and `q_init` are unconditionally active regardless of which
//! family a trial names (nothing in `conditions` gates them), and `mcgs`
//! is reported only when a game's own `supports_mcgs` flag is set -- none
//! of the three is "a field some family's `conditions` entry activates",
//! which is what this table exists to cover.

use game_host::TunerParameter;
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
        /// active-parameter set a SMAC3 harness builds from its search-space
        /// YAML. `family` selects which of the fields below are actually
        /// required; everything except `family`/`q_init` is `Option`
        /// because it's only meaningful for a subset of families (validated
        /// per-family in `to_search_spec`, the same way missing required
        /// fields were already rejected before `family` existed).
        #[derive(Debug, serde::Deserialize)]
        pub struct TrialParams {
            pub(crate) family: String,
            pub(crate) q_init: String,
            $(
                $(#[$doc])*
                pub(crate) $field: Option<$ty>,
            )+
            pub(crate) mcgs: Option<bool>,
        }

        /// `strategy_tuner_info`'s `TunerParameter` entries for every row in
        /// this table, in declaration order. `family`/`q_init`/`mcgs` are
        /// appended by hand in `strategy_tuner_info_with_mcgs` -- see this
        /// module's doc comment for why those three don't belong here.
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
}
