//! The `register_field!` table: the single source for every tunable
//! variant parameter's JSON-schema bounds and default. `tuner_info.rs`
//! reuses `tunable_field_parameters()` for the scalar/sub-categorical knobs
//! while declaring the schema's own shape -- the `algorithm` categorical
//! and the four policy axes -- directly.
//!
//! There is no per-composition catalog here anymore: `dispatch.rs` builds a
//! `config_ir::SearchSpec` straight from the axis categoricals, so the only
//! thing this table still owns is field metadata, not construction.
//! `algorithm`/`select`/`simulate`/`backprop`/`final_action`/`q_init`/`mcgs`/
//! `state_only_keying` are declared by hand in `tuner_info.rs`.

use game_host::{TunerCondition, TunerParameter};
use serde_json::{json, Value};

/// Builds one `TunerParameter` from a name and its JSON-schema spec --
/// shared by this table's `tunable_field_parameters()` and the
/// `algorithm`/axis rows `tuner_info.rs` declares by hand.
pub(crate) fn param(name: &str, spec: Value) -> TunerParameter {
    TunerParameter {
        name: name.into(),
        spec,
    }
}

/// Builds one `TunerCondition` from an `if_` predicate (a single-entry
/// object `{parent: value | [values]}`) and the parameter names it
/// activates -- used throughout `tuner_info.rs` to gate each axis variant's
/// parameters on the axis categorical's sampled value.
pub(crate) fn condition(if_: Value, then: &[&str]) -> TunerCondition {
    TunerCondition {
        if_,
        then: then.iter().map(|s| s.to_string()).collect(),
    }
}

/// Generates `tunable_field_parameters()` -- every row's `TunerParameter`,
/// in declaration order -- from one table. `$spec` is evaluated once per
/// row to build that field's `TunerParameter::spec` JSON. The `$ty` in each
/// row is retained only as documentation of the value's runtime type; it is
/// not referenced by the expansion (nothing deserializes a struct out of
/// the params object anymore -- `dispatch.rs` reads fields off the raw
/// `serde_json::Value`).
macro_rules! register_field {
    (
        $(
            $(#[$doc:meta])*
            $field:ident : $ty:ty => $spec:expr
        ),+ $(,)?
    ) => {
        /// `strategy_tuner_info`'s `TunerParameter` entries for every row in
        /// this table, in declaration order. `algorithm`/the axis
        /// categoricals/`q_init`/`mcgs`/`state_only_keying` are declared by
        /// hand in `tuner_info.rs` -- see this module's doc comment.
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
    // `backprop::PowerMeanBackprop`'s power-mean exponent (Power-UCT, Dam et
    // al. IJCAI 2020). `1.0` = plain UCT arithmetic mean; larger biases the
    // backup toward the max over children. Guidance from Stochastic-Power-UCT
    // (arXiv 2406.02235): the useful range is small single digits, but sweep
    // wider per game.
    p: f64 => json!({"type": "float", "bounds": [1.0, 50.0], "default": 1.0}),
    // `backprop::PowerMeanBackprop`'s mean<->max blend weight (Full-Bellman
    // backup, Asai & Wissow AAAI 2025). `0.0` = pure power-mean; `1.0` = pure
    // max over children. EVT paper's caveat: pure max helps a weak baseline and
    // hurts a strong one, so sweep the interior, not just the endpoints.
    alpha: f64 => json!({"type": "float", "bounds": [0.0, 1.0], "default": 0.0}),
    // `backprop::TdBackprop`'s λ-return decay (Sarsa-UCT(λ), Vodopivec et al.
    // JAIR 2017). `1.0` = plain Monte-Carlo mean backup (== `Classic`); lower
    // bootstraps each node from its children's current estimates. Adversarial-
    // game guidance: the useful band is [0.8, 1.0], so sweep the top densely.
    lambda: f64 => json!({"type": "float", "bounds": [0.0, 1.0], "default": 1.0}),
    // MENTS/E2W softmax temperature (Xiao et al., NeurIPS 2019) -- shared by
    // `select::Ments` (E2W policy) and `backprop::SoftmaxBackprop` (soft
    // backup). `-> 0` is a max backup, `-> inf` the arithmetic mean. Sweep
    // log-scale in the search-space YAML (distribution: loguniform); the
    // schema itself has no log flag.
    tau: f64 => json!({"type": "float", "bounds": [0.05, 5.0], "default": 1.0}),
    // `backprop::TdBackprop`'s MaxMCTS(λ) toggle (Khandelwal et al. ICML 2016):
    // 1 bootstraps from max over children instead of the on-path child. Named
    // `td_max_child` (not `max_child`) to match how this table disambiguates.
    td_max_child: u32 => json!({"type": "int", "bounds": [0, 1], "default": 0}),
    // `select::ScoreBoundedUct`'s §3.4 bound-induced value-bias weights
    // (Cazenave & Saffidine, CG 2010): `gamma` on the pessimistic bound,
    // `delta` on the optimistic one. Cazenave found the useful values
    // game-dependent; `0.1` is a starting guess, sweep per game.
    gamma: f64 => json!({"type": "float", "bounds": [0.0, 1.0], "default": 0.1}),
    delta: f64 => json!({"type": "float", "bounds": [0.0, 1.0], "default": 0.1}),
    // `select::GpnUct`'s proof-number bias formula (Kowalski et al.,
    // arXiv:2506.13249): `max`/`sum` are Eq. 4/5, `rank` the 2023 rank bonus.
    // `max` is strongest at two players; `sum` is the safer choice in wider
    // fields, where it damps the per-player AND-branch blow-up.
    gpn_bias: String => json!({"type": "categorical", "choices": ["max", "sum", "rank"], "default": "max"}),
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
    // the real enum -- `dispatch.rs` matches the wire name so it never needs
    // `negamax::Replacement: Deserialize`.
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
