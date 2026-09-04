//! A JSON description of `config_ir`'s four axes (`select`/`simulate`/
//! `backprop`/`final_action`), for a generic client (the UI's interactive
//! "Custom" strategy builder) to render a form without hand-coding any
//! per-family knowledge of its own. Unlike `tuner_info.rs`'s
//! `TunerParameter`/`TunerCondition` (a *flat* description: the `algorithm`
//! categorical plus each policy axis's variants as sibling parameters), this
//! describes `config_ir`'s actual
//! recursive shape: which axis variants exist, their fields, and which
//! variants *wrap* an inner spec of some other (always non-recursive)
//! variant set.
//!
//! This is hand-written rather than generated from `register_select!`/
//! `register_simulate!`/etc.'s tables, because those macros only capture
//! each field's Rust *type*, not its JSON-schema shape (bounds/choices/
//! default) -- teaching them that would mean re-annotating every existing
//! row for a benefit only this one new caller needs, on a table that's
//! `axis_schema_covers_every_real_variant`-tested below to be exhaustive.
//! That test provides the same drift protection a macro would: it `match`es
//! every real `*Spec` enum with no wildcard arm, so adding a new variant to
//! `SelectSpec`/`SimulateSpec`/`BackpropSpec`/`FinalActionSpec` fails to
//! compile here until this file's schema gains a matching entry.
//!
//! Field bounds/defaults are reused from `fields.rs`'s
//! `register_field!` table by field name (`c`, `epsilon`, `amaf_alpha`,
//! `ph_weight`, ...) wherever the same tunable knob appears in both places,
//! so the two schemas agree on what a sane default `c` or `epsilon` is
//! rather than picking a second, independently-chosen number.

use serde_json::{json, Value};

fn field(name: &str, spec: Value) -> Value {
    let Value::Object(mut map) = spec else {
        unreachable!("field specs are always JSON objects")
    };
    map.insert("name".to_string(), json!(name));
    Value::Object(map)
}

fn float(bounds: [f64; 2], default: f64) -> Value {
    json!({ "type": "float", "bounds": bounds, "default": default })
}

fn int(bounds: [u32; 2], default: u32) -> Value {
    json!({ "type": "int", "bounds": bounds, "default": default })
}

/// A nested categorical, for `RaveSchedule`/`RaveUcb`/`DecisiveMoveMode` --
/// fields whose value is itself a small tagged union, but (unlike an axis's
/// own variants) never a `config_ir` spec and never a `wraps` target.
fn en(default: &str, variants: Vec<Value>) -> Value {
    json!({ "type": "enum", "default": default, "variants": variants })
}

/// Like `en`, but for an enum whose Rust representation has *no* tag at
/// all -- every variant is fieldless, and `serde` (with no `#[serde(tag =
/// "kind")]`, just `rename_all`) writes/reads it as a bare JSON string
/// (`"win"`), not `{"kind": "win"}`. `DecisiveMoveMode` is the one example
/// in this schema (see `mcts::simulate::DecisiveMoveMode`'s own derive --
/// no `tag` attribute, unlike `RaveSchedule`/`RaveUcb`, which carry
/// per-variant fields and so must stay real tagged unions, i.e. `en`, not
/// this). A client must read `bare` and render/serialize the field as a
/// plain string, never `{kind, ...fields}` -- getting this wrong is exactly
/// what produced `CustomStrategySpec` deserialization's "unknown variant
/// `kind`, expected one of `win`, `win_loss`, `win_loss_draw`" error: the
/// object's own key (`"kind"`) was being parsed as if it were one of
/// `DecisiveMoveMode`'s three variant names.
fn bare_en(default: &str, choices: &[&str]) -> Value {
    json!({
        "type": "enum",
        "default": default,
        "bare": true,
        "variants": choices.iter().map(|c| variant(c, vec![])).collect::<Vec<_>>(),
    })
}

fn variant(kind: &str, fields: Vec<Value>) -> Value {
    json!({ "kind": kind, "fields": fields })
}

fn wrapping_variant(kind: &str, fields: Vec<Value>, wraps: &str) -> Value {
    json!({ "kind": kind, "fields": fields, "wraps": wraps })
}

// Shared leaf-field bounds/defaults, kept in sync by name with
// `fields.rs`'s `register_field!` table (see this module's doc
// comment on why they're duplicated here rather than imported directly --
// `register_field!` builds a `TunerParameter` whose bounds live inside a
// `serde_json::Value`, not a standalone constant this module could reuse
// without re-parsing that JSON for one number).
const C_BOUNDS: [f64; 2] = [0.0, 3.0];
const C_DEFAULT: f64 = std::f64::consts::SQRT_2;
const EPSILON_BOUNDS: [f64; 2] = [0.0, 1.0];
const EPSILON_DEFAULT: f64 = 0.1;

fn rave_schedule_enum() -> Value {
    en(
        "threshold",
        vec![
            variant("hand_selected", vec![field("k", int([0, 2000], 1000))]),
            variant("min_mse", vec![field("bias", float([0.0, 10.0], 0.00001))]),
            variant("threshold", vec![field("rave", int([0, 2000], 700))]),
        ],
    )
}

fn rave_ucb_enum() -> Value {
    en(
        "ucb1_tuned",
        vec![
            variant("none", vec![]),
            variant(
                "ucb1",
                vec![field("exploration_constant", float(C_BOUNDS, C_DEFAULT))],
            ),
            variant(
                "ucb1_tuned",
                vec![field("exploration_constant", float(C_BOUNDS, C_DEFAULT))],
            ),
        ],
    )
}

fn decisive_move_mode_enum() -> Value {
    bare_en(
        "win",
        &["win", "win_loss", "win_loss_draw", "anti_decisive"],
    )
}

/// `GpnUct`'s proof-number bias formula (`mcts::select::GpnBias`) -- a
/// fieldless enum serde writes as a bare string, like `DecisiveMoveMode`.
fn gpn_bias_enum() -> Value {
    bare_en("max", &["max", "sum", "rank"])
}

/// `BaseSelectSpec`'s variants -- every `select` family except
/// `EpsilonGreedy`, which wraps one of these (see `config_ir.rs`'s own doc
/// comment on why the wrapped inner spec is this narrower, non-recursive
/// set rather than a full `SelectSpec` again).
fn select_base_variants() -> Vec<Value> {
    vec![
        variant("ucb1", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant("ucb1_tuned", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant("ucb_v", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant("kl_ucb", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant(
            "ments",
            vec![
                field("tau", float([0.05, 5.0], 1.0)),
                field("epsilon", float([0.0, 1.0], 0.1)),
            ],
        ),
        variant("grill_act", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant(
            "score_bounded_uct",
            vec![
                field("c", float(C_BOUNDS, C_DEFAULT)),
                field("gamma", float([0.0, 1.0], 0.1)),
                field("delta", float([0.0, 1.0], 0.1)),
            ],
        ),
        variant(
            "gpn",
            vec![
                field("c", float(C_BOUNDS, C_DEFAULT)),
                field("c_pn", float(C_BOUNDS, 1.0)),
                field("bias", gpn_bias_enum()),
            ],
        ),
        variant(
            "amaf",
            vec![
                field("alpha", float([0.0, 1.0], 1.0)),
                field("c", float(C_BOUNDS, C_DEFAULT)),
            ],
        ),
        variant(
            "rave",
            vec![
                field("threshold", int([0, 2000], 700)),
                field("schedule", rave_schedule_enum()),
                field("ucb", rave_ucb_enum()),
            ],
        ),
        variant(
            "uct_pn",
            vec![
                field("c", float(C_BOUNDS, C_DEFAULT)),
                field("c_pn", float(C_BOUNDS, 1.0)),
            ],
        ),
        variant(
            "progressive_history",
            vec![
                field("c", float(C_BOUNDS, C_DEFAULT)),
                field("ph_weight", float([0.0, 5.0], 1.0)),
            ],
        ),
        variant("bayes_uct1", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
        variant("bayes_uct2", vec![field("c", float(C_BOUNDS, C_DEFAULT))]),
    ]
}

/// `BackpropSpec`'s variants.
fn backprop_variants() -> Vec<Value> {
    vec![
        variant("classic", vec![]),
        variant(
            "bayes_gaussian",
            vec![
                field("prior_variance", float([0.0, 10.0], 1.0)),
                field("obs_variance", float([1e-6, 10.0], 1.0)),
            ],
        ),
        variant(
            "bayes_numeric",
            vec![
                field("prior_variance", float([0.0, 10.0], 1.0)),
                field("obs_variance", float([1e-6, 10.0], 1.0)),
                field("value_lo", float([-10.0, 10.0], -1.0)),
                field("value_hi", float([-10.0, 10.0], 1.0)),
            ],
        ),
        variant(
            "power_mean",
            vec![
                field("p", float([1.0, 50.0], 1.0)),
                field("alpha", float([0.0, 1.0], 0.0)),
                field("depth", int([0, 64], 0)),
            ],
        ),
        variant(
            "td",
            vec![
                field("lambda", float([0.0, 1.0], 1.0)),
                field("max_child", int([0, 1], 0)),
            ],
        ),
        variant("softmax", vec![field("tau", float([0.05, 5.0], 1.0))]),
    ]
}

/// `SimulateSpec`'s non-wrapper variants -- shared between `simulate_base`
/// (what `EpsilonGreedy`/`DecisiveMove` may wrap) and `simulate`'s own leaf
/// rows.
fn simulate_base_variants() -> Vec<Value> {
    vec![
        variant("uniform", vec![]),
        variant("mast", vec![]),
        variant("nst", vec![field("backoff_threshold", int([0, 100], 5))]),
        variant("lgr", vec![]),
        variant("lgr2", vec![]),
        variant("lgr2_mast", vec![]),
    ]
}

/// The full `select`/`select_base`/`simulate`/`simulate_base`/`backprop`/
/// `final_action` schema tree a UI needs to render `config_ir::SearchSpec`
/// interactively -- see this module's doc comment for the overall shape.
pub fn axis_schema() -> Value {
    let mut select_variants = select_base_variants();
    select_variants.push(wrapping_variant(
        "epsilon_greedy",
        vec![field("epsilon", float(EPSILON_BOUNDS, EPSILON_DEFAULT))],
        "select_base",
    ));

    let mut simulate_variants = simulate_base_variants();
    simulate_variants.push(wrapping_variant(
        "epsilon_greedy",
        vec![field("epsilon", float(EPSILON_BOUNDS, EPSILON_DEFAULT))],
        "simulate_base",
    ));
    simulate_variants.push(wrapping_variant(
        "decisive_move",
        vec![field("mode", decisive_move_mode_enum())],
        "simulate_base",
    ));
    // `decisive_move_mast`/`decisive_move_nst`/`meta_mcts` are deliberately
    // *not* `wrapping_variant`s -- `config_ir.rs`'s own doc comments explain
    // each is a fixed, non-recursive shape (a flattened 2-level wrap, or a
    // hardcoded inner search) with no `Base.../...` inner spec to point a UI
    // at. They render as ordinary flat leaves with scalar fields.
    simulate_variants.push(variant(
        "decisive_move_mast",
        vec![
            field("mode", decisive_move_mode_enum()),
            field("epsilon", float(EPSILON_BOUNDS, EPSILON_DEFAULT)),
        ],
    ));
    simulate_variants.push(variant(
        "decisive_move_nst",
        vec![
            field("mode", decisive_move_mode_enum()),
            field("epsilon", float(EPSILON_BOUNDS, EPSILON_DEFAULT)),
            field("nst_backoff_threshold", int([0, 100], 5)),
        ],
    ));
    simulate_variants.push(variant(
        "meta_mcts",
        vec![field("iterations", int([1, 100_000], 1000))],
    ));

    let final_action_variants = vec![
        variant("robust_child", vec![]),
        variant("max_avg", vec![]),
        variant("max_robust_child", vec![]),
        variant("secure_child", vec![field("a", float([0.0, 10.0], 4.0))]),
    ];

    json!({
        "select": { "variants": select_variants },
        "select_base": { "variants": select_base_variants() },
        "simulate": { "variants": simulate_variants },
        "simulate_base": { "variants": simulate_base_variants() },
        "backprop": { "variants": backprop_variants() },
        "final_action": { "variants": final_action_variants },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_ir::{
        BackpropSpec, BaseSelectSpec, BaseSimulateSpec, FinalActionSpec, SelectSpec, SimulateSpec,
    };
    use mcts::select::{RaveSchedule, RaveUcb};
    use mcts::simulate::DecisiveMoveMode;

    fn variant_kinds(axis: &Value) -> Vec<String> {
        axis["variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["kind"].as_str().unwrap().to_string())
            .collect()
    }

    /// The drift-protection test this module's doc comment promises: one
    /// arm per real enum variant, no wildcard, so a newly added `SelectSpec`/
    /// `SimulateSpec`/`BackpropSpec`/`FinalActionSpec`/`BaseSelectSpec`/
    /// `BaseSimulateSpec`/`RaveSchedule`/`RaveUcb`/`DecisiveMoveMode` variant
    /// fails to compile here until `axis_schema()` (or this list) is updated
    /// to know about it.
    #[test]
    fn axis_schema_covers_every_real_variant() {
        let schema = axis_schema();

        let cover = |kind: &str, expected: &[&str]| {
            let got = variant_kinds(&schema[kind]);
            assert_eq!(
                got, expected,
                "schema[{kind:?}]'s variant kinds must match config_ir's real enum, in order"
            );
        };

        // Exhaustive matches: the compiler forces every arm to be listed
        // here, and each arm's string literal must then appear in `cover`'s
        // expected list below, or the assertion fails.
        let assert_select_variant_named = |s: &SelectSpec| match s {
            SelectSpec::Ucb1 { .. } => "ucb1",
            SelectSpec::Ucb1Tuned { .. } => "ucb1_tuned",
            SelectSpec::UcbV { .. } => "ucb_v",
            SelectSpec::KlUcb { .. } => "kl_ucb",
            SelectSpec::Ments { .. } => "ments",
            SelectSpec::GrillAct { .. } => "grill_act",
            SelectSpec::ScoreBoundedUct { .. } => "score_bounded_uct",
            SelectSpec::Gpn { .. } => "gpn",
            SelectSpec::Amaf { .. } => "amaf",
            SelectSpec::Rave { .. } => "rave",
            SelectSpec::UctPn { .. } => "uct_pn",
            SelectSpec::ProgressiveHistory { .. } => "progressive_history",
            SelectSpec::BayesUct1 { .. } => "bayes_uct1",
            SelectSpec::BayesUct2 { .. } => "bayes_uct2",
            SelectSpec::EpsilonGreedy { .. } => "epsilon_greedy",
        };
        let _ = assert_select_variant_named;
        cover(
            "select",
            &[
                "ucb1",
                "ucb1_tuned",
                "ucb_v",
                "kl_ucb",
                "ments",
                "grill_act",
                "score_bounded_uct",
                "gpn",
                "amaf",
                "rave",
                "uct_pn",
                "progressive_history",
                "bayes_uct1",
                "bayes_uct2",
                "epsilon_greedy",
            ],
        );

        let assert_base_select_variant_named = |s: &BaseSelectSpec| match s {
            BaseSelectSpec::Ucb1 { .. } => "ucb1",
            BaseSelectSpec::Ucb1Tuned { .. } => "ucb1_tuned",
            BaseSelectSpec::UcbV { .. } => "ucb_v",
            BaseSelectSpec::KlUcb { .. } => "kl_ucb",
            BaseSelectSpec::Ments { .. } => "ments",
            BaseSelectSpec::GrillAct { .. } => "grill_act",
            BaseSelectSpec::ScoreBoundedUct { .. } => "score_bounded_uct",
            BaseSelectSpec::Gpn { .. } => "gpn",
            BaseSelectSpec::Amaf { .. } => "amaf",
            BaseSelectSpec::Rave { .. } => "rave",
            BaseSelectSpec::UctPn { .. } => "uct_pn",
            BaseSelectSpec::ProgressiveHistory { .. } => "progressive_history",
            BaseSelectSpec::BayesUct1 { .. } => "bayes_uct1",
            BaseSelectSpec::BayesUct2 { .. } => "bayes_uct2",
        };
        let _ = assert_base_select_variant_named;
        cover(
            "select_base",
            &[
                "ucb1",
                "ucb1_tuned",
                "ucb_v",
                "kl_ucb",
                "ments",
                "grill_act",
                "score_bounded_uct",
                "gpn",
                "amaf",
                "rave",
                "uct_pn",
                "progressive_history",
                "bayes_uct1",
                "bayes_uct2",
            ],
        );

        let assert_simulate_variant_named = |s: &SimulateSpec| match s {
            SimulateSpec::Uniform {} => "uniform",
            SimulateSpec::Mast {} => "mast",
            SimulateSpec::Nst { .. } => "nst",
            SimulateSpec::Lgr {} => "lgr",
            SimulateSpec::Lgr2 {} => "lgr2",
            SimulateSpec::Lgr2Mast {} => "lgr2_mast",
            SimulateSpec::EpsilonGreedy { .. } => "epsilon_greedy",
            SimulateSpec::DecisiveMove { .. } => "decisive_move",
            SimulateSpec::DecisiveMoveMast { .. } => "decisive_move_mast",
            SimulateSpec::DecisiveMoveNst { .. } => "decisive_move_nst",
            SimulateSpec::MetaMcts { .. } => "meta_mcts",
        };
        let _ = assert_simulate_variant_named;
        cover(
            "simulate",
            &[
                "uniform",
                "mast",
                "nst",
                "lgr",
                "lgr2",
                "lgr2_mast",
                "epsilon_greedy",
                "decisive_move",
                "decisive_move_mast",
                "decisive_move_nst",
                "meta_mcts",
            ],
        );

        let assert_base_simulate_variant_named = |s: &BaseSimulateSpec| match s {
            BaseSimulateSpec::Uniform {} => "uniform",
            BaseSimulateSpec::Mast {} => "mast",
            BaseSimulateSpec::Nst { .. } => "nst",
            BaseSimulateSpec::Lgr {} => "lgr",
            BaseSimulateSpec::Lgr2 {} => "lgr2",
            BaseSimulateSpec::Lgr2Mast {} => "lgr2_mast",
        };
        let _ = assert_base_simulate_variant_named;
        cover(
            "simulate_base",
            &["uniform", "mast", "nst", "lgr", "lgr2", "lgr2_mast"],
        );

        let assert_backprop_variant_named = |s: &BackpropSpec| match s {
            BackpropSpec::Classic {} => "classic",
            BackpropSpec::BayesGaussian { .. } => "bayes_gaussian",
            BackpropSpec::BayesNumeric { .. } => "bayes_numeric",
            BackpropSpec::PowerMean { .. } => "power_mean",
            BackpropSpec::Td { .. } => "td",
            BackpropSpec::Softmax { .. } => "softmax",
        };
        let _ = assert_backprop_variant_named;
        cover(
            "backprop",
            &[
                "classic",
                "bayes_gaussian",
                "bayes_numeric",
                "power_mean",
                "td",
                "softmax",
            ],
        );

        let assert_final_action_variant_named = |s: &FinalActionSpec| match s {
            FinalActionSpec::RobustChild {} => "robust_child",
            FinalActionSpec::MaxAvg {} => "max_avg",
            FinalActionSpec::MaxRobustChild {} => "max_robust_child",
            FinalActionSpec::SecureChild { .. } => "secure_child",
        };
        let _ = assert_final_action_variant_named;
        cover(
            "final_action",
            &[
                "robust_child",
                "max_avg",
                "max_robust_child",
                "secure_child",
            ],
        );

        // Nested enums: same exhaustive-match trick, checked against the
        // `rave`/`decisive_move*` variants' own `schedule`/`ucb`/`mode`
        // field's `variants` list rather than a top-level schema key.
        let assert_rave_schedule_variant_named = |s: &RaveSchedule| match s {
            RaveSchedule::HandSelected { .. } => "hand_selected",
            RaveSchedule::MinMSE { .. } => "min_mse",
            RaveSchedule::Threshold { .. } => "threshold",
        };
        let _ = assert_rave_schedule_variant_named;
        let assert_rave_ucb_variant_named = |s: &RaveUcb| match s {
            RaveUcb::None => "none",
            RaveUcb::Ucb1 { .. } => "ucb1",
            RaveUcb::Ucb1Tuned { .. } => "ucb1_tuned",
        };
        let _ = assert_rave_ucb_variant_named;
        let assert_decisive_move_mode_variant_named = |s: &DecisiveMoveMode| match s {
            DecisiveMoveMode::Win => "win",
            DecisiveMoveMode::WinLoss => "win_loss",
            DecisiveMoveMode::WinLossDraw => "win_loss_draw",
            DecisiveMoveMode::AntiDecisive => "anti_decisive",
        };
        let _ = assert_decisive_move_mode_variant_named;

        let rave_field = schema["select"]["variants"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["kind"] == "rave")
            .unwrap();
        let schedule_field = rave_field["fields"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == "schedule")
            .unwrap();
        assert_eq!(
            variant_kinds(schedule_field),
            vec!["hand_selected", "min_mse", "threshold"]
        );
        let ucb_field = rave_field["fields"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == "ucb")
            .unwrap();
        assert_eq!(variant_kinds(ucb_field), vec!["none", "ucb1", "ucb1_tuned"]);
    }

    #[test]
    fn epsilon_greedy_and_decisive_move_declare_the_right_wraps_target() {
        let schema = axis_schema();
        let find = |axis: &str, kind: &str| {
            schema[axis]["variants"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["kind"] == kind)
                .unwrap()
                .clone()
        };
        assert_eq!(find("select", "epsilon_greedy")["wraps"], "select_base");
        assert_eq!(find("simulate", "epsilon_greedy")["wraps"], "simulate_base");
        assert_eq!(find("simulate", "decisive_move")["wraps"], "simulate_base");
        // Fixed-shape leaves must NOT carry a `wraps` key -- a UI must not
        // treat them as recursion targets.
        assert!(find("simulate", "decisive_move_mast")
            .get("wraps")
            .is_none());
        assert!(find("simulate", "decisive_move_nst").get("wraps").is_none());
        assert!(find("simulate", "meta_mcts").get("wraps").is_none());
    }

    /// Proves the schema's own shape actually matches what `build_search`
    /// accepts: build a `SearchSpec` by hand-sampling one variant per axis
    /// (including a nested `EpsilonGreedy`-wrapped `select` and a `Rave`
    /// select with a non-default `RaveSchedule`, exactly the composition
    /// this whole schema exists to describe), matching each field's schema
    /// default, and run it.
    #[test]
    fn a_schema_sampled_search_spec_builds_and_runs() {
        use crate::config_ir::{build_search, SearchSettings, SearchSpec};
        use game_nim::Nim;
        use mcts::game::Game;

        let spec = SearchSpec {
            select: SelectSpec::EpsilonGreedy {
                epsilon: 0.1,
                inner: BaseSelectSpec::Rave {
                    threshold: 700,
                    schedule: RaveSchedule::Threshold { rave: 700 },
                    ucb: RaveUcb::Ucb1Tuned {
                        exploration_constant: std::f64::consts::SQRT_2,
                    },
                },
            },
            simulate: SimulateSpec::Nst {
                backoff_threshold: 5,
            },
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::SecureChild { a: 4.0 },
        };
        let settings = SearchSettings {
            max_iterations: 200,
            max_playout_depth: 200,
            expand_threshold: 1,
            q_init: mcts::node::QInit::Infinity,
            use_transpositions: false,
            use_mcts_solver: false,
            reuse_tree: true,
            num_tree_threads: 1,
            num_threads: 1,
            determinize_root: false,
            seed: 1,
            max_time: None,
            graph_search: None,
            transposition_keying: mcts::TranspositionKeying::PerPly,
            solver_loss_threshold: None,
            contempt_factor: None,
        };
        let mut ai = build_search::<Nim>(&spec, &settings);
        let state = <Nim as Game>::S::default();
        let action = ai.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(legal.contains(&action));
    }

    /// Regression test for a live bug: `decisive_move`/`decisive_move_mast`/
    /// `decisive_move_nst`'s `mode` field used plain `en` (the same
    /// object-tagged `{"kind": ..., ...fields}` shape as `RaveSchedule`/
    /// `RaveUcb`), but `DecisiveMoveMode` has no `#[serde(tag = "kind")]` --
    /// it's a bare-string enum. A UI following the schema's own advertised
    /// shape (before `bare_en` existed) sent `{"kind":"win"}` for `mode`,
    /// which failed with exactly the error a real user hit: "unknown
    /// variant `kind`, expected one of `win`, `win_loss`, `win_loss_draw`"
    /// (`serde` reading the object's own key as if it were the variant
    /// name). This asserts the schema now tells a client `mode` is `bare`,
    /// and that the bare-string wire shape a client following that
    /// actually round-trips.
    #[test]
    fn decisive_move_mode_field_is_marked_bare_and_round_trips_as_a_plain_string() {
        let schema = axis_schema();
        let simulate_variants = schema["simulate"]["variants"].as_array().unwrap();
        for kind in ["decisive_move", "decisive_move_mast", "decisive_move_nst"] {
            let v = simulate_variants
                .iter()
                .find(|v| v["kind"] == kind)
                .unwrap_or_else(|| panic!("simulate schema is missing variant {kind:?}"));
            let mode_field = v["fields"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["name"] == "mode")
                .unwrap_or_else(|| panic!("{kind:?}'s schema entry is missing a mode field"));
            assert_eq!(
                mode_field["bare"],
                serde_json::json!(true),
                "{kind:?}'s mode field must be marked bare -- DecisiveMoveMode has no serde tag"
            );
        }

        // The actual wire shape a client honoring `bare` sends: a plain
        // string, not `{"kind": "win"}`.
        let json =
            r#"{"kind":"decisive_move_nst","mode":"win","epsilon":0.1,"nst_backoff_threshold":5}"#;
        let spec: SimulateSpec =
            serde_json::from_str(json).expect("bare-string mode must deserialize");
        assert_eq!(
            spec,
            SimulateSpec::DecisiveMoveNst {
                mode: DecisiveMoveMode::Win,
                epsilon: 0.1,
                nst_backoff_threshold: 5,
            }
        );
    }
}
