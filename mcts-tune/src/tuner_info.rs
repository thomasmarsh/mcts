use game_host::{GameConfigSchema, TunerInfo, TunerParameter};
use serde_json::json;

fn int_param(name: &str, min: i64, max: i64, default: i64) -> TunerParameter {
    TunerParameter {
        name: name.to_string(),
        spec: json!({ "type": "int", "bounds": [min, max], "default": default }),
    }
}

/// A `game_config` schema for a game whose only setup axis is a single
/// square board `size` bounded `min..=max` (AtariGo).
pub fn square_board_config_schema(min: i64, max: i64, default: i64) -> GameConfigSchema {
    GameConfigSchema {
        parameters: vec![int_param("size", min, max, default)],
        conditions: vec![],
    }
}

/// A `game_config` schema for a game whose board is a `{w, h}` object, each
/// dimension bounded independently and rendered as dotted `size.w` /
/// `size.h` int fields (Druid).
pub fn dimensions_board_config_schema(
    min: i64,
    max: i64,
    default_w: i64,
    default_h: i64,
) -> GameConfigSchema {
    GameConfigSchema {
        parameters: vec![
            int_param("size.w", min, max, default_w),
            int_param("size.h", min, max, default_h),
        ],
        conditions: vec![],
    }
}

use crate::family_catalog::{condition, param, tunable_field_parameters};

/// Every `select`-axis categorical choice, in `config_ir_schema::axis_schema`
/// order. The `epsilon_greedy` wrap is a separate `select_epsilon_greedy`
/// boolean rather than a choice here.
const SELECT_CHOICES: &[&str] = &[
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
];

/// `select` variants that read the shared exploration constant `c`. `rave`
/// is absent -- it reads `c` only through its own `rave_ucb` sub-categorical
/// -- and so is `ments`, which has no exploration constant.
const C_SELECTS: &[&str] = &[
    "ucb1",
    "ucb1_tuned",
    "ucb_v",
    "kl_ucb",
    "grill_act",
    "score_bounded_uct",
    "gpn",
    "amaf",
    "uct_pn",
    "progressive_history",
    "bayes_uct1",
    "bayes_uct2",
];

/// The proof-number selects (Kowalski et al. 2023): a per-player PN bias
/// that only bites with MCTS-Solver on, so each also activates
/// `solver_loss_threshold` and the `contempt` on/off switch.
const PN_SELECTS: &[&str] = &["score_bounded_uct", "gpn", "uct_pn"];

/// `select` variants that leave the select<->backprop pairing free, so
/// `backprop` is an independent categorical for them. A `bayes_uct1`/
/// `bayes_uct2`/`ments` select instead pins its backprop (see `conditions`).
const FREE_BACKPROP_SELECTS: &[&str] = &[
    "ucb1",
    "ucb1_tuned",
    "ucb_v",
    "kl_ucb",
    "grill_act",
    "score_bounded_uct",
    "gpn",
    "amaf",
    "rave",
    "uct_pn",
    "progressive_history",
];

/// Every `simulate`-axis categorical choice.
const SIMULATE_CHOICES: &[&str] = &[
    "uniform",
    "mast",
    "nst",
    "lgr",
    "lgr2",
    "lgr2_mast",
    "decisive_move",
    "decisive_move_mast",
    "decisive_move_nst",
    "meta_mcts",
];

/// The top-level `algorithm` categorical: the move-choosing method.
const ALGORITHM_CHOICES: &[&str] = &["random", "flat_mc", "mcts", "negamax"];

/// Axis categoricals and engine settings a `mcts` algorithm activates
/// unconditionally (per-variant scalars are gated further, on the axis's
/// own sampled value). `backprop` is deliberately absent: it is pinned, not
/// free, for a `bayes_*`/`ments` select, so it is activated per-select
/// instead.
const MCTS_AXES: &[&str] = &[
    "select",
    "select_epsilon_greedy",
    "simulate",
    "simulate_epsilon_greedy",
    "final_action",
    "q_init",
];

/// Search-space metadata describing a fully specified configuration -- the
/// `algorithm` categorical plus, for `mcts`, the four policy-axis
/// categoricals and their per-variant parameters -- for `tune describe` to
/// report to a tuner harness or launch-form UI.
pub fn strategy_tuner_info(baselines: &[&str], eval_rounds: u32) -> TunerInfo {
    strategy_tuner_info_with_mcgs(baselines, eval_rounds, false)
}

/// Tuning schema for a game with a sound Zobrist hash. The `mcgs` boolean
/// selects the combined edge-and-node statistics graph mode; it is omitted
/// entirely for games that cannot safely create transposition tables.
pub fn strategy_tuner_info_with_mcgs(
    baselines: &[&str],
    eval_rounds: u32,
    supports_mcgs: bool,
) -> TunerInfo {
    let mut info = TunerInfo {
        id: "strategy".into(),
        baselines: baselines.iter().map(|s| s.to_string()).collect(),
        game_config: json!({}),
        // Filled in per-game (AtariGo/Druid) alongside `game_config`; every
        // fixed-board game keeps this empty default.
        game_config_schema: GameConfigSchema::default(),
        eval_rounds,
        parameters: {
            let mut parameters = vec![
                param(
                    "algorithm",
                    json!({"type": "categorical", "choices": ALGORITHM_CHOICES, "default": "mcts"}),
                ),
                param(
                    "select",
                    json!({"type": "categorical", "choices": SELECT_CHOICES, "default": "ucb1"}),
                ),
                param(
                    "select_epsilon_greedy",
                    json!({"type": "bool", "default": false}),
                ),
                param(
                    "simulate",
                    json!({"type": "categorical", "choices": SIMULATE_CHOICES, "default": "uniform"}),
                ),
                param(
                    "simulate_epsilon_greedy",
                    json!({"type": "bool", "default": false}),
                ),
                param(
                    "decisive_move_mode",
                    json!({"type": "categorical", "choices": ["win", "win_loss", "win_loss_draw", "anti_decisive"], "default": "win"}),
                ),
                // The four hard-wired select<->backprop couplings are
                // encoded as pinned constants, not `ForbiddenClause`s:
                // `backprop` is an independent categorical only for the
                // selects that leave the pairing free. A `bayes_uct1`/
                // `bayes_uct2`/`ments` select instead directly activates its
                // pinned backprop's parameters (see `conditions`), and
                // `bayes_gaussian`/`bayes_numeric`/`softmax` never appear as
                // selectable `backprop` choices at all.
                param(
                    "backprop",
                    json!({"type": "categorical", "choices": ["classic", "power_mean", "td"], "default": "classic"}),
                ),
                param(
                    "final_action",
                    json!({"type": "categorical", "choices": ["robust_child", "max_avg", "max_robust_child", "secure_child"], "default": "robust_child"}),
                ),
                param(
                    "q_init",
                    json!({"type": "categorical", "choices": ["Draw", "Infinity", "Loss", "Parent", "Win"], "default": "Infinity"}),
                ),
            ];
            // Every remaining scalar/sub-categorical knob comes from
            // `register_field!` -- still the single source for each
            // variant parameter's bounds and default. `final_action` is
            // re-declared above as a four-choice axis categorical (its
            // `register_field!` row predates `max_robust_child` being
            // reachable outside a hard-fixed family row), so its table row
            // is dropped here.
            parameters.extend(
                tunable_field_parameters()
                    .into_iter()
                    .filter(|p| p.name != "final_action"),
            );
            parameters
        },
        conditions: vec![
            // `mcts` activates the four axis categoricals plus the
            // orthogonal `q_init` (meaningless to `random`/`flat_mc`/
            // `negamax`, which have no Q-values to initialize).
            condition(json!({"algorithm": "mcts"}), MCTS_AXES),
            // Shared exploration constant.
            condition(json!({"select": C_SELECTS}), &["c"]),
            // The `epsilon_greedy` wraps and the `decisive_move*`
            // policies that carry their own `epsilon`.
            condition(json!({"select_epsilon_greedy": true}), &["epsilon"]),
            condition(json!({"simulate_epsilon_greedy": true}), &["epsilon"]),
            condition(
                json!({"simulate": ["decisive_move_mast", "decisive_move_nst"]}),
                &["epsilon"],
            ),
            // Per-`select` variant parameters.
            condition(json!({"select": "ments"}), &["tau", "epsilon"]),
            condition(json!({"select": "amaf"}), &["amaf_alpha"]),
            condition(json!({"select": "progressive_history"}), &["ph_weight"]),
            condition(
                json!({"select": ["bayes_uct1", "bayes_uct2"]}),
                &["prior_variance", "obs_variance"],
            ),
            condition(json!({"select": "bayes_uct2"}), &["value_lo", "value_hi"]),
            condition(json!({"select": "score_bounded_uct"}), &["gamma", "delta"]),
            condition(json!({"select": "gpn"}), &["gpn_bias", "c_pn"]),
            condition(json!({"select": "uct_pn"}), &["c_pn"]),
            condition(
                json!({"select": PN_SELECTS}),
                &["solver_loss_threshold", "contempt"],
            ),
            condition(
                json!({"select": "rave"}),
                &["threshold", "schedule", "rave_ucb"],
            ),
            // `backprop` is an independent axis only for the
            // free-pairing selects (the couplings pin it otherwise).
            condition(json!({"select": FREE_BACKPROP_SELECTS}), &["backprop"]),
            condition(json!({"backprop": "power_mean"}), &["p", "alpha"]),
            condition(json!({"backprop": "td"}), &["lambda", "td_max_child"]),
            // Per-`simulate` variant parameters.
            condition(
                json!({"simulate": ["nst", "decisive_move_nst"]}),
                &["nst_backoff_threshold"],
            ),
            condition(
                json!({"simulate": ["decisive_move", "decisive_move_mast", "decisive_move_nst"]}),
                &["decisive_move_mode"],
            ),
            // Gated by a sub-categorical's own sampled value
            // (`final_action`, `schedule`, `rave_ucb`, `contempt`).
            condition(json!({"final_action": "secure_child"}), &["a"]),
            condition(json!({"schedule": "hand_selected"}), &["k"]),
            condition(json!({"schedule": "min_mse"}), &["bias"]),
            condition(json!({"schedule": "threshold"}), &["rave"]),
            condition(json!({"rave_ucb": ["ucb1", "tuned"]}), &["c"]),
            condition(json!({"contempt": "on"}), &["contempt_factor"]),
            // The standalone (non-`mcts`) algorithms.
            condition(
                json!({"algorithm": "flat_mc"}),
                &["samples_per_move", "max_rollout_depth", "flat_mc_selection"],
            ),
            condition(json!({"flat_mc_selection": "ucb1"}), &["c"]),
            condition(
                json!({"algorithm": "negamax"}),
                &[
                    "max_depth",
                    "table_bits",
                    "negamax_replacement",
                    "principal_variation_search",
                    "history_heuristic",
                    "singular_extension",
                    "countermove_heuristic",
                    "negamax_aspiration",
                ],
            ),
            condition(json!({"negamax_aspiration": "on"}), &["aspiration_window"]),
        ],
    };
    if supports_mcgs {
        info.parameters
            .push(param("mcgs", json!({"type": "bool", "default": false})));
        info.parameters.push(param(
            "state_only_keying",
            json!({"type": "bool", "default": false}),
        ));
        // The transposition graph and its keying only apply to `mcts`;
        // `state_only_keying` is further meaningless (and rejected by
        // `resolve_graph_search`) unless `mcgs` is also sampled `true` --
        // see `TrialParams::state_only_keying`'s doc comment.
        info.conditions
            .push(condition(json!({"algorithm": "mcts"}), &["mcgs"]));
        info.conditions
            .push(condition(json!({"mcgs": true}), &["state_only_keying"]));
    }
    info
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
