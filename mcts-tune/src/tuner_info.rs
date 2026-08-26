use game_host::TunerInfo;
use serde_json::json;

use crate::family_catalog::{
    condition, direct_family_names, family_choices, family_conditions, param,
    tunable_field_parameters,
};

/// Search-space metadata for the full multi-family catalog above, for `tune
/// describe` to report to a tuner harness or launch-form UI.
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
        eval_rounds,
        parameters: {
            let mut parameters = vec![
                param(
                    "family",
                    json!({"type": "categorical", "choices": family_choices(), "default": "rave"}),
                ),
                param(
                    "q_init",
                    json!({"type": "categorical", "choices": ["Draw", "Infinity", "Loss", "Parent", "Win"], "default": "Infinity"}),
                ),
            ];
            parameters.extend(tunable_field_parameters());
            parameters
        },
        conditions: {
            let mut conditions = family_conditions();
            // `q_init` is meaningless to a `Direct` family (no Q-values for
            // it to initialize -- see `DirectFamily`'s doc comment), so its
            // activation is gated on `family` naming one of the *other*
            // (`Compose`) rows, rather than being an unconditional root like
            // `family` itself.
            let direct = direct_family_names();
            let compose_families: Vec<&str> = family_choices()
                .into_iter()
                .filter(|f| !direct.contains(f))
                .collect();
            conditions.push(condition(json!({"family": compose_families}), &["q_init"]));
            // Gated by another field's own sampled value (`final_action`,
            // `schedule`, `rave_ucb`, `contempt`, `flat_mc_selection`,
            // `negamax_aspiration`), not by `family` directly -- see
            // `register_family!`'s doc comment in `family_catalog.rs` for
            // why these stay hand-written instead of per-row table entries.
            conditions.extend([
                condition(json!({"final_action": "secure_child"}), &["a"]),
                condition(json!({"schedule": "hand_selected"}), &["k"]),
                condition(json!({"schedule": "min_mse"}), &["bias"]),
                condition(json!({"schedule": "threshold"}), &["rave"]),
                condition(json!({"rave_ucb": ["ucb1", "tuned"]}), &["c"]),
                condition(json!({"contempt": "on"}), &["contempt_factor"]),
                condition(json!({"flat_mc_selection": "ucb1"}), &["c"]),
                condition(json!({"negamax_aspiration": "on"}), &["aspiration_window"]),
            ]);
            conditions
        },
    };
    if supports_mcgs {
        info.parameters
            .push(param("mcgs", json!({"type": "bool", "default": false})));
        info.parameters.push(param(
            "state_only_keying",
            json!({"type": "bool", "default": false}),
        ));
        // Only meaningful (and only accepted by `resolve_graph_search`)
        // when `mcgs` is also sampled `true` -- see
        // `TrialParams::state_only_keying`'s doc comment.
        info.conditions
            .push(condition(json!({"mcgs": true}), &["state_only_keying"]));
    }
    info
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
