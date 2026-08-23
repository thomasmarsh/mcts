use crate::family_catalog::{family_choices, TrialParams};
use crate::*;
use game_host::{ConfiguredCandidateSide, ConfiguredOutcome, HostError, TunerInfo};
use game_nim::Nim;
use mcts::game::Game;
use mcts::strategies::mcts::select::{RaveSchedule, RaveUcb};
use mcts::strategies::mcts::{
    node::QInit, simulate, strategy, GraphSearch, GraphStats, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;
use serde_json::{json, Value};

fn nim_action_value(state: &<Nim as Game>::S, action: &<Nim as Game>::A) -> Value {
    Value::String(Nim::notation(state, action))
}

#[test]
fn search_report_and_legacy_analysis_agree_on_the_selected_action() {
    let state = <Nim as Game>::S::default();
    let mut search =
        TreeSearch::<Nim, strategy::Ucb1>::new().config(SearchConfig::new().max_iterations(20));
    let (selected_action, report) = choose_action_with_report(&mut search, &state, |action| {
        nim_action_value(&state, action)
    });
    let analysis = legacy_analysis_with_report(
        &search,
        &state,
        &selected_action,
        report.clone(),
        |action| nim_action_value(&state, action),
    );

    assert_eq!(report.status, game_host::SearchReportStatus::Available);
    assert_eq!(report.selected_action, analysis.suggested_move);
    assert_eq!(
        analysis.search.as_ref().unwrap().selected_action,
        analysis.suggested_move
    );
    for action in &report.actions {
        assert!(analysis.actions.iter().any(|legacy| {
            legacy.action == action.action
                && legacy.visits == action.visits
                && legacy.mean_value == action.mean_value
                && legacy.is_proven == action.is_proven
        }));
    }
}

#[test]
fn non_mcts_search_reports_explicit_unavailability() {
    let state = <Nim as Game>::S::default();
    let mut search = mcts::strategies::random::Random::<Nim>::new();
    let (selected_action, report) = choose_action_with_report(&mut search, &state, |action| {
        nim_action_value(&state, action)
    });
    let analysis =
        legacy_analysis_with_report(&search, &state, &selected_action, report, |action| {
            nim_action_value(&state, action)
        });

    let report = analysis.search.as_ref().unwrap();
    assert_eq!(report.status, game_host::SearchReportStatus::Unavailable);
    assert_eq!(
        report.reason,
        Some(game_host::SearchReportReason::StrategyUnsupported)
    );
    assert_eq!(
        analysis.suggested_move,
        Some(nim_action_value(&state, &selected_action))
    );
}

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

// Bounded, unlike production `baseline_build` callers (which always pass
// a real budgeted preset): the missing-field/unknown-value rejection
// tests below never reach real play, but the family round-trip tests do,
// and `TreeSearch::default()`'s `max_iterations` is `usize::MAX`.
fn baseline() -> Box<dyn Search<G = Nim>> {
    Box::new(
        TreeSearch::<Nim, strategy::Ucb1>::new().config(SearchConfig::new().max_iterations(50)),
    )
}

fn rave_params() -> Value {
    json!({
        "family": "rave",
        "threshold": 700,
        "c": 0.3,
        "epsilon": 0.1,
        "q_init": "Infinity",
        "final_action": "robust_child",
        "schedule": "threshold",
        "rave": 700,
        "rave_ucb": "tuned",
    })
}

fn pn_params() -> Value {
    json!({
        "family": "ucb1_pn",
        "q_init": "Infinity",
        "c": 1.4,
        "c_pn": 1.0,
        "final_action": "robust_child",
        "solver_loss_threshold": 5,
        "contempt": "off",
    })
}

fn comparison_params() -> Value {
    json!({
        "family": "ucb1",
        "c": 1.4,
        "q_init": "Infinity",
        "final_action": "robust_child",
    })
}

#[test]
fn mcgs_schema_is_available_only_to_hashing_games() {
    let plain = strategy_tuner_info(&["strong"], 1);
    assert!(!plain.parameters.iter().any(|p| p.name == "mcgs"));

    let graph = strategy_tuner_info_with_mcgs(&["strong"], 1, true);
    let mcgs = graph
        .parameters
        .iter()
        .find(|p| p.name == "mcgs")
        .expect("hashing games expose the MCGS switch");
    assert_eq!(mcgs.spec["type"], json!("bool"));
    assert_eq!(mcgs.spec["default"], json!(false));
}

#[test]
fn configured_eval_streams_alternating_results_and_matches_aggregate() {
    let params = comparison_params();
    let budget = SearchBudget {
        max_iterations: Some(3),
        ..Default::default()
    };
    let mut records = Vec::new();
    let mut sink = |record| {
        records.push(record);
        Ok(())
    };
    let outcome = strategy_tune_eval::<Nim>(
        &params,
        2,
        Some(42),
        false,
        budget,
        || build_search::<Nim>(&params, 0, false, &budget).unwrap(),
        <Nim as Game>::S::default(),
        None,
        &mut sink,
    )
    .unwrap();

    assert_eq!(records.len(), 4);
    assert_eq!(
        records.iter().map(|record| record.seq).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.candidate_side)
            .collect::<Vec<_>>(),
        vec![
            ConfiguredCandidateSide::First,
            ConfiguredCandidateSide::Second,
            ConfiguredCandidateSide::First,
            ConfiguredCandidateSide::Second,
        ]
    );
    let wins = records
        .iter()
        .filter(|record| record.outcome == ConfiguredOutcome::CandidateWin)
        .count() as u32;
    let losses = records
        .iter()
        .filter(|record| record.outcome == ConfiguredOutcome::BaselineWin)
        .count() as u32;
    let draws = records
        .iter()
        .filter(|record| record.outcome == ConfiguredOutcome::Draw)
        .count() as u32;
    assert_eq!(
        (wins, losses, draws),
        (outcome.wins, outcome.losses, outcome.draws)
    );
    for record in records {
        assert!(record.candidate.iterations_total > 0);
        assert!(record.baseline.iterations_total > 0);
        assert!(record.candidate.iterations_first_half <= record.candidate.iterations_total);
        assert!(record.baseline.iterations_first_half <= record.baseline.iterations_total);
    }
}

#[test]
fn configured_eval_sink_error_stops_before_later_games() {
    let params = comparison_params();
    let budget = SearchBudget {
        max_iterations: Some(3),
        ..Default::default()
    };
    let mut seen = 0;
    let mut sink = |_record| {
        seen += 1;
        Err(HostError::internal("stop streaming"))
    };
    let err = strategy_tune_eval::<Nim>(
        &params,
        2,
        Some(42),
        false,
        budget,
        || build_search::<Nim>(&params, 0, false, &budget).unwrap(),
        <Nim as Game>::S::default(),
        None,
        &mut sink,
    )
    .expect_err("sink failure should abort the comparison");
    assert_eq!(seen, 1);
    assert_eq!(err.message, "stop streaming");
}

#[test]
fn search_budget_time_and_default_iteration_limits_are_distinct() {
    assert_eq!(SearchBudget::default().iteration_limit(), MAX_ITER);
    assert_eq!(
        SearchBudget {
            max_time: Some(std::time::Duration::from_millis(1)),
            ..Default::default()
        }
        .iteration_limit(),
        usize::MAX
    );
    assert_eq!(
        SearchBudget {
            max_iterations: Some(7),
            max_time: Some(std::time::Duration::from_millis(1)),
            ..Default::default()
        }
        .iteration_limit(),
        7
    );
}

#[test]
fn test_tune_eval_rejects_params_missing_required_field() {
    // "schedule": "threshold" requires "rave", which is absent -- this
    // must fail fast during config construction, before any game is
    // played (no real MCTS search runs in this test).
    let mut params = rave_params();
    params.as_object_mut().unwrap().remove("rave");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("missing `rave` must be rejected");
    assert!(err.message.contains("rave"));
}

#[test]
fn zero_round_internal_validation_builds_candidate_without_playing() {
    let mut params = rave_params();
    params.as_object_mut().unwrap().remove("rave");
    let err = strategy_tune_eval::<Nim>(
        &params,
        0,
        Some(0),
        false,
        SearchBudget {
            max_iterations: Some(1),
            ..Default::default()
        },
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("zero-round validation must reach the strategy builder");
    assert!(err.message.contains("rave"));
}

#[test]
fn test_tune_eval_rejects_unknown_schedule() {
    let mut params = rave_params();
    params["schedule"] = json!("not_a_real_schedule");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("unknown schedule must be rejected");
    assert!(err.message.contains("schedule"));
}

#[test]
fn test_tune_eval_rejects_unknown_final_action() {
    let mut params = rave_params();
    params["final_action"] = json!("not_a_real_final_action");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("unknown final_action must be rejected");
    assert!(err.message.contains("final_action"));
}

#[test]
fn test_tune_eval_rejects_unknown_contempt() {
    let mut params = pn_params();
    params["contempt"] = json!("not_a_real_contempt_mode");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("unknown contempt must be rejected");
    assert!(err.message.contains("contempt"));
}

#[test]
fn test_tune_eval_rejects_contempt_on_missing_contempt_factor() {
    let mut params = pn_params();
    params["contempt"] = json!("on");
    params.as_object_mut().unwrap().remove("contempt_factor");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("contempt=on without contempt_factor must be rejected");
    assert!(err.message.contains("contempt_factor"));
}

#[test]
fn test_tune_eval_rejects_unknown_family() {
    let mut params = rave_params();
    params["family"] = json!("not_a_real_family");
    let err = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        SearchBudget::default(),
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect_err("unknown family must be rejected");
    assert!(err.message.contains("family"));
}

/// One hand-verified construction+round-trip test per new family arm,
/// each playing a single round of `Nim` (fast, deterministic) to prove
/// the concrete type actually builds and the declared params round-trip
/// through `make_candidate` without error. `cost_from_losses` itself is
/// already covered above -- this only exercises dispatch, so the
/// candidate is bounded to the same small iteration count as `baseline`
/// rather than left on `SearchBudget::default()`'s `MAX_ITER` (10,000),
/// which made each of these tests a real multi-second search.
fn assert_family_round_trips(mut params: Value) {
    params["q_init"] = json!("Infinity");
    let candidate_budget = SearchBudget {
        max_iterations: Some(50),
        ..SearchBudget::default()
    };
    let outcome = strategy_tune_eval::<Nim>(
        &params,
        1,
        Some(0),
        false,
        candidate_budget,
        baseline,
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .unwrap_or_else(|e| {
        panic!(
            "family {:?} should round-trip: {}",
            params["family"], e.message
        )
    });
    assert!(outcome.wins + outcome.losses + outcome.draws == 2);
}

#[test]
fn test_family_ucb1_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1", "c": 1.4, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_dm_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_dm", "c": 1.4, "final_action": "max_avg",
    }));
}

#[test]
fn test_family_ucb1_adm_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_adm", "c": 1.4, "final_action": "max_avg",
    }));
}

#[test]
fn test_family_ucb1_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_lgr_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_lgr", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_lgr2_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_lgr2", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_lgr2_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_lgr2_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_nst_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_nst", "c": 1.4, "epsilon": 0.2,
        "nst_backoff_threshold": 3, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_progressive_history_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5,
        "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_max_robust_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_max_robust", "c": 1.4,
    }));
}

#[test]
fn test_family_amaf_round_trips() {
    assert_family_round_trips(json!({
        "family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "final_action": "secure_child", "a": 4.0,
    }));
}

#[test]
fn test_family_amaf_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2,
        "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_tuned_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_tuned", "c": 1.4, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_tuned_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_tuned_mast", "c": 1.4, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_tuned_dm_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_tuned_dm", "c": 1.4, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_tuned_dm_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_dm_nst_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2,
        "nst_backoff_threshold": 3, "final_action": "robust_child",
    }));
}

#[test]
fn test_family_ucb1_adm_nst_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_adm_nst", "c": 1.4, "epsilon": 0.2,
        "nst_backoff_threshold": 3, "final_action": "robust_child",
    }));
}

// `meta_mcts`'s round trip is proven in `tests/stress.rs` instead of here:
// its inner nested search makes even one candidate-vs-baseline game
// noticeably slower than every other family's (multi-second, not the
// sub-second every sibling test above runs in), so it belongs in the
// slow/stress suite `cargo test --lib` never compiles, not this fast one.

#[test]
fn test_family_rave_round_trips() {
    assert_family_round_trips(rave_params());
}

#[test]
fn test_family_ucb1_pn_round_trips() {
    assert_family_round_trips(pn_params());
}

#[test]
fn test_family_ucb1_pn_mast_round_trips() {
    assert_family_round_trips(json!({
        "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
        "final_action": "robust_child", "solver_loss_threshold": 5,
        "contempt": "on", "contempt_factor": -0.5,
    }));
}

// -----------------------------------------------------------------
// `to_search_spec` -- config_ir conversion (step 4c). Not yet wired
// into `make_candidate` (that's step 4d); these tests pin the exact
// `SearchSpec`/`SearchSettings` shape each family converts to.
// -----------------------------------------------------------------

fn trial(params: Value) -> TrialParams {
    serde_json::from_value(params).unwrap()
}

#[test]
fn to_search_spec_ucb1() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
            simulate: config_ir::SimulateSpec::Uniform {},
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::RobustChild {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_dm() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_dm", "c": 1.4, "q_init": "Infinity", "final_action": "max_avg",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
            simulate: config_ir::SimulateSpec::DecisiveMove {
                mode: simulate::DecisiveMoveMode::Win,
                inner: config_ir::BaseSimulateSpec::Uniform {},
            },
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::MaxAvg {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_adm() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_adm", "c": 1.4, "q_init": "Infinity", "final_action": "max_avg",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
            simulate: config_ir::SimulateSpec::DecisiveMove {
                mode: simulate::DecisiveMoveMode::AntiDecisive,
                inner: config_ir::BaseSimulateSpec::Uniform {},
            },
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::MaxAvg {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_mast() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Mast {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_lgr() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_lgr", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Lgr {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_lgr2() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_lgr2", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Lgr2 {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_lgr2_mast() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_lgr2_mast", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Lgr2Mast {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_nst() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3,
            "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Nst {
                backoff_threshold: 3
            },
        }
    );
}

#[test]
fn to_search_spec_ucb1_dm_nst() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3,
            "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::DecisiveMoveNst {
            mode: simulate::DecisiveMoveMode::Win,
            epsilon: 0.2,
            nst_backoff_threshold: 3,
        }
    );
}

#[test]
fn to_search_spec_ucb1_adm_nst() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_adm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3,
            "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::DecisiveMoveNst {
            mode: simulate::DecisiveMoveMode::AntiDecisive,
            epsilon: 0.2,
            nst_backoff_threshold: 3,
        }
    );
}

#[test]
fn to_search_spec_ucb1_progressive_history() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5,
            "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.select,
        config_ir::SelectSpec::ProgressiveHistory {
            c: 1.4,
            ph_weight: 0.5
        }
    );
}

#[test]
fn to_search_spec_amaf() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "q_init": "Infinity",
            "final_action": "secure_child", "a": 4.0,
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Amaf { alpha: 0.5, c: 1.4 },
            simulate: config_ir::SimulateSpec::Uniform {},
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::SecureChild { a: 4.0 },
        }
    );
}

#[test]
fn to_search_spec_amaf_mast() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2,
            "q_init": "Infinity", "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Mast {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_tuned() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_tuned", "c": 1.4, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(spec.select, config_ir::SelectSpec::Ucb1Tuned { c: 1.4 });
    assert_eq!(spec.simulate, config_ir::SimulateSpec::Uniform {});
}

#[test]
fn to_search_spec_ucb1_tuned_mast() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_tuned_mast", "c": 1.4, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(spec.simulate, config_ir::SimulateSpec::Mast {});
}

#[test]
fn to_search_spec_ucb1_tuned_dm() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_tuned_dm", "c": 1.4, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::DecisiveMove {
            mode: simulate::DecisiveMoveMode::Win,
            inner: config_ir::BaseSimulateSpec::Uniform {},
        }
    );
}

#[test]
fn to_search_spec_ucb1_tuned_dm_mast() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
            "final_action": "robust_child",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::DecisiveMoveMast {
            mode: simulate::DecisiveMoveMode::Win,
            epsilon: 0.2,
        }
    );
}

#[test]
fn to_search_spec_rave() {
    let (spec, _) =
        to_search_spec(&trial(rave_params()), 0, false, &SearchBudget::default()).unwrap();
    assert_eq!(
        spec.select,
        config_ir::SelectSpec::Rave {
            threshold: 700,
            schedule: RaveSchedule::Threshold { rave: 700 },
            ucb: RaveUcb::Ucb1Tuned {
                exploration_constant: 0.3
            },
        }
    );
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::DecisiveMoveMast {
            mode: simulate::DecisiveMoveMode::WinLoss,
            epsilon: 0.1,
        }
    );
}

#[test]
fn to_search_spec_ucb1_pn() {
    let (spec, settings) =
        to_search_spec(&trial(pn_params()), 0, false, &SearchBudget::default()).unwrap();
    assert_eq!(
        spec.select,
        config_ir::SelectSpec::UctPn { c: 1.4, c_pn: 1.0 }
    );
    assert_eq!(settings.solver_loss_threshold, Some(5));
    assert_eq!(settings.contempt_factor, None);
}

#[test]
fn to_search_spec_ucb1_pn_mast() {
    let (spec, settings) = to_search_spec(
        &trial(json!({
            "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
            "q_init": "Infinity", "final_action": "robust_child",
            "solver_loss_threshold": 5, "contempt": "on", "contempt_factor": -0.5,
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec.simulate,
        config_ir::SimulateSpec::EpsilonGreedy {
            epsilon: 0.2,
            inner: config_ir::BaseSimulateSpec::Mast {},
        }
    );
    assert_eq!(settings.solver_loss_threshold, Some(5));
    assert_eq!(settings.contempt_factor, Some(-0.5));
}

#[test]
fn to_search_spec_ucb1_max_robust() {
    let (spec, _) = to_search_spec(
        &trial(json!({
            "family": "ucb1_max_robust", "c": 1.4, "q_init": "Infinity",
        })),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
            simulate: config_ir::SimulateSpec::Uniform {},
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::MaxRobustChild {},
        }
    );
}

#[test]
fn to_search_spec_meta_mcts() {
    let (spec, _) = to_search_spec(
        &trial(json!({"family": "meta_mcts", "c": 1.4, "q_init": "Infinity"})),
        0,
        false,
        &SearchBudget::default(),
    )
    .unwrap();
    assert_eq!(
        spec,
        config_ir::SearchSpec {
            select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
            simulate: config_ir::SimulateSpec::MetaMcts {
                iterations: META_MCTS_INNER_ITERATIONS
            },
            backprop: config_ir::BackpropSpec::Classic {},
            final_action: config_ir::FinalActionSpec::MaxAvg {},
        }
    );
}

#[test]
fn to_search_spec_settings_mirror_base_config() {
    let (_, settings) = to_search_spec(
        &trial(comparison_params()),
        7,
        true,
        &SearchBudget {
            max_iterations: Some(123),
            threads: 4,
            max_time: Some(std::time::Duration::from_secs(1)),
        },
    )
    .unwrap();
    assert_eq!(settings.max_iterations, 123);
    assert_eq!(settings.max_playout_depth, PLAYOUT_DEPTH);
    assert_eq!(settings.expand_threshold, EXPAND_THRESHOLD);
    assert!(matches!(settings.q_init, QInit::Infinity));
    assert!(settings.use_transpositions);
    assert!(settings.use_mcts_solver);
    assert!(settings.reuse_tree);
    assert_eq!(settings.num_tree_threads, 4);
    assert_eq!(settings.seed, 7);
    assert_eq!(settings.max_time, Some(std::time::Duration::from_secs(1)));
    assert_eq!(settings.graph_search, None);
}

#[test]
fn to_search_spec_mcgs_sets_graph_search_and_disables_transpositions_and_reuse() {
    let mut params = comparison_params();
    params["mcgs"] = json!(true);
    let (_, settings) = to_search_spec(&trial(params), 0, true, &SearchBudget::default()).unwrap();
    assert_eq!(
        settings.graph_search,
        Some(GraphSearch::Dag(GraphStats::Both))
    );
    assert!(!settings.use_transpositions);
    assert!(!settings.reuse_tree);
}

#[test]
fn to_search_spec_mcgs_without_transpositions_is_rejected() {
    let mut params = comparison_params();
    params["mcgs"] = json!(true);
    // `(SearchSpec, SearchSettings)` isn't `Debug`, so `expect_err` doesn't
    // apply here -- match instead (see `test_build_search_rejects_unknown_family`).
    let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("mcgs without a zobrist hash must be rejected"),
    };
    assert!(err.message.contains("zobrist"));
}

#[test]
fn to_search_spec_rejects_missing_required_field() {
    let mut params = rave_params();
    params.as_object_mut().unwrap().remove("rave");
    let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("missing `rave` must be rejected"),
    };
    assert!(err.message.contains("rave"));
}

#[test]
fn to_search_spec_rejects_unknown_family() {
    let mut params = rave_params();
    params["family"] = json!("not_a_real_family");
    let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("unknown family must be rejected"),
    };
    assert!(err.message.contains("family"));
}

/// Proves `build_search` (the public entry point `GameAdapter::
/// tune_eval`'s `baseline_config` path uses) works as a
/// `strategy_tune_eval` `baseline_build` source, not just as a
/// standalone constructor -- a UCB1-built opponent played against a RAVE
/// candidate for one round.
#[test]
fn test_strategy_tune_eval_with_config_built_baseline_round_trips() {
    let baseline_params = json!({
        "family": "ucb1", "c": 1.4, "final_action": "robust_child", "q_init": "Infinity",
    });
    let outcome = strategy_tune_eval::<Nim>(
        &rave_params(),
        1,
        Some(0),
        false,
        SearchBudget::default(),
        || {
            build_search::<Nim>(&baseline_params, 0, false, &SearchBudget::default())
                .expect("baseline_params is a valid ucb1 config")
        },
        <Nim as Game>::S::default(),
        None,
        &mut |_| Ok(()),
    )
    .expect("candidate vs config-built baseline should round-trip");
    assert_eq!(outcome.wins + outcome.losses + outcome.draws, 2);
}

/// `random`/`flat_mc` are floor families reachable only via
/// `build_search`/`--baseline-config` (a ladder's floor rung), never
/// sampled as a tuner candidate -- proven by their absence from
/// `strategy_tuner_info().parameters`'s `family` choices below.
#[test]
fn test_build_search_builds_random_floor_family() {
    build_search::<Nim>(
        &json!({"family": "random", "q_init": "Infinity"}),
        0,
        false,
        &SearchBudget::default(),
    )
    .expect("random should build with just family/q_init");
}

#[test]
fn test_build_search_builds_flat_mc_floor_family() {
    build_search::<Nim>(
        &json!({"family": "flat_mc", "q_init": "Infinity"}),
        0,
        false,
        &SearchBudget::default(),
    )
    .expect("flat_mc should build with just family/q_init");
}

#[test]
fn test_strategy_tuner_info_excludes_floor_families_from_searchable_choices() {
    let tuner = strategy_tuner_info(&["strong"], 1);
    let family = tuner
        .parameters
        .iter()
        .find(|p| p.name == "family")
        .expect("family param must exist");
    let choices = family.spec["choices"]
        .as_array()
        .expect("family choices must be an array")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(
        !choices.contains(&"random") && !choices.contains(&"flat_mc"),
        "floor families must never be tuner-searchable candidates: {choices:?}"
    );
}

#[test]
fn test_build_search_rejects_unknown_family() {
    let mut params = rave_params();
    params["family"] = json!("not_a_real_family");
    // `Box<dyn Search<G>>` isn't `Debug`, so `Result::expect_err` doesn't
    // apply here -- match instead.
    let err = match build_search::<Nim>(&params, 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("unknown family must be rejected"),
    };
    assert!(err.message.contains("family"));
}

/// The parameter set each family's `make_candidate` arm actually
/// requires -- mirrors the literals already passed to
/// `assert_family_round_trips` above, plus `meta_mcts` (whose own
/// round-trip lives in `tests/stress.rs` for cost reasons, but this
/// check is pure metadata with no MCTS search, so it's cheap to include
/// here too).
///
/// Deliberately still hand-written rather than generated from
/// `register_family!`'s per-row field lists (`family_conditions()`):
/// those rows only name *which* top-level fields a family reads, not
/// concrete values, so they can't exercise the nested conditions this
/// test also needs to cover -- `rave`'s `schedule`/`rave_ucb`-gated
/// fields, `final_action: secure_child`'s `a`, `contempt: on`'s
/// `contempt_factor` -- all of which are hand-written conditions
/// `strategy_tuner_info_with_mcgs` appends precisely because they
/// depend on a *child* field's own sampled value, not on `family`
/// alone (see `family_catalog.rs`'s `register_family!` doc comment).
/// Generating a fixture from the field-name list alone would only be
/// able to assert "this field is active", which `family_conditions()`
/// already guarantees by construction -- a tautology, not a check.
/// What would still silently drift is a *new* family being added to
/// `register_family!` without a matching entry here; that's covered by
/// `test_family_required_params_covers_every_registered_family` below
/// instead, which needs no concrete values.
fn family_required_params() -> Vec<(&'static str, Value)> {
    vec![
        (
            "ucb1",
            json!({"family": "ucb1", "c": 1.4, "final_action": "robust_child"}),
        ),
        (
            "ucb1_dm",
            json!({"family": "ucb1_dm", "c": 1.4, "final_action": "max_avg"}),
        ),
        (
            "ucb1_adm",
            json!({"family": "ucb1_adm", "c": 1.4, "final_action": "max_avg"}),
        ),
        (
            "ucb1_mast",
            json!({"family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        (
            "ucb1_lgr",
            json!({"family": "ucb1_lgr", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        (
            "ucb1_lgr2",
            json!({"family": "ucb1_lgr2", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        (
            "ucb1_lgr2_mast",
            json!({"family": "ucb1_lgr2_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        (
            "ucb1_nst",
            json!({"family": "ucb1_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3, "final_action": "robust_child"}),
        ),
        (
            "ucb1_progressive_history",
            json!({"family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5, "final_action": "robust_child"}),
        ),
        (
            "ucb1_max_robust",
            json!({"family": "ucb1_max_robust", "c": 1.4}),
        ),
        (
            "amaf",
            json!({"family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "final_action": "secure_child", "a": 4.0}),
        ),
        (
            "amaf_mast",
            json!({"family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        (
            "ucb1_tuned",
            json!({"family": "ucb1_tuned", "c": 1.4, "final_action": "robust_child"}),
        ),
        (
            "ucb1_tuned_mast",
            json!({"family": "ucb1_tuned_mast", "c": 1.4, "final_action": "robust_child"}),
        ),
        (
            "ucb1_tuned_dm",
            json!({"family": "ucb1_tuned_dm", "c": 1.4, "final_action": "robust_child"}),
        ),
        (
            "ucb1_tuned_dm_mast",
            json!({"family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
        ),
        ("rave", rave_params()),
        (
            "ucb1_dm_nst",
            json!({"family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3, "final_action": "robust_child"}),
        ),
        (
            "ucb1_adm_nst",
            json!({"family": "ucb1_adm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3, "final_action": "robust_child"}),
        ),
        ("meta_mcts", json!({"family": "meta_mcts", "c": 1.4})),
        ("ucb1_pn", pn_params()),
        (
            "ucb1_pn_mast",
            json!({
                "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
                "final_action": "robust_child", "solver_loss_threshold": 5,
                "contempt": "on", "contempt_factor": -0.5,
            }),
        ),
        (
            "bayes_uct1_gaussian",
            json!({
                "family": "bayes_uct1_gaussian", "c": 1.0, "prior_variance": 1.0,
                "obs_variance": 1.0, "final_action": "robust_child",
            }),
        ),
        (
            "bayes_uct2_numeric",
            json!({
                "family": "bayes_uct2_numeric", "c": 1.0, "prior_variance": 1.0,
                "obs_variance": 1.0, "value_lo": -1.0, "value_hi": 1.0,
                "final_action": "robust_child",
            }),
        ),
    ]
}

#[test]
fn test_family_required_params_covers_every_registered_family() {
    // The gap `family_required_params()`'s own doc comment identifies:
    // a family added to `register_family!` without a matching fixture
    // here wouldn't fail anything, it would just silently skip that
    // family in `test_tuner_info_conditions_cover_every_family_param_make_candidate_needs`
    // below. Comparing the two name sets closes that gap without
    // needing `family_required_params()` to become generated data.
    let registered: std::collections::HashSet<&str> = family_choices().into_iter().collect();
    let covered: std::collections::HashSet<&str> = family_required_params()
        .iter()
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(
            registered, covered,
            "family_required_params() must have exactly one fixture per family_catalog::family_choices() entry"
        );
}

/// The fixed point of "active" parameter names implied by
/// `TunerInfo.conditions` for one fully-assigned trial config -- the
/// same any-of/if-then evaluation a tuner `ConfigSpace` performs,
/// chasing multi-level conditions (e.g. `family: rave` activates
/// `schedule`, whose own sampled value in turn activates one of
/// `rave`/`k`/`bias`).
fn active_params(tuner: &TunerInfo, chosen: &Value) -> std::collections::HashSet<String> {
    let chosen = chosen.as_object().expect("params must be an object");
    let mut active: std::collections::HashSet<String> =
        ["family", "q_init"].iter().map(|s| s.to_string()).collect();
    loop {
        let mut added = false;
        for cond in &tuner.conditions {
            let (parent, expected) = cond
                .if_
                .as_object()
                .and_then(|m| m.iter().next())
                .expect("condition `if` is a single-entry object");
            if !active.contains(parent) {
                continue;
            }
            let Some(actual) = chosen.get(parent) else {
                continue;
            };
            let matches = match expected {
                Value::Array(vals) => vals.contains(actual),
                other => other == actual,
            };
            if matches {
                for name in &cond.then {
                    if active.insert(name.clone()) {
                        added = true;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    active
}

#[test]
fn test_tuner_info_conditions_cover_every_family_param_make_candidate_needs() {
    // Regression coverage for a real bug: `make_candidate`'s `rave` arm
    // always required `epsilon`, but `strategy_tuner_info`'s conditions
    // never activated `epsilon` for `family: rave` -- so a real tuner
    // search built from this metadata could (and did) sample seemingly
    // valid `rave` configs the binary then rejected as missing a param.
    // For every family, every key its own round-trip fixture supplies
    // must be reachable as "active" from `strategy_tuner_info`'s
    // declared conditions given that exact assignment, catching any
    // future family where a hand-written fixture and the declared
    // schema's activation drift apart the same way.
    let tuner = strategy_tuner_info(&["strong"], 1);
    for (family, params) in family_required_params() {
        let active = active_params(&tuner, &params);
        for key in params.as_object().unwrap().keys() {
            assert!(
                active.contains(key),
                "family {family:?}: param {key:?} is required by make_candidate but \
                     strategy_tuner_info's conditions never mark it active for this config"
            );
        }
    }
}
