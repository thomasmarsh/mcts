use crate::family_catalog::{
    dispatch_family, family_choices, DirectFamily, FamilySpec, TrialParams,
};
use crate::*;
use game_host::{ConfiguredCandidateSide, ConfiguredOutcome, HostError, TunerInfo};
use game_nim::Nim;
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::select::{RaveSchedule, RaveUcb};
use mcts::strategies::mcts::{
    node::QInit, simulate, strategy, GraphSearch, GraphStats, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;
use serde_json::{json, Value};

fn nim_action_value(state: &<Nim as Game>::S, action: &<Nim as Game>::A) -> Value {
    Value::String(Nim::notation(state, action))
}

fn nim_trace_state_value(_: &<Nim as Game>::S) -> Value {
    json!({"canonical": "nim"})
}

fn nim_trace_move_value(_: &<Nim as Game>::S, action: &<Nim as Game>::A) -> Option<Value> {
    Some(serde_json::to_value(action).expect("Nim action always serializes"))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct TraceState(u8);

impl std::fmt::Display for TraceState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "trace state {}", self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize)]
struct TraceAction;

#[derive(Clone, Debug)]
enum TracePlayer {
    First,
    Second,
}

impl PlayerIndex for TracePlayer {
    fn to_index(&self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }
}

#[derive(Clone)]
struct TraceGame;

impl Game for TraceGame {
    type S = TraceState;
    type A = TraceAction;
    type P = TracePlayer;

    fn apply(state: Self::S, _: &Self::A) -> Self::S {
        TraceState(state.0 + 1)
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        if state.0 < 2 {
            actions.push(TraceAction);
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.0 == 2
    }

    fn winner(_: &Self::S) -> Option<Self::P> {
        None
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        if state.0 == 0 {
            TracePlayer::First
        } else {
            TracePlayer::Second
        }
    }
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
fn one_iteration_mcts_search_selects_after_the_root_playout() {
    let state = <Nim as Game>::S::default();
    let mut search =
        TreeSearch::<Nim, strategy::Ucb1>::new().config(SearchConfig::new().max_iterations(1));

    let (selected_action, report) = choose_action_with_report(&mut search, &state, |action| {
        nim_action_value(&state, action)
    });

    let mut legal_actions = Vec::new();
    Nim::generate_actions(&state, &mut legal_actions);
    assert!(legal_actions.contains(&selected_action));
    assert_eq!(report.status, game_host::SearchReportStatus::Available);
    assert_eq!(report.completed_iterations, 1);
    assert_eq!(
        report.selected_action,
        Some(nim_action_value(&state, &selected_action))
    );
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

#[test]
fn renderer_trace_uses_canonical_values_and_final_reports_for_both_seats() {
    let path = std::env::temp_dir().join(format!(
        "mcts_tune_renderer_trace_{}_{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let params = json!({"family": "random", "q_init": "Infinity"});
    let budget = SearchBudget {
        max_iterations: Some(1),
        ..Default::default()
    };
    let mut records = Vec::new();
    strategy_tune_eval::<TraceGame>(
        &params,
        1,
        Some(7),
        false,
        budget,
        || Box::new(mcts::strategies::random::Random::<TraceGame>::new()),
        TraceState::default(),
        |state| json!({"position": state.0}),
        |state, _| Some(json!({"kind": "advance", "from": state.0})),
        Some(&path),
        Some(17),
        &mut |record| {
            records.push(record);
            Ok(())
        },
    )
    .unwrap();

    let rows: Vec<Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].trace_game_seq, Some(17));
    assert_eq!(records[1].trace_game_seq, Some(18));
    assert_eq!(rows.len(), 6);
    for (game, rows) in rows.chunks_exact(3).enumerate() {
        assert!(rows.iter().all(|row| row["game_seq"] == 17 + game as u64));
        assert!(rows.iter().all(|row| row["state"].is_object()));
        assert!(rows.iter().all(|row| row["trace_schema_version"] == 1));
        assert_eq!(rows[0]["ply"], 0);
        assert!(rows[0]["mv"].is_null());
        assert!(rows[0]["player"].is_null());
        assert!(rows[0]["search"].is_null());
        for row in &rows[1..] {
            assert!(row["mv"].is_object());
            assert!(row["search"].is_object());
            assert_eq!(
                row["state"]["position"],
                row["mv"]["from"].as_u64().unwrap() + 1
            );
        }
        let players = if game == 0 {
            ["candidate", "baseline"]
        } else {
            ["baseline", "candidate"]
        };
        for (row, player) in rows[1..].iter().zip(players) {
            assert_eq!(row["player"], player);
            assert_eq!(row["search"]["status"], "unavailable");
            assert_eq!(
                row["search"]["reason"], "strategy_unsupported",
                "{player} report must be explicit rather than null"
            );
        }
    }
    std::fs::remove_file(path).unwrap();
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
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

/// Unlike every other family exercised here, `random` resolves to
/// `FamilySpec::Direct` and is built by `direct_search::build_direct`
/// rather than `config_ir::build_search` -- this proves it still round-trips
/// through the exact same `strategy_tune_eval` pipeline every `Compose`
/// family does, with no special-cased caller-side handling.
#[test]
fn test_family_random_round_trips() {
    assert_family_round_trips(json!({"family": "random"}));
}

/// Like `random`, `flat_mc` resolves to `FamilySpec::Direct`. This exercises
/// its default win-rate move rule; `test_family_flat_mc_ucb1_round_trips`
/// below covers the UCB1 branch `flat_mc_selection` also allows.
#[test]
fn test_family_flat_mc_round_trips() {
    assert_family_round_trips(json!({
        "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
        "flat_mc_selection": "win_rate",
    }));
}

#[test]
fn test_family_flat_mc_ucb1_round_trips() {
    assert_family_round_trips(json!({
        "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
        "flat_mc_selection": "ucb1", "c": 1.4,
    }));
}

/// Like `random`/`flat_mc`, `negamax` resolves to `FamilySpec::Direct`.
/// `max_depth`/`table_bits` are kept small: `Nim`'s heaps allow arbitrary
/// splits, so its game tree is not the trivially shallow one a fixed-heap
/// take-only Nim would be, and an unbounded iterative-deepening search
/// (negamax's own default `max_depth: 64`) would make this a real
/// multi-second search rather than a fast round-trip check -- see
/// `AGENTS.md`'s "keep `cargo test --lib` fast" rule.
#[test]
fn test_family_negamax_round_trips() {
    assert_family_round_trips(json!({
        "family": "negamax", "max_depth": 3, "table_bits": 10,
        "negamax_replacement": "depth_preferred",
        "principal_variation_search": true, "history_heuristic": true,
        "singular_extension": true, "countermove_heuristic": true,
        "negamax_aspiration": "off",
    }));
}

#[test]
fn test_family_negamax_aspiration_round_trips() {
    assert_family_round_trips(json!({
        "family": "negamax", "max_depth": 3, "table_bits": 10,
        "negamax_replacement": "two_tier",
        "principal_variation_search": true, "history_heuristic": true,
        "singular_extension": true, "countermove_heuristic": true,
        "negamax_aspiration": "on", "aspiration_window": 50,
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

// `meta_mcts`'s round trip is proven in `examples/tune-stress.rs` instead of here:
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
fn trial_params_deserialize_rejects_missing_family() {
    let err = serde_json::from_value::<TrialParams>(json!({"q_init": "Infinity"})).unwrap_err();
    assert!(err.to_string().contains("family"), "{err}");
}

/// `q_init` is required for a `Compose` family (rejected by
/// `compose_settings`, not `TrialParams::deserialize` -- see
/// `to_search_spec_rejects_missing_q_init` below) but meaningless to a
/// `Direct` one, so unlike `family` it's `Option` at the deserialize level:
/// a trial naming `"ucb1"` with no `q_init` at all still parses.
#[test]
fn trial_params_deserialize_treats_absent_and_null_optional_fields_alike() {
    let absent = trial(json!({"family": "ucb1", "q_init": "Infinity"}));
    assert_eq!(absent.c, None);
    assert_eq!(absent.mcgs, None);

    let null = trial(json!({"family": "ucb1", "q_init": "Infinity", "c": null, "mcgs": null}));
    assert_eq!(null.c, None);
    assert_eq!(null.mcgs, None);

    let no_q_init = trial(json!({"family": "ucb1"}));
    assert_eq!(no_q_init.q_init, None);
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

/// `q_init` is `Option` at the `TrialParams` deserialize level (see
/// `trial_params_deserialize_treats_absent_and_null_optional_fields_alike`)
/// since a `Direct` family never needs it, but a `Compose` family still
/// requires a value -- `compose_settings` (not `TrialParams::deserialize`)
/// is what rejects it missing.
#[test]
fn to_search_spec_rejects_missing_q_init() {
    let params = json!({"family": "ucb1", "c": 1.4, "final_action": "robust_child"});
    let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("missing `q_init` must be rejected for a Compose family"),
    };
    assert!(err.message.contains("q_init"));
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

#[test]
fn dispatch_family_resolves_random_to_the_direct_variant() {
    let trial = trial(json!({"family": "random", "q_init": "Infinity"}));
    match dispatch_family("random", &trial) {
        Ok(FamilySpec::Direct(DirectFamily::Random)) => {}
        Ok(FamilySpec::Compose(_)) => panic!("random must not resolve to a Compose family"),
        Ok(FamilySpec::Direct(_)) => panic!("random must resolve to DirectFamily::Random"),
        Err(e) => panic!("random must dispatch: {}", e.message),
    }
}

#[test]
fn dispatch_family_resolves_flat_mc_to_the_direct_variant() {
    let trial = trial(json!({
        "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
        "flat_mc_selection": "win_rate",
    }));
    match dispatch_family("flat_mc", &trial) {
        Ok(FamilySpec::Direct(DirectFamily::FlatMc {
            samples_per_move: 20,
            max_rollout_depth: 50,
            ucb1: None,
        })) => {}
        Ok(_) => panic!("flat_mc must resolve to DirectFamily::FlatMc with the parsed fields"),
        Err(e) => panic!("flat_mc must dispatch: {}", e.message),
    }
}

#[test]
fn dispatch_family_resolves_flat_mc_ucb1_selection() {
    let trial = trial(json!({
        "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
        "flat_mc_selection": "ucb1", "c": 1.4,
    }));
    match dispatch_family("flat_mc", &trial) {
        Ok(FamilySpec::Direct(DirectFamily::FlatMc { ucb1: Some(c), .. })) => {
            assert_eq!(c, 1.4);
        }
        Ok(_) => panic!("flat_mc with ucb1 selection must carry the c value"),
        Err(e) => panic!("flat_mc must dispatch: {}", e.message),
    }
}

#[test]
fn dispatch_family_resolves_negamax_to_the_direct_variant() {
    let trial = trial(json!({
        "family": "negamax", "max_depth": 3, "table_bits": 10,
        "negamax_replacement": "always",
        "principal_variation_search": false, "history_heuristic": false,
        "singular_extension": false, "countermove_heuristic": false,
        "negamax_aspiration": "off",
    }));
    match dispatch_family("negamax", &trial) {
        Ok(FamilySpec::Direct(DirectFamily::Negamax {
            max_depth: 3,
            table_bits: 10,
            replacement: mcts::strategies::negamax::Replacement::Always,
            aspiration_window: None,
            principal_variation_search: false,
            history_heuristic: false,
            singular_extension: false,
            countermove_heuristic: false,
        })) => {}
        Ok(_) => panic!("negamax must resolve to DirectFamily::Negamax with the parsed fields"),
        Err(e) => panic!("negamax must dispatch: {}", e.message),
    }
}

#[test]
fn dispatch_family_resolves_negamax_aspiration_window() {
    let trial = trial(json!({
        "family": "negamax", "max_depth": 3, "table_bits": 10,
        "negamax_replacement": "depth_preferred",
        "principal_variation_search": true, "history_heuristic": true,
        "singular_extension": true, "countermove_heuristic": true,
        "negamax_aspiration": "on", "aspiration_window": 75,
    }));
    match dispatch_family("negamax", &trial) {
        Ok(FamilySpec::Direct(DirectFamily::Negamax {
            aspiration_window: Some(window),
            ..
        })) => {
            assert_eq!(window, 75);
        }
        Ok(_) => panic!("negamax with aspiration on must carry the window value"),
        Err(e) => panic!("negamax must dispatch: {}", e.message),
    }
}

/// `random` resolves to `FamilySpec::Direct`, which has no
/// `config_ir::SearchSpec` representation -- `to_search_spec` (unlike
/// `make_candidate`, which builds it via `direct_search::build_direct`)
/// rejects it rather than silently producing a nonsensical spec.
#[test]
fn to_search_spec_rejects_direct_family() {
    let params = json!({"family": "random", "q_init": "Infinity"});
    let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
        Err(e) => e,
        Ok(_) => panic!("a Direct family must be rejected by to_search_spec"),
    };
    assert!(err.message.contains("random"));
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
        nim_trace_state_value,
        nim_trace_move_value,
        None,
        None,
        &mut |_| Ok(()),
    )
    .expect("candidate vs config-built baseline should round-trip");
    assert_eq!(outcome.wins + outcome.losses + outcome.draws, 2);
}

/// `random` is an ordinary `family_catalog` row (a `DirectFamily`, built by
/// `direct_search::build_direct` rather than `config_ir::build_search`, but
/// otherwise reachable exactly like any other family) -- `build_search`
/// resolves it from just `family`/`q_init`, the same as an MCTS family with
/// no other required params (`ucb1_max_robust`, `meta_mcts`).
#[test]
fn test_build_search_builds_random_family() {
    build_search::<Nim>(
        &json!({"family": "random", "q_init": "Infinity"}),
        0,
        false,
        &SearchBudget::default(),
    )
    .expect("random should build with just family/q_init");
}

/// `flat_mc` is an ordinary `family_catalog` row like `random`, just one
/// with its own tunable fields (`samples_per_move`/`max_rollout_depth`/
/// `flat_mc_selection`) rather than none.
#[test]
fn test_build_search_builds_flat_mc_family() {
    build_search::<Nim>(
        &json!({
            "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
            "flat_mc_selection": "win_rate",
        }),
        0,
        false,
        &SearchBudget::default(),
    )
    .expect("flat_mc should build from its own required params");
}

/// `negamax` is a `DirectFamily` row too, built by
/// `direct_search::build_direct` rather than `config_ir::build_search`, and
/// (unlike `random`/`flat_mc`) actually reads `SearchBudget` (`threads`/
/// `max_time`) -- this only proves it builds, so `max_depth` doesn't need to
/// be tight here (`choose_action` is never called).
#[test]
fn test_build_search_builds_negamax_family() {
    build_search::<Nim>(
        &json!({
            "family": "negamax", "max_depth": 8, "table_bits": 16,
            "negamax_replacement": "depth_preferred",
            "principal_variation_search": true, "history_heuristic": true,
            "singular_extension": true, "countermove_heuristic": true,
            "negamax_aspiration": "off",
        }),
        0,
        false,
        &SearchBudget::default(),
    )
    .expect("negamax should build from its own required params");
}

#[test]
fn test_strategy_tuner_info_includes_every_family_choice() {
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
    for name in family_choices() {
        assert!(
            choices.contains(&name),
            "family {name:?} must be a tuner-searchable choice: {choices:?}"
        );
    }
}

/// A `Direct` family (`random`, `flat_mc`, `negamax`) never reads `q_init`,
/// so Optuna shouldn't waste trials sampling it across `q_init`'s five
/// otherwise-equivalent values for that family -- `active_params` (the same
/// fixed-point evaluation
/// `test_tuner_info_conditions_cover_every_family_param_make_candidate_needs`
/// uses) must not mark `q_init` active for any of them, but must for an
/// ordinary `Compose` family.
#[test]
fn test_tuner_info_gates_q_init_off_for_direct_families_only() {
    let tuner = strategy_tuner_info(&["strong"], 1);
    let random_active = active_params(&tuner, &json!({"family": "random"}));
    assert!(
        !random_active.contains("q_init"),
        "q_init must not be active for a Direct family: {random_active:?}"
    );
    let flat_mc_active = active_params(&tuner, &json!({"family": "flat_mc"}));
    assert!(
        !flat_mc_active.contains("q_init"),
        "q_init must not be active for a Direct family: {flat_mc_active:?}"
    );
    let negamax_active = active_params(&tuner, &json!({"family": "negamax"}));
    assert!(
        !negamax_active.contains("q_init"),
        "q_init must not be active for a Direct family: {negamax_active:?}"
    );
    let ucb1_active = active_params(&tuner, &json!({"family": "ucb1"}));
    assert!(
        ucb1_active.contains("q_init"),
        "q_init must be active for a Compose family: {ucb1_active:?}"
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
/// round-trip lives in `examples/tune-stress.rs` for cost reasons, but this
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
            "ucb_v",
            json!({"family": "ucb_v", "c": 1.4, "final_action": "robust_child"}),
        ),
        (
            "kl_ucb",
            json!({"family": "kl_ucb", "c": 1.4, "final_action": "robust_child"}),
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
        (
            "power_uct",
            json!({
                "family": "power_uct", "c": 1.4, "p": 4.0, "alpha": 0.5,
                "final_action": "robust_child",
            }),
        ),
        (
            "td_uct",
            json!({
                "family": "td_uct", "c": 1.4, "lambda": 0.8, "td_max_child": 0,
                "final_action": "robust_child",
            }),
        ),
        (
            "ments",
            json!({
                "family": "ments", "tau": 1.0, "epsilon": 0.1,
                "final_action": "robust_child",
            }),
        ),
        (
            "grill_act",
            json!({
                "family": "grill_act", "c": 1.4, "final_action": "robust_child",
            }),
        ),
        (
            "score_bounded_uct",
            json!({
                "family": "score_bounded_uct", "c": 1.4, "gamma": 0.1, "delta": 0.1,
                "final_action": "robust_child", "solver_loss_threshold": 5,
                "contempt": "off",
            }),
        ),
        (
            "gpn",
            json!({
                "family": "gpn", "c": 1.4, "c_pn": 1.0, "gpn_bias": "max",
                "final_action": "robust_child", "solver_loss_threshold": 5,
                "contempt": "off",
            }),
        ),
        ("random", json!({"family": "random"})),
        (
            "flat_mc",
            json!({
                "family": "flat_mc", "samples_per_move": 20, "max_rollout_depth": 50,
                "flat_mc_selection": "win_rate",
            }),
        ),
        (
            "negamax",
            json!({
                "family": "negamax", "max_depth": 8, "table_bits": 16,
                "negamax_replacement": "depth_preferred",
                "principal_variation_search": true, "history_heuristic": true,
                "singular_extension": true, "countermove_heuristic": true,
                "negamax_aspiration": "on", "aspiration_window": 50,
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
/// `rave`/`k`/`bias`). `family` is the only unconditional root -- `q_init`
/// is itself gated on `family` naming a `Compose` row (see
/// `strategy_tuner_info_with_mcgs`), so it has to reach "active" through
/// the same condition-chasing loop below, not be seeded here.
fn active_params(tuner: &TunerInfo, chosen: &Value) -> std::collections::HashSet<String> {
    let chosen = chosen.as_object().expect("params must be an object");
    let mut active: std::collections::HashSet<String> =
        ["family"].iter().map(|s| s.to_string()).collect();
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
