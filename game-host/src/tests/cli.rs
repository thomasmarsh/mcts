use super::support::{parse_response, FakeAdapter, ValidationCounts, VALIDATION_COUNTS};
use crate::*;
use serde_json::Value;
use std::io::{self, Cursor, Write};

struct TunableFakeAdapter;

impl GameAdapter for TunableFakeAdapter {
    fn kind(&self) -> &'static str {
        "tunable-fake"
    }
    fn label(&self) -> &'static str {
        "Tunable Fake Game"
    }
    fn description(&self) -> &'static str {
        "A fake adapter that supports tuning, for testing `tune` subcommands"
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        VALIDATION_COUNTS.with(|counts| counts.borrow_mut().new_state += 1);
        if config.get("invalid").and_then(Value::as_str) == Some("game") {
            return Err(HostError::bad_request("game rejected"));
        }
        Ok(serde_json::json!({}))
    }
    fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
        Ok(vec![])
    }
    fn apply(&self, state: &Value, _mv: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn view(&self, _state: &Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({"terminal": true}))
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        vec![]
    }
    fn ai_move(
        &self,
        _state: &Value,
        _preset: &str,
        _custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        VALIDATION_COUNTS.with(|counts| counts.borrow_mut().plays += 1);
        Err(HostError::not_found("not implemented in test fake"))
    }
    fn analyze(
        &self,
        _state: &Value,
        _preset: &str,
        _custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        Err(HostError::not_found("not implemented in test fake"))
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(TunerInfo {
            id: "test".into(),
            baselines: vec!["baseline".into()],
            eval_rounds: 5,
            parameters: vec![TunerParameter {
                name: "c".into(),
                spec: serde_json::json!({"type": "float", "bounds": [0, 3], "default": 1.4}),
            }],
            conditions: vec![],
            game_config: serde_json::json!({}),
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        baseline: Option<String>,
        baseline_config: Option<Value>,
        game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        VALIDATION_COUNTS.with(|counts| counts.borrow_mut().builds += 1);
        match params.get("invalid").and_then(Value::as_str) {
            Some("candidate") => return Err(HostError::bad_request("candidate rejected")),
            Some("baseline") => return Err(HostError::bad_request("baseline rejected")),
            _ => {}
        }
        let _ = (max_iterations, max_time_ms, trace_path);
        for round in 1..=rounds {
            on_game(ConfiguredMatchResult {
                record_type: "configured_match_result".into(),
                seq: (round * 2 - 1) as u64,
                round,
                seed: seed.unwrap_or(0),
                candidate_side: ConfiguredCandidateSide::First,
                outcome: ConfiguredOutcome::CandidateWin,
                trace_game_seq: None,
                plies: 0,
                elapsed_ms: 0,
                candidate: ConfiguredStrategyMetrics::default(),
                baseline: ConfiguredStrategyMetrics::default(),
            })?;
            on_game(ConfiguredMatchResult {
                record_type: "configured_match_result".into(),
                seq: (round * 2) as u64,
                round,
                seed: seed.unwrap_or(0),
                candidate_side: ConfiguredCandidateSide::Second,
                outcome: ConfiguredOutcome::BaselineWin,
                trace_game_seq: None,
                plies: 0,
                elapsed_ms: 0,
                candidate: ConfiguredStrategyMetrics::default(),
                baseline: ConfiguredStrategyMetrics::default(),
            })?;
        }
        Ok(serde_json::json!({
            "cost": 0.25,
            "params": params,
            "rounds": rounds,
            "seed": seed,
            "baseline": baseline,
            "baseline_config": baseline_config,
            "game_config": game_config,
            "wins": rounds,
            "losses": rounds,
            "draws": 0,
        }))
    }
}

pub(super) fn run_cli_capture_with<A: GameAdapter>(
    adapter: A,
    args: &[&str],
    stdin: &str,
) -> (String, i32) {
    let args = args.iter().map(|s| s.to_string());
    let input = Cursor::new(stdin.to_owned());
    let mut output = Cursor::new(Vec::new());
    let code = run_cli_with(args, input, &mut output, adapter);
    (String::from_utf8(output.into_inner()).unwrap(), code)
}

pub(super) fn run_cli_capture(args: &[&str], stdin: &str) -> (String, i32) {
    run_cli_capture_with(FakeAdapter, args, stdin)
}

#[test]
fn test_run_cli_no_args_drives_stdin_stdout_loop_unchanged() {
    let (out, code) = run_cli_capture(&[], "{\"id\":1,\"method\":\"kind\",\"params\":{}}\n");
    assert_eq!(code, 0);
    let resp = parse_response(out.lines().next().unwrap());
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 1);
            assert_eq!(result, "fake");
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_run_cli_describe_matches_adapter_fields() {
    let (out, code) = run_cli_capture(&["describe"], "");
    assert_eq!(code, 0);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 1);
    let description: GameDescription = serde_json::from_str(lines[0]).unwrap();

    let adapter = FakeAdapter;
    assert_eq!(description.kind, adapter.kind());
    assert_eq!(description.label, adapter.label());
    assert_eq!(description.description, adapter.description());
    assert_eq!(description.default_config, adapter.default_config());
    assert_eq!(description.ai_presets.len(), adapter.ai_presets().len());
    assert_eq!(description.ai_presets[0].id, adapter.ai_presets()[0].id);
    assert!(description.tuning.is_none());
}

#[test]
fn test_run_cli_describe_folds_in_tuning_when_present() {
    let (out, code) = run_cli_capture_with(TunableFakeAdapter, &["describe"], "");
    assert_eq!(code, 0);
    let description: GameDescription = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    let tuning = description.tuning.expect("expected tuning metadata");
    assert_eq!(tuning.id, "test");
    assert_eq!(tuning.eval_rounds, 5);
}

#[test]
fn test_run_cli_unknown_subcommand_falls_back_to_stdin_stdout_loop() {
    let (out, code) = run_cli_capture(
        &["some-unknown-flag"],
        "{\"id\":7,\"method\":\"kind\",\"params\":{}}\n",
    );
    assert_eq!(code, 0);
    let resp = parse_response(out.lines().next().unwrap());
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 7);
            assert_eq!(result, "fake");
        }
        _ => panic!("expected success response, describe-only args must not error"),
    }
}

#[test]
fn test_run_cli_tune_describe_unsupported_when_tuner_none() {
    let (out, code) = run_cli_capture(&["tune", "describe"], "");
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_tune_describe_prints_tuner_info() {
    let (out, code) = run_cli_capture_with(TunableFakeAdapter, &["tune", "describe"], "");
    assert_eq!(code, 0);
    let info: TunerInfo = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(info.id, "test");
    assert_eq!(info.baselines, vec!["baseline".to_string()]);
    assert_eq!(info.eval_rounds, 5);
    assert_eq!(info.parameters.len(), 1);
    assert_eq!(info.parameters[0].name, "c");
}

#[test]
fn test_run_cli_compare_describe_matches_tune_describe() {
    let (tune, tune_code) = run_cli_capture_with(TunableFakeAdapter, &["tune", "describe"], "");
    let (compare, compare_code) =
        run_cli_capture_with(TunableFakeAdapter, &["compare", "describe"], "");
    assert_eq!(tune_code, 0);
    assert_eq!(compare_code, 0);
    assert_eq!(compare, tune);
}

#[test]
fn test_run_cli_compare_eval_streams_games_then_summary() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "compare",
            "eval",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
            "--rounds",
            "1",
            "--seed",
            "42",
            "--max-iterations",
            "1",
        ],
        "",
    );
    assert_eq!(code, 0);
    let lines: Vec<Value> = out
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    let games = &lines[..2];
    assert!(games
        .iter()
        .all(|game| game["type"] == "configured_match_result"));
    assert_eq!(games[0]["seq"], 1);
    assert_eq!(games[1]["seq"], 2);
    assert_eq!(games[0]["candidate_side"], "first");
    assert_eq!(games[1]["candidate_side"], "second");
    assert!(games.iter().all(|game| game["round"] == 1));
    assert!(games.iter().all(|game| game["seed"] == derive_seed(42, 0)));
    assert_eq!(lines[2]["type"], "configured_comparison_summary");
    assert_eq!(lines[2]["games"], 2);
    assert_eq!(lines[2]["wins"], 1);
    assert_eq!(lines[2]["losses"], 1);
    assert_eq!(lines[2]["draws"], 0);
    assert_eq!(
        lines[2]["games"].as_u64().unwrap(),
        lines[2]["wins"].as_u64().unwrap()
            + lines[2]["losses"].as_u64().unwrap()
            + lines[2]["draws"].as_u64().unwrap()
    );
}

#[test]
fn compare_eval_uses_stable_round_seeds_and_run_sequences() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "compare",
            "eval",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
            "--rounds",
            "2",
            "--seed",
            "42",
            "--max-iterations",
            "1",
        ],
        "",
    );
    assert_eq!(code, 0);
    let lines: Vec<Value> = out
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 5);
    for (index, line) in lines[..4].iter().enumerate() {
        assert_eq!(line["seq"], (index + 1) as u64);
        assert_eq!(line["round"], (index / 2 + 1) as u64);
        assert_eq!(line["seed"], derive_seed(42, (index / 2) as u64));
        assert_eq!(line["seed"], lines[index ^ 1]["seed"]);
    }
    assert_eq!(lines[0]["seed"], derive_seed(42, 0));
    assert_ne!(lines[0]["seed"], lines[2]["seed"]);
    assert_eq!(lines[4]["games"], 4);
}

#[test]
fn compare_validate_returns_structured_success_without_matches() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "compare",
            "validate",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
        ],
        "",
    );
    assert_eq!(code, 0);
    let response: Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(response["valid"], true);
    assert_eq!(response["errors"], serde_json::json!([]));
}

fn validation_counts() -> ValidationCounts {
    VALIDATION_COUNTS.with(|counts| std::mem::take(&mut *counts.borrow_mut()))
}

#[test]
fn compare_validate_checks_game_and_strategies_without_playing() {
    let _ = validation_counts();
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "compare",
            "validate",
            "--candidate-config",
            "{}",
            "--candidate-config",
            "{}",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
        ],
        "",
    );
    assert_eq!(code, 0);
    let response: Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(response["valid"], true);
    let counts = validation_counts();
    assert_eq!(counts.new_state, 1);
    assert_eq!(counts.builds, 4);
    assert_eq!(counts.plays, 0);
}

#[test]
fn compare_validate_attributes_game_candidate_and_baseline_errors() {
    let _ = validation_counts();
    let cases = [
        (
            vec![
                "compare",
                "validate",
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
                "--game-config",
                r#"{"invalid":"game"}"#,
            ],
            "game_config",
            "game rejected",
        ),
        (
            vec![
                "compare",
                "validate",
                "--candidate-config",
                r#"{"invalid":"candidate"}"#,
                "--baseline-config",
                "{}",
            ],
            "candidate_config",
            "candidate rejected",
        ),
        (
            vec![
                "compare",
                "validate",
                "--candidate-config",
                "{}",
                "--baseline-config",
                r#"{"invalid":"baseline"}"#,
            ],
            "baseline_config",
            "baseline rejected",
        ),
    ];

    for (args, field, message) in cases {
        let (out, code) = run_cli_capture_with(TunableFakeAdapter, &args, "");
        assert_eq!(code, 1);
        let response: Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(response["valid"], false);
        assert_eq!(response["errors"][0]["field"], field);
        assert_eq!(response["errors"][0]["message"], message);
        assert!(!out.contains("configured_match_result"));
        assert!(!out.contains("configured_comparison_summary"));
        assert_eq!(validation_counts().plays, 0);
    }

    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "compare",
            "validate",
            "--candidate-config",
            "{}",
            "--candidate-config",
            r#"{"invalid":"candidate"}"#,
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
        ],
        "",
    );
    assert_eq!(code, 1);
    let response: Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(response["errors"][0]["field"], "candidate_config");
    assert_eq!(response["errors"][0]["candidate_index"], 1);
    assert_eq!(validation_counts().plays, 0);
}

#[test]
fn tune_eval_rejects_zero_rounds_before_calling_adapter() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &["tune", "eval", "--config", "{}", "--rounds", "0"],
        "",
    );
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_compare_eval_rejects_invalid_invocations_before_play() {
    let base = [
        "compare",
        "eval",
        "--candidate-config",
        "{}",
        "--baseline-config",
        "{}",
        "--rounds",
        "1",
        "--seed",
        "42",
    ];
    let mut invalid = vec![base.to_vec()];
    let mut zero_rounds = base.to_vec();
    zero_rounds[7] = "0";
    zero_rounds.extend(["--max-iterations", "1"]);
    invalid.push(zero_rounds);
    let mut malformed_candidate = base.to_vec();
    malformed_candidate[3] = "not json";
    malformed_candidate.extend(["--max-iterations", "1"]);
    invalid.push(malformed_candidate);
    let mut missing_value = base.to_vec();
    missing_value.push("--max-iterations");
    invalid.push(missing_value);
    for extra in [
        vec!["--max-iterations", "0"],
        vec!["--max-time-ms", "0"],
        vec!["--max-iterations", "1", "--max-time-ms", "1"],
        vec!["--max-iterations", "1", "--unknown"],
    ] {
        let mut args = base.to_vec();
        args.extend(extra);
        invalid.push(args);
    }
    for extra in invalid {
        let (out, code) = run_cli_capture_with(TunableFakeAdapter, &extra, "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }
}

struct FailingWriter;

impl Write for FailingWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("write failed"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("flush failed"))
    }
}

#[test]
fn test_run_cli_compare_eval_sink_failure_stops_without_summary() {
    let args = [
        "compare",
        "eval",
        "--candidate-config",
        "{}",
        "--baseline-config",
        "{}",
        "--rounds",
        "2",
        "--seed",
        "42",
        "--max-iterations",
        "1",
    ]
    .into_iter()
    .map(str::to_owned);
    let code = run_cli_with(args, Cursor::new(""), FailingWriter, TunableFakeAdapter);
    assert_eq!(code, 1);
}

#[test]
fn test_run_cli_tune_eval_prints_result_verbatim() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "tune",
            "eval",
            "--config",
            r#"{"rave":700,"c":0.3}"#,
            "--rounds",
            "3",
            "--seed",
            "42",
        ],
        "",
    );
    assert_eq!(code, 0);
    let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(result["cost"], 0.25);
    assert_eq!(result["params"]["rave"], 700);
    assert_eq!(result["rounds"], 3);
    assert_eq!(result["seed"], 42);
}

#[test]
fn test_run_cli_tune_eval_missing_rounds_errors() {
    let (out, code) =
        run_cli_capture_with(TunableFakeAdapter, &["tune", "eval", "--config", "{}"], "");
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_tune_eval_baseline_config_threads_through() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "tune",
            "eval",
            "--config",
            "{}",
            "--rounds",
            "1",
            "--baseline-config",
            r#"{"family":"ucb1","c":1.4}"#,
        ],
        "",
    );
    assert_eq!(code, 0);
    let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert!(result["baseline"].is_null());
    assert_eq!(result["baseline_config"]["family"], "ucb1");
    assert_eq!(result["baseline_config"]["c"], 1.4);
}

#[test]
fn test_run_cli_tune_eval_rejects_both_baseline_and_baseline_config() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "tune",
            "eval",
            "--config",
            "{}",
            "--rounds",
            "1",
            "--baseline",
            "strong",
            "--baseline-config",
            "{}",
        ],
        "",
    );
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_tune_eval_rejects_both_max_iterations_and_max_time_ms() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "tune",
            "eval",
            "--config",
            "{}",
            "--rounds",
            "1",
            "--max-iterations",
            "100",
            "--max-time-ms",
            "1000",
        ],
        "",
    );
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_tune_eval_rejects_invalid_baseline_config_json() {
    let (out, code) = run_cli_capture_with(
        TunableFakeAdapter,
        &[
            "tune",
            "eval",
            "--config",
            "{}",
            "--rounds",
            "1",
            "--baseline-config",
            "not json",
        ],
        "",
    );
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_tune_with_no_further_args_falls_back_to_stdin_stdout_loop() {
    let (out, code) = run_cli_capture(&["tune"], "{\"id\":9,\"method\":\"kind\",\"params\":{}}\n");
    assert_eq!(code, 0);
    let resp = parse_response(out.lines().next().unwrap());
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 9);
            assert_eq!(result, "fake");
        }
        _ => panic!("expected success response"),
    }
}

// -----------------------------------------------------------------------
// `book` subcommand tests
// -----------------------------------------------------------------------
