//! Versioned experiment definitions and the one-cell foreground coordinator.
//!
//! The coordinator deliberately knows nothing about DuckDB.  It validates a
//! saved definition, invokes the existing configured game comparison, and
//! translates that process's stream into the run log consumed by ingestion.

use std::io::{BufRead, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use game_host::{ConfiguredCandidateSide, ConfiguredComparisonSummary, ConfiguredMatchResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::games::find_game_binary;
use crate::log::LogRecord;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Budget {
    Iterations { value: u64 },
    TimePerMoveMs { value: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedStrategyConfig {
    pub id: String,
    pub label: String,
    pub config: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentGame {
    pub game: String,
    pub game_config: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentSpecV1 {
    pub version: u32,
    pub games: Vec<ExperimentGame>,
    pub baseline: NamedStrategyConfig,
    pub variants: Vec<NamedStrategyConfig>,
    pub budgets: Vec<Budget>,
    pub rounds_per_cell: u32,
    pub base_seed: u64,
    pub max_parallel_cells: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationField {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentValidationError {
    pub fields: Vec<ValidationField>,
}

impl std::fmt::Display for ExperimentValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, field) in self.fields.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{}: {}", field.path, field.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for ExperimentValidationError {}

impl ExperimentSpecV1 {
    /// Validate the deliberately narrow first vertical slice.
    pub fn validate_one_cell(&self) -> Result<u64, ExperimentValidationError> {
        let mut fields = Vec::new();
        let nonempty = |value: &str, path: String, fields: &mut Vec<ValidationField>| {
            if value.trim().is_empty() {
                fields.push(ValidationField {
                    path,
                    message: "must not be empty".into(),
                });
            }
        };

        if self.version != 1 {
            fields.push(ValidationField {
                path: "spec.version".into(),
                message: "must be 1".into(),
            });
        }
        if self.games.len() != 1 {
            fields.push(ValidationField {
                path: "spec.games".into(),
                message: "must contain exactly one game".into(),
            });
        }
        if self.variants.len() != 1 {
            fields.push(ValidationField {
                path: "spec.variants".into(),
                message: "must contain exactly one variant".into(),
            });
        }
        if self.budgets.len() != 1 {
            fields.push(ValidationField {
                path: "spec.budgets".into(),
                message: "must contain exactly one budget".into(),
            });
        }
        if self.rounds_per_cell == 0 {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "must be positive".into(),
            });
        }
        if self.max_parallel_cells != 1 {
            fields.push(ValidationField {
                path: "spec.max_parallel_cells".into(),
                message: "must be 1 in the one-cell slice".into(),
            });
        }

        nonempty(&self.baseline.id, "spec.baseline.id".into(), &mut fields);
        nonempty(
            &self.baseline.label,
            "spec.baseline.label".into(),
            &mut fields,
        );
        if !self.baseline.config.is_object() {
            fields.push(ValidationField {
                path: "spec.baseline.config".into(),
                message: "must be a JSON object".into(),
            });
        }
        if let Some(game) = self.games.first() {
            nonempty(&game.game, "spec.games[0].game".into(), &mut fields);
        }
        if let Some(variant) = self.variants.first() {
            nonempty(&variant.id, "spec.variants[0].id".into(), &mut fields);
            nonempty(&variant.label, "spec.variants[0].label".into(), &mut fields);
            if !variant.config.is_object() {
                fields.push(ValidationField {
                    path: "spec.variants[0].config".into(),
                    message: "must be a JSON object".into(),
                });
            }
            if variant.id == self.baseline.id {
                fields.push(ValidationField {
                    path: "spec.variants[0].id".into(),
                    message: "must differ from baseline.id".into(),
                });
            }
            if variant.label == self.baseline.label {
                fields.push(ValidationField {
                    path: "spec.variants[0].label".into(),
                    message: "must differ from baseline.label".into(),
                });
            }
        }
        if let Some(budget) = self.budgets.first() {
            let value = match budget {
                Budget::Iterations { value } | Budget::TimePerMoveMs { value } => *value,
            };
            if value == 0 {
                fields.push(ValidationField {
                    path: "spec.budgets[0].value".into(),
                    message: "must be positive".into(),
                });
            }
        }

        let planned_games = self.rounds_per_cell.checked_mul(2).map(u64::from);
        if planned_games.is_none() {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "planned game count overflows".into(),
            });
        }
        if fields.is_empty() {
            Ok(planned_games.expect("validated above"))
        } else {
            Err(ExperimentValidationError { fields })
        }
    }
}

#[derive(Debug)]
pub enum CoordinatorError {
    Io(std::io::Error),
    Json(String),
    Child(String),
    Validation(ExperimentValidationError),
}

impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "invalid game output: {error}"),
            Self::Child(error) => f.write_str(error),
            Self::Validation(error) => write!(f, "invalid experiment: {error}"),
        }
    }
}

impl std::error::Error for CoordinatorError {}
impl From<std::io::Error> for CoordinatorError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn experiment_command(
    spec: &ExperimentSpecV1,
    trace_path: Option<&Path>,
) -> Result<Vec<String>, CoordinatorError> {
    spec.validate_one_cell()
        .map_err(CoordinatorError::Validation)?;
    let game = &spec.games[0];
    let binary = find_game_binary(&game.game).ok_or_else(|| {
        CoordinatorError::Child(format!("game binary for '{}' was not found", game.game))
    })?;
    experiment_command_for_binary(spec, trace_path, &binary)
}

pub fn experiment_command_for_binary(
    spec: &ExperimentSpecV1,
    trace_path: Option<&Path>,
    binary: &Path,
) -> Result<Vec<String>, CoordinatorError> {
    spec.validate_one_cell()
        .map_err(CoordinatorError::Validation)?;
    let game = &spec.games[0];
    let variant = &spec.variants[0];
    let mut command = vec![
        binary.to_string_lossy().into_owned(),
        "compare".into(),
        "eval".into(),
        "--candidate-config".into(),
        variant.config.to_string(),
        "--baseline-config".into(),
        spec.baseline.config.to_string(),
        "--rounds".into(),
        spec.rounds_per_cell.to_string(),
        "--seed".into(),
        spec.base_seed.to_string(),
    ];
    match spec.budgets[0] {
        Budget::Iterations { value } => {
            command.extend(["--max-iterations".into(), value.to_string()]);
        }
        Budget::TimePerMoveMs { value } => {
            command.extend(["--max-time-ms".into(), value.to_string()]);
        }
    }
    if !game.game_config.is_null() {
        command.extend(["--game-config".into(), game.game_config.to_string()]);
    }
    if let Some(path) = trace_path {
        command.extend(["--trace-path".into(), path.to_string_lossy().into_owned()]);
    }
    Ok(command)
}

fn match_record(spec: &ExperimentSpecV1, result: ConfiguredMatchResult) -> LogRecord {
    let candidate = &spec.variants[0];
    let baseline = &spec.baseline;
    let (strategy_a, strategy_b, outcome, winner) = match result.candidate_side {
        ConfiguredCandidateSide::First => {
            let outcome = match result.outcome {
                game_host::ConfiguredOutcome::CandidateWin => {
                    ("win_a", Some(candidate.label.clone()))
                }
                game_host::ConfiguredOutcome::BaselineWin => {
                    ("win_b", Some(baseline.label.clone()))
                }
                game_host::ConfiguredOutcome::Draw => ("draw", None),
            };
            (
                candidate.label.clone(),
                baseline.label.clone(),
                outcome.0,
                outcome.1,
            )
        }
        ConfiguredCandidateSide::Second => {
            let outcome = match result.outcome {
                game_host::ConfiguredOutcome::CandidateWin => {
                    ("win_b", Some(candidate.label.clone()))
                }
                game_host::ConfiguredOutcome::BaselineWin => {
                    ("win_a", Some(baseline.label.clone()))
                }
                game_host::ConfiguredOutcome::Draw => ("draw", None),
            };
            (
                baseline.label.clone(),
                candidate.label.clone(),
                outcome.0,
                outcome.1,
            )
        }
    };
    let metrics = serde_json::json!({
        "round": result.round,
        "candidate_side": result.candidate_side,
        "outcome": result.outcome,
        "plies": result.plies,
        "elapsed_ms": result.elapsed_ms,
        "candidate": result.candidate,
        "baseline": result.baseline,
    });
    LogRecord::MatchResult {
        seq: result.seq,
        strategy_a,
        strategy_b,
        outcome: outcome.into(),
        winner,
        extra: None,
        cell_id: Some("cell-1".into()),
        seed: Some(result.seed),
        trace_game_seq: result.trace_game_seq,
        metrics: Some(metrics),
    }
}

/// Translate one complete child stream. Every emitted record is flushed so a
/// server tail sees progress without waiting for the cell to finish.
pub fn translate_child_output<R: BufRead, W: Write>(
    spec: &ExperimentSpecV1,
    reader: R,
    writer: &mut W,
) -> Result<(), CoordinatorError> {
    write_record(
        writer,
        &LogRecord::CellStarted {
            cell_id: "cell-1".into(),
        },
    )?;
    let mut completed_games = 0;
    let result = translate_child_output_inner(spec, reader, writer, &mut completed_games);
    if let Err(ref error) = result {
        let _ = write_record(
            writer,
            &LogRecord::CellFailed {
                cell_id: "cell-1".into(),
                completed_games,
                error: error.to_string(),
            },
        );
    }
    result
}

fn translate_child_output_inner<R: BufRead, W: Write>(
    spec: &ExperimentSpecV1,
    reader: R,
    writer: &mut W,
    completed_games: &mut u64,
) -> Result<(), CoordinatorError> {
    spec.validate_one_cell()
        .map_err(CoordinatorError::Validation)?;
    let expected_games = usize::try_from(spec.rounds_per_cell)
        .ok()
        .and_then(|rounds| rounds.checked_mul(2))
        .ok_or_else(|| CoordinatorError::Child("planned game count overflows".into()))?;
    let mut next_seq = 1_u64;
    let mut summary: Option<ConfiguredComparisonSummary> = None;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| CoordinatorError::Json(error.to_string()))?;
        let record_type = value.get("type").and_then(Value::as_str).unwrap_or("");
        match record_type {
            "configured_match_result" => {
                if summary.is_some() {
                    return Err(CoordinatorError::Child("match after summary".into()));
                }
                let result: ConfiguredMatchResult = serde_json::from_value(value)
                    .map_err(|error| CoordinatorError::Json(error.to_string()))?;
                if result.seq != next_seq || result.seq > expected_games as u64 {
                    return Err(CoordinatorError::Child(
                        "duplicate or out-of-order match sequence".into(),
                    ));
                }
                write_record(writer, &match_record(spec, result))?;
                *completed_games += 1;
                next_seq += 1;
            }
            "configured_comparison_summary" => {
                if summary.is_some() {
                    return Err(CoordinatorError::Child("duplicate summary".into()));
                }
                let parsed: ConfiguredComparisonSummary = serde_json::from_value(value)
                    .map_err(|error| CoordinatorError::Json(error.to_string()))?;
                summary = Some(parsed);
            }
            _ => {
                return Err(CoordinatorError::Child(
                    "unexpected child record type".into(),
                ))
            }
        }
    }
    let summary =
        summary.ok_or_else(|| CoordinatorError::Child("child did not emit a summary".into()))?;
    if next_seq != expected_games as u64 + 1 || summary.games != expected_games as u32 {
        return Err(CoordinatorError::Child(
            "child output did not contain the planned games".into(),
        ));
    }
    if summary.wins + summary.losses + summary.draws != summary.games {
        return Err(CoordinatorError::Child(
            "child summary counts do not add up".into(),
        ));
    }
    write_record(
        writer,
        &LogRecord::CellFinished {
            cell_id: "cell-1".into(),
            completed_games: expected_games as u64,
        },
    )?;
    Ok(())
}

fn write_record<W: Write>(writer: &mut W, record: &LogRecord) -> Result<(), CoordinatorError> {
    writeln!(writer, "{}", record.to_json_line())?;
    writer.flush()?;
    Ok(())
}

pub fn run_experiment<W: Write>(
    spec: &ExperimentSpecV1,
    trace_path: Option<&Path>,
    writer: &mut W,
) -> Result<(), CoordinatorError> {
    spec.validate_one_cell()
        .map_err(CoordinatorError::Validation)?;
    write_record(
        writer,
        &LogRecord::CellStarted {
            cell_id: "cell-1".into(),
        },
    )?;
    let command = match experiment_command(spec, trace_path) {
        Ok(command) => command,
        Err(error) => {
            let message = error.to_string();
            let _ = write_record(
                writer,
                &LogRecord::CellFailed {
                    cell_id: "cell-1".into(),
                    completed_games: 0,
                    error: message.clone(),
                },
            );
            eprintln!("{message}");
            return Err(error);
        }
    };
    let mut child = match Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let message = format!("failed to spawn game comparison: {error}");
            write_record(
                writer,
                &LogRecord::CellFailed {
                    cell_id: "cell-1".into(),
                    completed_games: 0,
                    error: message.clone(),
                },
            )?;
            eprintln!("{message}");
            return Err(CoordinatorError::Child(message));
        }
    };
    let stdout = child.stdout.take().expect("piped stdout");
    let mut deferred = DeferredFinishWriter::new(writer);
    let mut completed_games = 0;
    let result = translate_child_output_inner(
        spec,
        std::io::BufReader::new(stdout),
        &mut deferred,
        &mut completed_games,
    );
    deferred.flush()?;
    let status = child.wait()?;
    if !status.success() {
        if result.is_ok() {
            let message = format!("game comparison exited with {status}");
            deferred.discard_finish();
            write_record(
                writer,
                &LogRecord::CellFailed {
                    cell_id: "cell-1".into(),
                    completed_games,
                    error: message.clone(),
                },
            )?;
            eprintln!("{message}");
            return Err(CoordinatorError::Child(message));
        }
    }
    if let Err(ref error) = result {
        let _ = write_record(
            writer,
            &LogRecord::CellFailed {
                cell_id: "cell-1".into(),
                completed_games,
                error: error.to_string(),
            },
        );
        eprintln!("{}", error);
    } else {
        deferred.finish()?;
    }
    result
}

/// Hold the success marker until the child has also reported a zero exit
/// status. Match records still pass through immediately, so a live run keeps
/// showing progress while a failing child cannot leave a false success event
/// behind.
struct DeferredFinishWriter<'a, W: Write> {
    inner: &'a mut W,
    buffered: Vec<u8>,
    finish: Option<Vec<u8>>,
    completed_games: u64,
}

impl<'a, W: Write> DeferredFinishWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            buffered: Vec::new(),
            finish: None,
            completed_games: 0,
        }
    }

    fn process_lines(&mut self) -> std::io::Result<()> {
        while let Some(index) = self.buffered.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buffered.drain(..=index).collect();
            let is_finish = serde_json::from_slice::<Value>(&line)
                .ok()
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(Value::as_str)
                        .map(|kind| kind == "cell_finished")
                })
                .unwrap_or(false);
            if is_finish {
                self.finish = Some(line);
            } else {
                if serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("type")
                            .and_then(Value::as_str)
                            .map(|kind| kind == "match_result")
                    })
                    .unwrap_or(false)
                {
                    self.completed_games += 1;
                }
                self.inner.write_all(&line)?;
                self.inner.flush()?;
            }
        }
        Ok(())
    }

    fn discard_finish(&mut self) {
        self.finish = None;
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.process_lines()?;
        if let Some(line) = self.finish.take() {
            self.inner.write_all(&line)?;
            self.inner.flush()?;
        }
        Ok(())
    }
}

impl<W: Write> Write for DeferredFinishWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.buffered.extend_from_slice(bytes);
        self.process_lines()?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.process_lines()?;
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn spec(game_config: Value, budget: Budget) -> ExperimentSpecV1 {
        ExperimentSpecV1 {
            version: 1,
            games: vec![ExperimentGame {
                game: "nim".into(),
                game_config,
            }],
            baseline: NamedStrategyConfig {
                id: "base".into(),
                label: "Baseline".into(),
                config: serde_json::json!({"family":"ucb1"}),
            },
            variants: vec![NamedStrategyConfig {
                id: "candidate".into(),
                label: "Candidate".into(),
                config: serde_json::json!({"family":"ucb1"}),
            }],
            budgets: vec![budget],
            rounds_per_cell: 1,
            base_seed: 42,
            max_parallel_cells: 1,
        }
    }

    #[test]
    fn validation_rejects_wrong_cardinality() {
        let mut value = spec(Value::Null, Budget::Iterations { value: 1 });
        value.games.clear();
        let error = value.validate_one_cell().unwrap_err();
        assert!(error.fields.iter().any(|field| field.path == "spec.games"));
    }

    #[test]
    fn command_forwards_budget_and_game_config_rules() {
        let iterations = experiment_command_for_binary(
            &spec(Value::Null, Budget::Iterations { value: 4 }),
            None,
            Path::new("game-nim"),
        )
        .unwrap();
        assert!(iterations
            .windows(2)
            .any(|pair| pair == ["--max-iterations", "4"]));
        assert!(!iterations.contains(&"--game-config".into()));
        let time = experiment_command_for_binary(
            &spec(serde_json::json!({}), Budget::TimePerMoveMs { value: 5 }),
            None,
            Path::new("game-nim"),
        )
        .unwrap();
        assert!(time.windows(2).any(|pair| pair == ["--max-time-ms", "5"]));
        assert!(time.contains(&"--game-config".into()));
    }

    #[test]
    fn translates_fixture_and_maps_sides() {
        let input = r#"{"type":"configured_match_result","seq":1,"round":1,"seed":42,"candidate_side":"first","outcome":"candidate_win","trace_game_seq":99,"plies":2,"elapsed_ms":3,"candidate":{"iterations_total":4,"iterations_first_half":1,"move_time_ms":2},"baseline":{"iterations_total":5,"iterations_first_half":2,"move_time_ms":3}}
{"type":"configured_match_result","seq":2,"round":1,"seed":42,"candidate_side":"second","outcome":"baseline_win","trace_game_seq":100,"plies":2,"elapsed_ms":3,"candidate":{"iterations_total":4,"iterations_first_half":1,"move_time_ms":2},"baseline":{"iterations_total":5,"iterations_first_half":2,"move_time_ms":3}}
{"type":"configured_comparison_summary","games":2,"wins":1,"losses":1,"draws":0}"#;
        let mut output = Vec::new();
        let mut value = spec(Value::Null, Budget::Iterations { value: 1 });
        value.rounds_per_cell = 1;
        // The fixture has two games, so use two rounds.
        value.rounds_per_cell = 1;
        // One round is two games in the contract.
        translate_child_output(&value, Cursor::new(input), &mut output).unwrap();
        let lines: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["type"], "cell_started");
        assert_eq!(lines[1]["strategy_a"], "Candidate");
        assert_eq!(lines[2]["strategy_a"], "Baseline");
        assert_eq!(lines[3]["type"], "cell_finished");
    }

    #[test]
    fn malformed_or_out_of_order_output_fails_without_a_success_marker() {
        let input = r#"{"type":"configured_match_result","seq":2,"round":1,"seed":42,"candidate_side":"first","outcome":"draw","trace_game_seq":99,"plies":2,"elapsed_ms":3,"candidate":{"iterations_total":1,"iterations_first_half":1,"move_time_ms":1},"baseline":{"iterations_total":1,"iterations_first_half":1,"move_time_ms":1}}"#;
        let mut output = Vec::new();
        let error = translate_child_output(
            &spec(Value::Null, Budget::Iterations { value: 1 }),
            Cursor::new(input),
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("sequence"));
        let lines: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["type"], "cell_started");
        assert_eq!(lines[1]["type"], "cell_failed");
        assert!(lines.iter().all(|line| line["type"] != "cell_finished"));
    }
}
