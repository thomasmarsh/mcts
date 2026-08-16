//! Versioned experiment definitions and the foreground experiment coordinator.
//!
//! The coordinator deliberately knows nothing about DuckDB.  It validates a
//! saved definition, invokes the existing configured game comparison, and
//! translates that process's stream into the run log consumed by ingestion.

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use game_host::{
    derive_seed, ConfiguredCandidateSide, ConfiguredComparisonSummary, ConfiguredMatchResult,
};
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

pub const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentCellPlan {
    pub ordinal: u64,
    pub cell_id: String,
    pub game: String,
    pub game_config: Value,
    pub variant_id: String,
    pub variant_label: String,
    pub candidate_config: Value,
    pub baseline_id: String,
    pub baseline_label: String,
    pub baseline_config: Value,
    pub budget: Budget,
    pub rounds: u32,
    pub planned_games: u64,
    pub cell_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpandedExperimentPlan {
    pub cells: Vec<ExperimentCellPlan>,
    pub total_planned_games: u64,
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
    pub fn expand(&self) -> Result<ExpandedExperimentPlan, ExperimentValidationError> {
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
        if self.games.is_empty() {
            fields.push(ValidationField {
                path: "spec.games".into(),
                message: "must contain at least one game".into(),
            });
        }
        if self.variants.is_empty() {
            fields.push(ValidationField {
                path: "spec.variants".into(),
                message: "must contain at least one variant".into(),
            });
        }
        if self.budgets.is_empty() {
            fields.push(ValidationField {
                path: "spec.budgets".into(),
                message: "must contain at least one budget".into(),
            });
        }
        if self.rounds_per_cell == 0 {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "must be positive".into(),
            });
        }
        if self.max_parallel_cells == 0 {
            fields.push(ValidationField {
                path: "spec.max_parallel_cells".into(),
                message: "must be positive".into(),
            });
        }
        if u64::from(self.rounds_per_cell) > JS_MAX_SAFE_INTEGER {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "must be a JavaScript-safe integer".into(),
            });
        }
        if u64::from(self.max_parallel_cells) > JS_MAX_SAFE_INTEGER {
            fields.push(ValidationField {
                path: "spec.max_parallel_cells".into(),
                message: "must be a JavaScript-safe integer".into(),
            });
        }
        if self.base_seed > JS_MAX_SAFE_INTEGER {
            fields.push(ValidationField {
                path: "spec.base_seed".into(),
                message: "must be a JavaScript-safe integer".into(),
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
        let mut game_names = HashSet::new();
        for (index, game) in self.games.iter().enumerate() {
            let path = format!("spec.games[{index}].game");
            nonempty(&game.game, path.clone(), &mut fields);
            let key = game.game.trim().to_owned();
            if !key.is_empty() && !game_names.insert(key) {
                fields.push(ValidationField {
                    path,
                    message: "duplicate game".into(),
                });
            }
        }
        let mut strategy_ids = HashSet::new();
        let mut strategy_labels = HashSet::new();
        let baseline_id = self.baseline.id.trim().to_owned();
        let baseline_label = self.baseline.label.trim().to_owned();
        if !baseline_id.is_empty() && !strategy_ids.insert(baseline_id) {
            fields.push(ValidationField {
                path: "spec.baseline.id".into(),
                message: "duplicate strategy ID".into(),
            });
        }
        if !baseline_label.is_empty() && !strategy_labels.insert(baseline_label) {
            fields.push(ValidationField {
                path: "spec.baseline.label".into(),
                message: "duplicate strategy label".into(),
            });
        }
        for (index, variant) in self.variants.iter().enumerate() {
            let id_path = format!("spec.variants[{index}].id");
            let label_path = format!("spec.variants[{index}].label");
            nonempty(&variant.id, id_path.clone(), &mut fields);
            nonempty(&variant.label, label_path.clone(), &mut fields);
            let id = variant.id.trim().to_owned();
            if !id.is_empty() && !strategy_ids.insert(id) {
                fields.push(ValidationField {
                    path: id_path,
                    message: "duplicate strategy ID".into(),
                });
            }
            let label = variant.label.trim().to_owned();
            if !label.is_empty() && !strategy_labels.insert(label) {
                fields.push(ValidationField {
                    path: label_path,
                    message: "duplicate strategy label".into(),
                });
            }
            if !variant.config.is_object() {
                fields.push(ValidationField {
                    path: format!("spec.variants[{index}].config"),
                    message: "must be a JSON object".into(),
                });
            }
        }
        let mut budgets = HashSet::new();
        for (index, budget) in self.budgets.iter().enumerate() {
            let value = match budget {
                Budget::Iterations { value } | Budget::TimePerMoveMs { value } => *value,
            };
            if value == 0 {
                fields.push(ValidationField {
                    path: format!("spec.budgets[{index}].value"),
                    message: "must be positive".into(),
                });
            }
            if value > JS_MAX_SAFE_INTEGER {
                fields.push(ValidationField {
                    path: format!("spec.budgets[{index}].value"),
                    message: "must be a JavaScript-safe integer".into(),
                });
            }
            let key = match budget {
                Budget::Iterations { .. } => "iterations",
                Budget::TimePerMoveMs { .. } => "time_per_move_ms",
            };
            if !budgets.insert((key, value)) {
                fields.push(ValidationField {
                    path: format!("spec.budgets[{index}]"),
                    message: "duplicate budget".into(),
                });
            }
        }

        let (planned_games, cell_count, total_planned_games) = checked_plan_counts(
            self.games.len(),
            self.budgets.len(),
            self.variants.len(),
            self.rounds_per_cell,
        );
        if planned_games.is_none() {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "planned game count overflows".into(),
            });
        }
        if cell_count.is_none() {
            fields.push(ValidationField {
                path: "spec.games".into(),
                message: "cell count overflows".into(),
            });
        }
        if total_planned_games.is_none()
            || total_planned_games.is_some_and(|n| n > JS_MAX_SAFE_INTEGER)
        {
            fields.push(ValidationField {
                path: "spec.rounds_per_cell".into(),
                message: "total planned game count is not representable".into(),
            });
        }
        if !fields.is_empty() {
            return Err(ExperimentValidationError { fields });
        }

        let planned_games = planned_games.expect("checked above");
        let cell_count = cell_count.expect("checked above");
        let total_planned_games = total_planned_games.expect("checked above");
        let width = (cell_count.max(1) as u64).to_string().len().max(6);
        let mut cells = Vec::new();
        cells
            .try_reserve(cell_count)
            .map_err(|_| ExperimentValidationError {
                fields: vec![ValidationField {
                    path: "spec".into(),
                    message: "expanded plan allocation failed".into(),
                }],
            })?;
        let mut ordinal = 0_u64;
        for game in &self.games {
            for budget in &self.budgets {
                for variant in &self.variants {
                    let cell_id =
                        format!("cell-{number:0width$}", number = ordinal + 1, width = width);
                    cells.push(ExperimentCellPlan {
                        ordinal,
                        cell_id,
                        game: game.game.clone(),
                        game_config: game.game_config.clone(),
                        variant_id: variant.id.clone(),
                        variant_label: variant.label.clone(),
                        candidate_config: variant.config.clone(),
                        baseline_id: self.baseline.id.clone(),
                        baseline_label: self.baseline.label.clone(),
                        baseline_config: self.baseline.config.clone(),
                        budget: budget.clone(),
                        rounds: self.rounds_per_cell,
                        planned_games,
                        cell_seed: derive_seed(self.base_seed, ordinal),
                    });
                    ordinal += 1;
                }
            }
        }
        Ok(ExpandedExperimentPlan {
            cells,
            total_planned_games,
        })
    }
}

fn checked_plan_counts(
    game_count: usize,
    budget_count: usize,
    variant_count: usize,
    rounds: u32,
) -> (Option<u64>, Option<usize>, Option<u64>) {
    let planned_games = rounds.checked_mul(2).map(u64::from);
    let cell_count = game_count
        .checked_mul(budget_count)
        .and_then(|count| count.checked_mul(variant_count));
    let total_planned_games = cell_count
        .and_then(|count| u64::try_from(count).ok())
        .and_then(|count| planned_games.and_then(|games| count.checked_mul(games)));
    (planned_games, cell_count, total_planned_games)
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

pub fn cell_command_for_binary(
    plan: &ExperimentCellPlan,
    trace_path: Option<&Path>,
    binary: &Path,
) -> Vec<String> {
    let mut command = vec![
        binary.to_string_lossy().into_owned(),
        "compare".into(),
        "eval".into(),
        "--candidate-config".into(),
        plan.candidate_config.to_string(),
        "--baseline-config".into(),
        plan.baseline_config.to_string(),
        "--rounds".into(),
        plan.rounds.to_string(),
        "--seed".into(),
        plan.cell_seed.to_string(),
    ];
    match plan.budget {
        Budget::Iterations { value } => {
            command.extend(["--max-iterations".into(), value.to_string()]);
        }
        Budget::TimePerMoveMs { value } => {
            command.extend(["--max-time-ms".into(), value.to_string()]);
        }
    }
    if !plan.game_config.is_null() {
        command.extend(["--game-config".into(), plan.game_config.to_string()]);
    }
    if let Some(path) = trace_path {
        command.extend(["--trace-path".into(), path.to_string_lossy().into_owned()]);
    }
    command
}

fn match_record(plan: &ExperimentCellPlan, result: ConfiguredMatchResult) -> LogRecord {
    let candidate_label = &plan.variant_label;
    let baseline_label = &plan.baseline_label;
    let (strategy_a, strategy_b, outcome, winner) = match result.candidate_side {
        ConfiguredCandidateSide::First => {
            let outcome = match result.outcome {
                game_host::ConfiguredOutcome::CandidateWin => {
                    ("win_a", Some(candidate_label.clone()))
                }
                game_host::ConfiguredOutcome::BaselineWin => {
                    ("win_b", Some(baseline_label.clone()))
                }
                game_host::ConfiguredOutcome::Draw => ("draw", None),
            };
            (
                candidate_label.clone(),
                baseline_label.clone(),
                outcome.0,
                outcome.1,
            )
        }
        ConfiguredCandidateSide::Second => {
            let outcome = match result.outcome {
                game_host::ConfiguredOutcome::CandidateWin => {
                    ("win_b", Some(candidate_label.clone()))
                }
                game_host::ConfiguredOutcome::BaselineWin => {
                    ("win_a", Some(baseline_label.clone()))
                }
                game_host::ConfiguredOutcome::Draw => ("draw", None),
            };
            (
                baseline_label.clone(),
                candidate_label.clone(),
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
        cell_id: Some(plan.cell_id.clone()),
        seed: Some(result.seed),
        trace_game_seq: result.trace_game_seq,
        metrics: Some(metrics),
    }
}

/// Translate one complete child stream for compatibility callers that provide
/// a single-cell spec and a synchronous writer.
pub fn translate_child_output<R: BufRead, W: Write>(
    spec: &ExperimentSpecV1,
    reader: R,
    writer: &mut W,
) -> Result<(), CoordinatorError> {
    let plan = spec
        .expand()
        .map_err(CoordinatorError::Validation)?
        .cells
        .into_iter()
        .next()
        .ok_or_else(|| CoordinatorError::Child("experiment has no cells".into()))?;
    write_record(
        writer,
        &LogRecord::CellStarted {
            cell_id: plan.cell_id.clone(),
        },
    )?;
    let mut completed_games = 0;
    let result = translate_child_output_inner(
        &plan,
        reader,
        &mut |record| write_record(writer, &record),
        &mut completed_games,
    );
    if let Err(ref error) = result {
        let _ = write_record(
            writer,
            &LogRecord::CellFailed {
                cell_id: plan.cell_id,
                completed_games,
                error: error.to_string(),
            },
        );
    } else {
        write_record(
            writer,
            &LogRecord::CellFinished {
                cell_id: plan.cell_id,
                completed_games,
            },
        )?;
    }
    result
}

fn translate_child_output_inner<R: BufRead, F>(
    plan: &ExperimentCellPlan,
    reader: R,
    emit: &mut F,
    completed_games: &mut u64,
) -> Result<(), CoordinatorError>
where
    F: FnMut(LogRecord) -> Result<(), CoordinatorError>,
{
    let expected_games = usize::try_from(plan.planned_games)
        .map_err(|_| CoordinatorError::Child("planned game count overflows".into()))?;
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
                let expected_round = ((next_seq - 1) / 2) as u32 + 1;
                let expected_seed =
                    game_host::derive_seed(plan.cell_seed, u64::from(expected_round - 1));
                if result.round != expected_round
                    || result.seed != expected_seed
                    || result.candidate_side
                        != if next_seq % 2 == 1 {
                            ConfiguredCandidateSide::First
                        } else {
                            ConfiguredCandidateSide::Second
                        }
                {
                    return Err(CoordinatorError::Child(
                        "child match metadata does not match the cell plan".into(),
                    ));
                }
                emit(match_record(plan, result))?;
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
    Ok(())
}

fn write_record<W: Write>(writer: &mut W, record: &LogRecord) -> Result<(), CoordinatorError> {
    writeln!(writer, "{}", record.to_json_line())?;
    writer.flush()?;
    Ok(())
}

fn send_record(
    sender: &mpsc::Sender<WorkerEvent>,
    record: LogRecord,
) -> Result<(), CoordinatorError> {
    sender
        .send(WorkerEvent::Record(record))
        .map_err(|_| CoordinatorError::Child("coordinator output receiver closed".into()))
}

fn run_cell_process(
    plan: ExperimentCellPlan,
    trace_path: Option<PathBuf>,
    sender: &mpsc::Sender<WorkerEvent>,
) -> Result<(), CoordinatorError> {
    send_record(
        sender,
        LogRecord::CellStarted {
            cell_id: plan.cell_id.clone(),
        },
    )?;
    let binary = match find_game_binary(&plan.game) {
        Some(binary) => binary,
        None => {
            send_record(
                sender,
                LogRecord::CellFailed {
                    cell_id: plan.cell_id,
                    completed_games: 0,
                    error: format!("game binary for '{}' was not found", plan.game),
                },
            )?;
            return Ok(());
        }
    };
    let command = cell_command_for_binary(&plan, trace_path.as_deref(), &binary);
    let mut child = match Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            send_record(
                sender,
                LogRecord::CellFailed {
                    cell_id: plan.cell_id,
                    completed_games: 0,
                    error: format!("failed to spawn game comparison: {error}"),
                },
            )?;
            return Ok(());
        }
    };
    let stdout = child.stdout.take().expect("piped stdout");
    let mut completed_games = 0;
    let parse_result = translate_child_output_inner(
        &plan,
        std::io::BufReader::new(stdout),
        &mut |record| send_record(sender, record),
        &mut completed_games,
    );
    let status = child.wait()?;
    if let Err(error) = parse_result {
        send_record(
            sender,
            LogRecord::CellFailed {
                cell_id: plan.cell_id,
                completed_games,
                error: error.to_string(),
            },
        )?;
    } else if !status.success() {
        send_record(
            sender,
            LogRecord::CellFailed {
                cell_id: plan.cell_id,
                completed_games,
                error: format!("game comparison exited with {status}"),
            },
        )?;
    } else {
        send_record(
            sender,
            LogRecord::CellFinished {
                cell_id: plan.cell_id,
                completed_games,
            },
        )?;
    }
    Ok(())
}

pub enum WorkerEvent {
    Record(LogRecord),
    Infrastructure(CoordinatorError),
}

pub fn run_cells_with_runner<W, F>(
    plan: &ExpandedExperimentPlan,
    max_parallel: usize,
    writer: &mut W,
    runner: F,
) -> Result<(), CoordinatorError>
where
    W: Write,
    F: Fn(ExperimentCellPlan, &mpsc::Sender<WorkerEvent>) -> Result<(), CoordinatorError>
        + Send
        + Sync,
{
    let queue = Arc::new(Mutex::new(VecDeque::from(plan.cells.clone())));
    let runner = Arc::new(runner);
    let mut next_seq = 1_u64;
    let mut infrastructure_error = None;
    thread::scope(|scope| {
        let worker_count = max_parallel.max(1).min(plan.cells.len().max(1));
        let (event_sender, event_receiver) = mpsc::channel::<WorkerEvent>();
        for _ in 0..worker_count {
            let queue = Arc::clone(&queue);
            let runner = Arc::clone(&runner);
            let event_sender = event_sender.clone();
            let event_sender_for_runner = event_sender.clone();
            scope.spawn(move || loop {
                let next = queue.lock().unwrap().pop_front();
                let Some(cell) = next else { break };
                let result = runner(cell, &event_sender_for_runner);
                if let Err(error) = result {
                    let _ = event_sender.send(WorkerEvent::Infrastructure(error));
                    break;
                }
            });
        }
        drop(event_sender);
        for event in event_receiver {
            match event {
                WorkerEvent::Record(record) => {
                    if let Err(error) = write_global_record(writer, record, &mut next_seq) {
                        infrastructure_error = Some(error);
                        break;
                    }
                }
                WorkerEvent::Infrastructure(error) => {
                    infrastructure_error = Some(error);
                    break;
                }
            }
        }
    });
    infrastructure_error.map_or(Ok(()), Err)
}

fn write_global_record<W: Write>(
    writer: &mut W,
    mut record: LogRecord,
    next_seq: &mut u64,
) -> Result<(), CoordinatorError> {
    if let LogRecord::MatchResult { ref mut seq, .. } = record {
        *seq = *next_seq;
        *next_seq = next_seq
            .checked_add(1)
            .ok_or_else(|| CoordinatorError::Child("run-global match sequence overflow".into()))?;
    }
    write_record(writer, &record)
}

pub fn run_experiment<W: Write>(
    spec: &ExperimentSpecV1,
    trace_path: Option<&Path>,
    writer: &mut W,
) -> Result<(), CoordinatorError> {
    let trace_path = trace_path.map(Path::to_path_buf);
    run_experiment_with_runner(spec, writer, move |cell, sender| {
        run_cell_process(cell, trace_path.clone(), sender)
    })
}

pub fn run_experiment_with_runner<W, F>(
    spec: &ExperimentSpecV1,
    writer: &mut W,
    runner: F,
) -> Result<(), CoordinatorError>
where
    W: Write,
    F: Fn(ExperimentCellPlan, &mpsc::Sender<WorkerEvent>) -> Result<(), CoordinatorError>
        + Send
        + Sync,
{
    let plan = spec.expand().map_err(CoordinatorError::Validation)?;
    let max_parallel = usize::try_from(spec.max_parallel_cells)
        .map_err(|_| CoordinatorError::Child("parallelism does not fit usize".into()))?;
    run_cells_with_runner(&plan, max_parallel, writer, runner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

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
        let error = value.expand().unwrap_err();
        assert!(error.fields.iter().any(|field| field.path == "spec.games"));
    }

    fn grid_spec() -> ExperimentSpecV1 {
        ExperimentSpecV1 {
            version: 1,
            games: vec![
                ExperimentGame {
                    game: "game-a".into(),
                    game_config: serde_json::json!({"size": 5}),
                },
                ExperimentGame {
                    game: "game-b".into(),
                    game_config: Value::Null,
                },
            ],
            baseline: NamedStrategyConfig {
                id: "baseline".into(),
                label: "Baseline".into(),
                config: serde_json::json!({"family": "ucb1"}),
            },
            variants: vec![
                NamedStrategyConfig {
                    id: "v1".into(),
                    label: "Variant 1".into(),
                    config: serde_json::json!({"family": "rave"}),
                },
                NamedStrategyConfig {
                    id: "v2".into(),
                    label: "Variant 2".into(),
                    config: serde_json::json!({"family": "ucb1"}),
                },
                NamedStrategyConfig {
                    id: "v3".into(),
                    label: "Variant 3".into(),
                    config: serde_json::json!({"family": "random"}),
                },
            ],
            budgets: vec![
                Budget::Iterations { value: 10 },
                Budget::TimePerMoveMs { value: 20 },
            ],
            rounds_per_cell: 2,
            base_seed: 42,
            max_parallel_cells: 2,
        }
    }

    #[test]
    fn expansion_is_ordered_and_uses_stable_ids_counts_and_seeds() {
        let plan = grid_spec().expand().unwrap();
        assert_eq!(plan.cells.len(), 12);
        assert_eq!(plan.total_planned_games, 48);
        assert_eq!(plan.cells[0].cell_id, "cell-000001");
        assert_eq!(plan.cells[0].game, "game-a");
        assert_eq!(plan.cells[0].budget, Budget::Iterations { value: 10 });
        assert_eq!(plan.cells[0].variant_id, "v1");
        assert_eq!(plan.cells[1].variant_id, "v2");
        assert_eq!(plan.cells[3].budget, Budget::TimePerMoveMs { value: 20 });
        assert_eq!(plan.cells[6].game, "game-b");
        assert_eq!(plan.cells[11].cell_id, "cell-000012");
        assert_eq!(plan.cells[0].cell_seed, 7_294_331_206_661_666);
        assert_eq!(plan.cells[1].cell_seed, 6_529_064_058_449_557);
        assert_eq!(
            derive_seed(plan.cells[0].cell_seed, 0),
            8_360_105_604_253_074
        );
        assert_eq!(
            derive_seed(plan.cells[0].cell_seed, 1),
            5_482_876_856_761_435
        );
    }

    #[test]
    fn one_cell_specs_remain_compatible() {
        let plan = spec(Value::Null, Budget::Iterations { value: 1 })
            .expand()
            .unwrap();
        assert_eq!(plan.cells.len(), 1);
        assert_eq!(plan.cells[0].cell_id, "cell-000001");
        assert_eq!(plan.total_planned_games, 2);
    }

    #[test]
    fn validation_reports_indexed_duplicates_and_all_structural_errors() {
        let mut value = grid_spec();
        value.version = 2;
        value.games[0].game = " game-a ".into();
        value.games[1].game = "game-a".into();
        value.baseline.id = " same ".into();
        value.baseline.label = " same label ".into();
        value.variants[0].id = "same".into();
        value.variants[1].label = "same label".into();
        value.variants[2].config = Value::Null;
        value.budgets.push(Budget::Iterations { value: 10 });
        value.rounds_per_cell = 0;
        value.max_parallel_cells = 0;
        value.base_seed = JS_MAX_SAFE_INTEGER + 1;
        let error = value.expand().unwrap_err();
        let paths: HashSet<&str> = error
            .fields
            .iter()
            .map(|field| field.path.as_str())
            .collect();
        for path in [
            "spec.version",
            "spec.games[1].game",
            "spec.variants[0].id",
            "spec.variants[1].label",
            "spec.variants[2].config",
            "spec.budgets[2]",
            "spec.rounds_per_cell",
            "spec.max_parallel_cells",
            "spec.base_seed",
        ] {
            assert!(
                paths.contains(path),
                "missing validation path {path}: {paths:?}"
            );
        }
    }

    #[test]
    fn checked_counts_report_each_overflow_without_panicking() {
        assert_eq!(checked_plan_counts(usize::MAX, 2, 2, 1).1, None);
        assert_eq!(checked_plan_counts(1, 1, 1, u32::MAX).0, None);
        assert_eq!(checked_plan_counts(usize::MAX / 2 + 1, 2, 1, 1).2, None);
        assert_eq!(
            checked_plan_counts(1, 1, 1, u32::MAX / 2).2,
            Some(u64::from(u32::MAX - 1))
        );
    }

    #[test]
    fn command_forwards_budget_and_game_config_rules() {
        let iterations_spec = spec(Value::Null, Budget::Iterations { value: 4 });
        let iterations_plan = iterations_spec.expand().unwrap();
        let iterations =
            cell_command_for_binary(&iterations_plan.cells[0], None, Path::new("game-nim"));
        assert!(iterations
            .windows(2)
            .any(|pair| pair == ["--max-iterations", "4"]));
        assert!(!iterations.contains(&"--game-config".into()));
        let time_spec = spec(serde_json::json!({}), Budget::TimePerMoveMs { value: 5 });
        let time_plan = time_spec.expand().unwrap();
        let time = cell_command_for_binary(&time_plan.cells[0], None, Path::new("game-nim"));
        assert!(time.windows(2).any(|pair| pair == ["--max-time-ms", "5"]));
        assert!(time.contains(&"--game-config".into()));
    }

    #[test]
    fn translates_fixture_and_maps_sides() {
        let input = r#"{"type":"configured_match_result","seq":1,"round":1,"seed":8360105604253074,"candidate_side":"first","outcome":"candidate_win","trace_game_seq":99,"plies":2,"elapsed_ms":3,"candidate":{"iterations_total":4,"iterations_first_half":1,"move_time_ms":2},"baseline":{"iterations_total":5,"iterations_first_half":2,"move_time_ms":3}}
{"type":"configured_match_result","seq":2,"round":1,"seed":8360105604253074,"candidate_side":"second","outcome":"baseline_win","trace_game_seq":100,"plies":2,"elapsed_ms":3,"candidate":{"iterations_total":4,"iterations_first_half":1,"move_time_ms":2},"baseline":{"iterations_total":5,"iterations_first_half":2,"move_time_ms":3}}
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

    fn fixture_match(cell_id: &str) -> LogRecord {
        LogRecord::MatchResult {
            seq: 1,
            strategy_a: "Candidate".into(),
            strategy_b: "Baseline".into(),
            outcome: "win_a".into(),
            winner: Some("Candidate".into()),
            extra: None,
            cell_id: Some(cell_id.into()),
            seed: Some(1),
            trace_game_seq: None,
            metrics: Some(serde_json::json!({"outcome": "candidate_win"})),
        }
    }

    #[test]
    fn scheduler_assigns_unique_run_global_sequences_to_interleaved_cells() {
        let plan = grid_spec().expand().unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));
        let mut output = Vec::new();
        run_cells_with_runner(&plan, 2, &mut output, {
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            let barrier = Arc::clone(&barrier);
            move |cell, sender| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                barrier.wait();
                sender
                    .send(WorkerEvent::Record(LogRecord::CellStarted {
                        cell_id: cell.cell_id.clone(),
                    }))
                    .unwrap();
                sender
                    .send(WorkerEvent::Record(fixture_match(&cell.cell_id)))
                    .unwrap();
                sender
                    .send(WorkerEvent::Record(LogRecord::CellFinished {
                        cell_id: cell.cell_id,
                        completed_games: 1,
                    }))
                    .unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        let records: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let matches: Vec<&Value> = records
            .iter()
            .filter(|record| record["type"] == "match_result")
            .collect();
        assert_eq!(matches.len(), plan.cells.len());
        for (index, record) in matches.iter().enumerate() {
            assert_eq!(record["seq"], (index + 1) as u64);
            assert!(record["cell_id"].as_str().unwrap().starts_with("cell-"));
        }
    }

    #[test]
    fn scheduler_respects_sequential_limit_without_timing_dependencies() {
        let mut spec = grid_spec();
        spec.variants.truncate(2);
        let plan = spec.expand().unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        run_cells_with_runner(&plan, 1, &mut Vec::new(), {
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            move |cell, sender| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                sender
                    .send(WorkerEvent::Record(LogRecord::CellFinished {
                        cell_id: cell.cell_id,
                        completed_games: 0,
                    }))
                    .unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduler_keeps_partial_matches_and_continues_after_cell_failure() {
        let plan = grid_spec().expand().unwrap();
        let failed_cell = plan.cells[0].cell_id.clone();
        let failed_cell_for_runner = failed_cell.clone();
        let mut output = Vec::new();
        run_cells_with_runner(&plan, 2, &mut output, move |cell, sender| {
            sender
                .send(WorkerEvent::Record(LogRecord::CellStarted {
                    cell_id: cell.cell_id.clone(),
                }))
                .unwrap();
            sender
                .send(WorkerEvent::Record(fixture_match(&cell.cell_id)))
                .unwrap();
            if cell.cell_id == failed_cell_for_runner {
                sender
                    .send(WorkerEvent::Record(LogRecord::CellFailed {
                        cell_id: cell.cell_id,
                        completed_games: 1,
                        error: "fixture failure".into(),
                    }))
                    .unwrap();
            } else {
                sender
                    .send(WorkerEvent::Record(LogRecord::CellFinished {
                        cell_id: cell.cell_id,
                        completed_games: 1,
                    }))
                    .unwrap();
            }
            Ok(())
        })
        .unwrap();
        let records: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records
                .iter()
                .filter(|record| record["type"] == "cell_failed")
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record["type"] == "cell_finished")
                .count(),
            plan.cells.len() - 1
        );
        assert!(records.iter().any(|record| {
            record["type"] == "match_result" && record["cell_id"] == failed_cell
        }));
    }

    #[test]
    fn coordinator_infrastructure_failure_has_no_false_success_marker() {
        let spec = grid_spec();
        let mut output = Vec::new();
        let error = run_experiment_with_runner(&spec, &mut output, |_cell, _sender| {
            Err(CoordinatorError::Child("scheduler failed".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("scheduler failed"));
        assert!(output.is_empty());
    }
}
