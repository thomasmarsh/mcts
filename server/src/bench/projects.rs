#![allow(unused_imports)]
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{HeaderValue, Method, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};

use game_host::TunerInfo;
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::identity;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::{commands::*, types::*};
// Route handlers
// ---------------------------------------------------------------------------

pub(crate) fn generated_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}-{nanos:x}-{:x}", std::process::id())
}

pub(crate) fn validate_name(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: path.into(),
                message: "must not be empty".into(),
            }],
        })
    } else {
        Ok(())
    }
}

pub fn validate_experiment_spec(
    spec: &ExperimentSpecV1,
) -> Result<(), Vec<mcts_bench::experiment::ValidationField>> {
    spec.expand().map_err(|error| error.fields)?;
    let mut fields = Vec::new();
    for (game_index, game) in spec.games.iter().enumerate() {
        let Some(binary) = mcts_bench::games::find_game_binary(&game.game) else {
            fields.push(mcts_bench::experiment::ValidationField {
                path: format!("spec.games[{game_index}].game"),
                message: "not found".into(),
            });
            continue;
        };
        let mut command = vec![
            binary.to_string_lossy().into_owned(),
            "compare".into(),
            "validate".into(),
        ];
        for variant in &spec.variants {
            command.extend(["--candidate-config".into(), variant.config.to_string()]);
        }
        command.extend(["--baseline-config".into(), spec.baseline.config.to_string()]);
        if !game.game_config.is_null() {
            command.extend(["--game-config".into(), game.game_config.to_string()]);
        }
        let output = match std::process::Command::new(&command[0])
            .args(&command[1..])
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                fields.push(mcts_bench::experiment::ValidationField {
                    path: format!("spec.games[{game_index}].game"),
                    message: error.to_string(),
                });
                continue;
            }
        };
        let response = serde_json::from_slice::<Value>(&output.stdout).ok();
        let errors = response
            .as_ref()
            .and_then(|value| value.get("errors"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if errors.is_empty() && !output.status.success() {
            fields.push(mcts_bench::experiment::ValidationField {
                path: format!("spec.games[{game_index}].game"),
                message: "configured validation failed".into(),
            });
        }
        for error in errors {
            let field = error.get("field").and_then(Value::as_str).unwrap_or("");
            let candidate_index = error
                .get("candidate_index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < spec.variants.len())
                .unwrap_or(0);
            let path = match field {
                "game_config" => format!("spec.games[{game_index}].game_config"),
                "candidate_config" => format!("spec.variants[{candidate_index}].config"),
                "baseline_config" => "spec.baseline.config".into(),
                _ => format!("spec.games[{game_index}].game"),
            };
            fields.push(mcts_bench::experiment::ValidationField {
                path,
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("invalid configuration")
                    .into(),
            });
        }
    }
    if fields.is_empty() {
        Ok(())
    } else {
        Err(fields)
    }
}

pub(crate) fn project_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<ProjectResponse> {
    Ok(ProjectResponse {
        project_id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        archived: row.get(3)?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

pub(crate) async fn list_projects(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<Vec<ProjectResponse>>, BenchError> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare("SELECT project_id, name, description, archived, CAST(created_at AS TEXT), CAST(updated_at AS TEXT) FROM projects WHERE archived = false ORDER BY name")?;
    let rows = stmt
        .query_map([], project_from_row)?
        .filter_map(Result::ok)
        .collect();
    Ok(Json(rows))
}

pub(crate) async fn create_project(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(body): Json<ProjectCreateBody>,
) -> Result<(StatusCode, Json<ProjectResponse>), ValidationError> {
    validate_name("name", &body.name)?;
    let now = iso_timestamp_now();
    let id = generated_id("project");
    let db = state.db.lock().unwrap();
    let exists: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE name = ?1 AND archived = false",
            duckdb::params![body.name.trim()],
            |row| row.get(0),
        )
        .map_err(|_| ValidationError { fields: vec![] })?;
    if exists > 0 {
        return Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "name".into(),
                message: "duplicate active project name".into(),
            }],
        });
    }
    db.execute("INSERT INTO projects (project_id, name, description, archived, created_at, updated_at) VALUES (?1, ?2, ?3, false, ?4, ?4)", duckdb::params![id, body.name.trim(), body.description, now]).map_err(|_| ValidationError { fields: vec![] })?;
    Ok((
        StatusCode::CREATED,
        Json(ProjectResponse {
            project_id: id,
            name: body.name.trim().into(),
            description: body.description,
            archived: false,
            created_at: now.clone(),
            updated_at: now,
        }),
    ))
}

pub(crate) async fn get_project(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<ProjectResponse>, BenchError> {
    let db = state.db.lock().unwrap();
    match db.query_row("SELECT project_id, name, description, archived, CAST(created_at AS TEXT), CAST(updated_at AS TEXT) FROM projects WHERE project_id = ?1", duckdb::params![project_id], project_from_row) {
        Ok(project) => Ok(Json(project)),
        Err(duckdb::Error::QueryReturnedNoRows) => Err(BenchError { status: StatusCode::NOT_FOUND, message: "project not found".into() }),
        Err(error) => Err(error.into()),
    }
}

pub(crate) async fn update_project(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(project_id): AxumPath<String>,
    Json(body): Json<ProjectPatchBody>,
) -> Result<Json<ProjectResponse>, ValidationError> {
    if let Some(ref name) = body.name {
        validate_name("name", name)?;
    }
    let db = state.db.lock().unwrap();
    let current: (String, String, bool) = db
        .query_row(
            "SELECT name, description, archived FROM projects WHERE project_id = ?1",
            duckdb::params![project_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "project_id".into(),
                message: "not found".into(),
            }],
        })?;
    let name = body.name.unwrap_or(current.0);
    let description = body.description.unwrap_or(current.1);
    let archived = body.archived.unwrap_or(current.2);
    let duplicate: i64 = db.query_row("SELECT COUNT(*) FROM projects WHERE project_id <> ?1 AND name = ?2 AND archived = false", duckdb::params![project_id, name.trim()], |row| row.get(0)).map_err(|_| ValidationError { fields: vec![] })?;
    if duplicate > 0 && !archived {
        return Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "name".into(),
                message: "duplicate active project name".into(),
            }],
        });
    }
    let now = iso_timestamp_now();
    db.execute("UPDATE projects SET name = ?1, description = ?2, archived = ?3, updated_at = ?4 WHERE project_id = ?5", duckdb::params![name.trim(), description, archived, now, project_id]).map_err(|_| ValidationError { fields: vec![] })?;
    Ok(Json(ProjectResponse {
        project_id: project_id.clone(),
        name: name.trim().into(),
        description,
        archived,
        created_at: db
            .query_row(
                "SELECT CAST(created_at AS TEXT) FROM projects WHERE project_id = ?1",
                duckdb::params![&project_id],
                |row| row.get(0),
            )
            .unwrap_or_default(),
        updated_at: now,
    }))
}

pub(crate) fn experiment_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<ExperimentResponse> {
    let spec: Value = serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or(Value::Null);
    Ok(ExperimentResponse {
        experiment_id: row.get(0)?,
        project_id: row.get(1)?,
        name: row.get(2)?,
        description: row.get(3)?,
        spec: serde_json::from_value(spec).unwrap_or_else(|_| ExperimentSpecV1 {
            version: 1,
            games: vec![],
            baseline: mcts_bench::experiment::NamedStrategyConfig {
                id: String::new(),
                label: String::new(),
                config: Value::Null,
            },
            variants: vec![],
            budgets: vec![],
            rounds_per_cell: 0,
            base_seed: 0,
            max_parallel_cells: 0,
        }),
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

pub(crate) async fn list_experiments(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Vec<ExperimentResponse>>, BenchError> {
    let db = state.db.lock().unwrap();
    let parent: i64 = db.query_row(
        "SELECT COUNT(*) FROM projects WHERE project_id = ?1",
        duckdb::params![project_id],
        |row| row.get(0),
    )?;
    if parent == 0 {
        return Err(BenchError {
            status: StatusCode::NOT_FOUND,
            message: "project not found".into(),
        });
    }
    let mut stmt = db.prepare("SELECT experiment_id, project_id, name, description, CAST(spec AS TEXT), CAST(created_at AS TEXT), CAST(updated_at AS TEXT) FROM experiments WHERE project_id = ?1 ORDER BY name")?;
    Ok(Json(
        stmt.query_map(duckdb::params![project_id], experiment_from_row)?
            .filter_map(Result::ok)
            .collect(),
    ))
}

pub(crate) async fn create_experiment(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(project_id): AxumPath<String>,
    Json(body): Json<ExperimentBody>,
) -> Result<(StatusCode, Json<ExperimentResponse>), ValidationError> {
    validate_name("name", &body.name)?;
    (state.experiment_validator)(&body.spec).map_err(|fields| ValidationError { fields })?;
    let db = state.db.lock().unwrap();
    let parent: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE project_id = ?1",
            duckdb::params![project_id],
            |row| row.get(0),
        )
        .map_err(|_| ValidationError { fields: vec![] })?;
    if parent == 0 {
        return Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "project_id".into(),
                message: "not found".into(),
            }],
        });
    }
    let duplicate: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE project_id = ?1 AND name = ?2",
            duckdb::params![project_id, body.name.trim()],
            |row| row.get(0),
        )
        .map_err(|_| ValidationError { fields: vec![] })?;
    if duplicate > 0 {
        return Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "name".into(),
                message: "duplicate experiment name".into(),
            }],
        });
    }
    let id = generated_id("experiment");
    let now = iso_timestamp_now();
    let spec_json = serde_json::to_string(&body.spec).unwrap();
    db.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)", duckdb::params![id, project_id, body.name.trim(), body.description, spec_json]).map_err(|_| ValidationError { fields: vec![] })?;
    Ok((
        StatusCode::CREATED,
        Json(ExperimentResponse {
            experiment_id: id,
            project_id,
            name: body.name.trim().into(),
            description: body.description,
            spec: body.spec,
            created_at: now.clone(),
            updated_at: now,
        }),
    ))
}

pub(crate) async fn get_experiment(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(experiment_id): AxumPath<String>,
) -> Result<Json<ExperimentResponse>, BenchError> {
    let db = state.db.lock().unwrap();
    match db.query_row("SELECT experiment_id, project_id, name, description, CAST(spec AS TEXT), CAST(created_at AS TEXT), CAST(updated_at AS TEXT) FROM experiments WHERE experiment_id = ?1", duckdb::params![experiment_id], experiment_from_row) { Ok(value) => Ok(Json(value)), Err(duckdb::Error::QueryReturnedNoRows) => Err(BenchError { status: StatusCode::NOT_FOUND, message: "experiment not found".into() }), Err(error) => Err(error.into()) }
}

pub(crate) async fn update_experiment(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(experiment_id): AxumPath<String>,
    Json(body): Json<ExperimentBody>,
) -> Result<Json<ExperimentResponse>, ValidationError> {
    validate_name("name", &body.name)?;
    (state.experiment_validator)(&body.spec).map_err(|fields| ValidationError { fields })?;
    let db = state.db.lock().unwrap();
    let project_id: String = db
        .query_row(
            "SELECT project_id FROM experiments WHERE experiment_id = ?1",
            duckdb::params![experiment_id],
            |row| row.get(0),
        )
        .map_err(|_| ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "experiment_id".into(),
                message: "not found".into(),
            }],
        })?;
    let duplicate: i64 = db.query_row("SELECT COUNT(*) FROM experiments WHERE project_id = ?1 AND experiment_id <> ?2 AND name = ?3", duckdb::params![project_id, experiment_id, body.name.trim()], |row| row.get(0)).map_err(|_| ValidationError { fields: vec![] })?;
    if duplicate > 0 {
        return Err(ValidationError {
            fields: vec![mcts_bench::experiment::ValidationField {
                path: "name".into(),
                message: "duplicate experiment name".into(),
            }],
        });
    }
    let now = iso_timestamp_now();
    let spec_json = serde_json::to_string(&body.spec).unwrap();
    db.execute("UPDATE experiments SET name = ?1, description = ?2, spec = ?3, updated_at = ?4 WHERE experiment_id = ?5", duckdb::params![body.name.trim(), body.description, spec_json, now, experiment_id]).map_err(|_| ValidationError { fields: vec![] })?;
    Ok(Json(ExperimentResponse {
        experiment_id: experiment_id.clone(),
        project_id,
        name: body.name.trim().into(),
        description: body.description,
        spec: body.spec,
        created_at: db
            .query_row(
                "SELECT CAST(created_at AS TEXT) FROM experiments WHERE experiment_id = ?1",
                duckdb::params![&experiment_id],
                |row| row.get(0),
            )
            .unwrap_or_default(),
        updated_at: now,
    }))
}

pub(crate) async fn launch_experiment(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(experiment_id): AxumPath<String>,
    Json(_body): Json<Value>,
) -> Result<Json<LaunchResponse>, ExperimentRouteError> {
    let (project_id, name, spec): (String, String, ExperimentSpecV1) = {
        let db = state.db.lock().unwrap();
        let row = db.query_row(
            "SELECT project_id, name, CAST(spec AS TEXT) FROM experiments WHERE experiment_id = ?1",
            duckdb::params![&experiment_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        );
        let (project_id, name, spec_json) = match row {
            Ok(value) => value,
            Err(duckdb::Error::QueryReturnedNoRows) => {
                return Err(ExperimentRouteError::Bench(BenchError {
                    status: StatusCode::NOT_FOUND,
                    message: "experiment not found".into(),
                }))
            }
            Err(error) => return Err(error.into()),
        };
        let spec = serde_json::from_str(&spec_json).map_err(|error| {
            ExperimentRouteError::Bench(BenchError {
                status: StatusCode::BAD_REQUEST,
                message: format!("invalid saved experiment spec: {error}"),
            })
        })?;
        (project_id, name, spec)
    };
    (state.experiment_validator)(&spec)
        .map_err(|fields| ExperimentRouteError::Validation(ValidationError { fields }))?;
    let plan = spec.expand().map_err(|error| {
        ExperimentRouteError::Validation(ValidationError {
            fields: error.fields,
        })
    })?;
    let run_game = (spec.games.len() == 1).then(|| spec.games[0].game.clone());
    let run_game_segment = run_game.as_deref().unwrap_or("experiment-grid");
    let run_id = launch::generate_run_id("experiment", run_game_segment, crate::BUILD_INFO);
    let run_dir = Path::new(launch::BENCH_RUNS_DIR).join(&run_id);
    let log_path = run_dir.join("log.jsonl");
    let command = build_experiment_command(&spec, &run_id).map_err(|error| {
        ExperimentRouteError::Bench(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: error,
        })
    })?;
    let now = iso_timestamp_now();
    let start_request = StartRequest {
        run_id: run_id.clone(),
        game: run_game.clone(),
        project_id,
        experiment_id,
        spec_json: serde_json::to_string(&spec).unwrap(),
        label: name.clone(),
        git_sha: crate::BUILD_INFO.git_sha.into(),
        git_dirty: crate::BUILD_INFO.git_dirty,
        host: hostname(),
        started_at: now.clone(),
        log_path: log_path.to_string_lossy().into_owned(),
        cells: plan
            .cells
            .iter()
            .map(|cell| CellRequest {
                cell_id: cell.cell_id.clone(),
                cell_seed: cell.cell_seed,
                game: cell.game.clone(),
                game_config: cell.game_config.to_string(),
                variant_id: cell.variant_id.clone(),
                variant_label: cell.variant_label.clone(),
                candidate_config: cell.candidate_config.to_string(),
                baseline_id: cell.baseline_id.clone(),
                baseline_label: cell.baseline_label.clone(),
                baseline_config: cell.baseline_config.to_string(),
                budget: serde_json::to_string(&cell.budget).unwrap(),
                rounds: cell.rounds,
                planned_games: cell.planned_games as u32,
            })
            .collect(),
    };
    let descriptor = LaunchDescriptor {
        supervisor: find_bench_binary().into_os_string(),
        logical_run_id: run_id.clone(),
        attempt_id: run_id.clone(),
        parent_attempt_id: None,
        launch_nonce: format!("{run_id}-{}", now),
        workload_argv: command,
        journal_path: run_dir.join("lifecycle.jsonl"),
        stdout_path: log_path.clone(),
        stderr_path: run_dir.join("stdout.log"),
    };
    let launched = match state.runtime.start_projects(start_request, descriptor) {
        Ok(value) => value,
        Err(ProjectsError::Storage(message)) => {
            return Err(ExperimentRouteError::Bench(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("failed to launch experiment: {message}"),
            }));
        }
        Err(error) => return Err(ExperimentRouteError::Bench(attempt_bench_error(error))),
    };
    Ok(Json(LaunchResponse {
        run_id,
        pid: launched.pid,
        log_path: log_path.to_string_lossy().into_owned(),
        launch_error: launched.diagnostic,
    }))
}
