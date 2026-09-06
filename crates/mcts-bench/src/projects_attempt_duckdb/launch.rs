use duckdb::{params, OptionalExt, Transaction};

use super::{db_error, identity_error, record, store_error, Repository};
use crate::attempt_store;
use crate::orchestration::{AttemptAction, AttemptEvent};
use crate::projects_attempt::{
    self, LaunchRecord, LaunchResult, LaunchToken, PreviousLaunch, ProjectsError,
    StartAuthorization, StartRequest,
};
use crate::supervised_launch::{LaunchDescriptor, WrapperIdentity};

struct StoredLaunch {
    logical_run_id: String,
    parent_attempt_id: Option<String>,
    nonce: String,
    workload_argv: String,
    lifecycle_path: String,
    stdout_path: String,
    stderr_path: String,
    pid: Option<i64>,
    pgid: Option<i64>,
    result: Option<String>,
    diagnostic: Option<String>,
}

pub(super) fn authorize_start(
    repo: &Repository<'_>,
    request: &StartRequest,
    descriptor: &LaunchDescriptor,
) -> Result<StartAuthorization, ProjectsError> {
    if descriptor.attempt_id != request.run_id {
        return Err(ProjectsError::Conflict(
            "launch attempt ID does not match run ID".into(),
        ));
    }
    let tx = repo.tx()?;
    if let Some(stored) = load_stored(&tx, &request.run_id)? {
        let replay = replay(stored, descriptor);
        tx.commit().map_err(db_error)?;
        return replay;
    }
    create_attempt(&tx, request, descriptor)?;
    tx.commit().map_err(db_error)?;
    Ok(StartAuthorization::New)
}

fn load_stored(
    tx: &Transaction<'_>,
    attempt_id: &str,
) -> Result<Option<StoredLaunch>, ProjectsError> {
    tx.query_row(
        "SELECT logical_run_id, parent_attempt_id, launch_nonce, workload_argv, lifecycle_path, stdout_path, stderr_path, wrapper_pid, process_group_id, launch_result, launch_diagnostic FROM projects_launches WHERE attempt_id = ?1",
        params![attempt_id],
        |row| {
            Ok(StoredLaunch {
                logical_run_id: row.get(0)?,
                parent_attempt_id: row.get(1)?,
                nonce: row.get(2)?,
                workload_argv: row.get(3)?,
                lifecycle_path: row.get(4)?,
                stdout_path: row.get(5)?,
                stderr_path: row.get(6)?,
                pid: row.get(7)?,
                pgid: row.get(8)?,
                result: row.get(9)?,
                diagnostic: row.get(10)?,
            })
        },
    )
    .optional()
    .map_err(db_error)
}

fn replay(
    stored: StoredLaunch,
    descriptor: &LaunchDescriptor,
) -> Result<StartAuthorization, ProjectsError> {
    let workload: Vec<String> = serde_json::from_str(&stored.workload_argv).map_err(|error| {
        ProjectsError::Corrupt(format!("invalid persisted workload argv: {error}"))
    })?;
    if stored.logical_run_id != descriptor.logical_run_id
        || stored.parent_attempt_id != descriptor.parent_attempt_id
        || stored.nonce != descriptor.launch_nonce
        || workload != descriptor.workload_argv
        || stored.lifecycle_path != descriptor.journal_path.to_string_lossy()
        || stored.stdout_path != descriptor.stdout_path.to_string_lossy()
        || stored.stderr_path != descriptor.stderr_path.to_string_lossy()
    {
        return Err(ProjectsError::Conflict(
            "launch descriptor conflicts with existing attempt".into(),
        ));
    }
    Ok(StartAuthorization::Replay(PreviousLaunch {
        result: stored
            .result
            .map(|token| persisted_record(token, stored.pid, stored.pgid, stored.diagnostic))
            .transpose()?,
    }))
}

fn persisted_record(
    token: String,
    pid: Option<i64>,
    pgid: Option<i64>,
    diagnostic: Option<String>,
) -> Result<LaunchRecord, ProjectsError> {
    let token = match token.as_str() {
        "ready" => LaunchToken::Ready,
        "spawn_failed" => LaunchToken::SpawnFailed,
        "pending" => LaunchToken::Pending,
        "conflict" => LaunchToken::Conflict,
        _ => {
            return Err(ProjectsError::Corrupt(format!(
                "unknown launch result {token}"
            )))
        }
    };
    Ok(LaunchRecord {
        token,
        wrapper: pid
            .zip(pgid)
            .map(|(pid, process_group_id)| WrapperIdentity {
                pid: pid as u64,
                process_group_id: process_group_id as u64,
            }),
        diagnostic,
    })
}

fn create_attempt(
    tx: &Transaction<'_>,
    request: &StartRequest,
    descriptor: &LaunchDescriptor,
) -> Result<(), ProjectsError> {
    tx.execute("INSERT INTO runs (run_id, kind, game, project_id, experiment_id, experiment_spec, label, config, git_sha, git_dirty, host, pid, started_at, status, log_path) VALUES (?1, 'experiment', ?2, ?3, ?4, ?5, ?6, NULL, ?7, ?8, ?9, NULL, ?10, 'starting', ?11)", params![request.run_id, request.game, request.project_id, request.experiment_id, request.spec_json, request.label, request.git_sha, request.git_dirty, request.host, request.started_at, request.log_path])?;
    crate::identity::create_root_identity(
        tx,
        &request.run_id,
        "experiment",
        Some(&request.project_id),
        Some(&request.experiment_id),
        &request.started_at,
    )
    .map_err(identity_error)?;
    for cell in &request.cells {
        tx.execute("INSERT INTO experiment_cells (run_id, cell_id, cell_seed, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 'pending')", params![request.run_id, cell.cell_id, cell.cell_seed, cell.game, cell.game_config, cell.variant_id, cell.variant_label, cell.candidate_config, cell.baseline_id, cell.baseline_label, cell.baseline_config, cell.budget, cell.rounds, cell.planned_games])?;
    }
    attempt_store::initialize_attempt(tx, &request.run_id).map_err(store_error)?;
    let receipt = record(
        tx,
        &request.run_id,
        projects_attempt::START_REQUESTED_KEY,
        AttemptEvent::StartRequested,
        &request.started_at,
        &[AttemptAction::SpawnProcess],
    )?;
    if receipt.replay {
        return Err(ProjectsError::Conflict(
            "start request was already recorded".into(),
        ));
    }
    let workload_argv = serde_json::to_string(&descriptor.workload_argv)
        .map_err(|error| ProjectsError::Storage(format!("serialize workload argv: {error}")))?;
    tx.execute("INSERT INTO projects_launches (attempt_id, logical_run_id, parent_attempt_id, launch_nonce, workload_argv, lifecycle_path, stdout_path, stderr_path) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)", params![request.run_id, descriptor.logical_run_id, descriptor.parent_attempt_id, descriptor.launch_nonce, workload_argv, descriptor.journal_path.to_string_lossy(), descriptor.stdout_path.to_string_lossy(), descriptor.stderr_path.to_string_lossy()])?;
    Ok(())
}

pub(super) fn record_launch(
    repo: &Repository<'_>,
    run_id: &str,
    result: &LaunchResult,
    observed_at: &str,
) -> Result<LaunchRecord, ProjectsError> {
    let tx = repo.tx()?;
    let launch = launch_record(result);
    let stored = load_stored(&tx, run_id)?.ok_or(ProjectsError::NotFound)?;
    if stored.result.is_some() {
        if matches_stored(&stored, &launch) {
            tx.rollback().map_err(db_error)?;
            return Ok(launch);
        }
        return Err(ProjectsError::Conflict(
            "launch observation conflicts with persisted result".into(),
        ));
    }
    apply_result(&tx, run_id, result, observed_at)?;
    tx.commit().map_err(db_error)?;
    Ok(launch)
}

fn launch_record(result: &LaunchResult) -> LaunchRecord {
    match result {
        LaunchResult::Ready(wrapper) => LaunchRecord {
            token: LaunchToken::Ready,
            wrapper: Some(*wrapper),
            diagnostic: None,
        },
        LaunchResult::SpawnFailed(message) => LaunchRecord {
            token: LaunchToken::SpawnFailed,
            wrapper: None,
            diagnostic: Some(message.clone()),
        },
        LaunchResult::Pending {
            wrapper,
            diagnostic,
        } => LaunchRecord {
            token: LaunchToken::Pending,
            wrapper: Some(*wrapper),
            diagnostic: Some(diagnostic.clone()),
        },
        LaunchResult::Conflict {
            wrapper,
            diagnostic,
        } => LaunchRecord {
            token: LaunchToken::Conflict,
            wrapper: Some(*wrapper),
            diagnostic: Some(format!("{diagnostic:?}")),
        },
    }
}

fn matches_stored(stored: &StoredLaunch, launch: &LaunchRecord) -> bool {
    let token = match launch.token {
        LaunchToken::Ready => "ready",
        LaunchToken::SpawnFailed => "spawn_failed",
        LaunchToken::Pending => "pending",
        LaunchToken::Conflict => "conflict",
    };
    let identity_matches =
        launch
            .wrapper
            .map_or(stored.pid.is_none() && stored.pgid.is_none(), |wrapper| {
                stored.pid == Some(wrapper.pid as i64)
                    && stored.pgid == Some(wrapper.process_group_id as i64)
            });
    stored.result.as_deref() == Some(token)
        && identity_matches
        && stored.diagnostic.as_deref() == launch.diagnostic.as_deref()
}

fn apply_result(
    tx: &Transaction<'_>,
    run_id: &str,
    result: &LaunchResult,
    observed_at: &str,
) -> Result<(), ProjectsError> {
    match result {
        LaunchResult::Ready(wrapper) => {
            record(
                tx,
                run_id,
                projects_attempt::PROCESS_OBSERVED_KEY,
                AttemptEvent::ProcessObserved,
                observed_at,
                &[],
            )?;
            tx.execute(
                "UPDATE runs SET status = 'running', pid = ?1 WHERE run_id = ?2",
                params![wrapper.pid as i64, run_id],
            )?;
            tx.execute("UPDATE projects_launches SET wrapper_pid = ?1, process_group_id = ?2, launch_result = 'ready', launch_diagnostic = NULL WHERE attempt_id = ?3", params![wrapper.pid as i64, wrapper.process_group_id as i64, run_id])?;
        }
        LaunchResult::SpawnFailed(message) => {
            record(
                tx,
                run_id,
                projects_attempt::SPAWN_FAILED_KEY,
                AttemptEvent::SpawnFailed,
                observed_at,
                &[],
            )?;
            tx.execute("UPDATE experiment_cells SET status = 'failed', error = ?1, ended_at = ?2 WHERE run_id = ?3 AND status IN ('pending', 'running')", params![message, observed_at, run_id])?;
            tx.execute(
                "UPDATE runs SET status = 'crashed', ended_at = ?1 WHERE run_id = ?2",
                params![observed_at, run_id],
            )?;
            tx.execute("UPDATE projects_launches SET launch_result = 'spawn_failed', launch_diagnostic = ?1 WHERE attempt_id = ?2", params![message, run_id])?;
        }
        LaunchResult::Pending {
            wrapper,
            diagnostic,
        } => {
            tx.execute("UPDATE projects_launches SET wrapper_pid = ?1, process_group_id = ?2, launch_result = 'pending', launch_diagnostic = ?3 WHERE attempt_id = ?4", params![wrapper.pid as i64, wrapper.process_group_id as i64, diagnostic, run_id])?;
        }
        LaunchResult::Conflict {
            wrapper,
            diagnostic,
        } => {
            tx.execute("UPDATE projects_launches SET wrapper_pid = ?1, process_group_id = ?2, launch_result = 'conflict', launch_diagnostic = ?3 WHERE attempt_id = ?4", params![wrapper.pid as i64, wrapper.process_group_id as i64, format!("{diagnostic:?}"), run_id])?;
        }
    }
    Ok(())
}
