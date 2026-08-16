use crate::orchestration::{AttemptPhase, ExitObservation};
use crate::projects_attempt::{ProjectsError, ProjectsRepository, StartRequest};
use crate::projects_attempt_duckdb::Repository;
use crate::schema::ensure_schema;
use duckdb::Connection;

fn repository() -> Repository<'static> {
    let connection = Box::leak(Box::new(Connection::open_in_memory().unwrap()));
    ensure_schema(connection).unwrap();
    Repository::new(connection)
}

fn request() -> StartRequest {
    StartRequest {
        run_id: "a".into(),
        game: Some("nim".into()),
        project_id: "p".into(),
        experiment_id: "e".into(),
        spec_json: "{}".into(),
        label: "A".into(),
        git_sha: "sha".into(),
        git_dirty: false,
        host: "host".into(),
        started_at: "2026-01-01T00:00:00Z".into(),
        log_path: "/tmp/a.jsonl".into(),
        cells: vec![],
    }
}

#[test]
fn first_start_and_replay_are_distinct() {
    let repository = repository();
    repository.create_and_request_start(&request()).unwrap();
    let first = repository
        .observe_process("a", 42, "/tmp/a.jsonl", "2026-01-01T00:00:01Z")
        .unwrap();
    assert!(!first.replay);
    let replay = repository
        .observe_process("a", 42, "/tmp/a.jsonl", "2026-01-01T00:00:02Z")
        .unwrap();
    assert!(replay.replay);
}

#[test]
fn stop_authorization_and_exit_conflicts_are_durable() {
    let repository = repository();
    repository.create_and_request_start(&request()).unwrap();
    repository
        .observe_process("a", 42, "/tmp/a.jsonl", "2026-01-01T00:00:01Z")
        .unwrap();
    assert!(
        repository
            .request_operator_stop("a", "2026-01-01T00:00:02Z")
            .unwrap()
            .signal_process_group
    );
    assert!(
        !repository
            .request_operator_stop("a", "2026-01-01T00:00:03Z")
            .unwrap()
            .signal_process_group
    );
    repository
        .observe_signal("a", "2026-01-01T00:00:04Z")
        .unwrap();
    let exit = repository
        .observe_exit(
            "a",
            ExitObservation::Exited { code: Some(0) },
            "2026-01-01T00:00:05Z",
        )
        .unwrap();
    assert!(exit.finalize_output);
    let final_output = repository
        .finalize_output("a", "2026-01-01T00:00:06Z")
        .unwrap();
    assert_eq!(final_output.state.phase(), AttemptPhase::Stopped);
    let conflict = repository.observe_exit("a", ExitObservation::Lost, "2026-01-01T00:00:06Z");
    assert!(matches!(conflict, Err(ProjectsError::Conflict(_))));
}
