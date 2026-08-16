use crate::orchestration::{AttemptPhase, ExitObservation};
use crate::projects_attempt::{LaunchResult, ProjectsError, ProjectsRepository, StartRequest};
use crate::projects_attempt_duckdb::Repository;
use crate::schema::ensure_schema;
use crate::supervised_launch::{LaunchDescriptor, WrapperIdentity};
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

fn descriptor() -> LaunchDescriptor {
    LaunchDescriptor {
        supervisor: "bench".into(),
        logical_run_id: "a".into(),
        attempt_id: "a".into(),
        parent_attempt_id: None,
        launch_nonce: "nonce".into(),
        workload_argv: vec!["work".into()],
        journal_path: "/tmp/a.lifecycle".into(),
        stdout_path: "/tmp/a.jsonl".into(),
        stderr_path: "/tmp/a.err".into(),
    }
}

#[test]
fn first_start_and_replay_are_distinct() {
    let repository = repository();
    repository
        .authorize_start(&request(), &descriptor())
        .unwrap();
    let first = repository
        .record_launch(
            "a",
            &LaunchResult::Ready(WrapperIdentity {
                pid: 42,
                process_group_id: 42,
            }),
            "2026-01-01T00:00:01Z",
        )
        .unwrap();
    assert_eq!(first.token, crate::projects_attempt::LaunchToken::Ready);
    assert!(matches!(
        repository
            .authorize_start(&request(), &descriptor())
            .unwrap(),
        crate::projects_attempt::StartAuthorization::Replay(_)
    ));
}

#[test]
fn launch_observation_replays_exactly_and_conflicts_on_differences() {
    let repository = repository();
    repository
        .authorize_start(&request(), &descriptor())
        .unwrap();
    let result = LaunchResult::Pending {
        wrapper: WrapperIdentity {
            pid: 42,
            process_group_id: 43,
        },
        diagnostic: "waiting".into(),
    };
    repository
        .record_launch("a", &result, "2026-01-01T00:00:01Z")
        .unwrap();
    let replay = repository
        .record_launch("a", &result, "2026-01-01T00:00:02Z")
        .unwrap();
    assert_eq!(replay.token, crate::projects_attempt::LaunchToken::Pending);
    for changed in [
        LaunchResult::Pending {
            wrapper: WrapperIdentity {
                pid: 99,
                process_group_id: 43,
            },
            diagnostic: "waiting".into(),
        },
        LaunchResult::Pending {
            wrapper: WrapperIdentity {
                pid: 42,
                process_group_id: 43,
            },
            diagnostic: "changed".into(),
        },
        LaunchResult::Ready(WrapperIdentity {
            pid: 42,
            process_group_id: 43,
        }),
    ] {
        assert!(matches!(
            repository.record_launch("a", &changed, "2026-01-01T00:00:03Z"),
            Err(ProjectsError::Conflict(_))
        ));
    }
}

#[test]
fn launch_observation_requires_prior_authorization() {
    let repository = repository();
    assert_eq!(
        repository.record_launch(
            "missing",
            &LaunchResult::SpawnFailed("failed".into()),
            "2026-01-01T00:00:01Z",
        ),
        Err(ProjectsError::NotFound)
    );
}

#[test]
fn stop_authorization_and_exit_conflicts_are_durable() {
    let repository = repository();
    repository
        .authorize_start(&request(), &descriptor())
        .unwrap();
    repository
        .record_launch(
            "a",
            &LaunchResult::Ready(WrapperIdentity {
                pid: 42,
                process_group_id: 42,
            }),
            "2026-01-01T00:00:01Z",
        )
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
    let conflict =
        repository.observe_exit("a", ExitObservation::Unavailable, "2026-01-01T00:00:06Z");
    assert!(matches!(conflict, Err(ProjectsError::Conflict(_))));
}
