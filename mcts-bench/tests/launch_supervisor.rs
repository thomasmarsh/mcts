//! End-to-end test of `launch_supervisor`: spawns a real detached `sh`
//! subprocess and polls its registry log on disk for the recorded exit --
//! genuinely slow (up to a 10-second poll ceiling waiting on process exit
//! and file writes from another process), so this lives here as an
//! integration test rather than a `src/`-level `#[test]`, alongside this
//! crate's other real-subprocess/on-disk tests. `cargo test --lib` never
//! compiles or runs `tests/`, so this doesn't slow down the fast suite.

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use mcts_bench::launch::{launch_supervisor, BuildInfo};
use mcts_bench::supervised_launch::SupervisorCommand;

#[cfg(unix)]
#[test]
fn supervisor_uses_distinct_outputs_and_records_exit() {
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "mcts_supervisor_test_{}_{}",
        std::process::id(),
        TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let stdout = dir.join("outer.stdout");
    let stderr = dir.join("outer.stderr");
    let workload = dir.join("workload.log");
    let registry = dir.join("registry.log");
    let command = SupervisorCommand {
        executable: "sh".into(),
        arguments: vec!["-c".into(), "printf out; printf err >&2; exit 7".into()],
    };
    launch_supervisor(
        "test-supervisor",
        &command,
        &stdout,
        &stderr,
        &workload,
        &registry,
        BuildInfo {
            git_sha: "test",
            git_dirty: false,
        },
    )
    .unwrap();
    for _ in 0..10_000 {
        if let Ok(contents) = fs::read_to_string(&registry) {
            if contents
                .lines()
                .any(|line| line.contains("\"type\":\"stop\"") && line.contains("\"exit_code\":7"))
            {
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let contents = fs::read_to_string(&registry).unwrap();
    assert!(contents.contains("\"exit_code\":7"));
    assert_eq!(fs::read_to_string(&stdout).unwrap(), "out");
    assert_eq!(fs::read_to_string(&stderr).unwrap(), "err");
    assert!(!workload.exists());
    let _ = fs::remove_dir_all(&dir);
}
