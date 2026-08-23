//! Structured logging types shared by everything that produces or consumes
//! benchmark run output.  Every process a run spawns — Rust or Python — emits
//! one JSON object per line to stdout, each tagged by `"type"`.  The ingest
//! loop dispatches on that tag; it does not care what language wrote the line.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Run log records (written by the run process itself to log.jsonl)
// ---------------------------------------------------------------------------

/// One event in a run's JSONL log file, tagged by `"type"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LogRecord {
    /// A single completed game between two strategies in a round-robin
    /// tournament.
    MatchResult {
        /// Monotonically increasing sequence number within the run.
        seq: u64,
        /// Human-readable name of the first strategy.
        strategy_a: String,
        /// Human-readable name of the second strategy.
        strategy_b: String,
        /// `"win_a"`, `"win_b"`, or `"draw"`.
        outcome: String,
        /// Strategy name that won, or `None` for a draw.
        winner: Option<String>,
        /// Arbitrary extra metadata (moves list, timing, etc.).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extra: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cell_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_game_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metrics: Option<serde_json::Value>,
    },
    /// An experiment cell has begun execution.
    CellStarted { cell_id: String },
    /// An experiment cell completed successfully.
    CellFinished {
        cell_id: String,
        completed_games: u64,
    },
    /// A cell failed after preserving any already completed matches.
    CellFailed {
        cell_id: String,
        completed_games: u64,
        error: String,
    },
    /// A single trial from a hyperparameter-optimization run (tuner, etc.).
    Trial {
        trial_id: u64,
        config: serde_json::Value,
        seed: Option<u64>,
        cost: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extra: Option<serde_json::Value>,
    },
    /// The current incumbent (the tuner's own best-so-far config, as tracked by
    /// its intensifier) of a hyperparameter-optimization run. Emitted once
    /// per *change* of incumbent, not once per trial -- unlike `Trial`,
    /// only the latest one matters, so ingest overwrites rather than
    /// appends. Deliberately sourced from the optimizer's own incumbent
    /// tracking rather than reconstructed as `MIN(cost)` over `trials`: once
    /// a run uses multiple baseline instances, per-trial costs aren't
    /// directly comparable across instances, and only the intensifier
    /// itself aggregates them correctly.
    Incumbent {
        config: serde_json::Value,
        cost: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extra: Option<serde_json::Value>,
    },
    /// One ply of one game's move trace, for live spectating / post-hoc
    /// replay through the same per-game UI renderer used for interactive
    /// play. `game_seq` is the existing per-run game identifier: it's
    /// `MatchResult.seq` for a round_robin run, `Trial.trial_id` for a
    /// tuner run -- deliberately reused rather than minting a new id, so a
    /// trace joins straight onto `match_results`/`trials` with no mapping
    /// table.
    Move {
        /// Version of the per-ply trace envelope. Absent rows predate the
        /// renderer trace contract.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_schema_version: Option<u32>,
        game_seq: u64,
        ply: u32,
        /// Wire-format game state after this ply, in the same shape
        /// `GameAdapter::ai_move`'s `result.state` already produces --
        /// renderers need no new code to draw it.
        state: serde_json::Value,
        /// The move applied to reach `state`. `None` for the initial state
        /// (ply 0), before any move has been made.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mv: Option<serde_json::Value>,
        /// Preset/strategy id that made this move, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        player: Option<String>,
        /// Final-search evidence for the move. A null or absent value keeps
        /// older and report-less producers distinct from an unavailable
        /// strategy report.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        search: Option<serde_json::Value>,
    },
    /// Periodic liveness heartbeat.
    Heartbeat { games_played: u64 },
}

impl LogRecord {
    /// Serialize this record as a single line of JSON (no trailing newline).
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("LogRecord is always serializable")
    }
}

// ---------------------------------------------------------------------------
// Registry log (written by the launcher to registry.log)
// ---------------------------------------------------------------------------

/// One event in the master registry log, which tracks run lifecycle across
/// all runs (start, stop).  Written by the launcher only.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RegistryEvent {
    /// A run was launched.
    Start {
        run_id: String,
        kind: String,
        game: String,
        pid: u32,
        /// The full command vector.
        cmd: Vec<String>,
        /// Path to the run's JSONL log file.
        log_path: String,
        /// Git SHA of the launching binary.
        git_sha: String,
        /// Whether the worktree was dirty at compile time.
        git_dirty: bool,
        /// ISO-8601 timestamp of launch.
        started_at: String,
    },
    /// A run stopped (normally or via signal).
    Stop {
        run_id: String,
        exit_code: Option<i32>,
        ended_at: String,
    },
}

impl RegistryEvent {
    /// Serialize this event as a single line of JSON (no trailing newline).
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("RegistryEvent is always serializable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_record_match_result_round_trips() {
        let rec = LogRecord::MatchResult {
            seq: 1,
            strategy_a: "strong".into(),
            strategy_b: "master".into(),
            outcome: "win_a".into(),
            winner: Some("strong".into()),
            extra: None,
            cell_id: None,
            seed: None,
            trace_game_seq: None,
            metrics: None,
        };
        let json = rec.to_json_line();
        let parsed: LogRecord = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, LogRecord::MatchResult { seq: 1, .. }));
    }

    #[test]
    fn log_record_trial_round_trips() {
        let rec = LogRecord::Trial {
            trial_id: 42,
            config: serde_json::json!({"lr": 0.001}),
            seed: Some(7),
            cost: 0.375,
            extra: None,
        };
        let json = rec.to_json_line();
        let parsed: LogRecord = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, LogRecord::Trial { trial_id: 42, .. }));
    }

    #[test]
    fn log_record_move_round_trips() {
        let rec = LogRecord::Move {
            trace_schema_version: Some(1),
            game_seq: 3,
            ply: 5,
            state: serde_json::json!({"board": [1, 0, 2]}),
            mv: Some(serde_json::json!({"cell": 4})),
            player: Some("strong".into()),
            search: Some(serde_json::json!({"schema_version": 1})),
        };
        let json = rec.to_json_line();
        let parsed: LogRecord = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            LogRecord::Move {
                game_seq: 3,
                ply: 5,
                ..
            }
        ));
    }

    #[test]
    fn log_record_heartbeat_round_trips() {
        let rec = LogRecord::Heartbeat { games_played: 40 };
        let json = rec.to_json_line();
        let parsed: LogRecord = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, LogRecord::Heartbeat { games_played: 40 }));
    }

    #[test]
    fn registry_event_start_round_trips() {
        let ev = RegistryEvent::Start {
            run_id: "rr-druid-20260808T120000-6fe2387".into(),
            kind: "round_robin".into(),
            game: "druid".into(),
            pid: 12345,
            cmd: vec!["bench".into(), "round-robin".into()],
            log_path: "bench-runs/rr-druid-.../log.jsonl".into(),
            git_sha: "6fe2387".into(),
            git_dirty: false,
            started_at: "2026-08-08T12:00:00Z".into(),
        };
        let json = ev.to_json_line();
        let parsed: RegistryEvent = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, RegistryEvent::Start { ref run_id, .. } if run_id == "rr-druid-20260808T120000-6fe2387")
        );
    }

    #[test]
    fn registry_event_stop_round_trips() {
        let ev = RegistryEvent::Stop {
            run_id: "rr-druid-20260808T120000-6fe2387".into(),
            exit_code: Some(0),
            ended_at: "2026-08-08T14:00:00Z".into(),
        };
        let json = ev.to_json_line();
        let parsed: RegistryEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            RegistryEvent::Stop {
                exit_code: Some(0),
                ..
            }
        ));
    }
}
