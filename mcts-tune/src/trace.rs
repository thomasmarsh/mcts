//! Optional per-ply move tracing for `strategy_tune_eval`'s self-play
//! games, for live monitoring/sanity-checking a tuner tuning run in
//! progress.
//!
//! Deliberately dependency-light, matching this crate's own charter (see
//! `lib.rs`'s module doc: no `mcts-bench`, no `duckdb`, no `game-host`
//! adapter machinery). A `Game::S` has no `Serialize` bound -- only
//! `Display` -- so there's no wire-JSON shape to convert it to without
//! pulling in a per-game `GameAdapter`, which this crate is not going to
//! start depending on. Instead, the concrete game adapter supplies the
//! canonical state and move JSON while `MoveTracer` only owns the destination
//! and JSONL sequencing.
//!
//! The emitted lines use the same `{"type": "move", ...}` tagged shape as
//! `mcts_bench::log::LogRecord::Move`, so a run's existing ingest loop can
//! read this file too -- without this crate depending on `mcts_bench` to
//! know that shape; it's just conventionally the same tag/field names.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use game_host::SearchReport;
use serde_json::Value;

/// Appends move-trace JSON lines to a file, one per ply. Opens its
/// destination once and keeps it open (append mode) for the lifetime of
/// the tracer, so a single `strategy_tune_eval` call's many games all
/// share one file handle.
pub struct MoveTracer {
    writer: BufWriter<std::fs::File>,
    next_game_seq: u64,
}

impl MoveTracer {
    /// Opens (creating if needed) `path` for appending. Multiple
    /// concurrent trial subprocesses may open the same path -- each
    /// `write_ply` call is a single `write_all` of a short line, which is
    /// atomic enough in append mode for this use (occasional interleaving
    /// under heavy concurrency would only cost that one line, not corrupt
    /// the file).
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let next_game_seq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Ok(Self {
            writer: BufWriter::new(file),
            next_game_seq,
        })
    }

    /// Mints a fresh `game_seq` for one game's worth of plies. Distinct
    /// games traced by this tracer never share a `game_seq`; distinct
    /// tracers (different processes/trials) are seeded from a nanosecond
    /// timestamp so collisions across processes are practically
    /// negligible -- and harmless if one ever happens, since the ingest
    /// side's primary key just drops the duplicate ply.
    pub fn start_game(&mut self) -> u64 {
        let seq = self.next_game_seq;
        self.next_game_seq += 1;
        seq
    }

    /// Writes one already-encoded canonical game-host ply. `mv` and `search`
    /// are null only for the initial state.
    pub fn write_ply(
        &mut self,
        game_seq: u64,
        ply: u32,
        state: Value,
        mv: Option<Value>,
        player: Option<&str>,
        search: Option<SearchReport>,
    ) {
        let line = serde_json::json!({
            "type": "move",
            "trace_schema_version": 1,
            "game_seq": game_seq,
            "ply": ply,
            "state": state,
            "mv": mv,
            "player": player,
            "search": search,
        });
        let mut text = line.to_string();
        text.push('\n');
        let _ = self.writer.write_all(text.as_bytes());
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_ply_appends_one_json_line_per_call() {
        let dir = std::env::temp_dir().join(format!(
            "mcts_tune_trace_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("moves.jsonl");

        let mut tracer = MoveTracer::open(&path).unwrap();
        let seq = tracer.start_game();
        tracer.write_ply(
            seq,
            0,
            serde_json::json!({"turn": "first"}),
            None,
            None,
            None,
        );
        tracer.write_ply(
            seq,
            1,
            serde_json::json!({"turn": "second"}),
            Some(serde_json::json!({"ptn": "a1"})),
            Some("candidate"),
            Some(SearchReport {
                schema_version: 1,
                status: game_host::SearchReportStatus::Unavailable,
                reason: Some(game_host::SearchReportReason::StrategyUnsupported),
                elapsed_seconds: None,
                iteration_limit: None,
                time_limit_seconds: None,
                completed_iterations: 0,
                termination: None,
                selected_action: None,
                actions: vec![],
                principal_variation: vec![],
                root_visits: 0,
                tree_nodes: 0,
                mean_depth: None,
                max_depth: None,
                graph_mode: None,
                tt_reads: 0,
                tt_writes: 0,
                tt_hits: 0,
                tt_hit_ratio: None,
                iterations_per_second: None,
                warnings: vec![],
            }),
        );
        drop(tracer);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "move");
        assert_eq!(first["game_seq"], seq);
        assert_eq!(first["ply"], 0);
        assert_eq!(first["trace_schema_version"], 1);
        assert_eq!(first["state"], serde_json::json!({"turn": "first"}));
        assert!(first["mv"].is_null());
        assert!(first["search"].is_null());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["ply"], 1);
        assert_eq!(second["mv"], serde_json::json!({"ptn": "a1"}));
        assert_eq!(second["player"], "candidate");
        assert_eq!(second["search"]["status"], "unavailable");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn start_game_mints_distinct_sequential_seqs() {
        let dir = std::env::temp_dir().join(format!(
            "mcts_tune_trace_seq_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("moves.jsonl");

        let mut tracer = MoveTracer::open(&path).unwrap();
        let a = tracer.start_game();
        let b = tracer.start_game();
        assert_eq!(b, a + 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
