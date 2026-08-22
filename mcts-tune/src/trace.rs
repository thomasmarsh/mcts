//! Optional per-ply move tracing for `strategy_tune_eval`'s self-play
//! games, for live monitoring/sanity-checking a tuner tuning run in
//! progress.
//!
//! Deliberately dependency-light, matching this crate's own charter (see
//! `lib.rs`'s module doc: no `mcts-bench`, no `duckdb`, no `game-host`
//! adapter machinery). A `Game::S` has no `Serialize` bound -- only
//! `Display` -- so there's no wire-JSON shape to convert it to without
//! pulling in a per-game `GameAdapter`, which this crate is not going to
//! start depending on. Instead, `MoveTracer` only needs a *destination*
//! (a file path): it renders each state via `Display` and appends one JSON
//! line per ply. `Game::A: Action` already requires `Serialize`, so moves
//! (unlike states) round-trip as real structured JSON, not text.
//!
//! The emitted lines use the same `{"type": "move", ...}` tagged shape as
//! `mcts_bench::log::LogRecord::Move`, so a run's existing ingest loop can
//! read this file too -- without this crate depending on `mcts_bench` to
//! know that shape; it's just conventionally the same tag/field names.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

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

    /// Writes one ply. `state` is rendered via `Display`; `mv` (the move
    /// applied to reach `state`, `None` for the initial ply) via
    /// `Serialize`.
    pub fn write_ply<S: std::fmt::Display, A: Serialize>(
        &mut self,
        game_seq: u64,
        ply: u32,
        state: &S,
        mv: Option<&A>,
        player: Option<&str>,
    ) {
        let line = serde_json::json!({
            "type": "move",
            "game_seq": game_seq,
            "ply": ply,
            "state": state.to_string(),
            "mv": mv.map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null)),
            "player": player,
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
        tracer.write_ply::<String, u32>(seq, 0, &"initial".to_owned(), None, None);
        tracer.write_ply(
            seq,
            1,
            &"after one move".to_owned(),
            Some(&7u32),
            Some("candidate"),
        );
        drop(tracer);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "move");
        assert_eq!(first["game_seq"], seq);
        assert_eq!(first["ply"], 0);
        assert_eq!(first["state"], "initial");
        assert!(first["mv"].is_null());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["ply"], 1);
        assert_eq!(second["mv"], 7);
        assert_eq!(second["player"], "candidate");

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
