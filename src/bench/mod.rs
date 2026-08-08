//! Benchmark / tournament / SMAC3 harness library. Only the `server` process
//! opens `bench.duckdb` directly; `bin/bench` and Python tools communicate
//! via JSONL files and the registry log.

pub mod games;
pub mod ingest;
pub mod launch;
pub mod log;
pub mod schema;
pub mod tournament;