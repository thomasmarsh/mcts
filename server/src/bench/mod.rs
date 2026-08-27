//! HTTP routes for benchmark projects, runs, traces, and tuning sessions.
#![allow(unused_imports)]

pub(crate) mod lifecycle;
mod process;

mod commands;
mod router;
mod runs;
mod traces;
mod tuning;
mod tuning_types;
mod types;

pub use router::bench_router;
pub use types::{signal_process_group, BenchState, ProcessGroupSignaller, RunLauncher};

pub(crate) use commands::*;
pub(crate) use runs::*;
pub(crate) use traces::*;
pub(crate) use tuning::*;
pub(crate) use tuning_types::*;
pub(crate) use types::*;

#[cfg(test)]
mod tests;
