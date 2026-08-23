//! HTTP routes for benchmark projects, runs, traces, and legacy ladder advancement.
#![allow(unused_imports)]

pub(crate) mod lifecycle;
mod process;
pub(crate) mod supervisor_runtime;

mod commands;
mod ladder;
mod projects;
mod router;
mod runs;
mod traces;
mod tuning;
mod tuning_types;
mod types;

pub use projects::validate_experiment_spec;
pub use router::bench_router;
pub use types::{
    signal_process_group, BenchState, ExperimentValidator, ProcessGroupSignaller, RunLauncher,
};

pub(crate) use commands::*;
pub(crate) use ladder::*;
pub(crate) use projects::*;
pub(crate) use runs::*;
pub(crate) use traces::*;
pub(crate) use tuning::*;
pub(crate) use tuning_types::*;
pub(crate) use types::*;

#[cfg(test)]
mod tests;
