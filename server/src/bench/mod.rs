//! HTTP routes for benchmark projects, runs, traces, and tuning sessions.
#![allow(unused_imports)]

pub(crate) mod lifecycle;
mod process;

mod router;
mod runs;
mod traces;
mod tuner_api;
mod tuner_runs;
mod types;

pub use router::bench_router;
pub use tuner_api::shell_refresh;
pub use tuner_runs::{seed_tuner_objectives, shell_preflight_launch, shell_validate_objective};
pub use types::{
    signal_process_group, BenchState, LaunchPreflight, LaunchPreflighter, ObjectiveValidation,
    ObjectiveValidator, ProcessGroupSignaller, ProjectionRefresher,
};

pub(crate) use runs::*;
pub(crate) use traces::*;
pub(crate) use types::*;

#[cfg(test)]
mod tests;
