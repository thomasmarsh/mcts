//! Protocol helper for game subprocess binaries.
//!
//! Each game kind builds a standalone binary that speaks the JSON-line
//! subprocess protocol over stdin/stdout using the types and run_host
//! function in this crate.

mod adapter;
mod cli;
mod error;
mod protocol;
pub mod subprocess;
mod types;

pub use adapter::GameAdapter;
pub use cli::{run_cli, GameDescription};
pub use error::HostError;
pub use protocol::{run_host, run_stdin_stdout};
pub use types::{
    derive_seed, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, BookInfo,
    CompareValidationField, ConfiguredCandidateSide, ConfiguredComparisonSummary,
    ConfiguredMatchResult, ConfiguredOutcome, ConfiguredStrategyMetrics, ErrorBody, Request,
    Response, SearchActionReport, SearchGraphMode, SearchReport, SearchReportReason,
    SearchReportStatus, SearchTermination, SearchWarning, TunerCondition, TunerInfo,
    TunerParameter,
};

#[cfg(test)]
pub(crate) use cli::run_cli_with;

#[cfg(test)]
mod tests {
    mod book;
    mod cli;
    mod support;
}
