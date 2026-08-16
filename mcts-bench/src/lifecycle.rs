//! Durable, typed lifecycle evidence for detached benchmark attempts.
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LifecycleRecord {
    pub schema_version: u32,
    pub sequence: u64,
    pub attempt_id: String,
    pub launch_nonce: String,
    pub timestamp: String,
    pub payload: LifecyclePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "value")]
pub enum LifecyclePayload {
    #[serde(rename = "wrapper_started")]
    WrapperStarted(WrapperManifest),
    #[serde(rename = "child_started")]
    ChildStarted { child_pid: u64 },
    #[serde(rename = "child_spawn_failed")]
    ChildSpawnFailed { stage: String, error: String },
    #[serde(rename = "child_exited")]
    ChildExited { outcome: ExitEvidence },
    #[serde(rename = "outputs_closed")]
    OutputsClosed { outputs: Vec<OutputClosure> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WrapperManifest {
    pub logical_run_id: String,
    pub attempt_id: String,
    pub parent_attempt_id: Option<String>,
    pub argv: Vec<String>,
    pub wrapper_pid: u64,
    pub process_group_id: u64,
    pub hostname: String,
    pub boot_id: Option<String>,
    pub process_start_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value")]
pub enum ExitEvidence {
    #[serde(rename = "code")]
    Code { code: i32 },
    #[serde(rename = "signal")]
    Signal { signal: i32 },
    #[serde(rename = "wait_failed")]
    WaitFailed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutputClosure {
    pub path: String,
    pub byte_length: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalSnapshot {
    pub manifest: WrapperManifest,
    pub child: Option<u64>,
    pub terminal: Option<TerminalEvidence>,
    pub outputs: Option<Vec<OutputClosure>>,
    pub last_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalEvidence {
    SpawnFailed { stage: String, error: String },
    Exited(ExitEvidence),
}

#[derive(Debug, PartialEq, Eq)]
pub enum JournalRead {
    Missing,
    Incomplete(JournalSnapshot),
    Complete(JournalSnapshot),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidReason {
    JsonSyntax,
    UnsupportedSchemaVersion,
    UnsupportedRecordType,
    ClosedSchemaViolation,
    InvalidNamedField { field: &'static str },
    SequenceMismatch,
    FirstRecordNotWrapper,
    DuplicateWrapper,
    AttemptIdDrift,
    LaunchNonceDrift,
    InvalidTypedRecordOrdering,
    RecordsAfterClose,
    InvalidExitVariant,
    EmptyJournal,
    BlankRecord,
    UnterminatedRecord,
}

impl fmt::Display for InvalidReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonSyntax => f.write_str("invalid JSON syntax"),
            Self::UnsupportedSchemaVersion => f.write_str("unsupported schema version"),
            Self::UnsupportedRecordType => f.write_str("unsupported record type"),
            Self::ClosedSchemaViolation => f.write_str("closed schema violation"),
            Self::InvalidNamedField { field } => write!(f, "invalid {field}"),
            Self::SequenceMismatch => f.write_str("sequence mismatch"),
            Self::FirstRecordNotWrapper => f.write_str("first record is not wrapper_started"),
            Self::DuplicateWrapper => f.write_str("duplicate wrapper_started"),
            Self::AttemptIdDrift => f.write_str("attempt ID drift"),
            Self::LaunchNonceDrift => f.write_str("launch nonce drift"),
            Self::InvalidTypedRecordOrdering => f.write_str("invalid typed record ordering"),
            Self::RecordsAfterClose => f.write_str("record after outputs_closed"),
            Self::InvalidExitVariant => f.write_str("invalid exit variant"),
            Self::EmptyJournal => f.write_str("empty journal"),
            Self::BlankRecord => f.write_str("blank record"),
            Self::UnterminatedRecord => f.write_str("unterminated final record"),
        }
    }
}

#[derive(Debug)]
pub enum LifecycleError {
    Io {
        path: PathBuf,
        operation: &'static str,
        source: std::io::Error,
    },
    Conflict {
        path: PathBuf,
    },
    Invalid {
        path: PathBuf,
        line: Option<usize>,
        sequence: Option<u64>,
        reason: InvalidReason,
    },
    Poisoned {
        path: PathBuf,
    },
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                path,
                operation,
                source,
            } => write!(f, "{operation} {}: {source}", path.display()),
            Self::Conflict { path } => {
                write!(f, "lifecycle target already exists: {}", path.display())
            }
            Self::Invalid {
                path,
                line,
                sequence,
                reason,
            } => write!(
                f,
                "invalid lifecycle {} (line {line:?}, sequence {sequence:?}): {reason}",
                path.display()
            ),
            Self::Poisoned { path } => {
                write!(f, "lifecycle writer is poisoned: {}", path.display())
            }
        }
    }
}
impl std::error::Error for LifecycleError {}

pub(crate) fn invalid(
    path: &Path,
    line: Option<usize>,
    sequence: Option<u64>,
    reason: InvalidReason,
) -> LifecycleError {
    LifecycleError::Invalid {
        path: path.to_path_buf(),
        line,
        sequence,
        reason,
    }
}

pub(crate) fn named(value: &str, field: &'static str) -> Result<(), InvalidReason> {
    (!value.is_empty())
        .then_some(())
        .ok_or(InvalidReason::InvalidNamedField { field })
}

pub(crate) fn validate_manifest(manifest: &WrapperManifest) -> Result<(), InvalidReason> {
    named(&manifest.logical_run_id, "logical_run_id")?;
    named(&manifest.attempt_id, "manifest attempt_id")?;
    if manifest.argv.is_empty() || manifest.argv.iter().any(String::is_empty) {
        return Err(InvalidReason::InvalidNamedField { field: "argv" });
    }
    if manifest.wrapper_pid == 0 {
        return Err(InvalidReason::InvalidNamedField {
            field: "wrapper_pid",
        });
    }
    if manifest.process_group_id == 0 {
        return Err(InvalidReason::InvalidNamedField {
            field: "process_group_id",
        });
    }
    named(&manifest.hostname, "hostname")?;
    for value in [
        &manifest.parent_attempt_id,
        &manifest.boot_id,
        &manifest.process_start_id,
    ]
    .into_iter()
    .flatten()
    {
        named(value, "identity field")?;
    }
    Ok(())
}

pub(crate) fn validate_payload(payload: &LifecyclePayload) -> Result<(), InvalidReason> {
    match payload {
        LifecyclePayload::WrapperStarted(manifest) => validate_manifest(manifest),
        LifecyclePayload::ChildStarted { child_pid } if *child_pid == 0 => {
            Err(InvalidReason::InvalidNamedField { field: "child_pid" })
        }
        LifecyclePayload::ChildStarted { .. } => Ok(()),
        LifecyclePayload::ChildSpawnFailed { stage, error } => {
            named(stage, "stage")?;
            named(error, "error")
        }
        LifecyclePayload::ChildExited {
            outcome: ExitEvidence::WaitFailed { error },
        } => named(error, "wait failure"),
        LifecyclePayload::ChildExited { .. } => Ok(()),
        LifecyclePayload::OutputsClosed { outputs } => outputs
            .iter()
            .try_for_each(|output| named(&output.path, "output path")),
    }
}

mod reader;
mod writer;
pub use reader::read_journal;
pub use writer::LifecycleWriter;

#[cfg(test)]
mod tests;
