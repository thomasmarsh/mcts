use crate::projects_attempt;

#[derive(Debug)]
pub enum IngestError {
    DuckDb(duckdb::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidMoveReport { message: String },
    OrphanCell { run_id: String, cell_id: String },
    Attempt(projects_attempt::ProjectsError),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::DuckDb(e) => write!(f, "DuckDB error: {e}"),
            IngestError::Io(e) => write!(f, "I/O error: {e}"),
            IngestError::Json(e) => write!(f, "JSON error: {e}"),
            IngestError::InvalidMoveReport { message } => {
                write!(f, "invalid move search report: {message}")
            }
            IngestError::OrphanCell { run_id, cell_id } => {
                write!(f, "cell '{cell_id}' does not belong to run '{run_id}'")
            }
            IngestError::Attempt(error) => write!(f, "typed attempt ingest error: {error}"),
        }
    }
}

impl std::error::Error for IngestError {}

impl From<duckdb::Error> for IngestError {
    fn from(e: duckdb::Error) -> Self {
        IngestError::DuckDb(e)
    }
}

impl From<std::io::Error> for IngestError {
    fn from(e: std::io::Error) -> Self {
        IngestError::Io(e)
    }
}

impl From<serde_json::Error> for IngestError {
    fn from(e: serde_json::Error) -> Self {
        IngestError::Json(e)
    }
}

impl From<projects_attempt::ProjectsError> for IngestError {
    fn from(error: projects_attempt::ProjectsError) -> Self {
        Self::Attempt(error)
    }
}
