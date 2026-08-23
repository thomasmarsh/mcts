use std::fmt;

/// A simple, HTTP-style error type used by the `GameAdapter` trait methods.
///
/// Carries an integer code (matching HTTP status conventions) and a
/// human-readable message.  The `run_host` function converts these into
/// `Response::Error` when a method fails.  No external HTTP framework
/// dependency — the server crate wraps this in its own `AdapterError` if
/// axum integration is needed.
#[derive(Debug)]
pub struct HostError {
    pub code: u16,
    pub message: String,
}

impl HostError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            code: 400,
            message: message.into(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: 404,
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: 500,
            message: message.into(),
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for HostError {}
