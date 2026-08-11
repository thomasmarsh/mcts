//! Build-time attribution: git SHA and worktree-dirty flag baked in by
//! `build.rs`. Every run/experiment row in the database carries these so
//! regressions can be found by filtering on `git_sha` rather than squinting
//! at timestamps.

/// Full SHA of `HEAD` at compile time, or `"unknown"` if `git rev-parse`
/// failed (e.g. building outside a git checkout).
pub const GIT_SHA: &str = env!("GIT_SHA");

/// `"true"` if the worktree had uncommitted changes at compile time,
/// `"false"` if it was clean. Non-canonical builds (git unavailable) also
/// produce `"false"`.
pub const GIT_DIRTY: &str = env!("GIT_DIRTY");
