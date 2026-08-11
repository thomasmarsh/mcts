//! Shared test infrastructure for MCTS framework + game integration.
//!
//! This crate is a workspace member that depends on `mcts` and all
//! extracted game crates, providing a home for:
//!
//! - Stress tests (oracle comparisons, long-running randomised checks)
//! - Integration tests that cross the `mcts` ↔ game boundary
//! - Eventually, MCTS-internals tests that access `TreeSearch` fields
//!   (once those fields are made `pub` in the `mcts` crate)
