//! Batched MCTS: `B` game trees advanced together, one simulation round at
//! a time, so their leaf evaluations can be scored in a single batched
//! oracle call instead of one call per leaf. Ported from AlphaZero.jl's
//! `BatchedMcts` module (Guillaume Thomas' 2022 GSoC "batched MCTS" work,
//! `jonathan-laurent/AlphaZero.jl#147`).
//!
//! This is a deliberately separate implementation from `crates/mcts`'s
//! per-node arena `TreeSearch`, which the rest of this workspace's ~20
//! games use and which cannot be bent into this shape (one tree searched
//! to completion before the next begins) without breaking every one of
//! them. This crate shares only the `Game`/`Evaluator`/`PolicyLogits`
//! contracts, via `crate::othello`'s concrete oracle.

pub mod gumbel;
pub mod oracle;
pub mod othello;
pub mod search;
pub mod selfplay;
pub mod tree;

pub use gumbel::{gumbel_explore, gumbel_explore_with_noise, gumbel_selected_action};
pub use oracle::{EnvOracle, StepOutput, TransitionOutput};
pub use search::{explore, improved_policy, Config};
pub use tree::Tree;
