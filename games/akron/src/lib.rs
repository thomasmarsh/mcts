//! Akron (pyramidal connection game on `pyramid::Pyramid`).
//!
//! This crate currently holds only the over/under-cut-aware connectivity
//! module ([`connectivity`]) built on top of `pyramid`'s placement/movement
//! primitives and `pyramid::crossing`'s crossing geometry. The full
//! `mcts::game::Game` implementation (state, actions, legality, win
//! condition) is not wired up yet.

pub mod connectivity;
