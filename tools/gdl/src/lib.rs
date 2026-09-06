//! Ludii game description language support.
//!
//! [`core`] is a small, backend-agnostic Core IR (see `DESIGN.md`) that can interpret a
//! [`core::Program`] directly against a concrete board. [`style_c`] is its frontend: a direct
//! s-expression encoding of Core IR, built on [`parse::sexpr`]'s generic reader. [`codegen`] is a
//! second backend, lowering a [`core::Program`] to the text of a standalone Rust `games/*` crate
//! instead of interpreting it.
//!
//! `.lud` source (Ludii's own game description language, `database-1/lud/games/`) is spec/oracle
//! material read by a person or an LLM, not loaded by any code here; this crate no longer has a
//! `.lud`-parsing frontend.

pub mod codegen;
pub mod core;
pub mod parse;
pub mod style_c;
