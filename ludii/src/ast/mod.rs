//! The Ludii game description language AST, mirroring the structure of
//! `LudiiLanguageReference.md`. Parsing is not implemented yet -- this crate currently only
//! defines the target tree that a future S-expression parser will build.
//!
//! Module layout follows the reference document's own parts and chapters:
//!
//! - [`located`] / [`value`]: shared infrastructure (span tracking, and the generic fallback
//!   representation used for ludemes not yet given a dedicated type).
//! - [`types`] / [`common`]: cross-cutting enums and small utility ludemes referenced
//!   throughout the rest of the grammar.
//! - [`game`] / [`equipment`] / [`graph`] / [`numeric`] (chapters 2-6): the equipment half of a
//!   game description.
//! - [`rules`] / [`moves`] / [`boolean`] / [`region`] / [`direction`] (chapters 7-9, 12-13): the
//!   rules half.
//! - [`metadata`] / [`metalanguage`] (parts II-III): lightly modeled for now.

pub mod common;
pub mod equipment;
pub mod game;
pub mod graph;
pub mod located;
pub mod types;
pub mod value;

pub mod boolean;
pub mod direction;
pub mod numeric;
pub mod region;

pub mod moves;
pub mod rules;

pub mod metadata;
pub mod metalanguage;
