//! Ludii game description language support.
//!
//! [`parse`] turns `.lud` source text into a generic, semantics-free s-expression tree.
//! [`ast`] is the target typed tree for that syntax. [`elaborate`] turns one into the other,
//! built incrementally per language-reference chapter. [`core`] lowers a self-contained `ast`
//! game into a small backend-agnostic Core IR (see `DESIGN.md`) and can interpret it directly
//! against a concrete board.

pub mod ast;
pub mod core;
pub mod elaborate;
pub mod parse;
