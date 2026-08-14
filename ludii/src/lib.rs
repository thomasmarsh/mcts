//! Ludii game description language support.
//!
//! [`parse`] turns `.lud` source text into a generic, semantics-free s-expression tree.
//! [`ast`] is the target typed tree for that syntax. [`elaborate`] turns one into the other,
//! built incrementally per language-reference chapter. [`core`] lowers a self-contained `ast`
//! game into a small backend-agnostic Core IR (see `DESIGN.md`) and can interpret it directly
//! against a concrete board.
//!
//! [`style_c`] is a second, independent frontend onto the same [`core::Program`]: a direct
//! s-expression encoding of Core IR, reusing [`parse::sexpr`]'s reader but bypassing `ast`/
//! `elaborate` entirely -- see that module's doc for why.

pub mod ast;
pub mod core;
pub mod elaborate;
pub mod parse;
pub mod style_c;
