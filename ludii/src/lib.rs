//! Ludii game description language support.
//!
//! [`parse`] turns `.lud` source text into a generic, semantics-free s-expression tree.
//! [`ast`] is the target typed tree for that syntax. [`elaborate`] turns one into the other,
//! built incrementally per language-reference chapter.

pub mod ast;
pub mod elaborate;
pub mod parse;
