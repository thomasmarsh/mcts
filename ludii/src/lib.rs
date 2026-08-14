//! Ludii game description language support.
//!
//! [`parse`] turns `.lud` source text into a generic, semantics-free s-expression tree.
//! [`ast`] is the target typed tree for that syntax; elaborating parsed forms into it is future
//! work, done incrementally per language-reference chapter.

pub mod ast;
pub mod parse;
