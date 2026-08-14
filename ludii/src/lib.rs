//! Ludii game description language support.
//!
//! This crate currently provides only the target AST (see [`ast`]) for a future compiler
//! front end that will parse `.lud` S-expression source into it. There is no parser yet.

pub mod ast;
