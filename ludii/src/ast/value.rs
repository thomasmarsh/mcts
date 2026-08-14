//! Generic fallback representation for ludemes not yet given a dedicated type.
//!
//! Ludii's grammar (see `LudiiLanguageReference.md`) runs to hundreds of distinct ludemes
//! across metadata and AI configuration alone. This AST models the core game-logic ludemes
//! (game/equipment/graph/rules/moves/functions) concretely, but deliberately leaves the long
//! tail -- mostly graphics and AI metadata -- as this generic `Ludeme` s-expression shape, so
//! that a `.lud` file using them can still round-trip instead of failing to parse.

use crate::ast::located::LBox;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// A generic `(name arg1 arg2 key:value ...)` call, for ludemes without a dedicated type.
#[derive(Debug, Clone, PartialEq)]
pub struct Ludeme {
    pub name: String,
    pub args: Vec<Arg>,
}

/// One argument to a [`Ludeme`]: either positional (`name: None`) or named (`key:value`).
#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Literal(Literal),
    Ludeme(LBox<Ludeme>),
    List(Vec<Value>),
}
