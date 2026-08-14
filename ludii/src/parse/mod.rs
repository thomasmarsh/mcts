//! Front end for `.lud` source text: a lexer and a generic, semantics-free s-expression parser.
//!
//! This stage knows the surface syntax of the language (parens, `{}` lists, literals, `key:value`
//! named arguments, option references, ranges, define calls, option-priority markers) but nothing
//! about what any particular ludeme means. It exists so that later work -- turning a parsed form
//! into a [`crate::ast`] node -- can be built and tested per chapter against a [`sexpr::SExpr`]
//! value, independent of tokenizing and bracket-matching concerns.
//!
//! See [`sexpr::parse`] for the entry point.

mod lexer;
pub mod sexpr;

pub use sexpr::{parse, Arg, Call, Head, SExpr};

use crate::ast::located::Span;

/// A lex or parse failure, with the byte span in the source that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        ParseError {
            message: message.into(),
            span,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {}..{})", self.message, self.span.start, self.span.end)
    }
}

impl std::error::Error for ParseError {}
