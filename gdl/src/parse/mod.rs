//! A lexer and a generic, semantics-free s-expression parser: parens for calls, `{}` for lists,
//! ordinary literals, `key:value` named arguments. Originally built as Ludii's own `.lud` front
//! end (hence support for option references/define calls/priority markers this crate's own
//! grammar doesn't use), but nothing about the reader itself is Ludii-specific -- it's reused
//! as-is by [`crate::style_c`] for a completely different, non-Ludii s-expression vocabulary. See
//! [`sexpr::parse`] for the entry point.

mod lexer;
pub mod located;
pub mod sexpr;

pub use sexpr::{parse, Arg, Call, Head, SExpr};

use located::Span;

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
        write!(
            f,
            "{} (at byte {}..{})",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}
