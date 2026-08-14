//! Elaboration of a parsed [`crate::parse::SExpr`] tree into the typed [`crate::ast`].
//!
//! Built incrementally, one language-reference chapter at a time, mirroring [`crate::ast`]'s own
//! module layout -- a ludeme with no elaboration function yet simply isn't supported. See each
//! submodule for how far it currently reaches.

pub mod game;
pub mod numeric;
pub mod types;

use crate::ast::located::{Located, Span};
use crate::parse::{Call, Head, SExpr};

/// An elaboration failure, with the byte span in the source that produced it. Reuses
/// [`crate::parse::ParseError`]'s shape, since both stages just need a message tied to a span.
pub type ElaborateError = crate::parse::ParseError;

/// Checks that `v` is a `(name ...)` call, and returns the [`Call`].
fn call_ident<'a>(v: &'a Located<SExpr>, name: &str) -> Result<&'a Call, ElaborateError> {
    let SExpr::Call(call) = &v.node else {
        return Err(ElaborateError {
            message: format!("expected ({name} ...), found {:?}", v.node),
            span: v.span,
        });
    };
    match &call.head {
        Head::Ident(s) if s == name => Ok(call),
        other => Err(ElaborateError {
            message: format!("expected ({name} ...), found call head {other:?}"),
            span: v.span,
        }),
    }
}

/// Returns `call`'s single positional (unnamed) argument, or an error if it has zero or more
/// than one.
fn one_positional_arg(call: &Call, call_span: Span) -> Result<&Located<SExpr>, ElaborateError> {
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let arg = positional.next().ok_or_else(|| ElaborateError {
        message: "expected one positional argument, found none".into(),
        span: call_span,
    })?;
    if positional.next().is_some() {
        return Err(ElaborateError {
            message: "expected exactly one positional argument, found more than one".into(),
            span: call_span,
        });
    }
    Ok(&arg.value)
}
