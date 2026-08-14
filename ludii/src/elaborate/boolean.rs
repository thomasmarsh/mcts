//! Elaboration of [`crate::ast::boolean`] (Language Reference chapter 9): ludemes returning
//! true/false about the current game state.
//!
//! Only `(is Line <int>)` (9.7.1's `Line` form, with no `[<siteType>]`, `[<absoluteDirection>]`,
//! `through:`, `[<roleType>]`, `exact:`, `contiguous:`, `if:`, or `byLevel:`) is elaborated so
//! far, since it's all [`crate::elaborate::rules::end`] needs for `(is Line 3)`.

use crate::ast::boolean::{BooleanFunction, Is, IsLine};
use crate::ast::located::Located;
use crate::elaborate::numeric::int::elaborate_int_function;
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::SExpr;

/// `(is Line <int>)` (9.7.1).
fn elaborate_is(v: &Located<SExpr>) -> Result<Is, ElaborateError> {
    let call = call_ident(v, "is")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let kind_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(is ...) requires a kind argument".into(),
        span: v.span,
    })?;
    let SExpr::Ident(kind) = &kind_arg.value.node else {
        return Err(ElaborateError {
            message: format!(
                "expected an is-kind identifier, found {:?}",
                kind_arg.value.node
            ),
            span: kind_arg.value.span,
        });
    };
    match kind.as_str() {
        "Line" => {
            let min_length_arg = positional.next().ok_or_else(|| ElaborateError {
                message: "(is Line ...) requires a minLength argument".into(),
                span: v.span,
            })?;
            let min_length = elaborate_int_function(&min_length_arg.value)?;
            Ok(Is::Line(Box::new(IsLine {
                site_type: None,
                min_length: Box::new(Located::new(min_length, min_length_arg.value.span)),
                direction: None,
                through: None,
                owner: None,
                what: None,
                exact: None,
                contiguous: None,
                condition: None,
                by_level: None,
            })))
        }
        other => Err(ElaborateError {
            message: format!("unsupported (is {other} ...) -- only Line is elaborated so far"),
            span: kind_arg.value.span,
        }),
    }
}

/// Any ludeme computing a boolean value. Only [`elaborate_is`]'s `Line` form is wired up so far.
pub fn elaborate_boolean_function(v: &Located<SExpr>) -> Result<BooleanFunction, ElaborateError> {
    Ok(BooleanFunction::Is(Box::new(elaborate_is(v)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::numeric::int::IntFunction;
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn is_line() {
        let Is::Line(line) = elaborate_is(&parse_one("(is Line 3)")).unwrap() else {
            panic!("expected Is::Line");
        };
        assert_eq!(line.min_length.node, IntFunction::Int(3));
        assert_eq!(line.owner, None);
    }

    #[test]
    fn boolean_function_wraps_is() {
        assert!(matches!(
            elaborate_boolean_function(&parse_one("(is Line 3)")).unwrap(),
            BooleanFunction::Is(_)
        ));
    }

    #[test]
    fn unsupported_kind_errors() {
        assert!(elaborate_is(&parse_one("(is Full)")).is_err());
    }
}
