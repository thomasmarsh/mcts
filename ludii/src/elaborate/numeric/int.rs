//! Elaboration of [`crate::ast::numeric::int::IntFunction`] (Language Reference chapter 10) --
//! by far the largest ludeme in the grammar (board/state/math queries, ~60 variants).
//!
//! Only the bare integer literal is elaborated so far, since it's all [`crate::elaborate::game`]
//! needs for `(players <int>)`. The rest is future work, one sub-chapter at a time.

use crate::ast::located::Located;
use crate::ast::numeric::int::IntFunction;
use crate::elaborate::ElaborateError;
use crate::parse::SExpr;

pub fn elaborate_int_function(v: &Located<SExpr>) -> Result<IntFunction, ElaborateError> {
    match &v.node {
        SExpr::Int(n) => Ok(IntFunction::Int(*n)),
        other => Err(ElaborateError {
            message: format!(
                "unsupported intFunction {other:?} (only integer literals are elaborated so far)"
            ),
            span: v.span,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn int_literal() {
        let forms = parse("42").unwrap();
        assert_eq!(
            elaborate_int_function(&forms[0]).unwrap(),
            IntFunction::Int(42)
        );
    }

    #[test]
    fn negative_int_literal() {
        let forms = parse("-3").unwrap();
        assert_eq!(
            elaborate_int_function(&forms[0]).unwrap(),
            IntFunction::Int(-3)
        );
    }

    #[test]
    fn unsupported_form_errors() {
        let forms = parse("(mover)").unwrap();
        assert!(elaborate_int_function(&forms[0]).is_err());
    }
}
