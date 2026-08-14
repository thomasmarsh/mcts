//! Elaboration of [`crate::ast::numeric::dim::DimFunction`] (Language Reference chapter 5):
//! small integer-valued expressions used for board dimensions, e.g. the `3` in `(square 3)`.
//!
//! Only the bare integer literal is elaborated so far, since it's all [`crate::elaborate::graph`]
//! needs for a fixed-size board. The arithmetic variants (`Add`, `Mul`, ...) are future work.

use crate::ast::located::Located;
use crate::ast::numeric::dim::DimFunction;
use crate::elaborate::ElaborateError;
use crate::parse::SExpr;

pub fn elaborate_dim_function(v: &Located<SExpr>) -> Result<DimFunction, ElaborateError> {
    match &v.node {
        SExpr::Int(n) => Ok(DimFunction::Int(*n)),
        other => Err(ElaborateError {
            message: format!(
                "unsupported dimFunction {other:?} (only integer literals are elaborated so far)"
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
        let forms = parse("3").unwrap();
        assert_eq!(
            elaborate_dim_function(&forms[0]).unwrap(),
            DimFunction::Int(3)
        );
    }

    #[test]
    fn unsupported_form_errors() {
        let forms = parse("(add 1 2)").unwrap();
        assert!(elaborate_dim_function(&forms[0]).is_err());
    }
}
