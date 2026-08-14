//! Elaboration of [`crate::ast::graph`] (Language Reference chapter 4): board graph generators,
//! shapes and operators.
//!
//! Only `(square <dim>)` -- a single-dimension square tiling, e.g. `(square 3)` -- is elaborated
//! so far, since it's all [`crate::elaborate::equipment`] needs for a `(board (square ...))`.
//! The rest of chapter 4 (other generators, shapes, operators) is future work.

use crate::ast::graph::generator::{Extent, Square};
use crate::ast::graph::GraphFunction;
use crate::ast::located::Located;
use crate::elaborate::numeric::dim::elaborate_dim_function;
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::{Head, SExpr};

/// `(square <dim>)` (4.5.2), restricted to the single-dimension form (no explicit columns,
/// shape, or diagonals modifier).
pub fn elaborate_graph_function(v: &Located<SExpr>) -> Result<GraphFunction, ElaborateError> {
    let SExpr::Call(raw_call) = &v.node else {
        return Err(ElaborateError {
            message: format!("expected a graph function call, found {:?}", v.node),
            span: v.span,
        });
    };
    match &raw_call.head {
        Head::Ident(s) if s == "square" => {
            let call = call_ident(v, "square")?;
            let arg = one_positional_arg(call, v.span)?;
            let dim = elaborate_dim_function(arg)?;
            Ok(GraphFunction::Square(Square {
                shape: None,
                extent: Extent::Dims(Box::new(Located::new(dim, arg.span)), None),
                modifier: None,
            }))
        }
        other => Err(ElaborateError {
            message: format!("unsupported graph function head {other:?}"),
            span: v.span,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::numeric::dim::DimFunction;
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn square() {
        let GraphFunction::Square(sq) = elaborate_graph_function(&parse_one("(square 3)")).unwrap()
        else {
            panic!("expected a Square");
        };
        assert_eq!(sq.shape, None);
        assert_eq!(sq.modifier, None);
        let Extent::Dims(rows, cols) = sq.extent else {
            panic!("expected Dims extent");
        };
        assert_eq!(rows.node, DimFunction::Int(3));
        assert_eq!(cols, None);
    }

    #[test]
    fn unsupported_generator_errors() {
        assert!(elaborate_graph_function(&parse_one("(hex Diamond 9)")).is_err());
    }
}
