//! Elaboration of [`crate::ast::graph`] (Language Reference chapter 4): board graph generators,
//! shapes and operators.
//!
//! Only `(square <dim>)` -- a single-dimension square tiling, e.g. `(square 3)` -- and
//! `(hex Diamond <dim>)` -- a single-dimension rhombus hex tiling, e.g. `(hex Diamond 3)` -- are
//! elaborated so far, since between them that's all [`crate::elaborate::equipment`] needs for
//! `(board (square ...))`/`(board (hex Diamond ...))`. The rest of chapter 4 (other generators,
//! shapes, operators) is future work.

use crate::ast::graph::generator::{Extent, Hex, HexShapeType, Square};
use crate::ast::graph::GraphFunction;
use crate::ast::located::Located;
use crate::elaborate::numeric::dim::elaborate_dim_function;
use crate::elaborate::types::elaborate_hex_shape_type;
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::{Head, SExpr};

/// `(square <dim>)` (4.5.2), restricted to the single-dimension form (no explicit columns,
/// shape, or diagonals modifier).
fn elaborate_square(v: &Located<SExpr>) -> Result<Square, ElaborateError> {
    let call = call_ident(v, "square")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(square ...) requires a dim argument".into(),
        span: v.span,
    })?;
    if positional.next().is_some() {
        return Err(ElaborateError {
            message: "(square rows columns) isn't elaborated yet -- only square boards".into(),
            span: v.span,
        });
    }
    let dim = elaborate_dim_function(&arg.value)?;
    Ok(Square {
        shape: None,
        extent: Extent::Dims(Box::new(Located::new(dim, arg.value.span)), None),
        modifier: None,
    })
}

/// `(hex <hexShapeType> <dim>)` (4.3.1), restricted to `Diamond` with a single-dimension extent
/// (no explicit columns).
fn elaborate_hex(v: &Located<SExpr>) -> Result<Hex, ElaborateError> {
    let call = call_ident(v, "hex")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let shape_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(hex ...) requires a hexShapeType argument".into(),
        span: v.span,
    })?;
    let shape = elaborate_hex_shape_type(&shape_arg.value)?;
    if shape != HexShapeType::Diamond {
        return Err(ElaborateError {
            message: "only (hex Diamond ...) is elaborated so far".into(),
            span: shape_arg.value.span,
        });
    }
    let dim_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(hex Diamond ...) requires a dim argument".into(),
        span: v.span,
    })?;
    if positional.next().is_some() {
        return Err(ElaborateError {
            message: "(hex Diamond rows columns) isn't elaborated yet -- only diamond boards"
                .into(),
            span: v.span,
        });
    }
    let dim = elaborate_dim_function(&dim_arg.value)?;
    Ok(Hex {
        shape: Some(shape),
        extent: Extent::Dims(Box::new(Located::new(dim, dim_arg.value.span)), None),
    })
}

pub fn elaborate_graph_function(v: &Located<SExpr>) -> Result<GraphFunction, ElaborateError> {
    let SExpr::Call(raw_call) = &v.node else {
        return Err(ElaborateError {
            message: format!("expected a graph function call, found {:?}", v.node),
            span: v.span,
        });
    };
    match &raw_call.head {
        Head::Ident(s) if s == "square" => Ok(GraphFunction::Square(elaborate_square(v)?)),
        Head::Ident(s) if s == "hex" => Ok(GraphFunction::Hex(elaborate_hex(v)?)),
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
    fn hex_diamond() {
        let GraphFunction::Hex(hex) =
            elaborate_graph_function(&parse_one("(hex Diamond 3)")).unwrap()
        else {
            panic!("expected a Hex");
        };
        assert_eq!(
            hex.shape,
            Some(crate::ast::graph::generator::HexShapeType::Diamond)
        );
        let Extent::Dims(side, cols) = hex.extent else {
            panic!("expected Dims extent");
        };
        assert_eq!(side.node, DimFunction::Int(3));
        assert_eq!(cols, None);
    }

    #[test]
    fn hex_non_diamond_shape_errors() {
        assert!(elaborate_graph_function(&parse_one("(hex Hexagon 3)")).is_err());
    }

    #[test]
    fn unsupported_generator_errors() {
        assert!(elaborate_graph_function(&parse_one("(tri Diamond 3)")).is_err());
    }
}
