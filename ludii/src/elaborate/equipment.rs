//! Elaboration of [`crate::ast::equipment`] (Language Reference chapter 3): what a game is
//! played with.
//!
//! Only `(board <graphFunction>)` (3.4.1, graph only -- no tracks/values/use/largeStack),
//! `(piece <string> [<roleType>])` (3.2.4, name/owner only -- no facing/flips/moves/max*), and
//! `(regions <roleType> {<regionFunction>})` (3.7.4, an explicit region-function list only -- no
//! sites/single-region/static forms) are elaborated so far, since they're all
//! [`crate::elaborate::game`] needs for Tic-Tac-Toe's `(equipment { (board (square 3)) (piece
//! "Disc" P1) (piece "Cross" P2) })` and Hex's `(regions P1 {(sites Side NE) (sites Side SW)})`.

use crate::ast::equipment::component::Piece;
use crate::ast::equipment::container::Board;
use crate::ast::equipment::other::{Regions, RegionsSpec};
use crate::ast::equipment::{Equipment, Item};
use crate::ast::located::Located;
use crate::elaborate::graph::elaborate_graph_function;
use crate::elaborate::region::elaborate_region_function;
use crate::elaborate::types::elaborate_role_type;
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::{Call, Head, SExpr};

/// Returns `call`'s positional (unnamed) arguments, in order.
fn positional_args(call: &Call) -> impl Iterator<Item = &Located<SExpr>> {
    call.args
        .iter()
        .filter(|a| a.name.is_none())
        .map(|a| &a.value)
}

fn expect_str(v: &Located<SExpr>) -> Result<&str, ElaborateError> {
    match &v.node {
        SExpr::Str(s) => Ok(s.as_str()),
        other => Err(ElaborateError {
            message: format!("expected a string, found {other:?}"),
            span: v.span,
        }),
    }
}

/// `(board <graphFunction>)` (3.4.1).
pub fn elaborate_board(v: &Located<SExpr>) -> Result<Board, ElaborateError> {
    let call = call_ident(v, "board")?;
    let mut args = positional_args(call);
    let graph = args.next().ok_or_else(|| ElaborateError {
        message: "(board ...) requires a graph function argument".into(),
        span: v.span,
    })?;
    Ok(Board {
        graph: Box::new(Located::new(elaborate_graph_function(graph)?, graph.span)),
        tracks: Vec::new(),
        values: Vec::new(),
        use_site_type: None,
        large_stack: None,
    })
}

/// `(piece <string> [<roleType>])` (3.2.4), name and owner only.
pub fn elaborate_piece(v: &Located<SExpr>) -> Result<Piece, ElaborateError> {
    let call = call_ident(v, "piece")?;
    let mut args = positional_args(call);
    let name = args
        .next()
        .ok_or_else(|| ElaborateError {
            message: "(piece ...) requires a name argument".into(),
            span: v.span,
        })
        .and_then(expect_str)?
        .to_string();
    let owner = args.next().map(elaborate_role_type).transpose()?;
    Ok(Piece {
        name,
        owner,
        facing: None,
        flips: None,
        moves: None,
        max_state: None,
        max_count: None,
        max_value: None,
    })
}

/// `(regions <roleType> {<regionFunction>})` (3.7.4), an explicit region-function list only.
pub fn elaborate_regions(v: &Located<SExpr>) -> Result<Regions, ElaborateError> {
    let call = call_ident(v, "regions")?;
    let mut args = positional_args(call);
    let owner_arg = args.next().ok_or_else(|| ElaborateError {
        message: "(regions ...) requires an owner argument".into(),
        span: v.span,
    })?;
    let owner = elaborate_role_type(owner_arg)?;
    let list_arg = args.next().ok_or_else(|| ElaborateError {
        message: "(regions ...) requires a region-function list argument".into(),
        span: v.span,
    })?;
    let SExpr::List(items) = &list_arg.node else {
        return Err(ElaborateError {
            message: format!("expected a {{...}} region list, found {:?}", list_arg.node),
            span: list_arg.span,
        });
    };
    let regions = items
        .iter()
        .map(|item| {
            Ok(Box::new(Located::new(
                elaborate_region_function(item)?,
                item.span,
            )))
        })
        .collect::<Result<_, ElaborateError>>()?;
    Ok(Regions {
        name: None,
        owner: Some(owner),
        spec: RegionsSpec::Regions(regions),
        hint_name: None,
    })
}

/// A single `(equipment {...})` entry. Only [`Item::Board`], [`Item::Piece`], and
/// [`Item::Regions`] are elaborated so far -- add more arms as later chapters need them.
fn elaborate_item(v: &Located<SExpr>) -> Result<Item, ElaborateError> {
    let SExpr::Call(raw_call) = &v.node else {
        return Err(ElaborateError {
            message: format!("expected an equipment item call, found {:?}", v.node),
            span: v.span,
        });
    };
    match &raw_call.head {
        Head::Ident(s) if s == "board" => Ok(Item::Board(elaborate_board(v)?)),
        Head::Ident(s) if s == "piece" => Ok(Item::Piece(elaborate_piece(v)?)),
        Head::Ident(s) if s == "regions" => Ok(Item::Regions(elaborate_regions(v)?)),
        other => Err(ElaborateError {
            message: format!("unsupported equipment item head {other:?}"),
            span: v.span,
        }),
    }
}

/// `(equipment {<item>})` (3.1.1).
pub fn elaborate_equipment(v: &Located<SExpr>) -> Result<Equipment, ElaborateError> {
    let call = call_ident(v, "equipment")?;
    let mut args = positional_args(call);
    let list = args.next().ok_or_else(|| ElaborateError {
        message: "(equipment ...) requires an item list argument".into(),
        span: v.span,
    })?;
    let SExpr::List(items) = &list.node else {
        return Err(ElaborateError {
            message: format!("expected a {{...}} item list, found {:?}", list.node),
            span: list.span,
        });
    };
    Ok(Equipment {
        items: items.iter().map(elaborate_item).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::generator::Extent;
    use crate::ast::graph::GraphFunction;
    use crate::ast::numeric::dim::DimFunction;
    use crate::ast::types::RoleType;
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn board() {
        let board = elaborate_board(&parse_one("(board (square 3))")).unwrap();
        let GraphFunction::Square(sq) = &board.graph.node else {
            panic!("expected a Square graph");
        };
        let Extent::Dims(rows, _) = &sq.extent else {
            panic!("expected Dims extent");
        };
        assert_eq!(rows.node, DimFunction::Int(3));
        assert!(board.tracks.is_empty());
    }

    #[test]
    fn piece_with_owner() {
        let piece = elaborate_piece(&parse_one(r#"(piece "Disc" P1)"#)).unwrap();
        assert_eq!(piece.name, "Disc");
        assert_eq!(piece.owner, Some(RoleType::P1));
    }

    #[test]
    fn piece_without_owner() {
        let piece = elaborate_piece(&parse_one(r#"(piece "Ball")"#)).unwrap();
        assert_eq!(piece.name, "Ball");
        assert_eq!(piece.owner, None);
    }

    #[test]
    fn equipment_list() {
        let equipment = elaborate_equipment(&parse_one(
            r#"(equipment { (board (square 3)) (piece "Disc" P1) (piece "Cross" P2) })"#,
        ))
        .unwrap();
        assert_eq!(equipment.items.len(), 3);
        assert!(matches!(equipment.items[0], Item::Board(_)));
        assert!(matches!(equipment.items[1], Item::Piece(_)));
        assert!(matches!(equipment.items[2], Item::Piece(_)));
    }
}
