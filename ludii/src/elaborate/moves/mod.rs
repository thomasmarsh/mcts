//! Elaboration of [`crate::ast::moves`] (Language Reference chapter 8), the largest chapter in
//! the language.
//!
//! Only [`decision`] has any elaboration so far -- effects, requirements, state moves and
//! generator/combinator operators are all future work.

pub mod decision;

use crate::ast::located::Located;
use crate::ast::moves::Moves;
use crate::elaborate::ElaborateError;
use crate::parse::SExpr;

/// Any ludeme producing a move or list of moves. Only [`decision::elaborate_decision`] is wired
/// up so far.
pub fn elaborate_moves(v: &Located<SExpr>) -> Result<Moves, ElaborateError> {
    Ok(Moves::Decision(Box::new(decision::elaborate_decision(v)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::moves::decision::Decision;
    use crate::parse::parse;

    #[test]
    fn moves_wraps_decision() {
        let forms = parse("(move Add (to (sites Empty)))").unwrap();
        let Moves::Decision(decision) = elaborate_moves(&forms[0]).unwrap() else {
            panic!("expected a Decision");
        };
        assert!(matches!(*decision, Decision::Site { .. }));
    }
}
