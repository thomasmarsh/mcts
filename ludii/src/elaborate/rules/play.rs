//! Elaboration of [`crate::ast::rules::play`] (Language Reference 7.6): the legal-move
//! generator of a game or phase.

use crate::ast::located::Located;
use crate::ast::rules::play::Play;
use crate::elaborate::moves::elaborate_moves;
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::SExpr;

/// `(play <moves>)` (7.6.1).
pub fn elaborate_play(v: &Located<SExpr>) -> Result<Play, ElaborateError> {
    let call = call_ident(v, "play")?;
    let arg = one_positional_arg(call, v.span)?;
    let moves = elaborate_moves(arg)?;
    Ok(Play(Box::new(Located::new(moves, arg.span))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn play() {
        let forms = parse("(play (move Add (to (sites Empty))))").unwrap();
        elaborate_play(&forms[0]).unwrap();
    }
}
