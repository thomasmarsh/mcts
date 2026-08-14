//! Elaboration of [`crate::ast::game`] (Language Reference chapter 2): the `<players>`/`<mode>`
//! children of the root `(game ...)` ludeme.
//!
//! `Equipment` and `Rules` -- the other two children of [`crate::ast::game::Game`] -- aren't
//! elaborated yet, so an `elaborate_game`/`elaborate_description` covering the whole `(game ...)`
//! or `(match ...)` form doesn't exist yet either; this covers only the self-contained pieces.

use crate::ast::game::{Mode, Player, Players};
use crate::ast::located::Located;
use crate::elaborate::numeric::int::elaborate_int_function;
use crate::elaborate::types::{elaborate_compass_direction, elaborate_mode_type};
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::SExpr;

/// `(player <directionFacing>)` (2.4.1).
pub fn elaborate_player(v: &Located<SExpr>) -> Result<Player, ElaborateError> {
    let call = call_ident(v, "player")?;
    let facing = elaborate_compass_direction(one_positional_arg(call, v.span)?)?;
    Ok(Player { facing })
}

/// `(players ...)` (2.4.2): either an explicit `{(player ...) ...}` list, or a plain count.
pub fn elaborate_players(v: &Located<SExpr>) -> Result<Players, ElaborateError> {
    let call = call_ident(v, "players")?;
    let arg = one_positional_arg(call, v.span)?;
    match &arg.node {
        SExpr::List(items) => Ok(Players::List(
            items
                .iter()
                .map(elaborate_player)
                .collect::<Result<_, _>>()?,
        )),
        _ => Ok(Players::Count(Box::new(Located::new(
            elaborate_int_function(arg)?,
            arg.span,
        )))),
    }
}

/// `(mode <modeType>)` (2.3.1).
pub fn elaborate_mode(v: &Located<SExpr>) -> Result<Mode, ElaborateError> {
    let call = call_ident(v, "mode")?;
    Ok(Mode(elaborate_mode_type(one_positional_arg(
        call, v.span,
    )?)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::located::Span;
    use crate::ast::numeric::int::IntFunction;
    use crate::ast::types::{CompassDirection, ModeType};
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn player() {
        assert_eq!(
            elaborate_player(&parse_one("(player N)")).unwrap(),
            Player {
                facing: CompassDirection::N
            }
        );
    }

    #[test]
    fn players_count() {
        assert_eq!(
            elaborate_players(&parse_one("(players 2)")).unwrap(),
            Players::Count(Box::new(Located::new(
                IntFunction::Int(2),
                Span::new(9, 10)
            )))
        );
    }

    #[test]
    fn players_list() {
        let Players::List(players) =
            elaborate_players(&parse_one("(players { (player N) (player S) })")).unwrap()
        else {
            panic!("expected a list");
        };
        assert_eq!(
            players,
            vec![
                Player {
                    facing: CompassDirection::N
                },
                Player {
                    facing: CompassDirection::S
                },
            ]
        );
    }

    #[test]
    fn mode() {
        assert_eq!(
            elaborate_mode(&parse_one("(mode Simultaneous)")).unwrap(),
            Mode(ModeType::Simultaneous)
        );
    }

    #[test]
    fn wrong_head_errors() {
        assert!(elaborate_player(&parse_one("(mode Simultaneous)")).is_err());
    }
}
