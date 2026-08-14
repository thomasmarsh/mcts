//! Elaboration of [`crate::ast::game`] (Language Reference chapter 2): the root `(game ...)`
//! ludeme and its `<players>`/`<mode>`/`<equipment>`/`<rules>` children.
//!
//! `(match ...)` -- the other possible root of a `.lud` file -- isn't elaborated yet, so
//! [`elaborate_description`] only ever produces [`Description::Game`].

use crate::ast::game::{Description, Game, Mode, Player, Players};
use crate::ast::located::Located;
use crate::elaborate::equipment::elaborate_equipment;
use crate::elaborate::numeric::int::elaborate_int_function;
use crate::elaborate::rules::elaborate_rules;
use crate::elaborate::types::{elaborate_compass_direction, elaborate_mode_type};
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::{Head, SExpr};

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

/// `(game <string> <players> [<mode>] <equipment> <rules>)` (2.1.1).
pub fn elaborate_game(v: &Located<SExpr>) -> Result<Game, ElaborateError> {
    let call = call_ident(v, "game")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let name_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(game ...) requires a name argument".into(),
        span: v.span,
    })?;
    let SExpr::Str(name) = &name_arg.value.node else {
        return Err(ElaborateError {
            message: format!(
                "expected a game name string, found {:?}",
                name_arg.value.node
            ),
            span: name_arg.value.span,
        });
    };
    let name = name.clone();

    let mut players = None;
    let mut mode = None;
    let mut equipment = None;
    let mut rules = None;
    for arg in positional {
        let SExpr::Call(raw_call) = &arg.value.node else {
            return Err(ElaborateError {
                message: format!("expected a game child call, found {:?}", arg.value.node),
                span: arg.value.span,
            });
        };
        match &raw_call.head {
            Head::Ident(s) if s == "players" => players = Some(elaborate_players(&arg.value)?),
            Head::Ident(s) if s == "mode" => mode = Some(elaborate_mode(&arg.value)?),
            Head::Ident(s) if s == "equipment" => {
                equipment = Some(elaborate_equipment(&arg.value)?)
            }
            Head::Ident(s) if s == "rules" => rules = Some(elaborate_rules(&arg.value)?),
            other => {
                return Err(ElaborateError {
                    message: format!(
                        "unsupported (game ...) child head {other:?} -- only players, mode, \
                         equipment and rules are elaborated so far"
                    ),
                    span: arg.value.span,
                })
            }
        }
    }

    Ok(Game {
        name,
        players: players.ok_or_else(|| ElaborateError {
            message: "(game ...) requires a players argument".into(),
            span: v.span,
        })?,
        mode,
        equipment: equipment.ok_or_else(|| ElaborateError {
            message: "(game ...) requires an equipment argument".into(),
            span: v.span,
        })?,
        rules: rules.ok_or_else(|| ElaborateError {
            message: "(game ...) requires a rules argument".into(),
            span: v.span,
        })?,
    })
}

/// The root of a `.lud` file: `(game ...)` (2.1.1) or `(match ...)` (2.2.2). Only `(game ...)`
/// is elaborated so far.
pub fn elaborate_description(v: &Located<SExpr>) -> Result<Description, ElaborateError> {
    Ok(Description::Game(elaborate_game(v)?))
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

    #[test]
    fn tic_tac_toe_fixture_elaborates() {
        let src = include_str!("../../lud/Tic-Tac-Toe.lud");
        let forms = parse(src).unwrap();
        // (game ...), (metadata ...)
        assert_eq!(forms.len(), 2);

        let Description::Game(game) = elaborate_description(&forms[0]).unwrap() else {
            panic!("expected Description::Game");
        };
        assert_eq!(game.name, "Tic-Tac-Toe");
        let Players::Count(count) = &game.players else {
            panic!("expected Players::Count");
        };
        assert_eq!(count.node, IntFunction::Int(2));
        assert_eq!(game.mode, None);
        assert_eq!(game.equipment.items.len(), 3);
        assert!(game.rules.play.is_some());
        assert_eq!(game.rules.end.as_ref().unwrap().rules.len(), 1);
    }
}
