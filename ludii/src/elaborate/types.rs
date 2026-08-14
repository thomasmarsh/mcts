//! Elaboration of the plain enumerated ludemes in [`crate::ast::types`] -- constant identifiers
//! like `N` or `Simultaneous` that appear as bare argument values throughout the grammar.
//!
//! Only the enums actually needed by an implemented elaboration function are covered so far; add
//! more [`ident_enum`] invocations as later chapters need them.

use crate::ast::located::Located;
use crate::ast::types::{CompassDirection, ModeType, ResultType, RoleType};
use crate::elaborate::ElaborateError;
use crate::parse::SExpr;

/// Generates `fn $name(v: &Located<SExpr>) -> Result<$ty, ElaborateError>`, matching a bare
/// identifier against the enum's variant names (chosen throughout [`crate::ast::types`] to match
/// the `.lud` source vocabulary exactly).
macro_rules! ident_enum {
    ($name:ident -> $ty:ident { $($variant:ident),+ $(,)? }) => {
        pub fn $name(v: &Located<SExpr>) -> Result<$ty, ElaborateError> {
            let SExpr::Ident(s) = &v.node else {
                return Err(ElaborateError {
                    message: format!(
                        "expected a {} identifier, found {:?}",
                        stringify!($ty),
                        v.node
                    ),
                    span: v.span,
                });
            };
            match s.as_str() {
                $(stringify!($variant) => Ok($ty::$variant),)+
                _ => Err(ElaborateError {
                    message: format!("unknown {} {s:?}", stringify!($ty)),
                    span: v.span,
                }),
            }
        }
    };
}

ident_enum!(elaborate_compass_direction -> CompassDirection {
    N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW
});

ident_enum!(elaborate_mode_type -> ModeType {
    Alternating, Simultaneous, Simulation
});

ident_enum!(elaborate_role_type -> RoleType {
    Neutral, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16,
    Team1, Team2, Team3, Team4, Team5, Team6, Team7, Team8, Team9, Team10, Team11, Team12,
    Team13, Team14, Team15, Team16, TeamMover, Each, Shared, All, Mover, Next, Prev, NonMover,
    Enemy, Friend, Ally, Player,
});

ident_enum!(elaborate_result_type -> ResultType {
    Win, Loss, Draw, Tie, Abandon, Crash
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn compass_direction() {
        assert_eq!(
            elaborate_compass_direction(&parse_one("N")).unwrap(),
            CompassDirection::N
        );
        assert_eq!(
            elaborate_compass_direction(&parse_one("NNE")).unwrap(),
            CompassDirection::NNE
        );
    }

    #[test]
    fn mode_type() {
        assert_eq!(
            elaborate_mode_type(&parse_one("Simultaneous")).unwrap(),
            ModeType::Simultaneous
        );
    }

    #[test]
    fn role_type() {
        assert_eq!(elaborate_role_type(&parse_one("P1")).unwrap(), RoleType::P1);
        assert_eq!(
            elaborate_role_type(&parse_one("Mover")).unwrap(),
            RoleType::Mover
        );
    }

    #[test]
    fn result_type() {
        assert_eq!(
            elaborate_result_type(&parse_one("Win")).unwrap(),
            ResultType::Win
        );
    }

    #[test]
    fn unknown_ident_errors() {
        assert!(elaborate_compass_direction(&parse_one("Bogus")).is_err());
    }

    #[test]
    fn non_ident_errors() {
        assert!(elaborate_compass_direction(&parse_one("42")).is_err());
    }
}
