//! Elaboration of the plain enumerated ludemes in [`crate::ast::types`] -- constant identifiers
//! like `N` or `Simultaneous` that appear as bare argument values throughout the grammar.
//!
//! Only the enums actually needed by an implemented elaboration function are covered so far; add
//! more [`ident_enum`] invocations as later chapters need them.

use crate::ast::located::Located;
use crate::ast::types::{CompassDirection, ModeType};
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
    fn unknown_ident_errors() {
        assert!(elaborate_compass_direction(&parse_one("Bogus")).is_err());
    }

    #[test]
    fn non_ident_errors() {
        assert!(elaborate_compass_direction(&parse_one("42")).is_err());
    }
}
