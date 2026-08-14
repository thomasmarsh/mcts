//! Elaboration of [`crate::ast::rules`] (Language Reference chapter 7): how a game is played.
//!
//! Only a flat `(rules (play ...) (end ...))` pair is elaborated so far -- `meta`, `start` and
//! `phases` aren't needed until a game requires setup or phased play.

pub mod end;
pub mod play;

use crate::ast::located::Located;
use crate::ast::rules::Rules;
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::{Head, SExpr};

/// `(rules (play <moves>) (end <endRule>))` (7.1.1), the flat form only.
pub fn elaborate_rules(v: &Located<SExpr>) -> Result<Rules, ElaborateError> {
    let call = call_ident(v, "rules")?;
    let mut rules = Rules::default();
    for arg in call.args.iter().filter(|a| a.name.is_none()) {
        let SExpr::Call(raw_call) = &arg.value.node else {
            return Err(ElaborateError {
                message: format!("expected a rules child call, found {:?}", arg.value.node),
                span: arg.value.span,
            });
        };
        match &raw_call.head {
            Head::Ident(s) if s == "play" => rules.play = Some(play::elaborate_play(&arg.value)?),
            Head::Ident(s) if s == "end" => rules.end = Some(end::elaborate_end(&arg.value)?),
            other => {
                return Err(ElaborateError {
                    message: format!(
                        "unsupported (rules ...) child head {other:?} -- only play and end are \
                         elaborated so far"
                    ),
                    span: arg.value.span,
                })
            }
        }
    }
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn rules_play_and_end() {
        let forms = parse(
            "(rules (play (move Add (to (sites Empty)))) (end (if (is Line 3) (result Mover Win))))",
        )
        .unwrap();
        let rules = elaborate_rules(&forms[0]).unwrap();
        assert!(rules.play.is_some());
        assert!(rules.end.is_some());
        assert!(rules.start.is_empty());
        assert!(rules.phases.is_empty());
        assert!(rules.meta.is_none());
    }
}
