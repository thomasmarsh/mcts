//! Elaboration of [`crate::ast::rules::end`] (Language Reference 7.2): terminating conditions
//! and results.
//!
//! Only `(end (if <boolean> (result <roleType> <resultType>)))` -- a single `If` end rule with
//! no subconditions -- is elaborated so far, since it's all [`crate::elaborate::game`] needs for
//! `(end (if (is Line 3) (result Mover Win)))`.

use crate::ast::located::Located;
use crate::ast::rules::end::{End, EndRule, If, Result as EndResult};
use crate::elaborate::boolean::elaborate_boolean_function;
use crate::elaborate::types::{elaborate_result_type, elaborate_role_type};
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::{Head, SExpr};

/// `(result <roleType> <resultType>)` (7.2.6).
fn elaborate_result(v: &Located<SExpr>) -> Result<EndResult, ElaborateError> {
    let call = call_ident(v, "result")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let role_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(result ...) requires a roleType argument".into(),
        span: v.span,
    })?;
    let result_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(result ...) requires a resultType argument".into(),
        span: v.span,
    })?;
    Ok(EndResult {
        role: elaborate_role_type(&role_arg.value)?,
        result: elaborate_result_type(&result_arg.value)?,
    })
}

/// `(if <boolean> <result>)` (7.2.4), with no subconditions.
fn elaborate_if(v: &Located<SExpr>) -> Result<If, ElaborateError> {
    let call = call_ident(v, "if")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let condition_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(if ...) requires a condition argument".into(),
        span: v.span,
    })?;
    let condition = elaborate_boolean_function(&condition_arg.value)?;
    let result_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(if ...) requires a result argument -- subconditions aren't elaborated yet"
            .into(),
        span: v.span,
    })?;
    let result = elaborate_result(&result_arg.value)?;
    Ok(If {
        condition: Some(Box::new(Located::new(condition, condition_arg.value.span))),
        subconditions: Vec::new(),
        result: Some(result),
    })
}

/// A single `(end (<endRule> | {<endRule>}))` entry. Only [`EndRule::If`] is elaborated so far.
fn elaborate_end_rule(v: &Located<SExpr>) -> Result<EndRule, ElaborateError> {
    let SExpr::Call(raw_call) = &v.node else {
        return Err(ElaborateError {
            message: format!("expected an end rule call, found {:?}", v.node),
            span: v.span,
        });
    };
    match &raw_call.head {
        Head::Ident(s) if s == "if" => Ok(EndRule::If(Box::new(elaborate_if(v)?))),
        other => Err(ElaborateError {
            message: format!("unsupported end rule head {other:?} -- only if is elaborated so far"),
            span: v.span,
        }),
    }
}

/// `(end (<endRule> | {<endRule>}))` (7.2.2).
pub fn elaborate_end(v: &Located<SExpr>) -> Result<End, ElaborateError> {
    let call = call_ident(v, "end")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(end ...) requires an endRule argument".into(),
        span: v.span,
    })?;
    let rules = match &arg.value.node {
        SExpr::List(items) => items
            .iter()
            .map(elaborate_end_rule)
            .collect::<Result<_, _>>()?,
        _ => vec![elaborate_end_rule(&arg.value)?],
    };
    Ok(End { rules })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::types::{ResultType, RoleType};
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn end_if_line_win() {
        let end = elaborate_end(&parse_one("(end (if (is Line 3) (result Mover Win)))")).unwrap();
        assert_eq!(end.rules.len(), 1);
        let EndRule::If(if_rule) = &end.rules[0] else {
            panic!("expected EndRule::If");
        };
        assert!(if_rule.condition.is_some());
        assert!(if_rule.subconditions.is_empty());
        assert_eq!(
            if_rule.result,
            Some(EndResult {
                role: RoleType::Mover,
                result: ResultType::Win,
            })
        );
    }
}
