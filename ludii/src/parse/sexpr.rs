//! Generic s-expression parser: turns a token stream into an [`SExpr`] tree that mirrors the
//! source syntax exactly, with no knowledge of what any particular ludeme name means.
//!
//! See the [module doc](super) for why this sits between the lexer and the typed [`crate::ast`].

use crate::ast::located::{Located, Span};
use crate::parse::lexer::{self, Token};
use crate::parse::ParseError;

/// A parsed value: either an atom, a `{...}` list, or a `(...)` call.
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// `<lo>..<hi>` (23), unexpanded -- whether/how to expand it is up to a later pass.
    Range(i64, i64),
    /// A bare word used as a value in its own right, e.g. the `Win` in `(result Mover Win)`.
    Ident(String),
    /// `<Tag:arg>` / `<arg>` (21.1), used as a value in its own right rather than as a call head.
    OptionRef(String),
    List(Vec<Located<SExpr>>),
    Call(Call),
}

/// `(head arg...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub head: Head,
    pub args: Vec<Arg>,
    /// The number of `*` priority markers immediately following the closing `)` (21.2), e.g. 2
    /// for `(item ... )**`. Zero when absent.
    pub priority: u32,
}

/// The callee position of a [`Call`]: an ordinary ludeme name, a known-define invocation (20.4,
/// where the "name" is a quoted string, e.g. `("ReachWin" ...)`), or an option reference used as
/// a call head (21.1, e.g. `(<Tiling:type> <Board:size>)`).
#[derive(Debug, Clone, PartialEq)]
pub enum Head {
    Ident(String),
    Define(String),
    OptionRef(String),
}

/// One argument to a [`Call`]: positional, or named via `key:value`.
#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Located<SExpr>,
}

/// Parses every top-level form in a `.lud` file (a game description typically contains several
/// siblings: `(game ...)`, `(option ...)` declarations, `(metadata ...)`, etc).
pub fn parse(src: &str) -> Result<Vec<Located<SExpr>>, ParseError> {
    let tokens = lexer::lex(src)?;
    let mut p = Parser { tokens: &tokens, i: 0 };
    let mut out = Vec::new();
    while p.peek().is_some() {
        out.push(p.parse_value()?);
    }
    Ok(out)
}

struct Parser<'a> {
    tokens: &'a [Located<Token>],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Located<Token>> {
        self.tokens.get(self.i)
    }

    fn peek_at(&self, k: usize) -> Option<&'a Located<Token>> {
        self.tokens.get(self.i + k)
    }

    fn bump(&mut self) -> Option<&'a Located<Token>> {
        let t = self.tokens.get(self.i);
        if t.is_some() {
            self.i += 1;
        }
        t
    }

    fn eof_span(&self) -> Span {
        self.tokens.last().map(|t| t.span).unwrap_or_default()
    }

    fn parse_value(&mut self) -> Result<Located<SExpr>, ParseError> {
        let tok = self
            .peek()
            .ok_or_else(|| ParseError::new("unexpected end of input, expected a value", self.eof_span()))?;
        let span = tok.span;
        match &tok.node {
            Token::Str(s) => {
                let s = s.clone();
                self.bump();
                Ok(Located::new(SExpr::Str(s), span))
            }
            Token::Int(v) => {
                let v = *v;
                self.bump();
                Ok(Located::new(SExpr::Int(v), span))
            }
            Token::Float(v) => {
                let v = *v;
                self.bump();
                Ok(Located::new(SExpr::Float(v), span))
            }
            Token::Bool(v) => {
                let v = *v;
                self.bump();
                Ok(Located::new(SExpr::Bool(v), span))
            }
            Token::Range(a, b) => {
                let (a, b) = (*a, *b);
                self.bump();
                Ok(Located::new(SExpr::Range(a, b), span))
            }
            Token::OptionRef(s) => {
                let s = s.clone();
                self.bump();
                Ok(Located::new(SExpr::OptionRef(s), span))
            }
            Token::Ident(s) => {
                let s = s.clone();
                self.bump();
                Ok(Located::new(SExpr::Ident(s), span))
            }
            Token::LBrace => self.parse_list(),
            Token::LParen => self.parse_call(),
            other => Err(ParseError::new(format!("unexpected token {other:?}"), span)),
        }
    }

    fn parse_list(&mut self) -> Result<Located<SExpr>, ParseError> {
        let open = self.bump().unwrap().span; // '{'
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Some(t) if matches!(t.node, Token::RBrace) => break,
                Some(_) => items.push(self.parse_value()?),
                None => return Err(ParseError::new("unterminated list, expected '}'", open)),
            }
        }
        let close = self.bump().unwrap().span; // '}'
        Ok(Located::new(SExpr::List(items), Span::new(open.start, close.end)))
    }

    fn parse_call(&mut self) -> Result<Located<SExpr>, ParseError> {
        let open = self.bump().unwrap().span; // '('
        let head_tok = self
            .peek()
            .ok_or_else(|| ParseError::new("unterminated form, expected a ludeme name", open))?;
        let head = match &head_tok.node {
            Token::Ident(s) => Head::Ident(s.clone()),
            Token::Str(s) => Head::Define(s.clone()),
            Token::OptionRef(s) => Head::OptionRef(s.clone()),
            other => {
                return Err(ParseError::new(
                    format!("expected a ludeme name, found {other:?}"),
                    head_tok.span,
                ))
            }
        };
        self.bump();

        let mut args = Vec::new();
        loop {
            match self.peek() {
                Some(t) if matches!(t.node, Token::RParen) => break,
                Some(_) => args.push(self.parse_arg()?),
                None => return Err(ParseError::new("unterminated form, expected ')'", open)),
            }
        }
        let mut end = self.bump().unwrap().span.end; // ')'

        let mut priority = 0u32;
        while let Some(t) = self.peek() {
            let Token::Ident(s) = &t.node else { break };
            if s.is_empty() || !s.chars().all(|c| c == '*') || t.span.start != end {
                break;
            }
            priority += s.chars().count() as u32;
            end = t.span.end;
            self.bump();
        }

        Ok(Located::new(
            SExpr::Call(Call { head, args, priority }),
            Span::new(open.start, end),
        ))
    }

    fn parse_arg(&mut self) -> Result<Arg, ParseError> {
        if let Some(name) = self.try_take_named_key() {
            let value = self.parse_value()?;
            Ok(Arg { name: Some(name), value })
        } else {
            let value = self.parse_value()?;
            Ok(Arg { name: None, value })
        }
    }

    /// A named argument's key is a bare identifier directly (no whitespace) followed by `:`,
    /// e.g. `next:1`. Consumes both tokens and returns the key if so; otherwise leaves the
    /// parser untouched.
    fn try_take_named_key(&mut self) -> Option<String> {
        let key_tok = self.peek()?;
        let Token::Ident(name) = &key_tok.node else { return None };
        let colon_tok = self.peek_at(1)?;
        if !matches!(colon_tok.node, Token::Colon) || key_tok.span.end != colon_tok.span.start {
            return None;
        }
        let name = name.clone();
        self.i += 2;
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(src: &str) -> SExpr {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1, "expected exactly one top-level form in {src:?}");
        forms.remove(0).node
    }

    #[test]
    fn atoms() {
        assert_eq!(parse_one("42"), SExpr::Int(42));
        assert_eq!(parse_one("-1"), SExpr::Int(-1));
        assert_eq!(parse_one("1.5"), SExpr::Float(1.5));
        assert_eq!(parse_one("true"), SExpr::Bool(true));
        assert_eq!(parse_one(r#""Pawn""#), SExpr::Str("Pawn".into()));
        assert_eq!(parse_one("Win"), SExpr::Ident("Win".into()));
        assert_eq!(parse_one("0..9"), SExpr::Range(0, 9));
        assert_eq!(parse_one("<Board:size>"), SExpr::OptionRef("Board:size".into()));
    }

    #[test]
    fn list_of_atoms() {
        assert_eq!(
            parse_one("{FR FL}"),
            SExpr::List(vec![
                Located::new(SExpr::Ident("FR".into()), Span::new(1, 3)),
                Located::new(SExpr::Ident("FL".into()), Span::new(4, 6)),
            ])
        );
    }

    #[test]
    fn call_with_positional_and_named_args() {
        let SExpr::Call(call) = parse_one(r#"(subgame "Tic-Tac-Toe" next:1)"#) else {
            panic!("expected a call")
        };
        assert_eq!(call.head, Head::Ident("subgame".into()));
        assert_eq!(call.priority, 0);
        assert_eq!(call.args.len(), 2);
        assert_eq!(call.args[0].name, None);
        assert_eq!(call.args[0].value.node, SExpr::Str("Tic-Tac-Toe".into()));
        assert_eq!(call.args[1].name.as_deref(), Some("next"));
        assert_eq!(call.args[1].value.node, SExpr::Int(1));
    }

    #[test]
    fn nested_call_as_named_arg_value() {
        let SExpr::Call(call) = parse_one(r#"(to if:(is Empty (to)) (apply (remove (to))))"#) else {
            panic!("expected a call")
        };
        assert_eq!(call.args[0].name.as_deref(), Some("if"));
        assert!(matches!(call.args[0].value.node, SExpr::Call(_)));
    }

    #[test]
    fn define_call_head_is_a_string() {
        let SExpr::Call(call) = parse_one(r#"("ReachWin" (sites Mover) Mover)"#) else {
            panic!("expected a call")
        };
        assert_eq!(call.head, Head::Define("ReachWin".into()));
        assert_eq!(call.args.len(), 2);
    }

    #[test]
    fn option_ref_call_head() {
        let SExpr::Call(call) = parse_one("(<Tiling:type> <Board:size>)") else {
            panic!("expected a call")
        };
        assert_eq!(call.head, Head::OptionRef("Tiling:type".into()));
        assert_eq!(call.args[0].value.node, SExpr::OptionRef("Board:size".into()));
    }

    #[test]
    fn priority_markers_must_be_adjacent_to_the_close_paren() {
        let SExpr::Call(call) = parse_one(r#"(item "8x8" <8> "desc")*"#) else {
            panic!("expected a call")
        };
        assert_eq!(call.priority, 1);

        let SExpr::Call(call) = parse_one(r#"(item "8x8" <8> "desc")**"#) else {
            panic!("expected a call")
        };
        assert_eq!(call.priority, 2);

        // A '*' separated by whitespace is not a priority marker: it's the start of the next
        // top-level form (or, standalone, its own bare-ident atom).
        let forms = parse("(item) *").unwrap();
        assert_eq!(forms.len(), 2);
        let SExpr::Call(call) = &forms[0].node else {
            panic!("expected a call")
        };
        assert_eq!(call.priority, 0);
        assert_eq!(forms[1].node, SExpr::Ident("*".into()));
    }

    #[test]
    fn multiple_top_level_forms() {
        let forms = parse(r#"(game "G") (option "O") (metadata)"#).unwrap();
        assert_eq!(forms.len(), 3);
    }

    #[test]
    fn breakthrough_fixture_round_trips() {
        let src = include_str!("../../lud/Breakthrough.lud");
        let forms = parse(src).unwrap();
        // (game ...), (option "Board" ...), (option "Board Size" ...), (metadata ...)
        assert_eq!(forms.len(), 4);

        let SExpr::Call(game) = &forms[0].node else {
            panic!("expected a call")
        };
        assert_eq!(game.head, Head::Ident("game".into()));
        assert_eq!(game.args[0].value.node, SExpr::Str("Breakthrough".into()));

        let SExpr::Call(board_option) = &forms[1].node else {
            panic!("expected a call")
        };
        assert_eq!(board_option.head, Head::Ident("option".into()));
        // (item "Square" <square> "...")* has priority 1.
        let items = board_option
            .args
            .iter()
            .find_map(|a| match &a.value.node {
                SExpr::List(items) if a.name.is_none() => Some(items),
                _ => None,
            })
            .expect("option items list");
        let SExpr::Call(first_item) = &items[0].node else {
            panic!("expected a call")
        };
        assert_eq!(first_item.priority, 1);

        let SExpr::Call(metadata) = &forms[3].node else {
            panic!("expected a call")
        };
        assert_eq!(metadata.head, Head::Ident("metadata".into()));
    }

    #[test]
    fn unterminated_call_errors() {
        assert!(parse("(game \"x\"").is_err());
    }

    #[test]
    fn unterminated_list_errors() {
        assert!(parse("{1 2 3").is_err());
    }

    #[test]
    fn non_head_token_in_call_position_errors() {
        assert!(parse("(1 2)").is_err());
        assert!(parse("({} 2)").is_err());
    }
}
