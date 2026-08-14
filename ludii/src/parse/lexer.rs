//! Tokenizer for `.lud` source text (Language Reference chapters 1 and 20-24).
//!
//! Beyond the obvious literals and brackets, the language has a few pieces of syntax that don't
//! show up until the metalanguage chapters but appear in ordinary game files (see
//! `lud/Breakthrough.lud`): option references like `<Tiling:type>`, integer ranges like `0..9`,
//! and known-define calls whose "name" is a quoted string in head position, e.g.
//! `("ReachWin" (sites Mover) Mover)`. The lexer only needs to recognize the first two as their
//! own token kinds -- a string in head position is just an ordinary [`Token::Str`], distinguished
//! from a ludeme call by the parser, not the lexer.
//!
//! Anything written with symbol punctuation (`<`, `<=`, `+`, `~`, ...) is folded into
//! [`Token::Ident`] rather than enumerated one by one, since the language uses ordinary ludeme
//! dispatch (not fixed operator precedence) for things like `(< a b)` -- to the lexer these are
//! just names.

use crate::ast::located::{Located, Span};
use crate::parse::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Colon,
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// `<lo>..<hi>` (23, ranges), inclusive of both ends.
    Range(i64, i64),
    /// The raw tag content of `<...>` (21.1, option references), e.g. `Tiling:type`, `type`, `4`.
    OptionRef(String),
    Ident(String),
}

struct Lexer<'a> {
    src: &'a str,
    chars: Vec<(usize, char)>,
    i: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer {
            src,
            chars: src.char_indices().collect(),
            i: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.peek_at(0)
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.i + k).map(|&(_, c)| c)
    }

    /// Byte offset of the next unconsumed character, or the source length at end of input.
    fn pos(&self) -> u32 {
        self.chars
            .get(self.i)
            .map(|&(b, _)| b as u32)
            .unwrap_or(self.src.len() as u32)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }

    fn slice(&self, start: u32, end: u32) -> &'a str {
        &self.src[start as usize..end as usize]
    }
}

pub fn lex(src: &str) -> Result<Vec<Located<Token>>, ParseError> {
    let mut lx = Lexer::new(src);
    let mut out = Vec::new();
    loop {
        skip_trivia(&mut lx)?;
        let Some(c) = lx.peek() else { break };
        let start = lx.pos();
        let tok = match c {
            '(' => {
                lx.bump();
                Token::LParen
            }
            ')' => {
                lx.bump();
                Token::RParen
            }
            '{' => {
                lx.bump();
                Token::LBrace
            }
            '}' => {
                lx.bump();
                Token::RBrace
            }
            ':' => {
                lx.bump();
                Token::Colon
            }
            '"' => lex_string(&mut lx)?,
            '<' => lex_angle(&mut lx)?,
            '>' => {
                lx.bump();
                if lx.peek() == Some('=') {
                    lx.bump();
                    Token::Ident(">=".to_string())
                } else {
                    Token::Ident(">".to_string())
                }
            }
            c if c.is_ascii_digit() => lex_number(&mut lx)?,
            '-' if lx.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => lex_number(&mut lx)?,
            c if is_ident_start(c) => lex_ident(&mut lx),
            _ => lex_symbol(&mut lx),
        };
        let end = lx.pos();
        out.push(Located::new(tok, Span::new(start, end)));
    }
    Ok(out)
}

fn skip_trivia(lx: &mut Lexer) -> Result<(), ParseError> {
    loop {
        match (lx.peek(), lx.peek_at(1)) {
            (Some(c), _) if c.is_whitespace() => {
                lx.bump();
            }
            (Some('/'), Some('/')) => {
                lx.bump();
                lx.bump();
                while let Some(c) = lx.peek() {
                    if c == '\n' {
                        break;
                    }
                    lx.bump();
                }
            }
            (Some('/'), Some('*')) => {
                let start = lx.pos();
                lx.bump();
                lx.bump();
                loop {
                    match (lx.peek(), lx.peek_at(1)) {
                        (Some('*'), Some('/')) => {
                            lx.bump();
                            lx.bump();
                            break;
                        }
                        (Some(_), _) => {
                            lx.bump();
                        }
                        (None, _) => {
                            return Err(ParseError::new(
                                "unterminated block comment",
                                Span::new(start, lx.pos()),
                            ))
                        }
                    }
                }
            }
            _ => break,
        }
    }
    Ok(())
}

fn lex_string(lx: &mut Lexer) -> Result<Token, ParseError> {
    let start = lx.pos();
    lx.bump(); // opening quote
    let mut s = String::new();
    loop {
        match lx.bump() {
            Some('"') => return Ok(Token::Str(s)),
            Some('\\') => match lx.bump() {
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some('r') => s.push('\r'),
                Some(other) => s.push(other),
                None => {
                    return Err(ParseError::new(
                        "unterminated string literal",
                        Span::new(start, lx.pos()),
                    ))
                }
            },
            Some(c) => s.push(c),
            None => {
                return Err(ParseError::new(
                    "unterminated string literal",
                    Span::new(start, lx.pos()),
                ))
            }
        }
    }
}

/// Lexes either an option reference `<Tag:arg>` / `<arg>` / `<4>`, or a bare `<`/`<=` comparison
/// identifier. An option reference is distinguished by having no whitespace or brackets between
/// the `<` and a closing `>`.
fn lex_angle(lx: &mut Lexer) -> Result<Token, ParseError> {
    let start = lx.pos();
    lx.bump(); // '<'
    match lx.peek() {
        Some('=') => {
            lx.bump();
            return Ok(Token::Ident("<=".to_string()));
        }
        Some(c) if is_ident_start(c) || c.is_ascii_digit() => {}
        _ => return Ok(Token::Ident("<".to_string())),
    }
    let content_start = lx.pos();
    loop {
        match lx.peek() {
            Some('>') => {
                let content = lx.slice(content_start, lx.pos()).to_string();
                lx.bump();
                return Ok(Token::OptionRef(content));
            }
            Some(c) if c.is_whitespace() || matches!(c, '(' | ')' | '{' | '}' | '<') => {
                return Err(ParseError::new(
                    "unterminated option reference, expected '>'",
                    Span::new(start, lx.pos()),
                ))
            }
            Some(_) => {
                lx.bump();
            }
            None => {
                return Err(ParseError::new(
                    "unterminated option reference, expected '>'",
                    Span::new(start, lx.pos()),
                ))
            }
        }
    }
}

/// Lexes an integer, a float (`N.M`, always with digits on both sides of the dot -- 1.4), or a
/// range (`N..M`, inclusive -- 23), each optionally negative.
fn lex_number(lx: &mut Lexer) -> Result<Token, ParseError> {
    let start = lx.pos();
    if lx.peek() == Some('-') {
        lx.bump();
    }
    while lx.peek().is_some_and(|c| c.is_ascii_digit()) {
        lx.bump();
    }
    let int_end = lx.pos();

    if lx.peek() == Some('.') && lx.peek_at(1) == Some('.') {
        let first = parse_i64(lx, start, int_end)?;
        lx.bump();
        lx.bump();
        let second_start = lx.pos();
        if lx.peek() == Some('-') {
            lx.bump();
        }
        while lx.peek().is_some_and(|c| c.is_ascii_digit()) {
            lx.bump();
        }
        let second = parse_i64(lx, second_start, lx.pos())?;
        return Ok(Token::Range(first, second));
    }

    if lx.peek() == Some('.') && lx.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
        lx.bump();
        while lx.peek().is_some_and(|c| c.is_ascii_digit()) {
            lx.bump();
        }
        let text = lx.slice(start, lx.pos());
        let f: f64 = text
            .parse()
            .map_err(|_| ParseError::new(format!("invalid float {text:?}"), Span::new(start, lx.pos())))?;
        return Ok(Token::Float(f));
    }

    let v = parse_i64(lx, start, int_end)?;
    Ok(Token::Int(v))
}

fn parse_i64(lx: &Lexer, start: u32, end: u32) -> Result<i64, ParseError> {
    let text = lx.slice(start, end);
    text.parse()
        .map_err(|_| ParseError::new(format!("invalid integer {text:?}"), Span::new(start, end)))
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn lex_ident(lx: &mut Lexer) -> Token {
    let start = lx.pos();
    while lx.peek().is_some_and(is_ident_continue) {
        lx.bump();
    }
    match lx.slice(start, lx.pos()) {
        "true" => Token::Bool(true),
        "false" => Token::Bool(false),
        text => Token::Ident(text.to_string()),
    }
}

/// Any run of punctuation not otherwise reserved becomes a single identifier, e.g. `~`, `+`,
/// `!=`. This is deliberately unbounded rather than an enumerated operator list -- see the module
/// doc comment.
fn is_symbol_char(c: char) -> bool {
    !c.is_whitespace() && !matches!(c, '(' | ')' | '{' | '}' | '"' | ':' | '<' | '>')
}

fn lex_symbol(lx: &mut Lexer) -> Token {
    let start = lx.pos();
    while lx.peek().is_some_and(is_symbol_char) {
        lx.bump();
    }
    Token::Ident(lx.slice(start, lx.pos()).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Token> {
        lex(src).unwrap().into_iter().map(|t| t.node).collect()
    }

    #[test]
    fn brackets_and_colon() {
        assert_eq!(
            kinds("(a:{b})"),
            vec![
                Token::LParen,
                Token::Ident("a".into()),
                Token::Colon,
                Token::LBrace,
                Token::Ident("b".into()),
                Token::RBrace,
                Token::RParen,
            ]
        );
    }

    #[test]
    fn literals() {
        assert_eq!(kinds(r#""Pawn""#), vec![Token::Str("Pawn".into())]);
        assert_eq!(kinds("42"), vec![Token::Int(42)]);
        assert_eq!(kinds("-7"), vec![Token::Int(-7)]);
        assert_eq!(kinds("1.5"), vec![Token::Float(1.5)]);
        assert_eq!(kinds("-1.5"), vec![Token::Float(-1.5)]);
        assert_eq!(kinds("true false"), vec![Token::Bool(true), Token::Bool(false)]);
    }

    #[test]
    fn string_escapes() {
        assert_eq!(
            kinds(r#""a\"b\\c""#),
            vec![Token::Str("a\"b\\c".to_string())]
        );
    }

    #[test]
    fn ranges() {
        assert_eq!(kinds("0..9"), vec![Token::Range(0, 9)]);
        assert_eq!(kinds("3..-3"), vec![Token::Range(3, -3)]);
    }

    #[test]
    fn option_refs() {
        assert_eq!(kinds("<Tiling:type>"), vec![Token::OptionRef("Tiling:type".into())]);
        assert_eq!(kinds("<type>"), vec![Token::OptionRef("type".into())]);
        assert_eq!(kinds("<4>"), vec![Token::OptionRef("4".into())]);
    }

    #[test]
    fn comparison_operators_are_bare_idents() {
        assert_eq!(kinds("<"), vec![Token::Ident("<".into())]);
        assert_eq!(kinds("<="), vec![Token::Ident("<=".into())]);
        assert_eq!(kinds(">"), vec![Token::Ident(">".into())]);
        assert_eq!(kinds(">="), vec![Token::Ident(">=".into())]);
        assert_eq!(kinds("(< a b)")[1], Token::Ident("<".into()));
    }

    #[test]
    fn symbol_idents() {
        assert_eq!(kinds("~"), vec![Token::Ident("~".into())]);
        assert_eq!(kinds("+"), vec![Token::Ident("+".into())]);
        assert_eq!(kinds("!="), vec![Token::Ident("!=".into())]);
    }

    #[test]
    fn priority_star_is_an_ident() {
        assert_eq!(kinds(")*"), vec![Token::RParen, Token::Ident("*".into())]);
        assert_eq!(kinds(")**"), vec![Token::RParen, Token::Ident("**".into())]);
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(kinds("a // comment\nb"), vec![Token::Ident("a".into()), Token::Ident("b".into())]);
        assert_eq!(kinds("a /* comment */ b"), vec![Token::Ident("a".into()), Token::Ident("b".into())]);
    }

    #[test]
    fn spans_are_byte_ranges() {
        let toks = lex("  (foo)").unwrap();
        assert_eq!(toks[0].span, Span::new(2, 3)); // '('
        assert_eq!(toks[1].span, Span::new(3, 6)); // 'foo'
        assert_eq!(toks[2].span, Span::new(6, 7)); // ')'
    }

    #[test]
    fn unterminated_string_errors() {
        assert!(lex(r#""abc"#).is_err());
    }

    #[test]
    fn unterminated_option_ref_errors() {
        assert!(lex("<abc").is_err());
        assert!(lex("<abc (x)").is_err());
    }

    #[test]
    fn unterminated_block_comment_errors() {
        assert!(lex("a /* never closed").is_err());
    }
}
