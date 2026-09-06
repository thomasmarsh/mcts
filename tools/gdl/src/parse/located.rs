//! Source-span tracking for tokens and parsed values.

/// A byte range into the original source text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }
}

/// Wraps a value with the span of source text it was parsed from.
#[derive(Debug, Clone, PartialEq)]
pub struct Located<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Located<T> {
    pub fn new(node: T, span: Span) -> Self {
        Located { node, span }
    }
}
