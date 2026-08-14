//! Source-span tracking for AST nodes.
//!
//! The parser is not implemented yet, but every recursive ludeme reference in this AST is
//! wrapped in [`Located`] (via [`LBox`]) up front so that later diagnostics can point back at
//! the exact `(...)` form in the source `.lud` file that produced a given node, without having
//! to retrofit spans through the whole tree.

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

/// Wraps an AST node with the span of source text it was parsed from.
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

/// A boxed, spanned reference to a nested ludeme; the usual way one AST node refers to another.
pub type LBox<T> = Box<Located<T>>;
