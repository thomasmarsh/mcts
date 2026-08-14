//! Metalanguage features (Language Reference Part III, chapters 20-24): source-level
//! mechanisms layered on top of the ludeme grammar itself -- macro-like defines, game options,
//! rulesets built from option combinations, and named integer constants.
//!
//! These are compile-time text/token features rather than semantic game ludemes (a `(define
//! ...)` is expanded, and an `(option ...)`/`(ruleset ...)` selection resolved, before the
//! rest of this AST's types are ever built), so they are modeled more loosely than
//! [`crate::ast::game`] onward: [`Define`] and [`OptionItem`] keep their bodies as generic
//! [`Value`] trees rather than typed ludemes.

use crate::ast::value::Value;

/// `(define <string> <body>)` (chapter 20): a named, optionally parameterised macro. `body`
/// may contain `#N` parameter-insertion points and `~` null-parameter markers, which are
/// resolved by textual substitution at each call site before parsing continues -- not
/// represented separately here.
#[derive(Debug, Clone, PartialEq)]
pub struct Define {
    pub name: String,
    pub body: Value,
}

/// `(item "Name" <arg> ... "Description.")` (21.1), optionally suffixed with priority
/// asterisks (21.2): one selectable choice within a [`GameOption`] category.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionItem {
    pub name: String,
    pub args: Vec<Value>,
    pub description: String,
    /// Number of trailing `*` priority markers; higher wins when no option is user-selected.
    pub priority: u32,
}

/// `(option "Heading" <Tag> args:{...} {<item> ...})` (21.1): one category of alternative
/// rules/equipment, instantiated at compile time via `<Tag:arg>` references elsewhere in the
/// game description.
#[derive(Debug, Clone, PartialEq)]
pub struct GameOption {
    pub heading: String,
    pub tag: String,
    pub arg_names: Vec<String>,
    pub items: Vec<OptionItem>,
}

/// `(ruleset "Ruleset/Name" {"Category/Item" ...})` (22): a named combination of option
/// selections, optionally suffixed with priority asterisks.
#[derive(Debug, Clone, PartialEq)]
pub struct Ruleset {
    pub name: String,
    pub options: Vec<String>,
    pub priority: u32,
}

/// `(rulesets {<ruleset> ...})` (22): the user-selectable rulesets of a game.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Rulesets {
    pub rulesets: Vec<Ruleset>,
}

/// The predefined integer constants (chapter 24), usable anywhere an `<int>` is expected. A
/// parser can resolve these directly to their internal value rather than needing a dedicated
/// AST node (e.g. as [`crate::ast::numeric::int::IntFunction::Int`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constant {
    /// An off-board position. Internal value: -1.
    Off,
    /// The end of a track. Internal value: -2.
    End,
    /// A general "undefined" condition. Internal value: -1.
    Undefined,
}

impl Constant {
    pub const fn value(self) -> i64 {
        match self {
            Constant::Off => -1,
            Constant::End => -2,
            Constant::Undefined => -1,
        }
    }
}
