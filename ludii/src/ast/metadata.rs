//! Metadata ludemes (Language Reference Part II, chapters 17-19): information about a game
//! that lives outside its core logic -- database info, rendering hints, and AI configuration.
//!
//! [`Info`] (chapter 17) is modeled concretely, since its items are simple `(name <string>)`
//! ludemes. Graphics (chapter 18) and AI (chapter 19) metadata are a much larger surface --
//! piece/board styling, heuristics, feature trees -- not yet modeled here; they round-trip as
//! generic [`Ludeme`] calls instead. A future pass can give them dedicated types the same way
//! [`crate::ast::moves`] and [`crate::ast::boolean`] were.

use crate::ast::value::Ludeme;

/// `(aliases {<string>})` (17.3.1) through `(version <string>)` (17.3.12): the "database"
/// info items, automatically synchronised from the Ludii game database. All but `aliases` are
/// simple `(name <string>)` ludemes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Info {
    pub aliases: Vec<String>,
    pub author: Option<String>,
    pub classification: Option<String>,
    pub credit: Option<String>,
    pub date: Option<String>,
    pub description: Option<String>,
    pub id: Option<String>,
    pub origin: Option<String>,
    pub publisher: Option<String>,
    pub rules: Option<String>,
    pub source: Option<String>,
    pub version: Option<String>,
}

/// `(metadata ...)` (17.1.1): the metadata of a game.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Metadata {
    pub info: Option<Info>,
    /// `(graphics {...})` (chapter 18): rendering hints. Not yet modeled -- kept as raw
    /// ludeme calls.
    pub graphics: Vec<Ludeme>,
    /// `(ai ...)` (chapter 19): AI configuration (heuristics, features, best-agent hints). Not
    /// yet modeled -- kept as a raw ludeme call.
    pub ai: Option<Ludeme>,
}
