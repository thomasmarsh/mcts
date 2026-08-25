//! Shared helpers for macro-generated `Serialize`/`Deserialize` impls that
//! dispatch on a concrete `serde_json::Value` match instead of going
//! through serde's generic derive machinery (`Visitor`/`MapAccess`/its
//! `Content`-buffering internal representation). Nothing here is specific
//! to any one axis: `register_backprop!` (and potentially other
//! `register_*!` macros) can call these from their own generated impls,
//! keyed by their own tables.

use serde_json::Value;

/// Converts a `PascalCase` Rust variant identifier (as produced by
/// `stringify!` on a `register_*!` table's variant name) into the
/// `snake_case` wire tag `#[serde(rename_all = "snake_case")]` would have
/// produced. Computed at (de)serialize time rather than by a proc macro,
/// since (de)serialization here happens once per tuner trial or preset
/// load, never in the search hot loop.
pub(crate) fn to_snake_case(pascal: &str) -> String {
    let mut out = String::with_capacity(pascal.len() + 4);
    for (i, ch) in pascal.chars().enumerate() {
        if ch.is_uppercase() {
            if i != 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Pulls a required, typed field out of a JSON object `Value` -- the
/// concrete-match counterpart of what serde's derived `Deserialize` does
/// per struct-variant field, without its generic `Visitor` machinery.
pub(crate) fn field<T: serde::de::DeserializeOwned>(v: &Value, name: &str) -> Result<T, String> {
    let raw = v
        .get(name)
        .ok_or_else(|| format!("missing field `{name}`"))?;
    serde_json::from_value(raw.clone()).map_err(|e| format!("field `{name}`: {e}"))
}
