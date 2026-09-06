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

/// [`field`]'s counterpart for a field serde's derive would treat as
/// implicitly optional -- an `Option<T>`-typed struct field with no
/// `#[serde(default)]` annotation still defaults to `None` when the key is
/// missing (serde's derive special-cases `Option<T>` fields this way, not
/// just fields with an explicit `#[serde(default = ...)]`), and treats an
/// explicit JSON `null` the same as a missing key rather than trying to
/// deserialize `T` from `null`. Also doubles as the concrete counterpart of
/// `#[serde(default)]`/`#[serde(default = "...")]` on a non-`Option` field --
/// call this and fall back with `.unwrap_or_default()`/`.unwrap_or(...)`.
pub(crate) fn field_opt<T: serde::de::DeserializeOwned>(
    v: &Value,
    name: &str,
) -> Result<Option<T>, String> {
    match v.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(raw) => serde_json::from_value(raw.clone())
            .map(Some)
            .map_err(|e| format!("field `{name}`: {e}")),
    }
}
