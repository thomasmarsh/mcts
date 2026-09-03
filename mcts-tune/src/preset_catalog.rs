//! The preset catalog: named *partial* configurations that replace the
//! curated pre-cutover `family` rows whose shape was more than a single
//! axis swap away from the `mcts`/`ucb1`/`uniform`/`classic`/`robust_child`
//! baseline (`ucb1_dm_nst`, `rave`, `power_uct`, ...). A trivial one-axis
//! composition -- `ucb1_tuned`, `ments`, `gpn`, ... -- needs no preset; it
//! is reached by setting the `select` categorical alone.
//!
//! A preset is a small overlay of axis categoricals and mode selectors
//! (`select`/`simulate`/`backprop`/`final_action`/`decisive_move_mode`/the
//! `*_epsilon_greedy` toggles) -- never a scalar hyperparameter, which a
//! proposer explores from its schema default. A launch profile starts from
//! a preset and searches whichever axes the preset leaves unpinned.
//!
//! Each entry is one JSON file under `mcts-tune/presets/`, embedded at
//! build time. The data is `config_ir::SearchSpec`-adjacent but not itself
//! a full spec: `dispatch::to_search_spec` resolves the overlay (merged
//! over the schema defaults) the same way it resolves any params object.
//! `every_preset_matches_its_legacy_family_axes` in `tests.rs` pins every
//! entry to the `dispatch::legacy_family_to_axes` mapping for the
//! same-named pre-cutover family, which `algorithm_native_specs_match_family_goldens`
//! in turn pins to the exact `SearchSpec` that `family` row produced.

use game_host::HostError;
use serde::Deserialize;
use serde_json::{Map, Value};

/// One catalog entry: a stable `id` (matching the file stem), a human
/// description, and the partial `params` overlay it contributes.
#[derive(Debug, Clone, Deserialize)]
pub struct Preset {
    pub id: String,
    #[serde(default)]
    pub description: String,
    pub params: Map<String, Value>,
}

macro_rules! catalog {
    ($($id:literal),+ $(,)?) => {
        /// `(id, raw JSON)` for every embedded preset file, in a fixed
        /// order.
        const RAW: &[(&str, &str)] = &[
            $(($id, include_str!(concat!("../presets/", $id, ".json")))),+
        ];
    };
}

catalog! {
    "ucb1_dm",
    "ucb1_adm",
    "ucb1_tuned_dm",
    "ucb1_mast",
    "ucb1_lgr",
    "ucb1_lgr2",
    "ucb1_lgr2_mast",
    "ucb1_nst",
    "ucb1_dm_nst",
    "ucb1_adm_nst",
    "ucb1_max_robust",
    "meta_mcts",
    "amaf_mast",
    "ucb1_tuned_mast",
    "ucb1_tuned_dm_mast",
    "rave",
    "ucb1_pn_mast",
    "power_uct",
    "td_uct",
}

/// Parses every embedded preset file. Errors only on a malformed or
/// id-mismatched file -- an authoring bug in the checked-in data, not a
/// runtime condition.
pub fn load() -> Result<Vec<Preset>, HostError> {
    RAW.iter()
        .map(|(id, json)| {
            let preset: Preset = serde_json::from_str(json)
                .map_err(|e| HostError::internal(format!("preset {id}: {e}")))?;
            if preset.id != *id {
                return Err(HostError::internal(format!(
                    "preset file {id}.json declares mismatched id {:?}",
                    preset.id
                )));
            }
            Ok(preset)
        })
        .collect()
}

/// Looks up one preset by id.
pub fn preset(id: &str) -> Result<Preset, HostError> {
    load()?
        .into_iter()
        .find(|p| p.id == id)
        .ok_or_else(|| HostError::not_found("unknown preset"))
}

impl Preset {
    /// Overlays this preset's `params` onto `base`, replacing any key it
    /// sets and leaving the rest untouched -- the operation a launch
    /// profile applies before handing the merged params to the proposer.
    pub fn overlay(&self, base: &mut Map<String, Value>) {
        for (key, value) in &self.params {
            base.insert(key.clone(), value.clone());
        }
    }
}
