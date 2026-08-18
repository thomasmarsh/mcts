//! Runtime-loadable AI presets ("easy"/"strong"/...), independent of any
//! particular game's own hardcoded preset list. A
//! `PresetTable` is just a `Vec<PresetSpec>` parsed from JSON -- each entry
//! names a `TrialParams`-shaped `params` object (the same JSON
//! `family_catalog`'s families already accept) plus a time/iteration/thread
//! budget, resolved to a runnable search via the existing [`build_search`]
//! (this crate's own `TrialParams` -> `config_ir::SearchSpec` ->
//! `Box<dyn Search<G>>` pipeline) -- not a new mechanism, just a new
//! *source* for the JSON `build_search` already accepts.
//!
//! Lives in `mcts-tune`, not `game-host`, despite presets being a
//! `GameAdapter`-level concern: `game-host` has no dependency on
//! `mcts-tune` (every game crate depends on both, but `mcts-tune` itself
//! depends on `game-host` for `HostError`/`TunerInfo`), so a loader that
//! calls `build_search` can only live on the `mcts-tune` side of that edge.

use std::path::Path;
use std::time::Duration;

use game_host::{AiPresetInfo, HostError};
use mcts::game::Game;
use mcts::strategies::Search;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{build_search, SearchBudget};

fn one() -> usize {
    1
}

/// One preset's wire shape: `id`/`label`/`description` mirror
/// `game_host::AiPresetInfo` exactly (`to_info` just clones them into it);
/// `params` is a `TrialParams`-shaped JSON object (`family` plus whatever
/// fields that family needs, `q_init` as a string) -- the exact shape
/// `build_search` already parses, so nothing new to validate here beyond
/// what `build_search` itself already rejects at build time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
    pub params: Value,
    #[serde(default)]
    pub max_time_ms: Option<u64>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Tree-parallelism thread count; `0` means "use every available core"
    /// (mirrors `games/druid/src/main.rs`'s `ai_thread_count`/
    /// `preset_threads` convention -- `SearchBudget::threads` itself has no
    /// such meaning, so [`PresetSpec::budget`] resolves `0` before
    /// constructing one).
    #[serde(default = "one")]
    pub threads: usize,
    #[serde(default)]
    pub use_transpositions: bool,
}

impl PresetSpec {
    fn to_info(&self) -> AiPresetInfo {
        AiPresetInfo {
            id: self.id.clone(),
            label: self.label.clone(),
            description: self.description.clone(),
        }
    }

    fn budget(&self) -> SearchBudget {
        let threads = if self.threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.threads
        };
        SearchBudget {
            max_time: self.max_time_ms.map(Duration::from_millis),
            threads,
            max_iterations: self.max_iterations,
        }
    }
}

/// A game's full preset list, parsed once at startup from JSON -- the
/// runtime-configurable replacement for a hand-written `PresetEntry`
/// array/`match`. See this module's doc comment for the wire shape.
#[derive(Debug, Clone)]
pub struct PresetTable {
    presets: Vec<PresetSpec>,
}

impl PresetTable {
    /// Parses `json` (a JSON array of [`PresetSpec`]) into a `PresetTable`.
    pub fn from_json(json: &str) -> Result<Self, HostError> {
        let presets: Vec<PresetSpec> = serde_json::from_str(json)
            .map_err(|e| HostError::bad_request(format!("invalid presets json: {e}")))?;
        Ok(Self { presets })
    }

    /// [`Self::from_json`], preferring `override_path`'s file contents when
    /// it exists -- the "easy to configure at runtime" half of the design:
    /// a game binary embeds its shipped defaults via `include_str!` and
    /// passes them as `default_json`, but an operator can point
    /// `override_path` (typically from an env var the binary's own `main`
    /// reads, e.g. `NIM_PRESETS_PATH`) at an edited copy without a
    /// rebuild.
    pub fn load(default_json: &str, override_path: Option<&Path>) -> Result<Self, HostError> {
        match override_path {
            Some(path) if path.exists() => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| HostError::internal(format!("reading {}: {e}", path.display())))?;
                Self::from_json(&text)
            }
            _ => Self::from_json(default_json),
        }
    }

    /// The `GameAdapter::ai_presets` reply -- every preset's `{id, label,
    /// description}`, in file order.
    pub fn ai_presets(&self) -> Vec<AiPresetInfo> {
        self.presets.iter().map(PresetSpec::to_info).collect()
    }

    /// Looks up `id`'s full spec -- for a caller that needs one of its
    /// declared knobs (e.g. `max_time_ms`) directly, not just a built
    /// search.
    pub fn preset(&self, id: &str) -> Result<&PresetSpec, HostError> {
        self.presets
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| HostError::not_found("unknown preset"))
    }

    /// Resolves `id` to a runnable search, seeded and budgeted per that
    /// preset's own declared `params`/budget -- `GameAdapter::ai_move`/
    /// `analyze`'s replacement for `PRESETS.iter().find(...).build()`.
    pub fn build<G: Game + 'static>(
        &self,
        id: &str,
        seed: u64,
    ) -> Result<Box<dyn Search<G = G>>, HostError> {
        let preset = self.preset(id)?;
        build_search::<G>(
            &preset.params,
            seed,
            preset.use_transpositions,
            &preset.budget(),
        )
    }

    /// [`Self::build`], but with `f` applied to `preset`'s own resolved
    /// budget first -- for a caller that needs the same strategy shape as
    /// a named preset with one budget knob overridden, e.g.
    /// `GameAdapter::analyze`'s `budget_ms` argument (override `max_time`)
    /// or a `tune_eval` baseline that must run single-threaded regardless
    /// of what the preset itself deploys with (override `threads`).
    pub fn build_with<G: Game + 'static>(
        &self,
        id: &str,
        seed: u64,
        f: impl FnOnce(&mut SearchBudget),
    ) -> Result<Box<dyn Search<G = G>>, HostError> {
        let preset = self.preset(id)?;
        let mut budget = preset.budget();
        f(&mut budget);
        build_search::<G>(&preset.params, seed, preset.use_transpositions, &budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_nim::Nim;

    fn sample_json() -> &'static str {
        r#"[
            {
                "id": "easy",
                "label": "Easy",
                "description": "Quick and beatable.",
                "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child"},
                "max_iterations": 100
            },
            {
                "id": "strong",
                "label": "Strong",
                "description": "",
                "params": {"family": "ucb1", "c": 1.4, "q_init": "Loss", "final_action": "robust_child"},
                "max_iterations": 5000,
                "threads": 1
            }
        ]"#
    }

    #[test]
    fn from_json_parses_every_preset_and_reports_ai_presets_in_order() {
        let table = PresetTable::from_json(sample_json()).unwrap();
        let infos = table.ai_presets();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].id, "easy");
        assert_eq!(infos[1].id, "strong");
    }

    #[test]
    fn build_resolves_a_known_preset_and_rejects_an_unknown_one() {
        let table = PresetTable::from_json(sample_json()).unwrap();
        let mut ai = table.build::<Nim>("easy", 0).unwrap();
        let state = <Nim as Game>::S::default();
        let _ = ai.choose_action(&state);

        // `Box<dyn Search<G>>` isn't `Debug`, so `Result::unwrap_err` doesn't
        // apply here -- match instead.
        let err = match table.build::<Nim>("not_a_real_preset", 0) {
            Err(e) => e,
            Ok(_) => panic!("unknown preset must be rejected"),
        };
        assert_eq!(err.code, 404);
    }

    /// `"random"`/`"flat_mc"` are non-composable floor families -- direct
    /// arms in `make_candidate`, not rows in `config_ir`'s registries (see
    /// that function's own comment) -- but `PresetTable::build` is just
    /// `build_search` under a preset id, so a game's `presets.json` can name
    /// either one directly (e.g. a "baseline"/"random" preset), same as any
    /// composable family. This is the proof: nothing about the preset layer
    /// restricts `family` to composable strategies.
    #[test]
    fn build_resolves_non_composable_floor_families() {
        let table = PresetTable::from_json(
            r#"[
                {"id": "random", "label": "Random", "description": "", "params": {"family": "random", "q_init": "Infinity"}, "max_iterations": 1},
                {"id": "flat_mc", "label": "Flat MC", "description": "", "params": {"family": "flat_mc", "q_init": "Infinity"}, "max_iterations": 100}
            ]"#,
        )
        .unwrap();
        let state = <Nim as Game>::S::default();

        let mut random_ai = table.build::<Nim>("random", 0).unwrap();
        let _ = random_ai.choose_action(&state);

        let mut flat_mc_ai = table.build::<Nim>("flat_mc", 0).unwrap();
        let _ = flat_mc_ai.choose_action(&state);
    }

    #[test]
    fn load_prefers_an_existing_override_file_over_the_embedded_default() {
        let dir =
            std::env::temp_dir().join(format!("mcts_tune_preset_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("presets.json");
        std::fs::write(
            &path,
            r#"[{"id": "only", "label": "Only", "description": "", "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child"}, "max_iterations": 1}]"#,
        )
        .unwrap();

        let table = PresetTable::load(sample_json(), Some(&path)).unwrap();
        assert_eq!(table.ai_presets().len(), 1);
        assert_eq!(table.ai_presets()[0].id, "only");

        std::fs::remove_file(&path).ok();
        let fallback = PresetTable::load(sample_json(), Some(&path)).unwrap();
        assert_eq!(fallback.ai_presets().len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_threads_resolves_to_available_parallelism_not_a_literal_zero_thread_budget() {
        let table = PresetTable::from_json(
            r#"[{"id": "auto", "label": "Auto", "description": "", "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child"}, "max_iterations": 1, "threads": 0}]"#,
        )
        .unwrap();
        let budget = table.presets[0].budget();
        assert!(budget.threads >= 1);
    }
}
