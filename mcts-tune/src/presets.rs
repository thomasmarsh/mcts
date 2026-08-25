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
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use std::str::FromStr;

use mcts::node::QInit;

use crate::config_ir;
use crate::config_ir::codec::{field, field_opt};
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
///
/// `Serialize`/`Deserialize` are hand-implemented below (routed through
/// `serde_json::Value`, via the same `config_ir::codec` helpers the
/// `register_*!` axis macros use) rather than `#[derive]`d, for the same
/// compile-cost reason -- see `config_ir/backprop.rs`'s `BackpropSpec` doc
/// comment. A flat struct with no `kind` tag, so unlike the axis specs
/// there's only one shape to match, not one per enum variant.
#[derive(Debug, Clone)]
pub struct PresetSpec {
    pub id: String,
    pub label: String,
    pub description: String,
    pub params: Value,
    pub max_time_ms: Option<u64>,
    pub max_iterations: Option<usize>,
    /// Tree-parallelism thread count; `0` means "use every available core"
    /// (mirrors `games/druid/src/main.rs`'s `ai_thread_count`/
    /// `preset_threads` convention -- `SearchBudget::threads` itself has no
    /// such meaning, so [`PresetSpec::budget`] resolves `0` before
    /// constructing one).
    pub threads: usize,
    pub use_transpositions: bool,
}

impl Serialize for PresetSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("label", &self.label)?;
        map.serialize_entry("description", &self.description)?;
        map.serialize_entry("params", &self.params)?;
        map.serialize_entry("max_time_ms", &self.max_time_ms)?;
        map.serialize_entry("max_iterations", &self.max_iterations)?;
        map.serialize_entry("threads", &self.threads)?;
        map.serialize_entry("use_transpositions", &self.use_transpositions)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for PresetSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        Ok(PresetSpec {
            id: field(&v, "id").map_err(D::Error::custom)?,
            label: field(&v, "label").map_err(D::Error::custom)?,
            description: field_opt(&v, "description")
                .map_err(D::Error::custom)?
                .unwrap_or_default(),
            params: field(&v, "params").map_err(D::Error::custom)?,
            max_time_ms: field_opt(&v, "max_time_ms").map_err(D::Error::custom)?,
            max_iterations: field_opt(&v, "max_iterations").map_err(D::Error::custom)?,
            threads: field_opt(&v, "threads")
                .map_err(D::Error::custom)?
                .unwrap_or_else(one),
            use_transpositions: field_opt(&v, "use_transpositions")
                .map_err(D::Error::custom)?
                .unwrap_or_default(),
        })
    }
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

    /// [`Self::from_json`], reading `path`'s contents at call time -- game
    /// binaries do not embed `presets.json` via `include_str!` (that would
    /// make Cargo treat the file as a build dependency and rebuild on every
    /// edit); they read it fresh from disk on every startup instead, so
    /// editing presets never triggers a rebuild. `path` is typically the
    /// game's own `presets.json` next to `Cargo.toml`, or an operator
    /// override from an env var the binary's own `main` reads (e.g.
    /// `NIM_PRESETS_PATH`).
    pub fn load_from_path(path: &Path) -> Result<Self, HostError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| HostError::internal(format!("reading {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// The `GameAdapter::ai_presets` reply -- every preset's `{id, label,
    /// description}`, in file order.
    pub fn ai_presets(&self) -> Vec<AiPresetInfo> {
        self.presets.iter().map(PresetSpec::to_info).collect()
    }

    /// Every preset's `id`, in file order -- the dynamic replacement for a
    /// hand-written `&["strong"]` baseline list: a `tuner()` reports
    /// whichever presets this game's own `presets.json` actually declares,
    /// so a game with `easy`/`strong` (or more) exposes all of them as
    /// tuner baseline instances instead of one hardcoded name.
    pub fn ai_preset_ids(&self) -> Vec<&str> {
        self.presets.iter().map(|p| p.id.as_str()).collect()
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

/// Non-strategy budget/threading defaults [`build_custom`] applies --
/// mirrors `mcts-tune::to_search_spec`'s own constants (`PLAYOUT_DEPTH`/
/// `EXPAND_THRESHOLD`), reused here rather than re-picked, so a "Custom"
/// search behaves like every named-preset search along every axis it
/// doesn't expose as a field.
const PLAYOUT_DEPTH: usize = 200;
const EXPAND_THRESHOLD: u32 = 1;

/// An inline, JSON-driven `config_ir::SearchSpec` -- the "Custom" strategy
/// wire payload: unlike [`PresetSpec`] (`params` is `TrialParams`-shaped,
/// dispatched through `family_catalog`'s ~18 pre-composed family names),
/// `search` is a full `config_ir::SearchSpec`, built through
/// [`build_custom`] via `config_ir::build_search` directly -- true free
/// composition of all four axes, not a named combination. Field shape
/// otherwise mirrors [`PresetSpec`] minus `id`/`label`/`description` (which
/// only make sense for a table entry, not a one-off inline config).
///
/// `Serialize`/`Deserialize` are hand-implemented below, the same way and
/// for the same reason as [`PresetSpec`]'s -- see that struct's doc comment.
#[derive(Debug, Clone)]
pub struct CustomStrategySpec {
    pub search: config_ir::SearchSpec,
    pub max_time_ms: Option<u64>,
    pub max_iterations: Option<usize>,
    /// `0` means "every available core" -- same convention as
    /// [`PresetSpec::threads`], but defaulting to `1` here too rather than
    /// requiring every caller to name a thread count.
    pub threads: usize,
    pub use_transpositions: bool,
    /// `QInit`'s wire form is a name string (`"Parent"`/`"Win"`/`"Loss"`/
    /// `"Draw"`/`"Infinity"`), matching `TrialParams::q_init` -- `QInit`
    /// itself has no `Serialize`/`Deserialize` derive to reuse directly.
    pub q_init: String,
    /// Same wire name and semantics as `TrialParams::mcgs`
    /// ([`crate::family_catalog`]/`to_search_spec`): `true` switches on Monte
    /// Carlo *graph* search (`GraphSearch::Dag(GraphStats::Both)`) in place
    /// of plain tree search, and requires `use_transpositions` also be `true`
    /// (rejected by [`build_custom`] otherwise) since graph search only
    /// makes sense against a game with a real zobrist hash. See
    /// `crate::resolve_graph_search`, the shared derivation both this and
    /// `to_search_spec` call.
    pub mcgs: bool,
    /// Same wire name and semantics as `TrialParams::state_only_keying`:
    /// `true` selects `TranspositionKeying::StateOnly` over the default
    /// `PerPly`, and requires `mcgs` also be `true` (rejected by
    /// [`build_custom`] otherwise, via `crate::resolve_graph_search`). See
    /// `mcts::TranspositionKeying`'s doc comment for the per-game GHI
    /// precondition this asserts.
    pub state_only_keying: bool,
}

fn default_q_init() -> String {
    "Parent".to_string()
}

impl Serialize for CustomStrategySpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("search", &self.search)?;
        map.serialize_entry("max_time_ms", &self.max_time_ms)?;
        map.serialize_entry("max_iterations", &self.max_iterations)?;
        map.serialize_entry("threads", &self.threads)?;
        map.serialize_entry("use_transpositions", &self.use_transpositions)?;
        map.serialize_entry("q_init", &self.q_init)?;
        map.serialize_entry("mcgs", &self.mcgs)?;
        map.serialize_entry("state_only_keying", &self.state_only_keying)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for CustomStrategySpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        Ok(CustomStrategySpec {
            search: field(&v, "search").map_err(D::Error::custom)?,
            max_time_ms: field_opt(&v, "max_time_ms").map_err(D::Error::custom)?,
            max_iterations: field_opt(&v, "max_iterations").map_err(D::Error::custom)?,
            threads: field_opt(&v, "threads")
                .map_err(D::Error::custom)?
                .unwrap_or_else(one),
            use_transpositions: field_opt(&v, "use_transpositions")
                .map_err(D::Error::custom)?
                .unwrap_or_default(),
            q_init: field_opt(&v, "q_init")
                .map_err(D::Error::custom)?
                .unwrap_or_else(default_q_init),
            mcgs: field_opt(&v, "mcgs")
                .map_err(D::Error::custom)?
                .unwrap_or_default(),
            state_only_keying: field_opt(&v, "state_only_keying")
                .map_err(D::Error::custom)?
                .unwrap_or_default(),
        })
    }
}

impl CustomStrategySpec {
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

/// Builds a runnable search straight from a [`CustomStrategySpec`] --
/// bypassing `TrialParams`/`family_catalog` entirely and calling
/// `config_ir::build_search` directly, the first real (non-test) caller of
/// that function: `family_catalog`'s ~18 families are pre-composed
/// combinations, never free per-axis composition, so a "Custom" strategy
/// needs this parallel path rather than routing through [`build_search`]
/// (this crate's `TrialParams`-based one, re-exported at the crate root).
pub fn build_custom<G: Game + 'static>(
    spec: &CustomStrategySpec,
    seed: u64,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    let q_init = QInit::from_str(&spec.q_init)
        .map_err(|_| HostError::bad_request(format!("invalid q_init: {}", spec.q_init)))?;
    config_ir::validate_search_spec::<G>(&spec.search).map_err(HostError::bad_request)?;
    let budget = spec.budget();
    let (use_transpositions, reuse_tree, graph_search, transposition_keying) =
        crate::resolve_graph_search(spec.mcgs, spec.use_transpositions, spec.state_only_keying)?;
    let settings = config_ir::SearchSettings {
        max_iterations: budget.iteration_limit(),
        max_playout_depth: PLAYOUT_DEPTH,
        expand_threshold: EXPAND_THRESHOLD,
        q_init,
        use_transpositions,
        use_mcts_solver: true,
        reuse_tree,
        num_tree_threads: budget.threads,
        seed,
        max_time: budget.max_time,
        graph_search,
        transposition_keying,
        solver_loss_threshold: None,
        contempt_factor: None,
    };
    Ok(config_ir::build_search::<G>(&spec.search, &settings))
}

/// Resolves either a named preset or an inline [`CustomStrategySpec`] to a
/// runnable search -- the one call every game adapter's `ai_move`/`analyze`
/// makes instead of `table.build(preset, seed)` directly, so wiring in the
/// "Custom" wire-protocol path (see `game-host`'s `ai_move`/`analyze`
/// `custom` parameter) is a one-line swap per adapter rather than a
/// per-adapter reimplementation of this branch.
pub fn build_strategy<G: Game + 'static>(
    table: &PresetTable,
    preset: &str,
    custom: Option<&CustomStrategySpec>,
    seed: u64,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    match custom {
        Some(spec) => build_custom::<G>(spec, seed),
        None => table.build::<G>(preset, seed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_ir::{
        BackpropSpec, BaseSelectSpec, FinalActionSpec, SelectSpec, SimulateSpec,
    };
    use game_nim::Nim;
    use serde_json::json;

    fn sample_custom_spec() -> CustomStrategySpec {
        CustomStrategySpec {
            search: config_ir::SearchSpec {
                select: SelectSpec::EpsilonGreedy {
                    epsilon: 0.1,
                    inner: BaseSelectSpec::Ucb1 { c: 1.4 },
                },
                simulate: SimulateSpec::Uniform {},
                backprop: BackpropSpec::Classic {},
                final_action: FinalActionSpec::RobustChild {},
            },
            max_time_ms: None,
            max_iterations: Some(50),
            threads: 1,
            use_transpositions: false,
            q_init: "Infinity".to_string(),
            mcgs: false,
            state_only_keying: false,
        }
    }

    #[test]
    fn build_custom_resolves_a_free_axis_composition_not_reachable_via_any_family() {
        // `epsilon_greedy`-wrapped `ucb1` on the `select` axis, with a
        // `final_action` independently chosen -- `family_catalog` has no
        // single named family for this exact composition (its
        // `EpsilonGreedy`-wrapped select rows are all fixed simulate-axis
        // combos, e.g. `ucb1_mast`), so building it at all is the proof
        // `build_custom` reaches `config_ir::build_search` directly.
        let spec = sample_custom_spec();
        let mut ai = build_custom::<Nim>(&spec, 0).unwrap();
        let state = <Nim as Game>::S::default();
        let _ = ai.choose_action(&state);
    }

    #[test]
    fn build_custom_rejects_an_invalid_q_init() {
        let mut spec = sample_custom_spec();
        spec.q_init = "NotAReal QInit".to_string();
        let err = match build_custom::<Nim>(&spec, 0) {
            Err(e) => e,
            Ok(_) => panic!("invalid q_init must be rejected"),
        };
        assert_eq!(err.code, 400);
    }

    #[test]
    fn build_custom_rejects_mcgs_without_transpositions() {
        let mut spec = sample_custom_spec();
        spec.mcgs = true;
        spec.use_transpositions = false;
        let err = match build_custom::<Nim>(&spec, 0) {
            Err(e) => e,
            Ok(_) => panic!("mcgs without use_transpositions must be rejected"),
        };
        assert_eq!(err.code, 400);
    }

    #[test]
    fn build_custom_resolves_mcgs_to_graph_search() {
        // Doesn't run the built search: `Nim`'s `zobrist_hash` is the
        // default constant `0` (see `strategy_tune_eval`'s doc comment on
        // why that corrupts an actual graph search), so this only proves
        // `build_custom` accepts `mcgs: true` and constructs a search --
        // not that running an MCGS search against a hash-less game is safe,
        // which it isn't and never has been (see `resolve_graph_search`'s
        // "requires a game with a zobrist hash" guard, which is the actual
        // safeguard callers must honor by only setting `mcgs` for a game
        // with a real hash, same precondition `use_transpositions` already
        // carries).
        let mut spec = sample_custom_spec();
        spec.mcgs = true;
        spec.use_transpositions = true;
        let _ai = build_custom::<Nim>(&spec, 0).unwrap();
    }

    #[test]
    fn build_strategy_picks_named_preset_or_custom_spec() {
        let table = PresetTable::from_json(sample_json()).unwrap();
        let state = <Nim as Game>::S::default();

        let mut named = build_strategy::<Nim>(&table, "easy", None, 0).unwrap();
        let _ = named.choose_action(&state);

        let custom = sample_custom_spec();
        let mut via_custom = build_strategy::<Nim>(&table, "custom", Some(&custom), 0).unwrap();
        let _ = via_custom.choose_action(&state);
    }

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
    fn load_from_path_reads_the_file_and_errors_when_it_is_missing() {
        let dir =
            std::env::temp_dir().join(format!("mcts_tune_preset_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("presets.json");
        std::fs::write(
            &path,
            r#"[{"id": "only", "label": "Only", "description": "", "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child"}, "max_iterations": 1}]"#,
        )
        .unwrap();

        let table = PresetTable::load_from_path(&path).unwrap();
        assert_eq!(table.ai_presets().len(), 1);
        assert_eq!(table.ai_presets()[0].id, "only");

        std::fs::remove_file(&path).ok();
        assert!(PresetTable::load_from_path(&path).is_err());

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

    #[test]
    fn preset_spec_round_trips_through_json() {
        let json = json!({
            "id": "strong",
            "label": "Strong",
            "description": "beatable but tries",
            "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity"},
            "max_time_ms": 500,
            "max_iterations": 1000,
            "threads": 4,
            "use_transpositions": true,
        });
        let spec: PresetSpec = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(spec.id, "strong");
        assert_eq!(spec.threads, 4);
        assert!(spec.use_transpositions);
        assert_eq!(serde_json::to_value(&spec).unwrap(), json);
    }

    #[test]
    fn preset_spec_deserialize_applies_defaults_for_omitted_fields() {
        let spec: PresetSpec = serde_json::from_value(json!({
            "id": "easy",
            "label": "Easy",
            "params": {"family": "ucb1", "c": 1.4, "q_init": "Infinity"},
        }))
        .unwrap();
        assert_eq!(spec.description, "");
        assert_eq!(spec.max_time_ms, None);
        assert_eq!(spec.max_iterations, None);
        assert_eq!(spec.threads, 1);
        assert!(!spec.use_transpositions);
    }

    #[test]
    fn preset_spec_deserialize_rejects_missing_required_field() {
        let err = serde_json::from_value::<PresetSpec>(json!({"label": "Easy", "params": {}}))
            .unwrap_err();
        assert!(err.to_string().contains("id"), "{err}");
    }

    #[test]
    fn custom_strategy_spec_round_trips_through_json() {
        let spec = sample_custom_spec();
        let json = serde_json::to_value(&spec).unwrap();
        let back: CustomStrategySpec = serde_json::from_value(json).unwrap();
        assert_eq!(back.search, spec.search);
        assert_eq!(back.max_iterations, spec.max_iterations);
        assert_eq!(back.q_init, spec.q_init);
    }

    #[test]
    fn custom_strategy_spec_deserialize_applies_defaults_for_omitted_fields() {
        let spec: CustomStrategySpec = serde_json::from_value(json!({
            "search": {
                "select": {"kind": "ucb1", "c": 1.4},
                "simulate": {"kind": "uniform"},
                "backprop": {"kind": "classic"},
                "final_action": {"kind": "robust_child"},
            },
        }))
        .unwrap();
        assert_eq!(spec.max_time_ms, None);
        assert_eq!(spec.max_iterations, None);
        assert_eq!(spec.threads, 1);
        assert!(!spec.use_transpositions);
        assert_eq!(spec.q_init, "Parent");
        assert!(!spec.mcgs);
        assert!(!spec.state_only_keying);
    }

    #[test]
    fn custom_strategy_spec_deserialize_rejects_missing_required_field() {
        let err = serde_json::from_value::<CustomStrategySpec>(json!({})).unwrap_err();
        assert!(err.to_string().contains("search"), "{err}");
    }
}
