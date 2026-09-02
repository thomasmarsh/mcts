use crate::{
    AiMoveResult, AiPresetInfo, Analysis, BookInfo, CompareValidationField, ConfiguredMatchResult,
    GameConfigSchema, HostError, TunerInfo,
};
use serde_json::Value;

/// Type-erased, per-game-kind adapter over `mcts::game::Game` +
/// `mcts::strategies::Search`.
///
/// Every method is stateless: state flows in as a JSON `Value` and back out
/// as another. Concrete adapters deserialize `Value` arguments into real
/// game types, call through to `Game`/ `Search`, and re-serialize the result.
/// This is the same shape as `server/adapters/`'s `GameAdapter` trait but
/// with a simpler error type (no axum dependency).
pub trait GameAdapter: Send + Sync {
    fn kind(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// A default/example config value for `new_state`.  Also serves as a
    /// config schema hint for generic new-game forms.
    fn default_config(&self) -> Value;

    /// The game-setup axis `new_state` / `tune_eval` / `book_build` accept
    /// as `game_config`, described (bounds, types) so a generic caller can
    /// render and validate a form without a per-game hardcode. The default
    /// is empty: the board is fixed at compile time, nothing to configure.
    fn config_schema(&self) -> GameConfigSchema {
        GameConfigSchema::default()
    }

    fn new_state(&self, config: Value) -> Result<Value, HostError>;
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError>;
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError>;
    fn view(&self, state: &Value) -> Result<Value, HostError>;

    fn ai_presets(&self) -> Vec<AiPresetInfo>;
    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError>;
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError>;

    /// Tunable strategy search-space metadata, for games that support
    /// tuner-style hyperparameter tuning via `tune_eval`. `None` (the
    /// default) for every game that doesn't -- tuning support is opt-in per
    /// game, not a universal requirement.
    fn tuner(&self) -> Option<TunerInfo> {
        None
    }

    /// Validate one configured comparison without playing a game. Concrete
    /// adapters retain ownership of game setup and strategy construction;
    /// this default implementation exercises those same adapter paths with
    /// zero evaluation rounds.
    fn validate_compare(
        &self,
        candidate_config: Value,
        baseline_config: Value,
        game_config: Option<Value>,
    ) -> Vec<CompareValidationField> {
        let mut errors =
            self.validate_compare_many(vec![candidate_config], baseline_config, game_config);
        for error in &mut errors {
            error.candidate_index = None;
        }
        errors
    }

    /// Validate several candidate configurations against one baseline without
    /// playing a game. Game setup and the baseline are built once; each
    /// candidate is built independently so every configuration is checked.
    fn validate_compare_many(
        &self,
        candidate_configs: Vec<Value>,
        baseline_config: Value,
        game_config: Option<Value>,
    ) -> Vec<CompareValidationField> {
        let mut errors = Vec::new();
        if self.tuner().is_none() {
            errors.push(CompareValidationField {
                field: "candidate_config".into(),
                message: "game does not support configured strategy validation".into(),
                candidate_index: None,
            });
            return errors;
        }
        let setup = game_config.clone().unwrap_or_else(|| self.default_config());
        if let Err(error) = self.new_state(setup) {
            errors.push(CompareValidationField {
                field: "game_config".into(),
                message: error.message,
                candidate_index: None,
            });
        }
        let baseline_name = self
            .tuner()
            .and_then(|info| info.baselines.first().cloned());
        if !errors.iter().any(|error| error.field == "game_config") {
            for (candidate_index, candidate_config) in candidate_configs.into_iter().enumerate() {
                let mut sink = |_result: ConfiguredMatchResult| Ok(());
                if let Err(error) = self.tune_eval(
                    candidate_config,
                    0,
                    Some(0),
                    baseline_name.clone(),
                    None,
                    game_config.clone(),
                    Some(1),
                    None,
                    None,
                    None,
                    &mut sink,
                ) {
                    errors.push(CompareValidationField {
                        field: "candidate_config".into(),
                        message: error.message,
                        candidate_index: Some(candidate_index),
                    });
                }
            }
            let mut sink = |_result: ConfiguredMatchResult| Ok(());
            if let Err(error) = self.tune_eval(
                baseline_config,
                0,
                Some(0),
                None,
                None,
                game_config,
                Some(1),
                None,
                None,
                None,
                &mut sink,
            ) {
                errors.push(CompareValidationField {
                    field: "baseline_config".into(),
                    message: error.message,
                    candidate_index: None,
                });
            }
        }
        errors
    }

    /// Play `rounds` games of a `params`-built candidate strategy against an
    /// opponent and return a cost (lower is better) plus win/loss/draw
    /// counts, as a JSON object. The opponent comes from exactly one of:
    /// `baseline`, a named entry of `tuner()`'s `baselines` list (`None`
    /// means the first/default entry), or `baseline_config`, a raw params
    /// JSON object (same schema as `params`) built the same way the
    /// candidate is -- e.g. a previously discovered config used as the next
    /// run's opponent instead of a hand-authored preset. The only real
    /// caller (`run_tune_eval`) guarantees at most one of the two is
    /// `Some` before invoking this method. CLI-only (`tune eval`) -- never
    /// dispatched over the JSONL loop, since a full multi-game match is a
    /// batch job, not a per-move request.
    ///
    /// `game_config` pins every game in this call to a non-default game
    /// setup (same schema as `new_state`'s `config` argument, e.g. Druid's
    /// `{"size": {...}}`), falling back to `default_config()`'s value when
    /// `None`. A game whose board is fixed at compile time (`default_config`
    /// returns `{}`) has nothing to vary here and ignores this argument.
    ///
    /// `max_iterations` is an operator-set, per-*run* compute budget (not a
    /// per-trial hyperparameter tuner searches over -- see
    /// `mcts_tune::SearchBudget`'s doc comment for why), forwarded verbatim
    /// from `--max-iterations`. `None` means "use `mcts-tune`'s own
    /// historical default." An implementation threading this into
    /// `mcts_tune::SearchBudget` must apply the *same* value to both the
    /// candidate and, for a `baseline_config`-backed opponent,
    /// `mcts_tune::build_search`'s own budget -- see that function's doc
    /// comment for why an asymmetric override there is a real bug, not a
    /// simplification.
    /// `trace_path`, forwarded verbatim from `--trace-path`, is a plain
    /// file path a `mcts_tune::trace::MoveTracer` appends move-trace JSON
    /// lines to as self-play games are played -- for live monitoring/
    /// sanity-checking a tuner run in progress. `None` disables tracing
    /// entirely (no file opened, no per-ply overhead).
    /// `trace_game_sequence_start` optionally assigns the first traced game
    /// sequence; it requires `trace_path` and is intended for isolated task
    /// bundles rather than shared experiment traces.
    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        baseline: Option<String>,
        baseline_config: Option<Value>,
        game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        trace_game_sequence_start: Option<u64>,
        on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        Err(HostError::not_found("tuning not supported"))
    }

    /// Opening-book metadata, for games that support Quasi-Best-First
    /// self-play book generation via `book_build`. `None` (the default) for
    /// every game that hasn't wired this up -- opt-in, same shape as
    /// `tuner()`.
    fn book(&self) -> Option<BookInfo> {
        None
    }

    /// Runs `rounds` self-play games (Chaslot et al.'s Quasi-Best-First
    /// algorithm) and returns the resulting opening book, serialized as
    /// JSON. `seed` seeds the run for reproducibility; `None` means
    /// whatever the game's own default is. `game_config` pins the run to a
    /// non-default game setup, same convention as `tune_eval`'s argument of
    /// the same name. CLI-only (`book build`), never dispatched over the
    /// JSONL loop -- same rationale as `tune_eval`: a many-game self-play
    /// run is a batch job, not a per-move request.
    #[allow(unused_variables)]
    fn book_build(
        &self,
        rounds: u32,
        seed: Option<u64>,
        game_config: Option<Value>,
    ) -> Result<Value, HostError> {
        Err(HostError::not_found(
            "opening book generation not supported",
        ))
    }
}
