use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
};
use serde_json::Value;

use game_count::{Count, CountingGame, Move as CountMove};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;

/// Fixed seed for every `ai_move`/`analyze` search built through
/// [`presets`] -- `GameAdapter::ai_move`/`analyze` take no seed argument,
/// so this is the only seed available to
/// `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/count/presets.json` (or the file named by `COUNT_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("COUNT_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path)
            .expect("games/count/presets.json must parse")
    })
}

fn state_to_value(s: &Count) -> Value {
    serde_json::json!({"value": s.0})
}

fn value_to_state(v: &Value) -> Result<Count, HostError> {
    let val = v["value"]
        .as_i64()
        .ok_or_else(|| HostError::bad_request("no value"))?;
    Ok(Count(val as i32))
}

struct CountAdapter;

impl GameAdapter for CountAdapter {
    fn kind(&self) -> &'static str {
        "count"
    }
    fn label(&self) -> &'static str {
        "Count"
    }
    fn description(&self) -> &'static str {
        "A counting game where you add or subtract toward 10."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&Count::default()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !CountingGame::is_terminal(&s) {
            CountingGame::generate_actions(&s, &mut mv);
        }
        Ok(mv
            .into_iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let m: CountMove = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        if CountingGame::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        CountingGame::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&CountingGame::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        presets().ai_presets()
    }
    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state(state)?;
        if CountingGame::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<CountingGame>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let action = ai.choose_action(&s);
        let next = CountingGame::apply(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(&action).unwrap(),
            state: state_to_value(&next),
        })
    }
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        _: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state(state)?;
        if CountingGame::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<CountingGame>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
            .principal_variation
            .first()
            .map(|a| serde_json::to_value(a).unwrap());
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: serde_json::to_value(&a.action).unwrap(),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| serde_json::to_value(a).unwrap())
                .collect(),
            total_visits: report.total_visits,
            suggested_move: suggested,
        })
    }

    // No `tuner()`/`tune_eval()` override: `CountingGame::num_players()` is
    // `1` (`player_to_move` always returns the same `Unit`), so `mcts-tune`'s
    // `play_one` -- which alternates on `player_to_move().to_index()` and
    // assumes a real 2-player contest -- would only ever let one side move
    // in every game, making "candidate vs baseline" meaningless (whichever
    // side is `first` just solves the puzzle alone; `second` never acts).
    // `winner()` also always returns `Some(Unit)` with no path-length
    // penalty, so nothing pushes a search toward reaching the target state
    // quickly -- confirmed by hand to occasionally wander for a very long
    // time before terminating. Left unsupported, the same as `null`/`unit`/
    // `shibumi` (no real 2-player search to tune here either).
}

fn main() {
    run_cli(CountAdapter);
}
