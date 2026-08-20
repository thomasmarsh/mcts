//! Standalone Hex game binary that speaks the JSON-line subprocess protocol
//! on stdin/stdout.
//!
//! Built by `cargo build -p game-hex-gen` and used by the server/bench
//! crates via `game_host::SubprocessAdapter`. Board size is chosen at
//! request time via `dispatch_size!`, the same const-generic-dispatch
//! pattern `games/gonnect/src/main.rs` hand-writes for its own multiple
//! board sizes -- `game_hex_gen::{Position, Hex, ...}` are themselves
//! generic over `<const N: usize, const WORDS: usize>` (see
//! `gdl/src/codegen/hex.rs`'s doc comment for why), so one generated crate
//! serves every size below rather than needing one generated crate per size.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_hex_gen::{HashedPosition, Hex, Move, Player, Position};
use mcts::game::{Game, PlayerIndex};
use mcts_tune::presets::PresetTable;

/// Fixed seed for every `ai_move`/`analyze` search built through [`presets`]
/// -- `GameAdapter::ai_move`/`analyze` take no seed argument, so this is the
/// only seed available to `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/hex-gen/presets.json` (or the file named by `HEX_GEN_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment). Presets
/// are size-invariant: `build_easy`/`build_strong` never varied by `N`/
/// `WORDS`, only by which `Hex<N, WORDS>` `PresetTable::build` is
/// monomorphized for at each call site.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("HEX_GEN_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/hex-gen/presets.json must parse")
    })
}

/// `(N, WORDS)` pairs this binary serves. Each is a distinct
/// `HashedPosition<N, WORDS>` monomorphization -- see `dispatch_size!` below
/// -- so board size is chosen at request time (via `new_state`'s
/// `{"size": N}` config, or inferred from an existing state's cell count)
/// rather than fixed at compile time. 11x11 is regulation tournament size
/// and stays the default.
const SUPPORTED_SIZES: &[(usize, usize)] = &[(5, 1), (7, 1), (11, 2)];
const DEFAULT_SIZE: usize = 11;

/// Runs `$body` with `$n`/`$words` bound as the matching `usize` consts for
/// board size `$size` (a runtime value). The match arms double as
/// validation: `$size` must be one of `SUPPORTED_SIZES` or the default arm
/// returns a `HostError::bad_request` -- so every caller of this macro
/// implicitly rejects an unsupported size before touching a `Position`.
macro_rules! dispatch_size {
    ($size:expr, $n:ident, $words:ident, $body:block) => {
        match $size {
            5 => {
                const $n: usize = 5;
                const $words: usize = 1;
                $body
            }
            7 => {
                const $n: usize = 7;
                const $words: usize = 1;
                $body
            }
            11 => {
                const $n: usize = 11;
                const $words: usize = 2;
                $body
            }
            other => {
                return Err(HostError::bad_request(format!(
                    "unsupported board size {other} (supported: 5, 7, 11)"
                )))
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

/// The wire shape for a position: a plain `N * N`-cell array (row-major,
/// `row * N + col`; a `Vec` rather than a fixed-size array since serde's
/// derive only implements `Serialize`/`Deserialize` for array lengths up to
/// 32) plus whose turn it is, deliberately not `Position`'s internal
/// per-player bitboard encoding.
#[derive(Serialize, Deserialize)]
struct WireState {
    turn: Player,
    cells: Vec<Option<Player>>,
}

#[derive(Serialize)]
struct GameView {
    turn: Player,
    cells: Vec<Option<Player>>,
    winner: Option<Player>,
    terminal: bool,
}

#[derive(Deserialize)]
struct NewGameConfig {
    size: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cells_of<const N: usize, const WORDS: usize>(
    position: &Position<N, WORDS>,
) -> Vec<Option<Player>> {
    (0..N * N)
        .map(|i| {
            if position.occupied[0].get_index(i) {
                Some(Player::P0)
            } else if position.occupied[1].get_index(i) {
                Some(Player::P1)
            } else {
                None
            }
        })
        .collect()
}

fn state_to_value<const N: usize, const WORDS: usize>(state: &HashedPosition<N, WORDS>) -> Value {
    serde_json::to_value(WireState {
        turn: state.position.turn,
        cells: cells_of(&state.position),
    })
    .expect("WireState always serializes")
}

fn value_to_state<const N: usize, const WORDS: usize>(
    v: &Value,
) -> Result<HashedPosition<N, WORDS>, HostError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    if wire.cells.len() != N * N {
        return Err(HostError::bad_request(format!(
            "invalid state: cells has {} entries, expected {}",
            wire.cells.len(),
            N * N
        )));
    }
    let mut state = HashedPosition::<N, WORDS>::new();
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(player) = cell {
            state.position.occupied[player.to_index()].set_index(i);
        }
    }
    state.position.turn = wire.turn;
    Ok(state)
}

/// Recovers `N` from a wire state's cell count by matching it against
/// `SUPPORTED_SIZES` -- no separate `size` field is needed on the state
/// wire format because `cells.len() == N * N` already determines `N`
/// uniquely.
fn size_from_cell_count(len: usize) -> Result<usize, HostError> {
    SUPPORTED_SIZES
        .iter()
        .map(|&(n, _)| n)
        .find(|&n| n * n == len)
        .ok_or_else(|| HostError::bad_request(format!("unexpected cell count {len}")))
}

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

struct HexGenAdapter;

impl GameAdapter for HexGenAdapter {
    fn kind(&self) -> &'static str {
        "hex-gen"
    }

    fn label(&self) -> &'static str {
        "Hex"
    }

    fn description(&self) -> &'static str {
        "Hex -- connect your two opposite edges. Generated by gdl's Core-IR-to-Rust codegen (see \
         gdl/ROADMAP.md phase 6), const-generic over board side so one generated crate serves \
         5x5, 7x7, and regulation 11x11 boards."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({ "size": DEFAULT_SIZE })
    }

    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        let config: NewGameConfig = serde_json::from_value(config)
            .map_err(|e| HostError::bad_request(format!("invalid config: {e}")))?;
        dispatch_size!(config.size, N, WORDS, {
            Ok(state_to_value(&HashedPosition::<N, WORDS>::new()))
        })
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let wire: WireState = serde_json::from_value(state.clone())
            .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
        let size = size_from_cell_count(wire.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: HashedPosition<N, WORDS> = value_to_state(state)?;
            let mut moves = Vec::new();
            if !Hex::<N, WORDS>::is_terminal(&s) {
                Hex::<N, WORDS>::generate_actions(&s, &mut moves);
            }
            Ok(moves.into_iter().map(|m| Value::from(m.0 as u64)).collect())
        })
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let wire: WireState = serde_json::from_value(state.clone())
            .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
        let size = size_from_cell_count(wire.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: HashedPosition<N, WORDS> = value_to_state(state)?;
            let idx = mv
                .as_u64()
                .ok_or_else(|| HostError::bad_request("move must be a cell index"))?;
            let action = Move(idx as u8);

            if Hex::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut legal = Vec::new();
            Hex::<N, WORDS>::generate_actions(&s, &mut legal);
            if !legal.contains(&action) {
                return Err(HostError::bad_request("illegal move"));
            }

            Ok(state_to_value(&Hex::<N, WORDS>::apply(s, &action)))
        })
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let wire: WireState = serde_json::from_value(state.clone())
            .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
        let size = size_from_cell_count(wire.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: HashedPosition<N, WORDS> = value_to_state(state)?;
            let view = GameView {
                turn: s.position.turn,
                cells: cells_of(&s.position),
                winner: s.position.winner(),
                terminal: Hex::<N, WORDS>::is_terminal(&s),
            };
            serde_json::to_value(view)
                .map_err(|e| HostError::internal(format!("serialize view: {e}")))
        })
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
        let wire: WireState = serde_json::from_value(state.clone())
            .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
        let size = size_from_cell_count(wire.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: HashedPosition<N, WORDS> = value_to_state(state)?;
            if Hex::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = mcts_tune::presets::build_strategy::<Hex<N, WORDS>>(
                presets(),
                preset,
                custom_spec.as_ref(),
                PRESET_SEED,
            )?;
            let action = ai.choose_action(&s);
            let next = Hex::<N, WORDS>::apply(s, &action);
            Ok(AiMoveResult {
                mv: Value::from(action.0 as u64),
                state: state_to_value(&next),
            })
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let wire: WireState = serde_json::from_value(state.clone())
            .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
        let size = size_from_cell_count(wire.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: HashedPosition<N, WORDS> = value_to_state(state)?;
            if Hex::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = mcts_tune::presets::build_strategy::<Hex<N, WORDS>>(
                presets(),
                preset,
                custom_spec.as_ref(),
                PRESET_SEED,
            )?;
            let _ = ai.choose_action(&s);
            let report = ai.root_report(&s);

            let suggested_move = report
                .principal_variation
                .first()
                .map(|a| Value::from(a.0 as u64));

            Ok(Analysis {
                actions: report
                    .actions
                    .into_iter()
                    .map(|a| AnalysisAction {
                        action: Value::from(a.action.0 as u64),
                        visits: a.visits,
                        mean_value: a.mean_value,
                        is_proven: a.is_proven,
                    })
                    .collect(),
                principal_variation: report
                    .principal_variation
                    .into_iter()
                    .map(|a| Value::from(a.0 as u64))
                    .collect(),
                total_visits: report.total_visits,
                suggested_move,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    run_cli(HexGenAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_supports_every_advertised_size() {
        for &(n, _) in SUPPORTED_SIZES {
            let v = HexGenAdapter
                .new_state(serde_json::json!({ "size": n }))
                .unwrap_or_else(|e| panic!("new_state({n}) failed: {e}"));
            assert_eq!(v["cells"].as_array().unwrap().len(), n * n);
        }
    }

    #[test]
    fn new_state_rejects_unsupported_size() {
        assert!(HexGenAdapter
            .new_state(serde_json::json!({ "size": 6 }))
            .is_err());
    }

    #[test]
    fn legal_moves_and_apply_round_trip_at_every_size() {
        for &(n, _) in SUPPORTED_SIZES {
            let state = HexGenAdapter
                .new_state(serde_json::json!({ "size": n }))
                .unwrap();
            let moves = HexGenAdapter.legal_moves(&state).unwrap();
            assert!(
                !moves.is_empty(),
                "size {n} should have legal moves from the empty board"
            );
            let next = HexGenAdapter.apply(&state, &moves[0]).unwrap();
            assert_eq!(next["cells"].as_array().unwrap().len(), n * n);
        }
    }
}
