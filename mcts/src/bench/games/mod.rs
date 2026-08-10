//! Subprocess-backed `BenchGame` implementations.
//!
//! Each game kind is a standalone binary (`games/<name>/`) that speaks the
//! JSON-line subprocess protocol.  This module wraps them in `BenchGame`
//! via `game_host::subprocess::SubprocessAdapter` so the benchmark harness
//! can play matches without compiling any game-specific code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use game_host::subprocess::SubprocessAdapter;
use game_host::{AiPresetInfo, GameAdapter};
use serde_json::Value;

use super::{BenchGame, MatchOutcome, StrategyInfo};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// All game kinds known to the benchmark harness, keyed by their
/// machine-readable kind string.
///
/// Each entry maps a kind (e.g. `"ttt"`) to a `SubprocessBenchGame` that
/// talks to the corresponding game binary.  The binary is discovered
/// relative to the running executable, under the name `game-<kind>`.
pub fn registry() -> HashMap<&'static str, Box<dyn BenchGame>> {
    let mut m: HashMap<&'static str, Box<dyn BenchGame>> = HashMap::new();

    for &(kind, pkg_name) in GAME_KINDS {
        let binary = find_game_binary(pkg_name);
        if let Some(path) = binary {
            // SubprocessAdapter::new panics if the binary can't be
            // spawned or doesn't respond — that's intentional (we already
            // verified the path exists above).
            let adapter = SubprocessAdapter::new(path);
            m.insert(kind, Box::new(SubprocessBenchGame { adapter }));
        } else {
            eprintln!(
                "warning: bench game '{kind}' not available \
                 (binary '{pkg_name}' not found)"
            );
        }
    }

    m
}

/// Known game kinds with their package name (also the binary name).
///
/// Must be kept in sync with the workspace members in `games/*`.
const GAME_KINDS: &[(&str, &str)] = &[
    ("atarigo", "game-atarigo"),
    ("bid_ttt", "game-bid_ttt"),
    ("breakthrough", "game-breakthrough"),
    ("count", "game-count"),
    ("druid", "game-druid"),
    ("gonnect", "game-gonnect"),
    ("knightthrough", "game-knightthrough"),
    ("nim", "game-nim"),
    ("null", "game-null"),
    ("othello", "game-othello"),
    ("shibumi", "game-shibumi"),
    ("tak", "game-tak"),
    ("tanbo", "game-tanbo"),
    ("traffic-lights", "game-traffic-lights"),
    ("ttt", "game-ttt"),
    ("unit", "game-unit"),
];

// ---------------------------------------------------------------------------
// SubprocessBenchGame
// ---------------------------------------------------------------------------

/// A `BenchGame` backed by a subprocess game binary.
struct SubprocessBenchGame {
    adapter: SubprocessAdapter,
}

impl BenchGame for SubprocessBenchGame {
    fn kind(&self) -> &'static str {
        self.adapter.kind()
    }

    fn strategies(&self) -> Vec<StrategyInfo> {
        self.adapter
            .ai_presets()
            .into_iter()
            .map(|p: AiPresetInfo| StrategyInfo {
                id: p.id,
                label: p.label,
                description: p.description,
            })
            .collect()
    }

    fn play_match(&self, strategy_a: &str, strategy_b: &str) -> MatchOutcome {
        play_match_via_adapter(&self.adapter, strategy_a, strategy_b)
    }
}

// ---------------------------------------------------------------------------
// Match logic
// ---------------------------------------------------------------------------

/// Play one full game using the subprocess adapter, alternating AI moves.
///
/// `preset_a` is used for the first player (player index 0), `preset_b`
/// for the second (player index 1).
fn play_match_via_adapter(
    adapter: &SubprocessAdapter,
    preset_a: &str,
    preset_b: &str,
) -> MatchOutcome {
    let config = adapter.default_config();
    let state = match adapter.new_state(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: new_state failed: {e}");
            return MatchOutcome {
                winner: None,
                extra: None,
            };
        }
    };

    play_match_inner(adapter, state, preset_a, preset_b, 0)
}

/// Play turns until terminal, alternating presets by turn parity.
fn play_match_inner(
    adapter: &SubprocessAdapter,
    state: Value,
    preset_a: &str,
    preset_b: &str,
    turn: usize,
) -> MatchOutcome {
    // Check if the current state is terminal.
    let view = match adapter.view(&state) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: view failed: {e}");
            return MatchOutcome {
                winner: None,
                extra: None,
            };
        }
    };

    if view.get("terminal").and_then(|t| t.as_bool()).unwrap_or(false) {
        return MatchOutcome {
            winner: view
                .get("winner")
                .and_then(|w| w.as_u64())
                .map(|w| w as usize),
            extra: None,
        };
    }

    // Choose which preset to use for this turn.
    let preset = if turn % 2 == 0 { preset_a } else { preset_b };

    let result = match adapter.ai_move(&state, preset) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: ai_move failed: {e}");
            return MatchOutcome {
                winner: None,
                extra: None,
            };
        }
    };

    play_match_inner(adapter, result.state, preset_a, preset_b, turn + 1)
}

// ---------------------------------------------------------------------------
// Binary discovery
// ---------------------------------------------------------------------------

/// Find a game binary by package name, looking next to the current
/// executable (standard Cargo sibling-binary convention), falling back
/// to bare `<name>` on PATH.
fn find_game_binary(pkg_name: &str) -> Option<PathBuf> {
    let exe_name = exe_name_for(pkg_name);

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(&exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    let candidate = Path::new(&exe_name);
    if candidate.exists() {
        return Some(candidate.to_path_buf());
    }

    None
}

fn exe_name_for(pkg_name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{pkg_name}.exe")
    } else {
        pkg_name.to_owned()
    }
}