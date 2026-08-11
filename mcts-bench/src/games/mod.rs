//! Subprocess-backed `BenchGame` implementations.
//!
//! Each game kind is a standalone binary (`games/<name>/`) that speaks the
//! JSON-line subprocess protocol.  This module wraps them in `BenchGame`
//! via `game_host::subprocess::SubprocessAdapter` so the benchmark harness
//! can play matches without compiling any game-specific code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use game_host::subprocess::SubprocessAdapter;
use game_host::{AiPresetInfo, GameAdapter, GameDescription};
use serde_json::Value;

use crate::{BenchGame, MatchOutcome, StrategyInfo};

pub fn registry() -> HashMap<&'static str, Box<dyn BenchGame>> {
    let mut m: HashMap<&'static str, Box<dyn BenchGame>> = HashMap::new();

    for &(kind, pkg_name) in GAME_KINDS {
        let binary = find_game_binary(pkg_name);
        if let Some(path) = binary {
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

/// Describe every registered game kind by spawning its binary once with
/// `describe` and exiting, rather than opening a persistent
/// [`SubprocessAdapter`] session as [`registry`] does for match play.  A
/// game whose binary is missing or fails to describe itself is skipped
/// with a warning on stderr, same as `registry`.
pub fn describe_games() -> Vec<GameDescription> {
    let mut descriptions = Vec::new();

    for &(kind, pkg_name) in GAME_KINDS {
        let Some(path) = find_game_binary(pkg_name) else {
            eprintln!(
                "warning: bench game '{kind}' not available \
                 (binary '{pkg_name}' not found)"
            );
            continue;
        };

        match describe_one(&path) {
            Ok(desc) => descriptions.push(desc),
            Err(e) => eprintln!("warning: bench game '{kind}' describe failed: {e}"),
        }
    }

    descriptions
}

/// Spawn `binary_path describe`, wait for it to exit, and parse its single
/// JSON line of output as a [`GameDescription`].
fn describe_one(binary_path: &Path) -> Result<GameDescription, String> {
    let output = Command::new(binary_path)
        .arg("describe")
        .output()
        .map_err(|e| format!("failed to spawn: {e}"))?;

    if !output.status.success() {
        return Err(format!("exited with {}", output.status));
    }

    serde_json::from_slice(&output.stdout).map_err(|e| format!("failed to parse output: {e}"))
}

const GAME_KINDS: &[(&str, &str)] = &[
    ("atarigo", "game-atarigo"),
    ("bid-ttt", "game-bid-ttt"),
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

fn play_match_inner(
    adapter: &SubprocessAdapter,
    state: Value,
    preset_a: &str,
    preset_b: &str,
    turn: usize,
) -> MatchOutcome {
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

    if view
        .get("terminal")
        .and_then(|t| t.as_bool())
        .unwrap_or(false)
    {
        return MatchOutcome {
            winner: view
                .get("winner")
                .and_then(|w| w.as_u64())
                .map(|w| w as usize),
            extra: None,
        };
    }

    let preset = if turn.is_multiple_of(2) {
        preset_a
    } else {
        preset_b
    };

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
