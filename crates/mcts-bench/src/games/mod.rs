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
use game_host::{AiPresetInfo, GameAdapter, GameDescription, TunerInfo};
use serde_json::Value;

use crate::{BenchGame, MatchOutcome, PlyEvent, StrategyInfo};

pub fn registry() -> HashMap<&'static str, Box<dyn BenchGame>> {
    let mut m: HashMap<&'static str, Box<dyn BenchGame>> = HashMap::new();

    for game in GAME_KINDS {
        let binary = game.binary_path();
        if let Some(path) = binary {
            let adapter = SubprocessAdapter::new(path);
            m.insert(game.kind, Box::new(SubprocessBenchGame { adapter }));
        } else {
            eprintln!(
                "warning: bench game '{}' not available (binary '{}' not found)",
                game.kind,
                game.binary_name(),
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

    for game in GAME_KINDS {
        let Some(path) = game.binary_path() else {
            eprintln!(
                "warning: bench game '{}' not available (binary '{}' not found)",
                game.kind,
                game.binary_name(),
            );
            continue;
        };

        match describe_one(&path) {
            Ok(desc) => descriptions.push(desc),
            Err(e) => eprintln!("warning: bench game '{}' describe failed: {e}", game.kind),
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

/// Describe every registered game kind's tuner metadata (search space,
/// baselines, eval rounds) by spawning its binary once with `tune describe`
/// and exiting -- same one-shot-per-binary approach as [`describe_games`],
/// so a game shows up here purely by shipping a `tuner()` implementation,
/// independent of whether it also has a live gameplay session open in
/// `server::adapter::registry()` (which only covers games with a UI
/// renderer). A game whose binary is missing, or that doesn't implement
/// `tuner()` (exits non-zero with "tuning not supported"), is silently
/// skipped, same as `describe_games` skips a missing binary.
pub fn describe_tuners() -> Vec<(&'static str, TunerInfo)> {
    let mut tuners = Vec::new();

    for game in GAME_KINDS {
        let Some(path) = game.binary_path() else {
            eprintln!(
                "warning: bench game '{}' not available (binary '{}' not found)",
                game.kind,
                game.binary_name(),
            );
            continue;
        };

        match tune_describe_one(&path) {
            Ok(Some(info)) => tuners.push((game.kind, info)),
            Ok(None) => {}
            Err(e) => eprintln!(
                "warning: bench game '{}' tune describe failed: {e}",
                game.kind
            ),
        }
    }

    tuners
}

/// Return the explicit host executable for an externally hosted tuner game.
/// Workspace games return `None` and use `bench tuner`'s standard
/// `target/release/game-<kind>` convention.
pub fn tuner_target_binary(kind: &str) -> Option<&'static str> {
    GAME_KINDS
        .iter()
        .find(|game| game.kind == kind)
        .and_then(GameKind::tuner_target_binary)
}

/// Spawn `binary_path tune describe`, wait for it to exit, and parse its
/// single JSON line of output as a [`TunerInfo`]. `Ok(None)` means the game
/// doesn't implement `tuner()` (`tune describe` exits 1 with "tuning not
/// supported" on stderr, per `game_host::run_cli_with`) -- not an error.
fn tune_describe_one(binary_path: &Path) -> Result<Option<TunerInfo>, String> {
    let output = Command::new(binary_path)
        .args(["tune", "describe"])
        .output()
        .map_err(|e| format!("failed to spawn: {e}"))?;

    if !output.status.success() {
        return Ok(None);
    }

    serde_json::from_slice(&output.stdout)
        .map(Some)
        .map_err(|e| format!("failed to parse output: {e}"))
}

struct GameKind {
    kind: &'static str,
    binary: GameBinary,
}

enum GameBinary {
    Workspace(&'static str),
    External(&'static str),
}

impl GameKind {
    fn binary_path(&self) -> Option<PathBuf> {
        match &self.binary {
            GameBinary::Workspace(binary) => find_binary_named(binary),
            GameBinary::External(path) => {
                let path = PathBuf::from(path);
                path.exists().then_some(path)
            }
        }
    }

    fn binary_name(&self) -> &str {
        match &self.binary {
            GameBinary::Workspace(binary) | GameBinary::External(binary) => binary,
        }
    }

    /// An explicit host argument is required only for games outside this
    /// workspace; workspace games retain the CLI's normal binary convention.
    fn tuner_target_binary(&self) -> Option<&'static str> {
        match &self.binary {
            GameBinary::Workspace(_) => None,
            GameBinary::External(path) => Some(*path),
        }
    }
}

const GAME_KINDS: &[GameKind] = &[
    GameKind {
        kind: "atarigo",
        binary: GameBinary::Workspace("game-atarigo"),
    },
    GameKind {
        kind: "bid-ttt",
        binary: GameBinary::Workspace("game-bid-ttt"),
    },
    GameKind {
        kind: "breakthrough",
        binary: GameBinary::Workspace("game-breakthrough"),
    },
    GameKind {
        kind: "congo",
        binary: GameBinary::Workspace("game-congo"),
    },
    GameKind {
        kind: "druid",
        binary: GameBinary::Workspace("game-druid"),
    },
    GameKind {
        kind: "focus-2p",
        binary: GameBinary::Workspace("game-focus-2p"),
    },
    GameKind {
        kind: "focus-3p",
        binary: GameBinary::Workspace("game-focus-3p"),
    },
    GameKind {
        kind: "focus-4p",
        binary: GameBinary::Workspace("game-focus-4p"),
    },
    GameKind {
        kind: "gonnect",
        binary: GameBinary::Workspace("game-gonnect"),
    },
    GameKind {
        kind: "ingenious",
        binary: GameBinary::Workspace("game-ingenious"),
    },
    GameKind {
        kind: "knightthrough",
        binary: GameBinary::Workspace("game-knightthrough"),
    },
    GameKind {
        kind: "margo",
        binary: GameBinary::Workspace("game-margo"),
    },
    GameKind {
        kind: "nego",
        binary: GameBinary::External("../nego/target/release/nego-host"),
    },
    GameKind {
        kind: "nim",
        binary: GameBinary::Workspace("game-nim"),
    },
    GameKind {
        kind: "othello",
        binary: GameBinary::Workspace("game-othello"),
    },
    GameKind {
        kind: "tak",
        binary: GameBinary::Workspace("game-tak"),
    },
    GameKind {
        kind: "tanbo",
        binary: GameBinary::Workspace("game-tanbo"),
    },
    GameKind {
        kind: "traffic-lights",
        binary: GameBinary::Workspace("game-traffic-lights"),
    },
    GameKind {
        kind: "ttt",
        binary: GameBinary::Workspace("game-ttt"),
    },
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

    fn play_match(
        &self,
        strategy_a: &str,
        strategy_b: &str,
        on_ply: &mut dyn FnMut(PlyEvent),
    ) -> MatchOutcome {
        play_match_via_adapter(&self.adapter, strategy_a, strategy_b, on_ply)
    }
}

fn play_match_via_adapter(
    adapter: &SubprocessAdapter,
    preset_a: &str,
    preset_b: &str,
    on_ply: &mut dyn FnMut(PlyEvent),
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

    on_ply(PlyEvent {
        ply: 0,
        state: &state,
        mv: None,
        player: None,
    });

    play_match_inner(adapter, state, preset_a, preset_b, 0, on_ply)
}

fn play_match_inner(
    adapter: &SubprocessAdapter,
    state: Value,
    preset_a: &str,
    preset_b: &str,
    turn: usize,
    on_ply: &mut dyn FnMut(PlyEvent),
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

    let result = match adapter.ai_move(&state, preset, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: ai_move failed: {e}");
            return MatchOutcome {
                winner: None,
                extra: None,
            };
        }
    };

    on_ply(PlyEvent {
        ply: (turn + 1) as u32,
        state: &result.state,
        mv: Some(&result.mv),
        player: Some(preset),
    });

    play_match_inner(adapter, result.state, preset_a, preset_b, turn + 1, on_ply)
}

/// Locate a sibling game binary using the same convention as the existing
/// registry and describe commands.
pub fn find_game_binary(kind: &str) -> Option<PathBuf> {
    match GAME_KINDS.iter().find(|game| game.kind == kind) {
        Some(game) => game.binary_path(),
        None => find_binary_named(kind),
    }
}

fn find_binary_named(binary: &str) -> Option<PathBuf> {
    let exe_name = exe_name_for(binary);

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Write a throwaway shell script standing in for a `game-*` binary's
    /// `tune describe` response, so `tune_describe_one`'s exit-code/JSON
    /// dispatch can be tested without a real, built game binary -- the
    /// thing that makes `describe_tuners()` itself impractical to exercise
    /// from a `cargo test` in this crate (`find_game_binary` looks next to
    /// `current_exe()`, which is a `target/*/deps/` test binary, not the
    /// `target/*/` dir the real `game-*` binaries land in). Returned path
    /// lives directly under the OS temp dir with a unique name -- the
    /// script itself is small enough not to warrant directory cleanup.
    fn fake_binary(body: &str) -> PathBuf {
        let n = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("mcts_bench_fake_game_{}_{n}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/bin/sh\n{body}").unwrap();
        drop(f);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn tune_describe_one_parses_tuner_info_on_success() {
        let json = r#"{"id":"rave","baselines":["strong"],"eval_rounds":20,"parameters":[],"conditions":[],"game_config":{}}"#;
        let path = fake_binary(&format!("echo '{json}'; exit 0"));

        let info = tune_describe_one(&path).unwrap().unwrap();
        assert_eq!(info.id, "rave");
        assert_eq!(info.baselines, vec!["strong".to_string()]);
        assert_eq!(info.eval_rounds, 20);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tune_describe_one_returns_none_when_tuning_unsupported() {
        // Mirrors `game_host::run_cli_with`'s `tune describe` arm for a
        // game whose `tuner()` returns `None`: stderr message, exit 1.
        let path = fake_binary("echo 'tuning not supported' >&2; exit 1");

        assert!(tune_describe_one(&path).unwrap().is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tune_describe_one_errors_on_unparseable_output() {
        let path = fake_binary("echo 'not json'; exit 0");

        assert!(tune_describe_one(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tune_describe_one_errors_when_binary_is_missing() {
        let missing = Path::new("/nonexistent/definitely-not-a-binary");

        assert!(tune_describe_one(missing).is_err());
    }
}
