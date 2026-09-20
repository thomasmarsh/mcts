//! Wrap a trained TD n-tuple model in play-time PUCT search and place it on the
//! Edax ladder through the shared gate, or time its moves.
//!
//! ```text
//! cargo run --release --example td_search_gate -p game-othello -- \
//!     --model-dir local/output/edax5/slice1/td4-s1 \
//!     [--search-config games/othello/ntuple/td-search.toml] [--set key=value]... \
//!     [--gate-config games/othello/edax/match.toml] [--gate-set key=value]... \
//!     [--label NAME] [--speed-games N] [--speed-out FILE.jsonl]
//! ```
//!
//! `--set` overrides search keys (`iterations`, `c_puct`, `prior_temperature`),
//! `--gate-set` overrides gate keys (`levels`, `games_per_level`, `out`, ...).
//! `--speed-games N` plays N self-play games (two instances of the agent, both
//! seats of the first N balanced openings) and reports seconds per move; it is
//! run before any gate. With `--gate-config` the agent is then gated, one JSONL
//! row per game and per level appended to the gate config's `out`.

use std::io::Write;
use std::sync::Arc;

use game_othello::td_cells::OthelloCells;
use ntuple::{CellFeatures, Model, PuctConfig, PuctPlayer};

mod common;
use common::{load_toml_config, play_from, run_edax_gate, GateConfig, Openings, Timed};

struct Args {
    model_dir: String,
    search_config: String,
    set: Vec<String>,
    gate_config: Option<String>,
    gate_set: Vec<String>,
    label: Option<String>,
    speed_games: usize,
    speed_out: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model_dir: String::new(),
        search_config: "games/othello/ntuple/td-search.toml".to_string(),
        set: Vec::new(),
        gate_config: None,
        gate_set: Vec::new(),
        label: None,
        speed_games: 0,
        speed_out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--model-dir" => a.model_dir = val(),
            "--search-config" => a.search_config = val(),
            "--set" => a.set.extend(["--set".to_string(), val()]),
            "--gate-config" => a.gate_config = Some(val()),
            "--gate-set" => a.gate_set.extend(["--set".to_string(), val()]),
            "--label" => a.label = Some(val()),
            "--speed-games" => a.speed_games = val().parse().expect("--speed-games takes a number"),
            "--speed-out" => a.speed_out = Some(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!a.model_dir.is_empty(), "--model-dir is required");
    a
}

fn speed(
    model: &Arc<Model>,
    cfg: &PuctConfig,
    gcfg: &GateConfig,
    games: usize,
    label: &str,
    out: &Option<String>,
) {
    let mut sized = gcfg.clone();
    sized.games_per_level = (2 * games) as u32;
    let openings = Openings::from_config(&sized);
    let make = || Timed::new(PuctPlayer::new(OthelloCells, model.clone(), cfg.clone()));
    let (mut a, mut b) = (make(), make());
    for p in 0..games {
        let (_, opening) = openings.pair(p);
        play_from(opening, &mut a, &mut b);
        play_from(opening, &mut b, &mut a);
    }
    let (moves, secs) = (a.moves + b.moves, a.secs + b.secs);
    let per_move = secs / moves.max(1) as f64;
    println!(
        "speed {label}: {} iterations/move, {} moves over {} self-play games, {:.1} ms/move \
         ({:.0} iterations/s counting forced moves as searched)",
        cfg.iterations,
        moves,
        2 * games,
        per_move * 1e3,
        cfg.iterations as f64 / per_move
    );
    if let Some(path) = out {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        writeln!(
            f,
            "{}",
            serde_json::json!({
                "type": "speed", "agent": label, "iterations": cfg.iterations,
                "c_puct": cfg.c_puct, "prior_temperature": cfg.prior_temperature,
                "games": 2 * games, "moves": moves, "secs": secs, "secs_per_move": per_move,
            })
        )
        .unwrap();
    }
}

fn main() {
    let args = parse_args();
    let cfg: PuctConfig = load_toml_config(&args.search_config, &args.set);
    let model = Arc::new(Model::load(
        std::path::Path::new(&args.model_dir),
        &OthelloCells.orientations(),
    ));
    let base = args.label.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model_dir)
            .file_name()
            .map_or("td".to_string(), |n| n.to_string_lossy().into_owned())
    });
    let label = format!("{base}+puct{}", cfg.iterations);
    let gcfg: Option<GateConfig> =
        args.gate_config.as_ref().map(|p| load_toml_config(p, &args.gate_set));

    if args.speed_games > 0 {
        let gate = gcfg.as_ref().expect("--speed-games needs --gate-config for the openings");
        speed(&model, &cfg, gate, args.speed_games, &label, &args.speed_out);
    }
    if let Some(gate) = &gcfg {
        run_edax_gate(gate, &label, Some(cfg.iterations as u64), |_seed| {
            PuctPlayer::new(OthelloCells, model.clone(), cfg.clone())
        });
    }
}
