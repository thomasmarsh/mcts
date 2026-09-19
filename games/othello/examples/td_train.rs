//! Train an Othello n-tuple value network by self-play TD(lambda) with temporal
//! coherence learning (no search), log progress as JSONL, and optionally place
//! the raw agent on the Edax ladder through the shared gate.
//!
//! ```text
//! cargo run --release --example td_train -p game-othello -- \
//!     [--config games/othello/ntuple/td-ref.toml] [--set key=value]... \
//!     [--gate-config games/othello/edax/match.toml] [--gate-set key=value]... \
//!     [--gate-only]
//! ```
//!
//! `--set` overrides training keys, `--gate-set` overrides gate keys (levels,
//! games_per_level, out, ...). With `--gate-config` the trained model is gated
//! after training; with `--gate-only` training is skipped and the model already
//! in `out_dir` is gated. Training writes `train.jsonl` (one row per
//! `log_every` episodes, appended as it goes), a checkpoint (`model.toml`,
//! `weights.bin`, `weights.meta.json`) at every row, and the final weights.
//! The gate appends per-game and per-level rows to `out_dir/gate.jsonl` unless
//! the gate config names its own `out`.
//!
//! Training is single-threaded; run several seeds as separate processes.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use game_othello::td_cells::OthelloCells;
use game_othello::{Move, State};
use ntuple::{play_match, uniform_random, CellFeatures, GreedyPlayer, Model, Trainer, TrainConfig};
use rand::rngs::SmallRng;
use rand::Rng;

mod common;
use common::{load_toml_config, run_edax_gate, GateConfig};

struct Args {
    config: String,
    set: Vec<String>,
    gate_config: Option<String>,
    gate_set: Vec<String>,
    gate_only: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        config: "games/othello/ntuple/td-ref.toml".to_string(),
        set: Vec::new(),
        gate_config: None,
        gate_set: Vec::new(),
        gate_only: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => a.config = val(),
            "--set" => a.set.extend(["--set".to_string(), val()]),
            "--gate-config" => a.gate_config = Some(val()),
            "--gate-set" => a.gate_set.extend(["--set".to_string(), val()]),
            "--gate-only" => a.gate_only = true,
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

/// Opponent that plays the move leaving it the most discs over the opponent's
/// (1-ply greedy on the immediate disc differential), random among ties.
fn greedy_discs(state: &State, actions: &[Move], rng: &mut SmallRng) -> usize {
    use game_othello::{Othello, Player};
    use mcts::game::Game;
    let mover = state.turn;
    let diff = |s: &State| {
        let (b, w) = (s.black.bits().count_ones() as i32, s.white.bits().count_ones() as i32);
        if mover == Player::Black {
            b - w
        } else {
            w - b
        }
    };
    let scores: Vec<i32> = actions.iter().map(|a| diff(&Othello::apply(*state, a))).collect();
    let best = *scores.iter().max().unwrap();
    let ties: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] == best).collect();
    ties[rng.gen_range(0..ties.len())]
}

/// `f32` as the shortest decimal that round-trips, so JSON shows 0.2 rather than 0.20000000298.
fn dec(x: f32) -> f64 {
    x.to_string().parse().unwrap()
}

fn append(path: &Path, row: &serde_json::Value) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    writeln!(f, "{row}").unwrap();
}

fn save(model: &Model, dir: &Path, cfg: &TrainConfig, episodes: u64) {
    model.save(
        dir,
        serde_json::json!({
            "trainer": "ntuple-td",
            "episodes": episodes,
            "seed": cfg.seed,
            "states_per_cell": cfg.states_per_cell,
        }),
    );
}

fn train(cfg: TrainConfig) {
    let out = std::path::PathBuf::from(&cfg.out_dir);
    std::fs::create_dir_all(&out).unwrap();
    let log = out.join("train.jsonl");
    let mut trainer = Trainer::new(OthelloCells, cfg.clone());
    let geom = trainer.model().geometry();
    append(
        &log,
        &serde_json::json!({
            "type": "config", "seed": cfg.seed, "episodes": cfg.episodes,
            "n_tuples": cfg.n_tuples, "tuple_len": cfg.tuple_len,
            "states_per_cell": cfg.states_per_cell, "alpha": dec(cfg.alpha),
            "lambda": dec(cfg.lambda), "gamma": dec(cfg.gamma),
            "epsilon_start": dec(cfg.epsilon_start), "epsilon_end": dec(cfg.epsilon_end),
            "tcl": cfg.tcl, "reset_traces_on_explore": cfg.reset_traces_on_explore,
            "trace_cutoff": dec(cfg.trace_cutoff), "n_weights": geom.n_weights(),
            "n_images": geom.n_images(), "model_toml_sha256": geom.sha256_hex(),
        }),
    );
    eprintln!(
        "td_train: {} episodes, {} tuples of {} cells, {} states/cell, {} weights",
        cfg.episodes,
        cfg.n_tuples,
        cfg.tuple_len,
        cfg.states_per_cell,
        geom.n_weights()
    );

    let started = Instant::now();
    let (mut train_secs, mut interval_secs) = (0.0f64, 0.0f64);
    let (mut td_abs, mut td_sum, mut td_n) = (0.0f64, 0.0f64, 0u64);
    let (mut moves, mut explored) = (0u64, 0u64);
    for ep in 1..=cfg.episodes {
        let t = Instant::now();
        let s = trainer.run_episode();
        let dt = t.elapsed().as_secs_f64();
        train_secs += dt;
        interval_secs += dt;
        td_abs += s.td_abs_sum;
        td_sum += s.td_sum;
        td_n += s.td_count as u64;
        moves += s.moves as u64;
        explored += s.explored as u64;

        if ep % cfg.log_every == 0 || ep == cfg.episodes {
            let n = if ep % cfg.log_every == 0 { cfg.log_every } else { ep % cfg.log_every };
            let model = trainer.model();
            let eval_seed = cfg.seed.wrapping_mul(1_000_003).wrapping_add(ep);
            let vs_random = play_match(&OthelloCells, model, cfg.eval_games, eval_seed, uniform_random);
            let vs_greedy =
                play_match(&OthelloCells, model, cfg.eval_games, eval_seed ^ 0x5eed, greedy_discs);
            append(
                &log,
                &serde_json::json!({
                    "type": "progress", "episodes": ep,
                    "episodes_per_s": n as f64 / interval_secs,
                    "train_secs": train_secs, "wall_secs": started.elapsed().as_secs_f64(),
                    "mean_td_error": td_sum / td_n.max(1) as f64,
                    "mean_abs_td_error": td_abs / td_n.max(1) as f64,
                    "moves_per_episode": moves as f64 / n as f64,
                    "explored_frac": explored as f64 / moves.max(1) as f64,
                    "epsilon": dec(cfg.epsilon_at(ep - 1)),
                    "eval_games": cfg.eval_games,
                    "score_vs_random": vs_random.score(),
                    "score_vs_greedy_discs": vs_greedy.score(),
                }),
            );
            eprintln!(
                "  ep {ep:>7}  {:.1} eps/s  |td| {:.4}  vs random {:.3}  vs greedy {:.3}",
                n as f64 / interval_secs,
                td_abs / td_n.max(1) as f64,
                vs_random.score(),
                vs_greedy.score()
            );
            save(model, &out, &cfg, ep);
            interval_secs = 0.0;
            td_abs = 0.0;
            td_sum = 0.0;
            td_n = 0;
            moves = 0;
            explored = 0;
        }
    }
    eprintln!(
        "td_train: done, {} episodes in {:.0}s of training ({:.1} episodes/s overall)",
        cfg.episodes,
        train_secs,
        cfg.episodes as f64 / train_secs
    );
}

fn gate(cfg: &TrainConfig, gate_path: &str, gate_args: &[String]) {
    let mut gcfg: GateConfig = load_toml_config(gate_path, gate_args);
    let out = Path::new(&cfg.out_dir);
    if gcfg.out.is_none() {
        gcfg.out = Some(out.join("gate.jsonl").to_string_lossy().into_owned());
    }
    let model = Arc::new(Model::load(out, &OthelloCells.orientations()));
    run_edax_gate(&gcfg, &format!("td-seed{}", cfg.seed), Some(1), |_seed| {
        GreedyPlayer::new(OthelloCells, model.clone())
    });
}

fn main() {
    let args = parse_args();
    let cfg: TrainConfig = load_toml_config(&args.config, &args.set);
    if !args.gate_only {
        train(cfg.clone());
    }
    if let Some(gate_path) = &args.gate_config {
        gate(&cfg, gate_path, &args.gate_set);
    }
}
