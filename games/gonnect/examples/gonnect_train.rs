//! Train a 5x5 Gonnect n-tuple value network by self-play TD(lambda) with
//! temporal coherence learning (no search), logging progress as JSONL.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example gonnect_train -p game-gonnect -- \
//!     [--config games/gonnect/ntuple/td-5x5.toml] [--set key=value]...
//! ```
//!
//! Training writes `train.jsonl` (one row per `log_every` episodes, appended as it
//! goes), a checkpoint (`model.toml`, `weights.bin`, `weights.meta.json`) at every
//! row, and the final weights. Each row carries the raw greedy agent's score vs
//! uniform random and vs a "tactical" opponent (takes an immediately winning move
//! if there is one, otherwise uniform random). Single-threaded; run several seeds
//! as separate processes. The gate is `gonnect_gate`.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use game_gonnect::sized::{SizedGonnect, SizedState};
use game_gonnect::td_cells::GonnectCells;
use game_gonnect::Move;
use mcts::game::{Game, PlayerIndex};
use ntuple::{play_match, uniform_random, TrainConfig, Trainer};
use rand::rngs::SmallRng;
use rand::Rng;

mod common;
use common::load_toml_config;

type G = SizedGonnect<5>;

/// Plays an immediately winning move when one exists, otherwise a uniformly random one.
fn tactical(state: &SizedState<5>, actions: &[Move], rng: &mut SmallRng) -> usize {
    let me = G::player_to_move(state).to_index();
    let wins: Vec<usize> = (0..actions.len())
        .filter(|&i| {
            let next = G::apply(state.clone(), &actions[i]);
            G::is_terminal(&next) && G::winner(&next).map(|p| p.to_index()) == Some(me)
        })
        .collect();
    if wins.is_empty() {
        rng.gen_range(0..actions.len())
    } else {
        wins[rng.gen_range(0..wins.len())]
    }
}

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

fn main() {
    let (mut config, mut sets) = ("games/gonnect/ntuple/td-5x5.toml".to_string(), Vec::new());
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => config = val(),
            "--set" => sets.push(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    let cfg: TrainConfig = load_toml_config(&config, &sets);
    let out = std::path::PathBuf::from(&cfg.out_dir);
    std::fs::create_dir_all(&out).unwrap();
    let log = out.join("train.jsonl");
    let feats = GonnectCells::<5>;
    let mut trainer = Trainer::new(feats, cfg.clone());
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
        "gonnect_train: {} episodes, {} tuples of {} cells, {} states/cell, {} weights",
        cfg.episodes, cfg.n_tuples, cfg.tuple_len, cfg.states_per_cell, geom.n_weights()
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
            let vs_random = play_match(&feats, model, cfg.eval_games, eval_seed, uniform_random);
            let vs_tactical = play_match(&feats, model, cfg.eval_games, eval_seed ^ 0x5eed, tactical);
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
                    "score_vs_tactical": vs_tactical.score(),
                }),
            );
            eprintln!(
                "  ep {ep:>7}  {:.1} eps/s  {:.1} moves/ep  |td| {:.4}  vs random {:.3}  vs tactical {:.3}",
                n as f64 / interval_secs,
                moves as f64 / n as f64,
                td_abs / td_n.max(1) as f64,
                vs_random.score(),
                vs_tactical.score()
            );
            model.save(
                &out,
                serde_json::json!({
                    "trainer": "ntuple-td", "game": "gonnect-5x5", "episodes": ep,
                    "seed": cfg.seed, "states_per_cell": cfg.states_per_cell,
                }),
            );
            interval_secs = 0.0;
            td_abs = 0.0;
            td_sum = 0.0;
            td_n = 0;
            moves = 0;
            explored = 0;
        }
    }
    eprintln!(
        "gonnect_train: done, {} episodes in {:.0}s of training ({:.1} episodes/s overall)",
        cfg.episodes,
        train_secs,
        cfg.episodes as f64 / train_secs
    );
}
