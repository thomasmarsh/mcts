//! Paired-opening net-vs-net matches for the Druid CNN track (the progress gate, the round robin).
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example druid_gate -p game-druid -- \
//!     --config match.toml [--resume]
//! ```
//!
//! The config has the same shape as `gonnect_gate`'s net-only configs: `size`, `openings`,
//! `opening_plies`, `max_plies`, `seed`, `workers`, `out`, `pairs = ["A:B", ...]` and one
//! `[[agent]]` table per net (`name`, `weights`, `iterations`, and optionally `considered_actions`,
//! `value_scale`, `max_visit_init`, `chunk_size`). Each pairing plays every seeded random opening
//! from both seats (`2 * openings` games) on `workers` threads. One JSONL row per game and one per
//! pairing (same fields as `gonnect_gate`, so `yardstick.pairs_from_rows` reads them unchanged) is
//! appended to `out` as the run goes; `--resume` skips pairings already summarised there.
//!
//! An opening is `opening_plies` uniformly random plies (single sub-decisions, not whole turns) from
//! the start position, so it can end in the middle of a turn; both seats still play it from that
//! exact position. A ply cap ends a game as a draw and counts it as `capped`.

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use game_druid::cnn::agent::{CnnAgent, Kind};
use game_druid::cnn::encode::legal_moves;
use game_druid::{DruidSplit, HashedState, Player, Size};
use grid_cnn::Weights;
use mcts::algorithms::Search;
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct AgentSpec {
    name: String,
    weights: String,
    iterations: usize,
    considered_actions: Option<usize>,
    value_scale: Option<f32>,
    max_visit_init: Option<i32>,
    chunk_size: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct Config {
    size: usize,
    openings: usize,
    opening_plies: usize,
    max_plies: usize,
    seed: u64,
    workers: usize,
    out: String,
    #[serde(default)]
    pairs: Vec<String>,
    agent: Vec<AgentSpec>,
}

type Maker<const N: usize> = Box<dyn Fn() -> CnnAgent<N> + Send + Sync>;

fn maker<const N: usize>(spec: &AgentSpec) -> Maker<N> {
    let weights = Arc::new(Weights::load(&spec.weights).unwrap_or_else(|e| panic!("{}: {e}", spec.weights)));
    assert_eq!(weights.geometry.size, N, "{} is not a {N}x{N} net", spec.weights);
    let cfg = mcts_batch::Config {
        num_simulations: spec.iterations,
        num_considered_actions: spec.considered_actions.unwrap_or(16),
        value_scale: spec.value_scale.unwrap_or(0.1),
        max_visit_init: spec.max_visit_init.unwrap_or(50),
    };
    let (name, chunk) = (spec.name.clone(), spec.chunk_size.unwrap_or(64));
    // The agent is a deterministic function of the position, so there is no per-game seed.
    Box::new(move || CnnAgent::<N>::new(&name, &weights, cfg, Kind::Gumbel, chunk, Duration::default(), 0x51))
}

/// A non-terminal position `plies` uniformly random plies from the start; opening `index` is the
/// same in every run.
fn opening<const N: usize>(seed: u64, index: usize, plies: usize) -> HashedState {
    let mut attempt = 0u64;
    loop {
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(index as u64 * 1_000_003 + attempt),
        );
        let mut s = HashedState::new(Size { w: N as u8, h: N as u8 });
        for _ in 0..plies {
            let (moves, _) = legal_moves(&s);
            s = DruidSplit::apply(s, &moves[rng.gen_range(0..moves.len())]);
            if DruidSplit::is_terminal(&s) {
                break;
            }
        }
        if !DruidSplit::is_terminal(&s) {
            return s;
        }
        attempt += 1;
    }
}

struct Outcome {
    winner: Option<Player>,
    plies: usize,
    capped: bool,
    /// Seconds and moves per seat, indexed by colour (Black = 0).
    secs: [f64; 2],
    moves: [u64; 2],
}

fn play<const N: usize>(
    start: &HashedState,
    black: &mut CnnAgent<N>,
    white: &mut CnnAgent<N>,
    max_plies: usize,
) -> Outcome {
    let mut s = start.clone();
    let (mut secs, mut moves) = ([0.0f64; 2], [0u64; 2]);
    let mut plies = 0;
    while !DruidSplit::is_terminal(&s) {
        if plies >= max_plies {
            return Outcome { winner: None, plies, capped: true, secs, moves };
        }
        let seat = usize::from(DruidSplit::player_to_move(&s) != Player::Black);
        let agent = if seat == 0 { &mut *black } else { &mut *white };
        let t = Instant::now();
        let mv = agent.choose_action(&s);
        secs[seat] += t.elapsed().as_secs_f64();
        moves[seat] += 1;
        s = DruidSplit::apply(s, &mv);
        plies += 1;
    }
    Outcome { winner: DruidSplit::winner(&s), plies, capped: false, secs, moves }
}

fn wilson(wins: f64, games: f64) -> (f64, f64) {
    let z = 1.96f64;
    if games == 0.0 {
        return (0.0, 1.0);
    }
    let p = wins / games;
    let denom = 1.0 + z * z / games;
    let centre = (p + z * z / (2.0 * games)) / denom;
    let half = z * (p * (1.0 - p) / games + z * z / (4.0 * games * games)).sqrt() / denom;
    (centre - half, centre + half)
}

#[derive(Default)]
struct Tally {
    wins: u32,
    losses: u32,
    draws: u32,
    capped: u32,
    secs: [f64; 2],
    moves: [u64; 2],
}

fn paired_match<const N: usize>(
    a: &(String, Maker<N>),
    b: &(String, Maker<N>),
    cfg: &Config,
    out: &Mutex<std::fs::File>,
) -> Tally {
    let next = AtomicUsize::new(0);
    let total = 2 * cfg.openings;
    let tally = Mutex::new(Tally::default());
    std::thread::scope(|scope| {
        for _ in 0..cfg.workers.max(1) {
            scope.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= total {
                    break;
                }
                let (op, a_black) = (g / 2, g.is_multiple_of(2));
                let start = opening::<N>(cfg.seed, op, cfg.opening_plies);
                // `a_black` names who plays the colour that moves first at the opening; the
                // opening's own mover is Black or White depending on its parity, so seat by colour.
                let a_moves_first = a_black;
                let (mut x, mut y) = ((a.1)(), (b.1)());
                let mover = DruidSplit::player_to_move(&start);
                let a_colour = if a_moves_first { mover } else { other(mover) };
                let o = if a_colour == Player::Black {
                    play(&start, &mut x, &mut y, cfg.max_plies)
                } else {
                    play(&start, &mut y, &mut x, cfg.max_plies)
                };
                let (a_seat, b_seat) = if a_colour == Player::Black { (0, 1) } else { (1, 0) };
                let verdict = match o.winner {
                    None => "draw",
                    Some(w) if w == a_colour => "a",
                    Some(_) => "b",
                };
                {
                    let mut f = out.lock().unwrap();
                    writeln!(
                        f,
                        "{}",
                        serde_json::json!({
                            "type": "game", "a": a.0, "b": b.0, "opening": op, "a_first": a_moves_first,
                            "a_black": a_colour == Player::Black, "winner": verdict, "plies": o.plies,
                            "capped": o.capped,
                            "a_secs": o.secs[a_seat], "a_moves": o.moves[a_seat],
                            "b_secs": o.secs[b_seat], "b_moves": o.moves[b_seat],
                        })
                    )
                    .unwrap();
                    f.flush().unwrap();
                }
                let mut t = tally.lock().unwrap();
                match verdict {
                    "a" => t.wins += 1,
                    "b" => t.losses += 1,
                    _ => t.draws += 1,
                }
                t.capped += u32::from(o.capped);
                t.secs[0] += o.secs[a_seat];
                t.moves[0] += o.moves[a_seat];
                t.secs[1] += o.secs[b_seat];
                t.moves[1] += o.moves[b_seat];
            });
        }
    });
    tally.into_inner().unwrap()
}

fn other(p: Player) -> Player {
    if p == Player::Black {
        Player::White
    } else {
        Player::Black
    }
}

fn run<const N: usize>(cfg: &Config, resume: bool) {
    let makers: Vec<(String, Maker<N>)> = cfg.agent.iter().map(|s| (s.name.clone(), maker::<N>(s))).collect();
    let pairs: Vec<(String, String)> = if cfg.pairs.is_empty() {
        (0..makers.len())
            .flat_map(|i| (i + 1..makers.len()).map(move |j| (i, j)))
            .map(|(i, j)| (makers[i].0.clone(), makers[j].0.clone()))
            .collect()
    } else {
        cfg.pairs
            .iter()
            .map(|p| {
                let (a, b) = p.split_once(':').expect("pairs entries are A:B");
                (a.to_string(), b.to_string())
            })
            .collect()
    };
    let done: HashSet<(String, String)> = if resume {
        std::fs::read_to_string(&cfg.out)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|v| v["type"] == "pairing")
            .filter_map(|v| Some((v["a"].as_str()?.to_string(), v["b"].as_str()?.to_string())))
            .collect()
    } else {
        HashSet::new()
    };
    if let Some(dir) = std::path::Path::new(&cfg.out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.out)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", cfg.out));
    let out = Mutex::new(file);
    let find = |n: &str| makers.iter().find(|(name, _)| name == n).unwrap_or_else(|| panic!("no agent {n:?}"));
    for (an, bn) in &pairs {
        if done.contains(&(an.clone(), bn.clone())) {
            println!("{an:>14} vs {bn:<14} already in {}, skipped", cfg.out);
            continue;
        }
        let started = Instant::now();
        let t = paired_match(find(an), find(bn), cfg, &out);
        let games = f64::from(t.wins + t.losses + t.draws);
        let score = (f64::from(t.wins) + 0.5 * f64::from(t.draws)) / games.max(1.0);
        let (lo, hi) = wilson(f64::from(t.wins) + 0.5 * f64::from(t.draws), games);
        let ms = |i: usize| 1e3 * t.secs[i] / t.moves[i].max(1) as f64;
        let row = serde_json::json!({
            "type": "pairing", "a": an, "b": bn, "games": games as u32,
            "a_wins": t.wins, "b_wins": t.losses, "draws": t.draws, "capped": t.capped,
            "score_a": score, "wilson_lo": lo, "wilson_hi": hi,
            "a_ms_per_move": ms(0), "b_ms_per_move": ms(1),
            "seconds": started.elapsed().as_secs_f64(),
        });
        {
            let mut f = out.lock().unwrap();
            writeln!(f, "{row}").unwrap();
            f.flush().unwrap();
        }
        println!(
            "{an:>14} vs {bn:<14} {games:>3} games  W-L-D {}-{}-{}  score {score:.3} [{lo:.3}, {hi:.3}]  \
             capped {}  {:.1} / {:.1} ms per move  ({:.0}s)",
            t.wins,
            t.losses,
            t.draws,
            t.capped,
            ms(0),
            ms(1),
            started.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    let (mut config, mut resume) = (String::new(), false);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--config" => config = it.next().expect("--config needs a value"),
            "--resume" => resume = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let text = std::fs::read_to_string(&config).unwrap_or_else(|e| panic!("cannot read {config}: {e}"));
    let cfg: Config = toml::from_str(&text).unwrap_or_else(|e| panic!("{config}: {e}"));
    game_druid::with_board_size!(
        cfg.size,
        N => run::<N>(&cfg, resume),
        n => panic!("size {n} is not compiled in ({:?})", game_druid::cnn::SUPPORTED_SIZES),
    );
}
