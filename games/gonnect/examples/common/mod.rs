//! Paired-match scaffolding, generic over any two-player `Game`: seeded random
//! openings, both seats of every opening, a ply cap (a deterministic agent can
//! repeat a position forever under Gonnect's positional-ko rule), Wilson
//! intervals and per-move timing, one JSONL row per game written as it finishes.

#![allow(dead_code)] // each example uses a subset.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts_bench::tournament::Result as Tally;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub type Agent<G> = Box<dyn Search<G = G>>;

/// Builds a fresh agent for one game from a seed (so no search state, tree or
/// random stream ever crosses games).
pub type Maker<G> = Box<dyn Fn(u64) -> Agent<G> + Send + Sync>;

pub struct Outcome {
    /// 0 if the agent that moved first at the opening won, 1 if the other did,
    /// `None` for a draw (only possible through the ply cap).
    pub winner: Option<usize>,
    pub plies: usize,
    pub capped: bool,
    pub secs: [f64; 2],
    pub moves: [u64; 2],
    /// Notation of every move the agents chose, in order (the opening plies are not included).
    pub log: Vec<String>,
}

/// Play one game from `start`; `agents[0]` controls the side to move at `start`.
pub fn play_game<G: Game>(start: G::S, agents: &mut [Agent<G>; 2], max_plies: usize) -> Outcome {
    let first = G::player_to_move(&start).to_index();
    let mut state = start;
    let (mut secs, mut moves) = ([0.0f64; 2], [0u64; 2]);
    let mut log = Vec::new();
    let mut plies = 0;
    while !G::is_terminal(&state) {
        if plies >= max_plies {
            return Outcome { winner: None, plies, capped: true, secs, moves, log };
        }
        let seat = usize::from(G::player_to_move(&state).to_index() != first);
        let t = Instant::now();
        let a = agents[seat].choose_action(&state);
        secs[seat] += t.elapsed().as_secs_f64();
        moves[seat] += 1;
        log.push(G::notation(&state, &a));
        state = G::apply(state, &a);
        plies += 1;
    }
    let winner = G::winner(&state).map(|p| usize::from(p.to_index() != first));
    Outcome { winner, plies, capped: false, secs, moves, log }
}

/// A non-terminal position reached by `plies` uniformly random legal actions
/// from the game's default state; opening `index` is the same in every run.
pub fn opening<G: Game>(seed: u64, index: usize, plies: usize) -> G::S {
    let mut attempt = 0u64;
    loop {
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(index as u64 * 1_000_003 + attempt),
        );
        let mut s = G::S::default();
        let mut actions = Vec::new();
        for _ in 0..plies {
            actions.clear();
            G::generate_actions(&s, &mut actions);
            let a = actions[rng.gen_range(0..actions.len())].clone();
            s = G::apply(s, &a);
            if G::is_terminal(&s) {
                break;
            }
        }
        if !G::is_terminal(&s) {
            return s;
        }
        attempt += 1;
    }
}

#[derive(Clone, Debug)]
pub struct PairedConfig {
    pub openings: usize,
    pub opening_plies: usize,
    pub max_plies: usize,
    pub seed: u64,
    pub workers: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PairedResult {
    pub tally: Tally,
    pub capped: usize,
    pub secs: [f64; 2],
    pub moves: [u64; 2],
}

impl PairedResult {
    pub fn ms_per_move(&self, side: usize) -> f64 {
        1e3 * self.secs[side] / self.moves[side].max(1) as f64
    }

    pub fn summary_json(&self, a: &str, b: &str) -> serde_json::Value {
        let (score, (lo, hi)) = self.tally.win_rate_ci(1.96);
        serde_json::json!({
            "type": "pairing", "a": a, "b": b, "games": self.tally.total(),
            "a_wins": self.tally.wins, "b_wins": self.tally.losses, "draws": self.tally.draws,
            "capped": self.capped, "score_a": score, "wilson_lo": lo, "wilson_hi": hi,
            "a_ms_per_move": self.ms_per_move(0), "b_ms_per_move": self.ms_per_move(1),
        })
    }
}

/// Every opening played from both seats: `2 * cfg.openings` games of `a` against `b`, score
/// from `a`'s side. Games run on `cfg.workers` threads; each finished game is appended to
/// `out` immediately.
pub fn paired_match<G: Game>(
    a_name: &str,
    make_a: &Maker<G>,
    b_name: &str,
    make_b: &Maker<G>,
    cfg: &PairedConfig,
    out: &Mutex<std::fs::File>,
) -> PairedResult {
    let next = AtomicUsize::new(0);
    let total = 2 * cfg.openings;
    let result = Mutex::new(PairedResult::default());
    let pair_seed = hash_name(a_name) ^ hash_name(b_name).rotate_left(17);
    std::thread::scope(|scope| {
        for _ in 0..cfg.workers.max(1) {
            scope.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= total {
                    break;
                }
                let (op, a_first) = (g / 2, g.is_multiple_of(2));
                let start = opening::<G>(cfg.seed, op, cfg.opening_plies);
                let seed = cfg.seed ^ pair_seed ^ (g as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let (ma, mb) = (make_a(seed), make_b(seed ^ 0x5bd1_e995));
                // agents[0] plays the side to move at the opening, so a_first decides who that is.
                let mut agents = if a_first { [ma, mb] } else { [mb, ma] };
                let o = play_game::<G>(start, &mut agents, cfg.max_plies);
                let a_seat = usize::from(!a_first);
                let b_seat = 1 - a_seat;
                let verdict = match o.winner {
                    None => "draw",
                    Some(w) if w == a_seat => "a",
                    Some(_) => "b",
                };
                {
                    let mut f = out.lock().unwrap();
                    writeln!(
                        f,
                        "{}",
                        serde_json::json!({
                            "type": "game", "a": a_name, "b": b_name, "opening": op,
                            "a_first": a_first, "winner": verdict, "plies": o.plies,
                            "capped": o.capped, "log": o.log.join(" "),
                            "a_secs": o.secs[a_seat], "a_moves": o.moves[a_seat],
                            "b_secs": o.secs[b_seat], "b_moves": o.moves[b_seat],
                        })
                    )
                    .unwrap();
                    f.flush().unwrap();
                }
                let mut r = result.lock().unwrap();
                match verdict {
                    "a" => r.tally.wins += 1,
                    "b" => r.tally.losses += 1,
                    _ => r.tally.draws += 1,
                }
                r.capped += usize::from(o.capped);
                r.secs[0] += o.secs[a_seat];
                r.moves[0] += o.moves[a_seat];
                r.secs[1] += o.secs[b_seat];
                r.moves[1] += o.moves[b_seat];
            });
        }
    });
    result.into_inner().unwrap()
}

fn hash_name(s: &str) -> u64 {
    s.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// Read a TOML file into `T`, first applying `--set key=value` overrides (top-level keys; the
/// value is parsed as a TOML value, a bare word falls back to a string).
pub fn load_toml_config<T: serde::de::DeserializeOwned>(path: &str, sets: &[String]) -> T {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut table: toml::Table = text.parse().unwrap_or_else(|e| panic!("{path}: {e}"));
    for set in sets {
        let (k, v) = set.split_once('=').unwrap_or_else(|| panic!("--set needs key=value: {set}"));
        let value = format!("v = {v}")
            .parse::<toml::Table>()
            .ok()
            .and_then(|mut t| t.remove("v"))
            .unwrap_or_else(|| toml::Value::String(v.to_string()));
        table.insert(k.trim().to_string(), value);
    }
    table.try_into().unwrap_or_else(|e| panic!("{path}: {e}"))
}

pub fn open_append(path: &str) -> std::fs::File {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"))
}
