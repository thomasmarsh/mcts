//! Paired-game measurements on N x N Gonnect (`size` in the config, 5 by default; 5, 7 and 9 are
//! compiled in): the MCTS preset ladder, and (through
//! the same config) any n-tuple agent placed against it.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example gonnect_gate -p game-gonnect -- \
//!     [--config games/gonnect/ntuple/ladder.toml] [--set key=value]... [--pair A:B]...
//! ```
//!
//! Every pairing plays each of `openings` seeded random openings from both seats
//! (`2 * openings` games); games run on `workers` threads. `[[agent]]` tables
//! define the field; `--pair` (repeatable) or the config's `pairs = ["A:B", ...]` pick the
//! pairings, and with neither every unordered pair plays. One JSONL row per
//! game and one per pairing is appended to `out` as the run goes.

use std::sync::{Arc, Mutex, OnceLock};

use game_gonnect::sized::SizedGonnect;
use game_gonnect::td_cells::GonnectCells;
use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use mcts_tune::SearchBudget;
use ntuple::{CellFeatures, GreedyPlayer, Model, PuctConfig, PuctPlayer};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

mod common;
use common::{load_toml_config, open_append, paired_match, Agent, Maker, PairedConfig};

#[derive(Deserialize, Clone, Debug)]
struct AgentSpec {
    name: String,
    kind: String,
    preset: Option<String>,
    iterations: Option<usize>,
    model_dir: Option<String>,
    c_puct: Option<f32>,
    prior_temperature: Option<f32>,
}

fn default_size() -> usize {
    5
}

#[derive(Deserialize, Debug)]
struct Config {
    openings: usize,
    opening_plies: usize,
    max_plies: usize,
    seed: u64,
    workers: usize,
    #[serde(default = "default_size")]
    size: usize,
    out: String,
    presets: String,
    #[serde(default)]
    pairs: Vec<String>,
    agent: Vec<AgentSpec>,
}

struct RandomPlayer<const N: usize> {
    rng: SmallRng,
    name: String,
}

impl<const N: usize> Search for RandomPlayer<N> {
    type G = SizedGonnect<N>;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &<SizedGonnect<N> as Game>::S) -> <SizedGonnect<N> as Game>::A {
        let mut actions = Vec::new();
        SizedGonnect::<N>::generate_actions(state, &mut actions);
        actions[self.rng.gen_range(0..actions.len())]
    }
}

fn maker<const N: usize>(spec: &AgentSpec, presets: &Arc<PresetTable>) -> Maker<SizedGonnect<N>> {
    type G<const N: usize> = SizedGonnect<N>;
    match spec.kind.as_str() {
        "random" => Box::new(|seed| -> Agent<G<N>> {
            Box::new(RandomPlayer::<N> { rng: SmallRng::seed_from_u64(seed), name: "random".into() })
        }),
        "preset" => {
            let presets = presets.clone();
            let id = spec.preset.clone().expect("preset agents need `preset`");
            let iterations = spec.iterations.expect("preset agents need `iterations`");
            Box::new(move |seed| -> Agent<G<N>> {
                presets
                    .build_with::<G<N>>(&id, seed, |b: &mut SearchBudget| {
                        b.threads = 1;
                        b.max_iterations = Some(iterations);
                        b.max_time = None;
                    })
                    .unwrap_or_else(|e| panic!("preset {id:?}: {e}"))
            })
        }
        "ntuple-greedy" | "ntuple-puct" => {
            let dir = spec.model_dir.clone().expect("n-tuple agents need `model_dir`");
            let model: Arc<OnceLock<Arc<Model>>> = Arc::new(OnceLock::new());
            let puct = (spec.kind == "ntuple-puct").then(|| PuctConfig {
                iterations: spec.iterations.expect("ntuple-puct needs `iterations`") as u32,
                c_puct: spec.c_puct.unwrap_or(1.0),
                prior_temperature: spec.prior_temperature.unwrap_or(1.0),
                empties_exact: 0,
            });
            Box::new(move |_seed| -> Agent<G<N>> {
                let m = model
                    .get_or_init(|| {
                        Arc::new(Model::load(
                            std::path::Path::new(&dir),
                            &GonnectCells::<N>.orientations(),
                        ))
                    })
                    .clone();
                match &puct {
                    Some(cfg) => Box::new(PuctPlayer::new(GonnectCells::<N>, m, cfg.clone())),
                    None => Box::new(GreedyPlayer::new(GonnectCells::<N>, m)),
                }
            })
        }
        other => panic!("unknown agent kind {other:?}"),
    }
}

fn main() {
    let (mut config, mut sets, mut pairs) =
        ("games/gonnect/ntuple/ladder.toml".to_string(), Vec::new(), Vec::<(String, String)>::new());
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--config" => config = val(),
            "--set" => sets.push(val()),
            "--pair" => {
                let v = val();
                let (a, b) = v.split_once(':').expect("--pair takes A:B");
                pairs.push((a.to_string(), b.to_string()));
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let cfg: Config = load_toml_config(&config, &sets);
    match cfg.size {
        5 => run::<5>(cfg, pairs),
        7 => run::<7>(cfg, pairs),
        9 => run::<9>(cfg, pairs),
        n => panic!("size {n} is not compiled in (5, 7, 9)"),
    }
}

fn run<const N: usize>(cfg: Config, mut pairs: Vec<(String, String)>) {
    let presets = Arc::new(
        PresetTable::load_from_path(std::path::Path::new(&cfg.presets))
            .unwrap_or_else(|e| panic!("{}: {e}", cfg.presets)),
    );
    let makers: Vec<(String, Maker<SizedGonnect<N>>)> =
        cfg.agent.iter().map(|s| (s.name.clone(), maker::<N>(s, &presets))).collect();
    if pairs.is_empty() {
        for p in &cfg.pairs {
            let (a, b) = p.split_once(':').expect("pairs entries are A:B");
            pairs.push((a.to_string(), b.to_string()));
        }
    }
    if pairs.is_empty() {
        for i in 0..makers.len() {
            for j in i + 1..makers.len() {
                pairs.push((makers[i].0.clone(), makers[j].0.clone()));
            }
        }
    }
    let find = |n: &str| {
        makers.iter().find(|(name, _)| name == n).unwrap_or_else(|| panic!("no agent {n:?}"))
    };
    let paired = PairedConfig {
        openings: cfg.openings,
        opening_plies: cfg.opening_plies,
        max_plies: cfg.max_plies,
        seed: cfg.seed,
        workers: cfg.workers,
    };
    let out = Mutex::new(open_append(&cfg.out));
    for (a, b) in &pairs {
        let (an, am) = find(a);
        let (bn, bm) = find(b);
        let t = std::time::Instant::now();
        let r = paired_match::<SizedGonnect<N>>(an, am, bn, bm, &paired, &out);
        let row = r.summary_json(an, bn);
        {
            use std::io::Write;
            let mut f = out.lock().unwrap();
            writeln!(f, "{row}").unwrap();
            f.flush().unwrap();
        }
        let (score, (lo, hi)) = r.tally.win_rate_ci(1.96);
        println!(
            "{an:>14} vs {bn:<14} {:>3} games  W-L-D {}-{}-{}  score {score:.3} [{lo:.3}, {hi:.3}]  \
             capped {}  {:.1} / {:.1} ms per move  ({:.0}s)",
            r.tally.total(),
            r.tally.wins,
            r.tally.losses,
            r.tally.draws,
            r.capped,
            r.ms_per_move(0),
            r.ms_per_move(1),
            t.elapsed().as_secs_f64()
        );
    }
}
