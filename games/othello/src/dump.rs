//! `game-othello dump` -- write self-play positions as fixed-width
//! little-endian records for the offline learned-evaluation pipeline.
//!
//! This is Othello-specific scaffolding, deliberately kept out of the
//! generic `game-host` CLI: `main.rs` intercepts `dump` as the first
//! argument and calls [`run`] before `game_host::run_cli` ever sees the
//! args.
//!
//! ## Record format
//!
//! 22 bytes, packed (no padding), little-endian:
//!
//! | field  | type   | bytes |
//! |--------|--------|-------|
//! | black  | u64 LE | 8     |
//! | white  | u64 LE | 8     |
//! | side   | u8     | 1     |
//! | ply    | u8     | 1     |
//! | target | f32 LE | 4     |
//!
//! `side` is 0 for black to move, 1 for white to move. `ply` is the number
//! of discs on the board minus 4 (0 at the opening, up to 60 at a full
//! board). `target` is the final game result **from the side-to-move
//! player's perspective**: `+1.0` that player won, `-1.0` they lost, `0.0`
//! a draw.
//!
//! ## Label modes
//!
//! - `--label outcome` (default): seeded self-play, every non-terminal
//!   position labelled with the final game outcome.
//! - `--label harvest --out <dir>`: one search-per-move self-play pass
//!   (config from `games/othello/ntuple/harvest.toml`, or
//!   `--harvest-config`) writing four label streams from the *same*
//!   searches -- `arm_a` (played root <- outcome), `arm_b` (played root <-
//!   its searched value), `arm_c` (every internal node <- its searched
//!   value, TreeStrap; see `harvest.rs`), `arm_d` (played root <- the
//!   TD(lambda) return along the game). Each `arm_*.bin` has a matching
//!   `arm_*.json` manifest; `harvest.json` records the counts and the
//!   harvest ratio.
//!
//! ## Position source
//!
//! Without `--engine`, moves are uniform-random. With `--engine <preset>`,
//! moves come from that `games/othello/presets.json` engine, except that
//! with probability `--epsilon` (default 0.1) a uniform-random legal move
//! is played instead -- diversity so a deterministic seeded engine doesn't
//! emit the same game repeatedly. The label is unchanged either way.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{Move, Othello, Player, State};

/// One dumped position. See the module docs for field semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Record {
    pub black: u64,
    pub white: u64,
    pub side: u8,
    pub ply: u8,
    pub target: f32,
}

/// Serialized size of one [`Record`], in bytes.
pub const RECORD_BYTES: usize = 22;

impl Record {
    /// Append this record's 22 little-endian bytes to `buf`. Fields are
    /// written one at a time -- never a `#[repr(C)]` struct cast, which
    /// could introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.black.to_le_bytes());
        buf.extend_from_slice(&self.white.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.target.to_le_bytes());
    }

    /// Decode one record from exactly [`RECORD_BYTES`] bytes.
    pub fn decode(b: &[u8; RECORD_BYTES]) -> Record {
        Record {
            black: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            white: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            side: b[16],
            ply: b[17],
            target: f32::from_le_bytes(b[18..22].try_into().unwrap()),
        }
    }
}

fn side_of(p: Player) -> u8 {
    match p {
        Player::Black => 0,
        Player::White => 1,
    }
}

/// Build a [`Record`] for a mid-game `state` given the final `winner`
/// (`None` == draw). `target` is from `state.turn`'s perspective.
pub fn record_for(state: &State, winner: Option<Player>) -> Record {
    let side = side_of(state.turn);
    let ply = (state.occupied().count_ones() as u8).saturating_sub(4);
    let target = match winner {
        None => 0.0,
        Some(w) if side_of(w) == side => 1.0,
        Some(_) => -1.0,
    };
    Record {
        black: state.black.bits(),
        white: state.white.bits(),
        side,
        ply,
        target,
    }
}

struct Config {
    out: PathBuf,
    manifest: Option<PathBuf>,
    games: u64,
    seed: u64,
    engine: Option<String>,
    epsilon: f64,
    presets_path: PathBuf,
    label: String,
    harvest_config: PathBuf,
    games_overridden: bool,
    seed_overridden: bool,
    /// `--label harvest` target oracle: `mcts` (the searched value read from
    /// the tree) or `edax` (an independent Edax evaluation of each position).
    /// Arm A (outcome) is unaffected either way.
    oracle: String,
    edax_config: PathBuf,
    /// `--edax-level N` overrides the config's `edax_level` (used by the
    /// `edax_level_sweep.sh` diagnostic and the higher-level held-out relabel).
    edax_level_override: Option<u32>,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Config {
    let mut out = None;
    let mut manifest = None;
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut label = "outcome".to_string();
    let mut engine = None;
    let mut epsilon = 0.1f64;
    let mut presets_path = PathBuf::from("games/othello/presets.json");
    let mut harvest_config = PathBuf::from("games/othello/ntuple/harvest.toml");
    let mut oracle = "mcts".to_string();
    let mut edax_config = PathBuf::from("games/othello/ntuple/harvest_edax.toml");
    let mut edax_level_override = None;
    let mut games_overridden = false;
    let mut seed_overridden = false;
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--manifest" => manifest = Some(PathBuf::from(val())),
            "--games" => {
                games = val().parse().expect("--games must be an integer");
                games_overridden = true;
            }
            "--seed" => {
                seed = val().parse().expect("--seed must be an integer");
                seed_overridden = true;
            }
            "--label" => label = val(),
            "--engine" => engine = Some(val()),
            "--epsilon" => epsilon = val().parse().expect("--epsilon must be a float"),
            "--presets" => presets_path = PathBuf::from(val()),
            "--harvest-config" => harvest_config = PathBuf::from(val()),
            "--oracle" => oracle = val(),
            "--edax-config" => edax_config = PathBuf::from(val()),
            "--edax-level" => {
                edax_level_override = Some(val().parse().expect("--edax-level must be an integer"))
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-othello dump --out <path> [--games N] [--seed N] \
                     [--label outcome|harvest] [--engine <preset>] [--epsilon P] \
                     [--presets <path>] [--manifest <path>] [--harvest-config <path>]\n\
                     \n\
                     --label harvest: --out is a directory; writes arm_{{a,b,c,d}}.bin \
                     + manifests + harvest.json"
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    match oracle.as_str() {
        "mcts" | "edax" => {}
        other => panic!("unknown --oracle {other:?} (want mcts | edax)"),
    }
    match label.as_str() {
        "outcome" | "harvest" => {}
        "treestrap" | "root_value" => panic!(
            "--label {label} is superseded by --label harvest, which emits arms a/b/c/d \
             (outcome, root searched value, TreeStrap, TD(lambda)) from one self-play pass"
        ),
        other => panic!("unknown --label mode: {other}"),
    }
    assert!(
        (0.0..=1.0).contains(&epsilon),
        "--epsilon must be in [0, 1], got {epsilon}"
    );
    Config {
        out: out.expect("--out is required"),
        manifest,
        games,
        seed,
        engine,
        epsilon,
        presets_path,
        label,
        harvest_config,
        games_overridden,
        seed_overridden,
        oracle,
        edax_config,
        edax_level_override,
    }
}

/// Backfill the final-outcome target into every record pushed for this
/// game (those from index `first` onward), now that `winner` is known.
fn finish_game(records: &mut [Record], first: usize, winner: Option<Player>) {
    for rec in &mut records[first..] {
        let side_player = if rec.side == 0 {
            Player::Black
        } else {
            Player::White
        };
        rec.target = match winner {
            None => 0.0,
            Some(w) if w == side_player => 1.0,
            Some(_) => -1.0,
        };
    }
}

/// Play one uniform-random game, pushing a [`Record`] for every
/// non-terminal position onto `records` once the outcome is known.
fn dump_one_game(rng: &mut SmallRng, records: &mut Vec<Record>) {
    let mut state = State::default();
    let mut actions = Vec::new();
    let first = records.len();
    while !Othello::is_terminal(&state) {
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        // Placeholder target, backfilled below once the winner is known.
        records.push(record_for(&state, None));
        let action = actions[rng.gen_range(0..actions.len())];
        state = Othello::apply(state, &action);
    }
    finish_game(records, first, Othello::winner(&state));
}

/// Play one game driven by `engine`, except that with probability `epsilon`
/// a uniform-random legal move is substituted. Same labelling as
/// [`dump_one_game`].
fn dump_one_game_engine(
    rng: &mut SmallRng,
    records: &mut Vec<Record>,
    engine: &mut dyn Search<G = Othello>,
    epsilon: f64,
) {
    let mut state = State::default();
    let mut actions = Vec::new();
    let first = records.len();
    while !Othello::is_terminal(&state) {
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        records.push(record_for(&state, None));
        let action = if actions == [Move::PASS] {
            Move::PASS
        } else if rng.gen_bool(epsilon) {
            actions[rng.gen_range(0..actions.len())]
        } else {
            engine.choose_action(&state)
        };
        state = Othello::apply(state, &action);
    }
    finish_game(records, first, Othello::winner(&state));
}

/// Entry point for `game-othello dump ...`. `args` is the argument iterator
/// positioned just past the `dump` token.
pub fn run(args: impl Iterator<Item = String>) {
    let cfg = parse_args(args);
    if cfg.label == "harvest" {
        return run_harvest(&cfg);
    }
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut records = Vec::new();

    let preset_table = cfg.engine.as_ref().map(|_| {
        PresetTable::load_from_path(&cfg.presets_path)
            .unwrap_or_else(|e| panic!("cannot load {}: {e}", cfg.presets_path.display()))
    });

    for g in 0..cfg.games {
        match (&cfg.engine, &preset_table) {
            (Some(preset), Some(table)) => {
                // Rebuild per game with a game-specific seed so a persistent
                // search tree can't carry across games.
                let mut engine = table
                    .build::<Othello>(preset, cfg.seed.wrapping_add(g).wrapping_add(1))
                    .unwrap_or_else(|e| panic!("preset {preset:?} did not resolve: {e}"));
                dump_one_game_engine(&mut rng, &mut records, &mut *engine, cfg.epsilon);
            }
            _ => dump_one_game(&mut rng, &mut records),
        }
        if (g + 1) % 100 == 0 || g + 1 == cfg.games {
            eprintln!(
                "  dumped {}/{} games ({} records)",
                g + 1,
                cfg.games,
                records.len()
            );
        }
    }

    let mut buf = Vec::with_capacity(records.len() * RECORD_BYTES);
    for r in &records {
        r.encode(&mut buf);
    }
    let mut w = BufWriter::new(File::create(&cfg.out).expect("cannot create --out file"));
    w.write_all(&buf).expect("write failed");
    w.flush().expect("flush failed");

    if let Some(path) = &cfg.manifest {
        let mut json = String::from("[\n");
        for (i, r) in records.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"black\": \"{:016x}\", \"white\": \"{:016x}\", \"side\": {}, \"ply\": {}, \"target\": {}}}",
                r.black, r.white, r.side, r.ply, r.target
            ));
            json.push_str(if i + 1 == records.len() { "\n" } else { ",\n" });
        }
        json.push_str("]\n");
        std::fs::write(path, json).expect("cannot write --manifest file");
    }

    eprintln!(
        "wrote {} records ({} bytes) from {} games to {}",
        records.len(),
        buf.len(),
        cfg.games,
        cfg.out.display()
    );
}

// ---------------------------------------------------------------------------
// `--label harvest`: one search-per-move self-play pass, four label streams
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct HarvestParams {
    #[allow(dead_code)]
    engine: String,
    label_iters: usize,
    epsilon: f64,
    opening_plies: usize,
    games: u64,
    seed: u64,
    min_visits: u32,
    max_per_search: usize,
    dedup: bool,
    td_lambda: f64,
}

/// Parameters for the `--oracle edax` target teacher
/// (`games/othello/ntuple/harvest_edax.toml`). The self-play corpus params
/// still come from `harvest.toml`; only the label oracle changes.
#[derive(serde::Deserialize)]
struct EdaxParams {
    edax_level: u32,
    edax_exact_ply: u32,
    target_mode: String,
    squash_t: f32,
    /// Wall-clock ceiling on a single Edax search; a slower one is aborted
    /// and relabelled neutral rather than wedging the run.
    eval_timeout_s: f64,
    edax_binary: String,
    edax_data_dir: String,
}

/// A [`TargetOracle`] that labels each position with an independent Edax
/// evaluation, mapped into `[-1, 1]`. Near the endgame it raises the search
/// to a full solve.
struct EdaxLabel {
    edax: crate::edax::EdaxEval,
    level: u32,
    exact_ply: u32,
    mode: EdaxMode,
    squash_t: f32,
    /// Positions that came back as exact solves (label-quality cross-check).
    exact_hits: u64,
}

#[derive(Clone, Copy)]
enum EdaxMode {
    Sign,
    Squash,
}

impl EdaxLabel {
    fn new(p: &EdaxParams, level_override: Option<u32>) -> Self {
        let mode = match p.target_mode.as_str() {
            "sign" => EdaxMode::Sign,
            "squash" => EdaxMode::Squash,
            other => panic!("target_mode must be \"sign\" or \"squash\", got {other:?}"),
        };
        let level = level_override.unwrap_or(p.edax_level);
        EdaxLabel {
            edax: crate::edax::EdaxEval::spawn(
                &p.edax_binary,
                &p.edax_data_dir,
                level,
                std::time::Duration::from_secs_f64(p.eval_timeout_s),
            ),
            level,
            exact_ply: p.edax_exact_ply,
            mode,
            squash_t: p.squash_t,
            exact_hits: 0,
        }
    }

    fn map_score(&self, score: f32) -> f32 {
        match self.mode {
            EdaxMode::Sign => {
                if score > 0.0 {
                    1.0
                } else if score < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            EdaxMode::Squash => (score / self.squash_t).tanh(),
        }
    }
}

/// Repeatedly applies a forced pass -- a position whose *only* legal action
/// is `Move::PASS` -- until the side to move has a real action or the game
/// is over, folding the perspective flip each pass causes. A pass changes
/// no discs, so `value(pre-pass state) == -value(post-pass state)`.
///
/// Edax auto-plays a lone forced pass on its own when it has *some* other
/// legal continuation to search into, but a `go` on a position whose only
/// legal action is that pass does not reliably return at all (observed as
/// a `go` that never completes, traced from the D1c level-sweep "no score"
/// flood -- a harvested MCTS tree node can be exactly such a position, even
/// though it never arises in ordinary alternating self-play). Passing
/// ourselves first sidesteps asking Edax to search a lone-pass position at
/// all.
fn skip_forced_passes(mut state: State) -> (State, f32) {
    let mut sign = 1.0f32;
    loop {
        if Othello::is_terminal(&state) {
            return (state, sign);
        }
        let mut acts = Vec::new();
        Othello::generate_actions(&state, &mut acts);
        if acts.len() == 1 && acts[0] == Move::PASS {
            state = Othello::apply(state, &Move::PASS);
            sign = -sign;
        } else {
            return (state, sign);
        }
    }
}

impl crate::harvest::TargetOracle for EdaxLabel {
    fn target(&mut self, state: &State) -> f32 {
        let (state, sign) = skip_forced_passes(*state);

        // A harvested tree node can be a genuinely terminal position (no
        // legal move for either side). Handing Edax a `setboard` on an
        // already-over game does not reliably reach its `*** Game Over
        // ***` report either; the final margin is already exact and known
        // without a search, so skip Edax entirely.
        if Othello::is_terminal(&state) {
            self.exact_hits += 1;
            return sign * self.map_score(crate::edax::terminal_disc_diff(&state) as f32);
        }

        let empties = 64 - state.occupied().count_ones();
        let level = if empties <= self.exact_ply {
            60
        } else {
            self.level
        };
        let s = self.edax.eval(&state, level);
        if s.exact {
            self.exact_hits += 1;
        }
        sign * self.map_score(s.score)
    }
}

/// Encode `records` to `<dir>/<name>.bin` and a matching JSON manifest at
/// `<dir>/<name>.json`.
fn write_arm(dir: &std::path::Path, name: &str, records: &[Record]) {
    let mut buf = Vec::with_capacity(records.len() * RECORD_BYTES);
    for r in records {
        r.encode(&mut buf);
    }
    std::fs::write(dir.join(format!("{name}.bin")), &buf).expect("cannot write arm .bin");

    let mut json = String::from("[\n");
    for (i, r) in records.iter().enumerate() {
        json.push_str(&format!(
            "  {{\"black\": \"{:016x}\", \"white\": \"{:016x}\", \"side\": {}, \"ply\": {}, \"target\": {}}}",
            r.black, r.white, r.side, r.ply, r.target
        ));
        json.push_str(if i + 1 == records.len() { "\n" } else { ",\n" });
    }
    json.push_str("]\n");
    std::fs::write(dir.join(format!("{name}.json")), json).expect("cannot write arm .json");
}

/// Play `opening_plies` uniform-random real (non-pass) plies from the start,
/// retrying if a line ends early.
fn random_opening(rng: &mut SmallRng, opening_plies: usize) -> State {
    'outer: loop {
        let mut state = State::default();
        let mut actions = Vec::new();
        for _ in 0..opening_plies {
            if Othello::is_terminal(&state) {
                continue 'outer;
            }
            actions.clear();
            Othello::generate_actions(&state, &mut actions);
            if actions == [Move::PASS] {
                continue 'outer;
            }
            let a = actions[rng.gen_range(0..actions.len())];
            state = Othello::apply(state, &a);
        }
        return state;
    }
}

fn run_harvest(cfg: &Config) {
    use crate::harvest::{
        harvest_tree_scored, harvest_tree_scored_oracle, label_search, root_value, HarvestFilter,
        TargetOracle,
    };
    use mcts::algorithms::Search;
    use mcts::game::PlayerIndex;
    use std::time::{Duration, Instant};

    let use_edax = cfg.oracle == "edax";
    let mut edax: Option<EdaxLabel> = if use_edax {
        let etext = std::fs::read_to_string(&cfg.edax_config)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.edax_config.display()));
        let ep: EdaxParams = toml::from_str(&etext).expect("edax config must parse");
        eprintln!(
            "oracle=edax  level={}  exact_ply={}  mode={}",
            cfg.edax_level_override.unwrap_or(ep.edax_level),
            ep.edax_exact_ply,
            ep.target_mode
        );
        Some(EdaxLabel::new(&ep, cfg.edax_level_override))
    } else {
        None
    };

    // CPU accounting: the self-play search is the controlled variable and is
    // paid by every arm; the Edax label cost is broken out per arm (root
    // evals feed B and D, the per-node harvest is arm C's alone).
    let mut t_selfplay = Duration::ZERO;
    let mut t_edax_root = Duration::ZERO;
    let mut t_edax_harvest = Duration::ZERO;

    let text = std::fs::read_to_string(&cfg.harvest_config)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.harvest_config.display()));
    let mut params: HarvestParams = toml::from_str(&text).expect("harvest config must parse");
    if cfg.games_overridden {
        params.games = cfg.games;
    }
    if cfg.seed_overridden {
        params.seed = cfg.seed;
    }
    assert!(
        (0.0..=1.0).contains(&params.epsilon),
        "harvest epsilon must be in [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&params.td_lambda),
        "td_lambda must be in [0, 1]"
    );

    let dir = &cfg.out;
    std::fs::create_dir_all(dir).expect("cannot create --out directory");

    let filter = HarvestFilter {
        min_visits: params.min_visits,
        max_per_search: params.max_per_search,
        max_depth: None,
    };

    let mut arm_a: Vec<Record> = Vec::new();
    let mut arm_b: Vec<Record> = Vec::new();
    let mut arm_c: Vec<(u32, Record)> = Vec::new();
    let mut arm_d: Vec<Record> = Vec::new();

    for g in 0..params.games {
        let mut walk_rng = SmallRng::seed_from_u64(params.seed.wrapping_add(g).wrapping_add(1));
        let mut engine = label_search(
            params.label_iters,
            params.seed.wrapping_add(g).wrapping_add(1),
        );
        let mut state = random_opening(&mut walk_rng, params.opening_plies);
        let a_first = arm_a.len();

        // Per label-search ply: (arm_a record index, side, root value in the
        // fixed Black-to-move perspective).
        let mut traj: Vec<(usize, u8, f64)> = Vec::new();
        let mut actions = Vec::new();

        while !Othello::is_terminal(&state) {
            actions.clear();
            Othello::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }
            if actions == [Move::PASS] {
                state = Othello::apply(state, &Move::PASS);
                continue;
            }
            if walk_rng.gen_bool(params.epsilon) {
                let a = actions[walk_rng.gen_range(0..actions.len())];
                state = Othello::apply(state, &a);
                continue;
            }

            let t0 = Instant::now();
            let action = engine.choose_action(&state);
            t_selfplay += t0.elapsed();

            let pidx = state.turn.to_index();
            let rv: f64 = if let Some(o) = edax.as_mut() {
                let t = Instant::now();
                let v = o.target(&state) as f64;
                t_edax_root += t.elapsed();
                v
            } else {
                root_value(&engine, &state, pidx)
            };
            let rv_black = if state.turn == Player::Black { rv } else { -rv };

            let mut rec = record_for(&state, None);
            arm_a.push(rec); // outcome target backfilled below
            rec.target = rv as f32;
            arm_b.push(rec);
            if let Some(o) = edax.as_mut() {
                let t = Instant::now();
                arm_c.extend(harvest_tree_scored_oracle(&engine, &state, &filter, o));
                t_edax_harvest += t.elapsed();
            } else {
                arm_c.extend(harvest_tree_scored(&engine, &state, &filter));
            }
            traj.push((arm_a.len() - 1, rec.side, rv_black));

            state = Othello::apply(state, &action);
        }

        let winner = Othello::winner(&state);
        finish_game(&mut arm_a, a_first, winner);
        let z_black = match winner {
            None => 0.0,
            Some(Player::Black) => 1.0,
            Some(Player::White) => -1.0,
        };

        // Forward-view TD(lambda) return, computed backward along the played
        // line: G_t = (1-lambda) V(s_{t+1}) + lambda G_{t+1}, with the value
        // past the last recorded ply pinned to the terminal outcome.
        let mut g_next = z_black;
        for t in (0..traj.len()).rev() {
            let (a_idx, side, _) = traj[t];
            let v_next = if t + 1 == traj.len() {
                z_black
            } else {
                traj[t + 1].2
            };
            let g = (1.0 - params.td_lambda) * v_next + params.td_lambda * g_next;
            let mut rec = arm_a[a_idx];
            rec.target = (if side == 0 { g } else { -g }) as f32;
            arm_d.push(rec);
            g_next = g;
        }

        if (g + 1) % 100 == 0 || g + 1 == params.games {
            eprintln!(
                "  harvest {}/{} games  arm_a={} arm_c={} (pre-dedup)",
                g + 1,
                params.games,
                arm_a.len(),
                arm_c.len()
            );
        }
    }

    // arm C dedup: collapse (black, white, side) across the whole file,
    // keeping the highest-visit target.
    let arm_c_raw_count = arm_c.len();
    let harvest_ratio_raw = arm_c_raw_count as f64 / arm_a.len().max(1) as f64;
    if params.dedup {
        use std::collections::HashMap;
        let mut best: HashMap<(u64, u64, u8), (u32, f32)> = HashMap::new();
        for (v, r) in arm_c.drain(..) {
            let key = (r.black, r.white, r.side);
            let e = best.entry(key).or_insert((0, 0.0));
            if v >= e.0 {
                *e = (v, r.target);
            }
        }
        arm_c = best
            .into_iter()
            .map(|((black, white, side), (v, target))| {
                let ply = ((black | white).count_ones() as u8).saturating_sub(4);
                (
                    v,
                    Record {
                        black,
                        white,
                        side,
                        ply,
                        target,
                    },
                )
            })
            .collect();
    }
    let arm_c_records: Vec<Record> = arm_c.iter().map(|(_, r)| *r).collect();

    write_arm(dir, "arm_a", &arm_a);
    write_arm(dir, "arm_b", &arm_b);
    write_arm(dir, "arm_c", &arm_c_records);
    write_arm(dir, "arm_d", &arm_d);

    // CPU accounting. For `--oracle mcts` the arms differ only in training
    // cost (bakeoff.sh times that); for `--oracle edax` the plot needs the
    // Edax label bill broken out -- root evals feed arms B/D, the per-node
    // harvest is arm C's alone, and arm A pays no Edax at all.
    let cpu_block = if use_edax {
        let o = edax.as_ref().unwrap();
        format!(
            ",\n  \"oracle\": \"edax\",\n  \"cpu\": {{\n    \
             \"selfplay_s\": {:.2},\n    \"edax_root_s\": {:.2},\n    \
             \"edax_harvest_s\": {:.2},\n    \"edax_calls\": {},\n    \
             \"edax_nodes\": {},\n    \"edax_exact_hits\": {},\n    \"edax_timeouts\": {}\n  }}",
            t_selfplay.as_secs_f64(),
            t_edax_root.as_secs_f64(),
            t_edax_harvest.as_secs_f64(),
            o.edax.calls(),
            o.edax.total_nodes(),
            o.exact_hits,
            o.edax.timeouts(),
        )
    } else {
        format!(
            ",\n  \"oracle\": \"mcts\",\n  \"selfplay_s\": {:.2}",
            t_selfplay.as_secs_f64()
        )
    };
    let summary = format!(
        "{{\n  \"games\": {},\n  \"label_iters\": {},\n  \"epsilon\": {},\n  \"td_lambda\": {},\n  \
         \"min_visits\": {},\n  \"max_per_search\": {},\n  \"dedup\": {},\n  \
         \"arm_a\": {},\n  \"arm_b\": {},\n  \"arm_c\": {},\n  \"arm_c_raw\": {},\n  \"arm_d\": {},\n  \
         \"harvest_ratio\": {:.2},\n  \"harvest_ratio_raw\": {:.2}{}\n}}\n",
        params.games,
        params.label_iters,
        params.epsilon,
        params.td_lambda,
        params.min_visits,
        params.max_per_search,
        params.dedup,
        arm_a.len(),
        arm_b.len(),
        arm_c_records.len(),
        arm_c_raw_count,
        arm_d.len(),
        arm_c_records.len() as f64 / arm_a.len().max(1) as f64,
        harvest_ratio_raw,
        cpu_block,
    );
    std::fs::write(dir.join("harvest.json"), &summary).expect("cannot write harvest.json");

    eprintln!(
        "wrote arm_a={} arm_b={} arm_c={} arm_d={} to {}  (harvest ratio {:.2}x)",
        arm_a.len(),
        arm_b.len(),
        arm_c_records.len(),
        arm_d.len(),
        dir.display(),
        arm_c_records.len() as f64 / arm_a.len().max(1) as f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_states() -> Vec<State> {
        let d3 = Othello::apply(State::default(), &Move(19));
        let d3c5 = Othello::apply(d3, &Move(34));
        let hand = State {
            black: crate::BB::from_bits((1 << 0) | (1 << 63)),
            white: crate::BB::from_bits((1 << 7) | (1 << 56)),
            turn: Player::White,
            last_pass: true,
            ..State::default()
        };
        vec![State::default(), d3, d3c5, hand]
    }

    #[test]
    fn records_round_trip_through_encode_decode() {
        for (i, st) in sample_states().into_iter().enumerate() {
            for winner in [None, Some(Player::Black), Some(Player::White)] {
                let rec = record_for(&st, winner);
                let mut buf = Vec::new();
                rec.encode(&mut buf);
                assert_eq!(buf.len(), RECORD_BYTES, "state {i}");
                let back = Record::decode(&buf.as_slice().try_into().unwrap());
                assert_eq!(back, rec, "state {i} winner {winner:?}");
            }
        }
    }

    #[test]
    fn target_is_from_side_to_move_perspective() {
        // Opening: black to move. Black wins -> +1; white wins -> -1.
        let s = State::default();
        assert_eq!(record_for(&s, Some(Player::Black)).target, 1.0);
        assert_eq!(record_for(&s, Some(Player::White)).target, -1.0);
        assert_eq!(record_for(&s, None).target, 0.0);
        assert_eq!(record_for(&s, None).side, 0);
        assert_eq!(record_for(&s, None).ply, 0);
    }

    #[test]
    fn a_dumped_game_labels_every_position_consistently() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut recs = Vec::new();
        dump_one_game(&mut rng, &mut recs);
        assert!(!recs.is_empty());
        // Every target is one of the three legal values, and ply is
        // non-decreasing across the game.
        let mut last_ply = 0u8;
        for r in &recs {
            assert!([1.0f32, -1.0, 0.0].contains(&r.target));
            assert!(r.ply >= last_ply);
            last_ply = r.ply;
        }
    }

    /// Decode an Edax `setboard` string (64 square chars + side-to-move
    /// token) into a `State`. `hashes` are left at the default zeroed
    /// value -- fine for tests that don't touch Zobrist lookups.
    fn state_from_edax_board(board: &str) -> State {
        let (squares, turn_tok) = board.split_once(' ').unwrap();
        let bytes = squares.as_bytes();
        assert_eq!(bytes.len(), 64);
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, &c) in bytes.iter().enumerate() {
            match c {
                b'X' => black |= 1 << i,
                b'O' => white |= 1 << i,
                b'-' => {}
                other => panic!("unexpected board char {other}"),
            }
        }
        State {
            black: crate::BB::from_bits(black),
            white: crate::BB::from_bits(white),
            turn: if turn_tok == "X" {
                Player::Black
            } else {
                Player::White
            },
            ..State::default()
        }
    }

    /// The exact board from the Phase 2.5 D1c level-sweep run that wedged
    /// `EdaxEval::eval` for a full 60s (two 30s timeouts, then a "no
    /// score" 0.0 fallback): White to move has no real action, only
    /// `Move::PASS`, and Edax's `go` never returns for a lone-pass
    /// position. `skip_forced_passes` must resolve it locally instead of
    /// ever handing it to Edax.
    #[test]
    fn skip_forced_passes_resolves_a_lone_pass_without_asking_edax() {
        let white_to_move_only_pass = state_from_edax_board(
            "OOOOOOOOXOXXXOOOXXOXXOOOXOXXOXOOXOOXXOOOXOOOOXOXXOOOOOXXXOOXO-XX O",
        );
        let mut acts = Vec::new();
        Othello::generate_actions(&white_to_move_only_pass, &mut acts);
        assert_eq!(acts, vec![Move::PASS], "fixture must be a lone-pass position");

        let (resolved, sign) = skip_forced_passes(white_to_move_only_pass);
        assert_eq!(sign, -1.0, "one pass folds one perspective flip");
        assert_eq!(resolved.turn, Player::Black);
        let mut resolved_acts = Vec::new();
        Othello::generate_actions(&resolved, &mut resolved_acts);
        assert_ne!(
            resolved_acts,
            vec![Move::PASS],
            "must stop once the side to move has a real action"
        );
        // Passing changes no discs.
        assert_eq!(resolved.black, white_to_move_only_pass.black);
        assert_eq!(resolved.white, white_to_move_only_pass.white);
    }

    #[test]
    fn skip_forced_passes_is_a_no_op_when_a_real_move_exists() {
        let (resolved, sign) = skip_forced_passes(State::default());
        assert_eq!(sign, 1.0);
        assert_eq!(resolved, State::default());
    }

    #[test]
    fn skip_forced_passes_stops_immediately_at_an_already_terminal_state() {
        let full_board = State {
            black: crate::BB::from_bits(u64::MAX),
            white: crate::BB::from_bits(0),
            turn: Player::Black,
            ..State::default()
        };
        let (resolved, sign) = skip_forced_passes(full_board);
        assert_eq!(sign, 1.0);
        assert!(Othello::is_terminal(&resolved));
    }
}
