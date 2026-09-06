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
            eprintln!("  dumped {}/{} games ({} records)", g + 1, cfg.games, records.len());
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
    use crate::harvest::{harvest_tree_scored, label_search, root_value, HarvestFilter};
    use mcts::algorithms::Search;
    use mcts::game::PlayerIndex;

    let text = std::fs::read_to_string(&cfg.harvest_config)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.harvest_config.display()));
    let mut params: HarvestParams =
        toml::from_str(&text).expect("harvest config must parse");
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

            let action = engine.choose_action(&state);
            let pidx = state.turn.to_index();
            let rv = root_value(&engine, &state, pidx);
            let rv_black = if state.turn == Player::Black { rv } else { -rv };

            let mut rec = record_for(&state, None);
            arm_a.push(rec); // outcome target backfilled below
            rec.target = rv as f32;
            arm_b.push(rec);
            arm_c.extend(harvest_tree_scored(&engine, &state, &filter));
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

    let summary = format!(
        "{{\n  \"games\": {},\n  \"label_iters\": {},\n  \"epsilon\": {},\n  \"td_lambda\": {},\n  \
         \"min_visits\": {},\n  \"max_per_search\": {},\n  \"dedup\": {},\n  \
         \"arm_a\": {},\n  \"arm_b\": {},\n  \"arm_c\": {},\n  \"arm_c_raw\": {},\n  \"arm_d\": {},\n  \
         \"harvest_ratio\": {:.2},\n  \"harvest_ratio_raw\": {:.2}\n}}\n",
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
    );
    std::fs::write(dir.join("harvest.json"), &summary).expect("cannot write harvest.json");

    eprintln!(
        "wrote arm_a={} arm_b={} arm_c={} arm_d={} to {}  (harvest ratio {:.1}x)",
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
}
