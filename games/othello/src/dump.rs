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
//! Only `--label outcome` is implemented: seeded self-play, every
//! non-terminal position labelled with the final outcome. The
//! `treestrap` / `root_value` search-labelled harvest modes are not built
//! yet -- the `match` arm for them `unimplemented!()`s with a pointer.
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
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--manifest" => manifest = Some(PathBuf::from(val())),
            "--games" => games = val().parse().expect("--games must be an integer"),
            "--seed" => seed = val().parse().expect("--seed must be an integer"),
            "--label" => label = val(),
            "--engine" => engine = Some(val()),
            "--epsilon" => epsilon = val().parse().expect("--epsilon must be a float"),
            "--presets" => presets_path = PathBuf::from(val()),
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-othello dump --out <path> [--games N] [--seed N] \
                     [--label outcome] [--engine <preset>] [--epsilon P] \
                     [--presets <path>] [--manifest <path>]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    match label.as_str() {
        "outcome" => {}
        "treestrap" | "root_value" => unimplemented!(
            "--label {label} (search-labelled harvest) is not implemented yet; only \
             --label outcome is available"
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
