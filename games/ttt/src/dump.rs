//! `game-ttt dump` -- write self-play positions as little-endian records for
//! an offline self-play training loop (Gumbel-style: a value/policy target
//! per position, refit between generations).
//!
//! This is kept out of the generic `game-host` CLI -- `main.rs` intercepts
//! `dump` as the first argument and calls [`run`] before `game_host::run_cli`
//! sees the args, the same split `games/othello/src/dump.rs` uses.
//!
//! ## Record format v2
//!
//! A fixed 11-byte head followed by a variable-length policy tail, packed
//! (no padding), little-endian:
//!
//! | field    | type          | bytes        |
//! |----------|---------------|--------------|
//! | board    | u32 LE        | 4            |
//! | side     | u8            | 1            |
//! | ply      | u8            | 1            |
//! | value    | f32 LE        | 4            |
//! | n_policy | u8            | 1            |
//! | policy   | n * (u8, f32) | n_policy * 5 |
//!
//! `board` is `game_ttt::Position`'s packed-`u32` encoding (2 bits per cell,
//! digit 1 = X, 2 = O). `side` is 0 for X to move, 1 for O. `ply` is the
//! number of pieces on the board. `value` is the final game result **from
//! the side-to-move player's perspective**: `+1.0` win, `-1.0` loss, `0.0`
//! draw. The policy tail is the improved-policy training target -- pairs of
//! `(cell_index, probability)` from the Gumbel Sequential-Halving visit
//! distribution at the root. It is empty for `--label outcome` dumps, which
//! record positions from uniform-random or preset-engine self-play with no
//! search-derived policy.
//!
//! v2 records are variable-width, so a reader must walk them sequentially --
//! `np.fromfile` with a fixed dtype does not apply (see
//! `research/az-train/`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{HashedPosition, Piece, Position, TicTacToe};

/// One dumped position. See the module docs for field semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub board: u32,
    pub side: u8,
    pub ply: u8,
    pub value: f32,
    /// `(cell_index, probability)` pairs -- the improved-policy target.
    /// Empty when no search produced a policy for this position.
    pub policy: Vec<(u8, f32)>,
}

/// Size of a [`Record`]'s fixed head, in bytes (everything up to and
/// including the `n_policy` count byte, before the policy tail).
pub const RECORD_HEAD_BYTES: usize = 11;

/// Size of one policy tail entry, in bytes: `(action: u8, prob: f32 LE)`.
pub const POLICY_ENTRY_BYTES: usize = 5;

impl Record {
    /// Append this record's little-endian bytes to `buf`. Fields are written
    /// one at a time -- never a `#[repr(C)]` struct cast, which could
    /// introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.board.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.value.to_le_bytes());
        let n: u8 = self
            .policy
            .len()
            .try_into()
            .expect("policy tail longer than 255 entries");
        buf.push(n);
        for (action, prob) in &self.policy {
            buf.push(*action);
            buf.extend_from_slice(&prob.to_le_bytes());
        }
    }

    /// Decode one record from the front of `bytes`, returning it and the
    /// number of bytes it consumed. Returns `None` if `bytes` is too short
    /// to hold a complete record.
    pub fn decode(bytes: &[u8]) -> Option<(Record, usize)> {
        if bytes.len() < RECORD_HEAD_BYTES {
            return None;
        }
        let board = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let side = bytes[4];
        let ply = bytes[5];
        let value = f32::from_le_bytes(bytes[6..10].try_into().unwrap());
        let n_policy = bytes[10] as usize;
        let tail_bytes = n_policy * POLICY_ENTRY_BYTES;
        let total = RECORD_HEAD_BYTES + tail_bytes;
        if bytes.len() < total {
            return None;
        }
        let mut policy = Vec::with_capacity(n_policy);
        for i in 0..n_policy {
            let off = RECORD_HEAD_BYTES + i * POLICY_ENTRY_BYTES;
            let action = bytes[off];
            let prob = f32::from_le_bytes(bytes[off + 1..off + 5].try_into().unwrap());
            policy.push((action, prob));
        }
        Some((
            Record {
                board,
                side,
                ply,
                value,
                policy,
            },
            total,
        ))
    }
}

fn side_of(p: Piece) -> u8 {
    match p {
        Piece::X => 0,
        Piece::O => 1,
    }
}

fn piece_count(board: u32) -> u8 {
    (0..9).filter(|&i| (board >> (i << 1)) & 0b11 != 0).count() as u8
}

/// Build a [`Record`] for `state` given the final `winner` (`None` == draw)
/// and an optional policy target. `value` is from `state`'s player-to-move
/// perspective.
pub fn record_for(state: &Position, winner: Option<Piece>, policy: Vec<(u8, f32)>) -> Record {
    let side = side_of(state.turn);
    let value = match winner {
        None => 0.0,
        Some(w) if side_of(w) == side => 1.0,
        Some(_) => -1.0,
    };
    Record {
        board: state.board,
        side,
        ply: piece_count(state.board),
        value,
        policy,
    }
}

struct Config {
    out: PathBuf,
    games: u64,
    seed: u64,
    label: String,
    engine: Option<String>,
    epsilon: f64,
    presets_path: PathBuf,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Config {
    let mut out = None;
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut label = "outcome".to_string();
    let mut engine = None;
    let mut epsilon = 0.1f64;
    let mut presets_path = PathBuf::from("games/ttt/presets.json");
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--games" => games = val().parse().expect("--games must be an integer"),
            "--seed" => seed = val().parse().expect("--seed must be an integer"),
            "--label" => label = val(),
            "--engine" => engine = Some(val()),
            "--epsilon" => epsilon = val().parse().expect("--epsilon must be a float"),
            "--presets" => presets_path = PathBuf::from(val()),
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-ttt dump --out <path> [--games N] [--seed N] \
                     [--label outcome] [--engine <preset>] [--epsilon P] [--presets <path>]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    match label.as_str() {
        "outcome" => {}
        "gumbel" => panic!(
            "--label gumbel (Gumbel Sequential-Halving self-play) is not implemented yet"
        ),
        other => panic!("unknown --label mode: {other}"),
    }
    assert!(
        (0.0..=1.0).contains(&epsilon),
        "--epsilon must be in [0, 1], got {epsilon}"
    );
    Config {
        out: out.expect("--out is required"),
        games,
        seed,
        label,
        engine,
        epsilon,
        presets_path,
    }
}

/// Backfill the final-outcome value into every record pushed for this game
/// (those from index `first` onward), now that `winner` is known.
fn finish_game(records: &mut [Record], first: usize, winner: Option<Piece>) {
    for rec in &mut records[first..] {
        let side_piece = if rec.side == 0 { Piece::X } else { Piece::O };
        rec.value = match winner {
            None => 0.0,
            Some(w) if w == side_piece => 1.0,
            Some(_) => -1.0,
        };
    }
}

fn winner_of(state: &HashedPosition) -> Option<Piece> {
    if TicTacToe::is_terminal(state) {
        TicTacToe::winner(state)
    } else {
        None
    }
}

/// Play one game, pushing a [`Record`] for every non-terminal position onto
/// `records`. Moves are uniform-random unless `engine` is set, in which case
/// they come from that engine except with probability `epsilon`.
fn dump_one_game(
    rng: &mut SmallRng,
    records: &mut Vec<Record>,
    mut engine: Option<&mut dyn Search<G = TicTacToe>>,
    epsilon: f64,
) {
    let mut state = HashedPosition::new();
    let mut actions = Vec::new();
    let first = records.len();
    while !TicTacToe::is_terminal(&state) {
        actions.clear();
        TicTacToe::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        records.push(record_for(&state.position, None, Vec::new()));
        let action = match engine.as_deref_mut() {
            Some(e) if !rng.gen_bool(epsilon) => e.choose_action(&state),
            _ => actions[rng.gen_range(0..actions.len())],
        };
        state = TicTacToe::apply(state, &action);
    }
    finish_game(records, first, winner_of(&state));
}

/// Entry point for `game-ttt dump ...`. `args` is the argument iterator
/// positioned just past the `dump` token.
pub fn run(args: impl Iterator<Item = String>) {
    let cfg = parse_args(args);
    debug_assert_eq!(cfg.label, "outcome");

    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut records = Vec::new();

    let preset_table = cfg.engine.as_ref().map(|_| {
        PresetTable::load_from_path(&cfg.presets_path)
            .unwrap_or_else(|e| panic!("cannot load {}: {e}", cfg.presets_path.display()))
    });

    for g in 0..cfg.games {
        match (&cfg.engine, &preset_table) {
            (Some(preset), Some(table)) => {
                let mut engine = table
                    .build::<TicTacToe>(preset, cfg.seed.wrapping_add(g).wrapping_add(1))
                    .unwrap_or_else(|e| panic!("preset {preset:?} did not resolve: {e}"));
                dump_one_game(&mut rng, &mut records, Some(&mut *engine), cfg.epsilon);
            }
            _ => dump_one_game(&mut rng, &mut records, None, cfg.epsilon),
        }
        if (g + 1) % 200 == 0 || g + 1 == cfg.games {
            eprintln!("  dumped {}/{} games ({} records)", g + 1, cfg.games, records.len());
        }
    }

    let mut buf = Vec::with_capacity(records.len() * RECORD_HEAD_BYTES);
    for r in &records {
        r.encode(&mut buf);
    }
    if let Some(parent) = cfg.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).expect("cannot create --out parent directory");
    }
    let mut w = BufWriter::new(File::create(&cfg.out).expect("cannot create --out file"));
    w.write_all(&buf).expect("write failed");
    w.flush().expect("flush failed");

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
    use crate::Move;

    fn sample_positions() -> Vec<Position> {
        let mut p1 = Position::new();
        p1.apply(Move(4));
        let mut p2 = p1;
        p2.apply(Move(0));
        p2.apply(Move(8));
        vec![Position::new(), p1, p2]
    }

    #[test]
    fn records_round_trip_through_encode_decode() {
        let policies: Vec<Vec<(u8, f32)>> = vec![
            Vec::new(),
            vec![(4, 1.0)],
            vec![(0, 0.25), (1, 0.5), (8, 0.25)],
        ];
        for pos in sample_positions() {
            for winner in [None, Some(Piece::X), Some(Piece::O)] {
                for policy in &policies {
                    let rec = record_for(&pos, winner, policy.clone());
                    let mut buf = Vec::new();
                    rec.encode(&mut buf);
                    assert_eq!(buf.len(), RECORD_HEAD_BYTES + policy.len() * POLICY_ENTRY_BYTES);
                    let (back, consumed) = Record::decode(&buf).unwrap();
                    assert_eq!(consumed, buf.len());
                    assert_eq!(back, rec);
                }
            }
        }
    }

    #[test]
    fn decode_walks_a_concatenated_stream() {
        let recs: Vec<Record> = sample_positions()
            .into_iter()
            .map(|p| record_for(&p, Some(Piece::X), vec![(3, 1.0)]))
            .collect();
        let mut buf = Vec::new();
        for r in &recs {
            r.encode(&mut buf);
        }
        let mut cursor = &buf[..];
        let mut decoded = Vec::new();
        while let Some((rec, n)) = Record::decode(cursor) {
            decoded.push(rec);
            cursor = &cursor[n..];
        }
        assert!(cursor.is_empty());
        assert_eq!(decoded, recs);
    }

    #[test]
    fn value_is_from_side_to_move_perspective() {
        let p = Position::new(); // X to move
        assert_eq!(record_for(&p, Some(Piece::X), Vec::new()).value, 1.0);
        assert_eq!(record_for(&p, Some(Piece::O), Vec::new()).value, -1.0);
        assert_eq!(record_for(&p, None, Vec::new()).value, 0.0);
        assert_eq!(record_for(&p, None, Vec::new()).side, 0);
        assert_eq!(record_for(&p, None, Vec::new()).ply, 0);
    }

    #[test]
    fn a_dumped_game_labels_every_position_consistently() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut recs = Vec::new();
        dump_one_game(&mut rng, &mut recs, None, 0.0);
        assert!(!recs.is_empty());
        let mut last_ply = 0u8;
        for r in &recs {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(r.ply >= last_ply);
            last_ply = r.ply;
        }
    }
}
