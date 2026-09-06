//! `game-connect4 dump` -- write self-play positions as little-endian
//! records for an offline self-play training loop: one value (and, later,
//! policy) target per position, with the value net refit between
//! generations.
//!
//! This is the Connect Four counterpart of `games/ttt/src/dump.rs`. The
//! tic-tac-toe record packs the whole board into a `u32` (2 bits x 9
//! cells); the standard 6x7 board has 42 cells and does not fit, so the
//! head carries the two occupancy bitboards verbatim instead.
//!
//! ## Record format (v2-connect4)
//!
//! A fixed 23-byte head followed by a variable-length policy tail, packed
//! (no padding), little-endian:
//!
//! | field    | type          | bytes        |
//! |----------|---------------|--------------|
//! | black    | u64 LE        | 8            |
//! | white    | u64 LE        | 8            |
//! | side     | u8            | 1            |
//! | ply      | u8            | 1            |
//! | value    | f32 LE        | 4            |
//! | n_policy | u8            | 1            |
//! | policy   | n * (u8, f32) | n_policy * 5 |
//!
//! `black` / `white` are `game_connect4::BitBoard<6, 7>` raw words
//! (`Board::bits`): bit `row * 7 + col` set where that player holds the
//! cell, row 0 at the bottom. `side` is 0 for Black to move, 1 for White.
//! `ply` is the disc count. `value` is the final game result **from the
//! side-to-move player's perspective**: `+1.0` win, `-1.0` loss, `0.0`
//! draw. The policy tail is the improved-policy training target -- pairs of
//! `(column_index, probability)` from a Gumbel Sequential-Halving visit
//! distribution -- and is empty for `--label outcome` dumps, which record
//! positions from uniform-random self-play with no search-derived policy.
//! The Gumbel self-play path is a later slice of the port; this module
//! establishes the record format and its reader.
//!
//! v2 records are variable-width, so a reader must walk them sequentially
//! (`research/az-train/`'s `az_train.records_c4`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{BitBoard, Move, Player, Standard, State};

/// Bottom-row-origin cell count of the standard board.
const CELLS: usize = 42;

/// One dumped position. See the module docs for field semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub black: u64,
    pub white: u64,
    pub side: u8,
    pub ply: u8,
    pub value: f32,
    /// `(column_index, probability)` pairs -- the improved-policy target.
    /// Empty when no search produced a policy for this position.
    pub policy: Vec<(u8, f32)>,
}

/// Size of a [`Record`]'s fixed head, in bytes (everything up to and
/// including the `n_policy` count byte, before the policy tail).
pub const RECORD_HEAD_BYTES: usize = 23;

/// Size of one policy tail entry, in bytes: `(column: u8, prob: f32 LE)`.
pub const POLICY_ENTRY_BYTES: usize = 5;

impl Record {
    /// Append this record's little-endian bytes to `buf`. Fields are
    /// written one at a time -- never a `#[repr(C)]` struct cast, which
    /// could introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.black.to_le_bytes());
        buf.extend_from_slice(&self.white.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.value.to_le_bytes());
        let n: u8 = self
            .policy
            .len()
            .try_into()
            .expect("policy tail longer than 255 entries");
        buf.push(n);
        for (col, prob) in &self.policy {
            buf.push(*col);
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
        let black = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let white = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let side = bytes[16];
        let ply = bytes[17];
        let value = f32::from_le_bytes(bytes[18..22].try_into().unwrap());
        let n_policy = bytes[22] as usize;
        let total = RECORD_HEAD_BYTES + n_policy * POLICY_ENTRY_BYTES;
        if bytes.len() < total {
            return None;
        }
        let mut policy = Vec::with_capacity(n_policy);
        for i in 0..n_policy {
            let off = RECORD_HEAD_BYTES + i * POLICY_ENTRY_BYTES;
            let col = bytes[off];
            let prob = f32::from_le_bytes(bytes[off + 1..off + 5].try_into().unwrap());
            policy.push((col, prob));
        }
        Some((
            Record {
                black,
                white,
                side,
                ply,
                value,
                policy,
            },
            total,
        ))
    }
}

fn side_of(p: Player) -> u8 {
    match p {
        Player::Black => 0,
        Player::White => 1,
    }
}

/// Build a [`Record`] for `state` given an optional policy target. `value`
/// is filled in later by [`finish_game`] once the outcome is known.
fn record_for(state: &State<6, 7>, policy: Vec<(u8, f32)>) -> Record {
    let ply = (state.black().count_ones() + state.white().count_ones()) as u8;
    Record {
        black: state.black().bits(),
        white: state.white().bits(),
        side: side_of(state.turn()),
        ply,
        value: 0.0,
        policy,
    }
}

/// Backfill the final-outcome value into every record pushed for this game
/// (those from index `first` onward), now that `winner` is known. `value`
/// is from each record's own side-to-move perspective.
fn finish_game(records: &mut [Record], first: usize, winner: Option<Player>) {
    for rec in &mut records[first..] {
        let side_player = if rec.side == 0 {
            Player::Black
        } else {
            Player::White
        };
        rec.value = match winner {
            None => 0.0,
            Some(w) if w == side_player => 1.0,
            Some(_) => -1.0,
        };
    }
}

/// Play one uniform-random game, pushing a [`Record`] for every
/// non-terminal position onto `records`.
fn dump_one_game(rng: &mut SmallRng, records: &mut Vec<Record>) {
    let mut state = State::<6, 7>::default();
    let mut actions = Vec::new();
    let first = records.len();
    while !Standard::is_terminal(&state) {
        actions.clear();
        Standard::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        records.push(record_for(&state, Vec::new()));
        let action = actions[rng.gen_range(0..actions.len())];
        state = Standard::apply(state, &action);
    }
    let winner = if state.has_winner() {
        Some(Standard::winner(&state).expect("has_winner implies a winner"))
    } else {
        None
    };
    finish_game(records, first, winner);
}

struct Config {
    out: PathBuf,
    games: u64,
    seed: u64,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Config {
    let mut out = None;
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut label = "outcome".to_string();
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--games" => games = val().parse().expect("--games must be an integer"),
            "--seed" => seed = val().parse().expect("--seed must be an integer"),
            "--label" => label = val(),
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-connect4 dump --out <path> [--games N] [--seed N] \
                     [--label outcome]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    assert_eq!(
        label, "outcome",
        "only --label outcome is supported so far (Gumbel self-play is a later slice)"
    );
    Config {
        out: out.expect("--out is required"),
        games,
        seed,
    }
}

/// Entry point for `game-connect4 dump ...`. `args` is the argument
/// iterator positioned just past the `dump` token.
pub fn run(args: impl Iterator<Item = String>) {
    let cfg = parse_args(args);
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut records = Vec::new();

    for g in 0..cfg.games {
        dump_one_game(&mut rng, &mut records);
        if (g + 1) % 200 == 0 || g + 1 == cfg.games {
            eprintln!(
                "  dumped {}/{} games ({} records)",
                g + 1,
                cfg.games,
                records.len()
            );
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

/// `(me, opp)` occupancy planes relative to the side to move, as flat
/// row-major `[f32; 42]` arrays (bit `row * 7 + col`, row 0 at the bottom)
/// -- the Connect Four analogue of `az_train.records.me_opp_planes`, and
/// the exact input `az_train.ntuple_c4` consumes. `me[c]` is 1.0 where the
/// mover holds cell `c`, `opp[c]` where the opponent does.
pub fn me_opp_planes(black: u64, white: u64, side: u8) -> ([f32; CELLS], [f32; CELLS]) {
    let (mover, other) = if side == 0 { (black, white) } else { (white, black) };
    let plane = |bits: u64| std::array::from_fn(|c| ((bits >> c) & 1) as f32);
    (plane(mover), plane(other))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_states() -> Vec<State<6, 7>> {
        let mut a = State::<6, 7>::default();
        let b = Standard::apply(a, &Move(3));
        let mut c = b;
        for col in [3u8, 2, 2, 4, 0] {
            c = Standard::apply(c, &Move(col));
        }
        a = Standard::apply(a, &Move(0));
        vec![State::<6, 7>::default(), a, b, c]
    }

    #[test]
    fn records_round_trip_through_encode_decode() {
        let policies: Vec<Vec<(u8, f32)>> = vec![
            Vec::new(),
            vec![(3, 1.0)],
            vec![(0, 0.25), (3, 0.5), (6, 0.25)],
        ];
        for state in sample_states() {
            for policy in &policies {
                let mut rec = record_for(&state, policy.clone());
                rec.value = -1.0;
                let mut buf = Vec::new();
                rec.encode(&mut buf);
                assert_eq!(
                    buf.len(),
                    RECORD_HEAD_BYTES + policy.len() * POLICY_ENTRY_BYTES
                );
                let (back, consumed) = Record::decode(&buf).unwrap();
                assert_eq!(consumed, buf.len());
                assert_eq!(back, rec);
            }
        }
    }

    #[test]
    fn decode_walks_a_concatenated_stream() {
        let recs: Vec<Record> = sample_states()
            .iter()
            .map(|s| {
                let mut r = record_for(s, vec![(3, 1.0)]);
                r.value = 1.0;
                r
            })
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
    fn decode_rejects_a_truncated_record() {
        let mut buf = Vec::new();
        record_for(&State::<6, 7>::default(), vec![(3, 1.0)]).encode(&mut buf);
        buf.pop();
        assert!(Record::decode(&buf).is_none());
    }

    #[test]
    fn value_and_side_track_the_perspective() {
        // Black opens; White to move at ply 1.
        let after_black = Standard::apply(State::<6, 7>::default(), &Move(3));
        let mut recs = vec![
            record_for(&State::<6, 7>::default(), Vec::new()),
            record_for(&after_black, Vec::new()),
        ];
        assert_eq!(recs[0].side, 0);
        assert_eq!(recs[0].ply, 0);
        assert_eq!(recs[1].side, 1);
        assert_eq!(recs[1].ply, 1);
        finish_game(&mut recs, 0, Some(Player::Black));
        assert_eq!(recs[0].value, 1.0); // Black to move, Black won
        assert_eq!(recs[1].value, -1.0); // White to move, Black won
    }

    #[test]
    fn a_dumped_game_labels_every_position_consistently() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut recs = Vec::new();
        dump_one_game(&mut rng, &mut recs);
        assert!(!recs.is_empty());
        let mut last_ply = 0u8;
        for r in &recs {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(r.ply >= last_ply);
            assert!(r.black & r.white == 0, "a cell can't hold both colors");
            last_ply = r.ply;
        }
    }

    #[test]
    fn me_opp_planes_follow_the_side_to_move() {
        let state = {
            let mut s = State::<6, 7>::default();
            for col in [3u8, 2, 4] {
                s = Standard::apply(s, &Move(col));
            }
            s
        };
        // Black has two discs (cols 3, 4 on the bottom row), White one (col 2).
        let rec = record_for(&state, Vec::new());
        assert_eq!(rec.side, 1); // White to move after B, W, B
        let (me, opp) = me_opp_planes(rec.black, rec.white, rec.side);
        assert_eq!(me.iter().sum::<f32>(), 1.0); // White (mover)
        assert_eq!(opp.iter().sum::<f32>(), 2.0); // Black
        assert_eq!(me[2], 1.0); // White's disc, bottom row col 2
        assert_eq!(opp[3], 1.0);
        assert_eq!(opp[4], 1.0);
    }
}
