//! Self-play shards: a small header, then fixed-size little-endian records, one per position
//! (one per sub-decision ply, so a turn is up to three records).
//!
//! Header: the magic, then `size`, `actions` and `record_bytes` (`u32` each).
//!
//! Record (`3 * cells + 16 + 4 * actions` bytes): stack heights (`u16` per cell), cell owners
//! (`u8` per cell: 0 empty, 1 Black, 2 White), the hands (`u8` x 4: Black sarsens, Black lintels,
//! White sarsens, White lintels), the pending phase (`u8`, see [`Fields::pending`]), the side to
//! move (`u8`: 0 Black, 1 White), ply (`u16`), the mover's outcome (`f32`: `+1` win, `-1` loss,
//! `0` draw), game index (`u32`), then the dense improved-policy target (`f32` per action).
//! Cells are row-major, as in the encoder.

use std::io;
use std::path::Path;

use super::encode::num_actions;
use crate::{
    HashedState, Hand, Orientation, Pending, PieceKind, Player, Size, Square, State,
};

/// Version 1 of Druid's shard format; a layout change gets a new magic, never silent drift.
const MAGIC: &[u8; 8] = b"DRDSHRD1";
const HEADER_BYTES: usize = 8 + 3 * 4;

/// A position as stored: the raw fields of [`State`], no derived caches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fields {
    pub heights: Vec<u16>,
    pub owners: Vec<u8>,
    pub hands: [u8; 4],
    /// 0 none, 1 sarsen chosen, 2 lintel chosen, 3 horizontal lintel, 4 vertical lintel.
    pub pending: u8,
    /// 0 Black, 1 White.
    pub player: u8,
}

fn pending_code(p: Pending) -> u8 {
    match p {
        Pending::None => 0,
        Pending::Piece(PieceKind::Sarsen) => 1,
        Pending::Piece(PieceKind::Lintel) => 2,
        Pending::Oriented(Orientation::Horizontal) => 3,
        Pending::Oriented(Orientation::Vertical) => 4,
    }
}

fn pending_from(code: u8) -> Pending {
    match code {
        0 => Pending::None,
        1 => Pending::Piece(PieceKind::Sarsen),
        2 => Pending::Piece(PieceKind::Lintel),
        3 => Pending::Oriented(Orientation::Horizontal),
        4 => Pending::Oriented(Orientation::Vertical),
        _ => panic!("bad pending code {code}"),
    }
}

impl Fields {
    pub fn of(state: &HashedState) -> Fields {
        let s = state.state();
        Fields {
            heights: s.board.iter().map(|q| q.height).collect(),
            owners: s
                .board
                .iter()
                .map(|q| match q.piece {
                    None => 0,
                    Some(Player::Black) => 1,
                    Some(Player::White) => 2,
                })
                .collect(),
            hands: [
                s.hand_black.sarsens,
                s.hand_black.lintels,
                s.hand_white.sarsens,
                s.hand_white.lintels,
            ],
            pending: pending_code(s.pending),
            player: u8::from(s.player == Player::White),
        }
    }

    pub fn is_black_to_move(&self) -> bool {
        self.player == 0
    }

    /// The position back as a full game state (caches rebuilt).
    pub fn to_state(&self, size: usize) -> HashedState {
        let n = size as u8;
        HashedState::from_state(State {
            player: if self.player == 0 { Player::Black } else { Player::White },
            board: self
                .heights
                .iter()
                .zip(&self.owners)
                .map(|(&height, &o)| Square {
                    height,
                    piece: match o {
                        0 => None,
                        1 => Some(Player::Black),
                        _ => Some(Player::White),
                    },
                })
                .collect(),
            hand_black: Hand { sarsens: self.hands[0], lintels: self.hands[1] },
            hand_white: Hand { sarsens: self.hands[2], lintels: self.hands[3] },
            size: Size { w: n, h: n },
            pending: pending_from(self.pending),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub fields: Fields,
    pub value: f32,
    pub game: u32,
    pub ply: u16,
    pub policy: Vec<f32>,
}

pub fn record_bytes(size: usize) -> usize {
    3 * size * size + 16 + 4 * num_actions(size)
}

impl Record {
    fn write_to(&self, out: &mut Vec<u8>) {
        for h in &self.fields.heights {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out.extend_from_slice(&self.fields.owners);
        out.extend_from_slice(&self.fields.hands);
        out.extend_from_slice(&[self.fields.pending, self.fields.player]);
        out.extend_from_slice(&self.ply.to_le_bytes());
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.game.to_le_bytes());
        for p in &self.policy {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    fn read_from(b: &[u8], cells: usize) -> Record {
        let heights_end = 2 * cells;
        let owners_end = 3 * cells;
        let t = &b[owners_end..];
        Record {
            fields: Fields {
                heights: b[..heights_end]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                owners: b[heights_end..owners_end].to_vec(),
                hands: t[0..4].try_into().unwrap(),
                pending: t[4],
                player: t[5],
            },
            ply: u16::from_le_bytes(t[6..8].try_into().unwrap()),
            value: f32::from_le_bytes(t[8..12].try_into().unwrap()),
            game: u32::from_le_bytes(t[12..16].try_into().unwrap()),
            policy: t[16..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        }
    }
}

/// Write a whole shard atomically (temp file, then rename), so a killed run never leaves a
/// half-written shard behind under the final name.
pub fn write_shard(path: &Path, size: usize, records: &[Record]) -> io::Result<()> {
    let mut out = Vec::with_capacity(HEADER_BYTES + records.len() * record_bytes(size));
    out.extend_from_slice(MAGIC);
    for v in [size as u32, num_actions(size) as u32, record_bytes(size) as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for r in records {
        assert_eq!(r.policy.len(), num_actions(size));
        assert_eq!(r.fields.heights.len(), size * size);
        r.write_to(&mut out);
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(tmp, path)
}

pub fn read_shard(path: &Path) -> io::Result<(usize, Vec<Record>)> {
    let bytes = std::fs::read(path)?;
    let bad = |m: &str| {
        io::Error::new(io::ErrorKind::InvalidData, format!("{}: {m}", path.display()))
    };
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(bad("not a DRDSHRD1 shard"));
    }
    let word =
        |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (size, actions, per) = (word(0), word(1), word(2));
    if actions != num_actions(size) || per != record_bytes(size) {
        return Err(bad("header disagrees with the record layout"));
    }
    let body = &bytes[HEADER_BYTES..];
    if body.len() % per != 0 {
        return Err(bad("truncated record"));
    }
    Ok((
        size,
        body.chunks_exact(per)
            .map(|b| Record::read_from(b, size * size))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::encode::{action_id, legal_moves, planes};
    use crate::Move;
    use crate::DruidSplit;
    use mcts::game::Game;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Positions from a random game, so heights, both owners, hands and every pending phase occur.
    fn positions(size: usize, plies: usize, seed: u64) -> Vec<HashedState> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = HashedState::new(Size { w: size as u8, h: size as u8 });
        let mut out = Vec::new();
        for _ in 0..plies {
            if DruidSplit::is_terminal(&s) {
                break;
            }
            out.push(s.clone());
            let (moves, _) = legal_moves(&s);
            s = DruidSplit::apply(s, &moves[rng.gen_range(0..moves.len())]);
        }
        out
    }

    fn record_of(state: &HashedState, i: u32) -> Record {
        let size = usize::from(state.state().size.w);
        let mut policy = vec![0.0f32; num_actions(size)];
        policy[legal_moves(state).1[0] as usize] = 1.0;
        Record {
            fields: Fields::of(state),
            value: [1.0, -1.0, 0.0][i as usize % 3],
            game: i * 3,
            ply: 250 + i as u16,
            policy,
        }
    }

    #[test]
    fn shard_round_trips_with_the_documented_record_size_and_rebuilds_the_same_planes() {
        for (size, bytes) in [(5, 75 + 16 + 116), (7, 147 + 16 + 212)] {
            assert_eq!(record_bytes(size), bytes, "size {size}");
            let states = positions(size, 40, 7);
            let records: Vec<Record> =
                states.iter().enumerate().map(|(i, s)| record_of(s, i as u32)).collect();
            let path = std::env::temp_dir()
                .join(format!("druid-shard-{size}-{}.bin", std::process::id()));
            write_shard(&path, size, &records).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len() as usize,
                HEADER_BYTES + records.len() * bytes
            );
            let (read_size, back) = read_shard(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(read_size, size);
            assert_eq!(back, records, "size {size}");
            assert!(back.iter().any(|r| r.ply > 255), "the ply is wider than a byte");
            assert!(
                back.iter().map(|r| r.fields.pending).collect::<std::collections::HashSet<_>>().len() >= 3,
                "the fixture covers several pending phases"
            );
            for (r, s) in back.iter().zip(&states) {
                assert_eq!(planes(&r.fields.to_state(size)), planes(s));
            }
        }
    }

    /// Positions sampled every `stride` plies from `games` random games, so heights, both owners,
    /// hands and every pending phase occur, in a fixed order.
    fn sample_positions(size: usize, games: usize, stride: usize, seed: u64) -> Vec<HashedState> {
        (0..games as u64)
            .flat_map(|g| positions(size, 400, seed + g).into_iter().step_by(stride))
            .collect()
    }

    /// Real positions written by Rust and read back by `research/az-train/tests/test_druid_records.py`,
    /// so the two encoders cannot drift. The policy is uniform over the legal ids, which lets the
    /// Python side check its own legality derivation against the support. Regenerate with
    /// `UPDATE_FIXTURE=1 cargo test -p game-druid --lib fixture`.
    fn fixture_matches_the_encoding(size: usize, games: usize) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cnn/fixtures");
        let shard_path = dir.join(format!("encode-{size}.shard.bin"));
        let planes_path = dir.join(format!("encode-{size}.planes.bin"));
        let states = sample_positions(size, games, 5, 100);
        let records: Vec<Record> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let ids = legal_moves(s).1;
                let mut policy = vec![0.0f32; num_actions(size)];
                for &id in &ids {
                    policy[usize::from(id)] = 1.0 / ids.len() as f32;
                }
                Record {
                    fields: Fields::of(s),
                    value: if i % 2 == 0 { 1.0 } else { -1.0 },
                    game: i as u32,
                    ply: (i % 200) as u16,
                    policy,
                }
            })
            .collect();
        let planes: Vec<u8> = states.iter().flat_map(planes).flat_map(f32::to_le_bytes).collect();
        let phases: std::collections::HashSet<u8> =
            records.iter().map(|r| r.fields.pending).collect();
        assert_eq!(phases.len(), 5, "the fixture must cover every pending phase");
        assert!(records.iter().any(|r| r.fields.heights.iter().any(|&h| h >= 2)), "stacked cells");
        if std::env::var("UPDATE_FIXTURE").is_ok() {
            std::fs::create_dir_all(&dir).unwrap();
            write_shard(&shard_path, size, &records).unwrap();
            std::fs::write(&planes_path, &planes).unwrap();
        }
        let (read_size, stored) =
            read_shard(&shard_path).expect("fixture shard (run with UPDATE_FIXTURE=1)");
        assert_eq!(read_size, size);
        assert_eq!(stored, records, "the stored shard fixture is stale");
        assert_eq!(std::fs::read(&planes_path).unwrap(), planes, "the stored planes fixture is stale");
    }

    #[test]
    fn fixtures_match_the_encoding() {
        fixture_matches_the_encoding(5, 6);
        fixture_matches_the_encoding(7, 4);
    }

    /// The board reflected across the horizontal axis (`flip_rows`) and/or the vertical one.
    fn mirror(f: &Fields, size: usize, flip_rows: bool, flip_cols: bool) -> Fields {
        let mut out = f.clone();
        for r in 0..size {
            for c in 0..size {
                let (r2, c2) = (
                    if flip_rows { size - 1 - r } else { r },
                    if flip_cols { size - 1 - c } else { c },
                );
                out.heights[r2 * size + c2] = f.heights[r * size + c];
                out.owners[r2 * size + c2] = f.owners[r * size + c];
            }
        }
        out
    }

    /// The action id `mv` becomes under [`mirror`]. A lintel's cell id is its anchor (its first
    /// cell, the left or top end), so a reflection along the lintel's own axis moves the anchor to
    /// the other end: `n - 3 - x`. This is the table `druid_records.py` implements.
    fn mirror_move(mv: Move, pending: Pending, size: usize, flip_rows: bool, flip_cols: bool) -> Move {
        let Move::Cell(cell) = mv else { return mv };
        let (n, x, y) = (size as i32, i32::from(cell) % size as i32, i32::from(cell) / size as i32);
        let refl = |v: i32, flip: bool, span: i32| if !flip { v } else { (n - span - v).rem_euclid(n) };
        let (along_x, along_y) = match pending {
            Pending::Oriented(Orientation::Horizontal) => (3, 1),
            Pending::Oriented(Orientation::Vertical) => (1, 3),
            _ => (1, 1),
        };
        Move::Cell((refl(y, flip_rows, along_y) * n + refl(x, flip_cols, along_x)) as u8)
    }

    /// Druid is invariant under the reflections that keep each colour's axis (Black joins top and
    /// bottom, White left and right): a game replayed through one has the mapped legal moves, the
    /// mapped successor position and the same outcome at every ply. Quarter turns and transposes
    /// swap the axes, so they are not symmetries.
    #[test]
    fn the_rules_are_invariant_under_axis_preserving_reflections() {
        let mut rng = SmallRng::seed_from_u64(9);
        for size in [5usize, 7] {
            for (flip_rows, flip_cols) in [(true, false), (false, true), (true, true)] {
                for _ in 0..6 {
                    let mut s = HashedState::new(Size { w: size as u8, h: size as u8 });
                    loop {
                        let m = mirror(&Fields::of(&s), size, flip_rows, flip_cols).to_state(size);
                        assert_eq!(DruidSplit::is_terminal(&s), DruidSplit::is_terminal(&m));
                        assert_eq!(DruidSplit::winner(&s), DruidSplit::winner(&m));
                        if DruidSplit::is_terminal(&s) {
                            break;
                        }
                        let pending = s.state().pending;
                        let (moves, _) = legal_moves(&s);
                        let mut mapped: Vec<u16> = moves
                            .iter()
                            .map(|&mv| action_id(&mirror_move(mv, pending, size, flip_rows, flip_cols), size))
                            .collect();
                        mapped.sort_unstable();
                        assert_eq!(mapped, legal_moves(&m).1, "size {size} phase {pending:?}");
                        let mv = moves[rng.gen_range(0..moves.len())];
                        let mv2 = mirror_move(mv, pending, size, flip_rows, flip_cols);
                        s = DruidSplit::apply(s, &mv);
                        let m = DruidSplit::apply(m, &mv2);
                        assert_eq!(
                            Fields::of(&m),
                            mirror(&Fields::of(&s), size, flip_rows, flip_cols),
                            "size {size} {mv:?} in {pending:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_truncated_or_foreign_shard_is_rejected() {
        let path = std::env::temp_dir().join(format!("druid-shard-bad-{}.bin", std::process::id()));
        std::fs::write(&path, b"DRDSHRD0 old format").unwrap();
        assert!(read_shard(&path).is_err());
        let records = vec![record_of(&positions(5, 3, 1)[0], 0)];
        write_shard(&path, 5, &records).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(read_shard(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
