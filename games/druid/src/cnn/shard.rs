//! Self-play shards: a small header, then fixed-size little-endian records, one per position
//! (one per sub-decision ply, so a turn is up to three records).
//!
//! Header: the magic, then `size`, `actions` and `record_bytes` (`u32` each). `record_bytes`
//! selects the record layout: v1 (`record_bytes_v1`) or v2 (`record_bytes`, the current writer),
//! so a v1 shard from before the root search value was recorded stays readable.
//!
//! Record (`3 * cells + 16 + 4 * actions` bytes in v1, plus one trailing `f32` in v2): stack
//! heights (`u16` per cell), cell owners (`u8` per cell: 0 empty, 1 Black, 2 White), the hands
//! (`u8` x 4: Black sarsens, Black lintels, White sarsens, White lintels), the pending phase (`u8`,
//! see [`Fields::pending`]), the side to move (`u8`: 0 Black, 1 White), ply (`u16`), the mover's
//! outcome (`f32`: `+1` win, `-1` loss, `0` draw), game index (`u32`), then in v2 the mover's root
//! search value `q` (`f32`, same sign convention as the outcome; a v1 shard has no `q`, so the
//! reader sets it equal to the outcome), then the dense improved-policy target (`f32` per action).
//! Cells are row-major, as in the encoder.
//!
//! Alongside a shard, [`write_terminal_sidecar`] writes one JSON line per game played (finished or
//! capped at `max_plies`): the game id, the winner (or `null` for a draw or a capped game), and the
//! final board's per-cell heights and top owners, in the same encoding as a record's fields.

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
    /// The root search value of this position at self-play time, from the mover's point of view
    /// (same sign convention as `value`); see `cnn::selfplay::play_games`. A shard read as v1 sets
    /// this equal to `value` (no search value was recorded then).
    pub q: f32,
    pub policy: Vec<f32>,
}

/// Version 1's record size (no `q`): the layout every shard used before the root search value was
/// recorded.
pub fn record_bytes_v1(size: usize) -> usize {
    3 * size * size + 16 + 4 * num_actions(size)
}

/// The current (v2) record size: v1 plus one trailing `f32`, the root search value `q`.
pub fn record_bytes(size: usize) -> usize {
    record_bytes_v1(size) + 4
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
        out.extend_from_slice(&self.q.to_le_bytes());
        for p in &self.policy {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    /// `has_q` selects the v2 tail layout (`q` before the policy) versus v1 (no `q`, so `q` reads
    /// as `value`).
    fn read_from(b: &[u8], cells: usize, has_q: bool) -> Record {
        let heights_end = 2 * cells;
        let owners_end = 3 * cells;
        let t = &b[owners_end..];
        let value = f32::from_le_bytes(t[8..12].try_into().unwrap());
        let (q, policy_start) = if has_q {
            (f32::from_le_bytes(t[16..20].try_into().unwrap()), 20)
        } else {
            (value, 16)
        };
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
            value,
            game: u32::from_le_bytes(t[12..16].try_into().unwrap()),
            q,
            policy: t[policy_start..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        }
    }
}

/// Write a whole shard atomically (temp file, then rename), so a killed run never leaves a
/// half-written shard behind under the final name. Always writes the current (v2) layout.
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
    let has_q = per == record_bytes(size);
    if actions != num_actions(size) || (!has_q && per != record_bytes_v1(size)) {
        return Err(bad("header disagrees with the record layout"));
    }
    let body = &bytes[HEADER_BYTES..];
    if body.len() % per != 0 {
        return Err(bad("truncated record"));
    }
    Ok((
        size,
        body.chunks_exact(per)
            .map(|b| Record::read_from(b, size * size, has_q))
            .collect(),
    ))
}

/// One game's final position, from a self-play generation: written alongside its shard as a JSON
/// line by [`write_terminal_sidecar`]. `winner` is `None` for a draw or a game dropped at
/// `max_plies` (`capped`); `heights`/`owners` are the final board, in [`Fields`]'s own encoding.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Terminal {
    pub game: u32,
    /// 0 Black, 1 White.
    pub winner: Option<u8>,
    pub capped: bool,
    pub heights: Vec<u16>,
    pub owners: Vec<u8>,
}

/// Write a shard's terminal sidecar (one JSON line per game) atomically, matching
/// [`write_shard`]'s temp-file-then-rename.
pub fn write_terminal_sidecar(path: &Path, games: &[Terminal]) -> io::Result<()> {
    let mut out = String::new();
    for g in games {
        out.push_str(&serde_json::to_string(g).expect("Terminal always serializes"));
        out.push('\n');
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::config::SearchSettings;
    use crate::cnn::encode::{action_id, legal_moves, planes, CONNECT_PLANES, IN_PLANES};
    use crate::cnn::oracle::DruidOracle;
    use crate::cnn::selfplay::root_values;
    use crate::Move;
    use crate::DruidSplit;
    use crate::{apply_placed, Piece, PlacedPiece, Pos};
    use grid_cnn::{Geometry, Net, Weights};
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
            // Deliberately different from `value`, so the round trip below actually exercises `q`
            // as its own stored field rather than accidentally matching it.
            q: [0.4, -0.6, 0.05][i as usize % 3],
            policy,
        }
    }

    #[test]
    fn shard_round_trips_with_the_documented_record_size_and_rebuilds_the_same_planes() {
        for (size, bytes) in [(5, 75 + 16 + 4 + 116), (7, 147 + 16 + 4 + 212)] {
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
                assert_eq!(planes(&r.fields.to_state(size), IN_PLANES), planes(s, IN_PLANES));
            }
        }
    }

    /// A v1 shard (written before the root search value existed, `record_bytes_v1` per record, no
    /// trailing `q`) still reads: the header's own `record_bytes` selects the layout, and every
    /// record's `q` comes back equal to its `value`.
    #[test]
    fn a_v1_shard_with_no_q_field_reads_with_q_equal_to_value() {
        let size = 5;
        let states = positions(size, 20, 11);
        let records: Vec<Record> =
            states.iter().enumerate().map(|(i, s)| record_of(s, i as u32)).collect();

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for v in [size as u32, num_actions(size) as u32, record_bytes_v1(size) as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for r in &records {
            for h in &r.fields.heights {
                out.extend_from_slice(&h.to_le_bytes());
            }
            out.extend_from_slice(&r.fields.owners);
            out.extend_from_slice(&r.fields.hands);
            out.extend_from_slice(&[r.fields.pending, r.fields.player]);
            out.extend_from_slice(&r.ply.to_le_bytes());
            out.extend_from_slice(&r.value.to_le_bytes());
            out.extend_from_slice(&r.game.to_le_bytes());
            for p in &r.policy {
                out.extend_from_slice(&p.to_le_bytes());
            }
        }
        let path =
            std::env::temp_dir().join(format!("druid-shard-v1-{}.bin", std::process::id()));
        std::fs::write(&path, &out).unwrap();
        let (read_size, back) = read_shard(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read_size, size);
        assert_eq!(back.len(), records.len());
        for (got, want) in back.iter().zip(&records) {
            assert_eq!(got.q, want.value, "a v1 record has no stored q, so it reads as value");
            assert_eq!(got.value, want.value);
            assert_eq!(got.fields, want.fields);
            assert_eq!(got.policy, want.policy);
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
                let value = if i % 2 == 0 { 1.0 } else { -1.0 };
                Record {
                    fields: Fields::of(s),
                    value,
                    game: i as u32,
                    ply: (i % 200) as u16,
                    // The committed fixture shard is still v1 (no recorded search value), whose
                    // reader sets `q` equal to `value`; keep this in-memory copy the same so the
                    // round trip below does not need `UPDATE_FIXTURE=1`.
                    q: value,
                    policy,
                }
            })
            .collect();
        let encode = |width: usize| -> Vec<u8> {
            states.iter().flat_map(|s| planes(s, width)).flat_map(f32::to_le_bytes).collect()
        };
        let (planes, connect_planes) = (encode(IN_PLANES), encode(CONNECT_PLANES));
        let connect_path = dir.join(format!("encode-{size}.planes20.bin"));
        let phases: std::collections::HashSet<u8> =
            records.iter().map(|r| r.fields.pending).collect();
        assert_eq!(phases.len(), 5, "the fixture must cover every pending phase");
        assert!(records.iter().any(|r| r.fields.heights.iter().any(|&h| h >= 2)), "stacked cells");
        if std::env::var("UPDATE_FIXTURE").is_ok() {
            std::fs::create_dir_all(&dir).unwrap();
            write_shard(&shard_path, size, &records).unwrap();
            std::fs::write(&planes_path, &planes).unwrap();
            std::fs::write(&connect_path, &connect_planes).unwrap();
        }
        let (read_size, stored) =
            read_shard(&shard_path).expect("fixture shard (run with UPDATE_FIXTURE=1)");
        assert_eq!(read_size, size);
        assert_eq!(stored, records, "the stored shard fixture is stale");
        assert_eq!(std::fs::read(&planes_path).unwrap(), planes, "the stored planes fixture is stale");
        assert_eq!(
            std::fs::read(&connect_path).unwrap(),
            connect_planes,
            "the stored connectivity planes fixture is stale"
        );
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

    /// Black's column at column 2 is complete except its middle cell, with Black already
    /// committed to "sarsen": the one ply left is which cell to place it in, and the gap wins
    /// outright. Pins `cnn::selfplay::root_values`'s sign convention (the same one
    /// `druid_value_label` and `Record.value` use) against the mcts-batch backprop sign bug: a
    /// search from here must find the forced win and report it as positive, from the mover's own
    /// point of view.
    fn wins_next_ply() -> HashedState {
        let size = Size { w: 5, h: 5 };
        let place = |state: HashedState, pos: Pos| -> HashedState {
            let mut state = state;
            state.0.player = Player::Black;
            apply_placed(state, PlacedPiece(Piece::Sarsen, pos.index(size.w) as u8))
        };
        let mut state = HashedState::new(size);
        for y in [0, 1, 3, 4] {
            state = place(state, Pos(2, y));
        }
        state.0.player = Player::Black;
        DruidSplit::apply(state, &Move::Piece(PieceKind::Sarsen))
    }

    #[test]
    fn root_value_of_a_forced_win_is_positive_and_matches_records_own_sign() {
        let state = wins_next_ply();
        assert!(!DruidSplit::is_terminal(&state), "the gap is still open");
        assert!(Fields::of(&state).is_black_to_move(), "Black is the one about to win");

        // A zero-weight net, so any positive value can only come from the search actually
        // discovering the terminal win, not from a learned prior.
        let geometry = Geometry {
            size: 5,
            in_planes: IN_PLANES,
            channels: 4,
            blocks: 1,
            policy_planes: 2,
            policy_out: num_actions(5),
            value_planes: 1,
            value_hidden: 4,
        };
        let oracle = DruidOracle::<5>::new(Net::new(&Weights::zeros(geometry)), 8);
        // More than the 21 legal cells left, so every one of them (including the winning gap) is
        // visited at least once regardless of how the simulation budget narrows afterwards.
        let search = SearchSettings { simulations: 0, considered_actions: 32, value_scale: 0.1, max_visit_init: 50 };

        let values = root_values(&oracle, std::slice::from_ref(&state), &search, 64, 1, 7);
        assert!(values[0] > 0.0, "the mover about to win must show a positive root value, got {}", values[0]);

        // `selfplay::play_games`'s own formula for a record at this position (the recorded
        // mover, Black, matches the eventual winner) would set `Record.value` to `+1.0`; the
        // search value above must agree in sign.
        assert_eq!(values[0].signum(), 1.0);

        // `play_games` does not call `root_values`: it takes `Record.q` straight off the noise
        // search tree it already ran to pick the move (`gumbel_explore_with_noise`, then
        // `tree.value(bid, tree.root())`). Run that same pair of calls directly, so the sign
        // convention above is pinned on the actual mechanism a self-play record's `q` comes from.
        let batch = mcts_batch::Config { num_simulations: 64, ..search.batch_config() };
        let mut rng = SmallRng::seed_from_u64(7);
        let (tree, _noise) =
            mcts_batch::gumbel_explore_with_noise(&batch, &oracle, std::slice::from_ref(&state), &mut rng);
        let q = tree.value(0, tree.root());
        assert!(q > 0.0, "play_games' own search mechanism must also see the forced win, got {q}");
    }
}
