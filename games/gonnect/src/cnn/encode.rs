//! Input encoding. A position becomes seven `size x size` planes, all from the side to move's
//! point of view (the win condition, connecting opposite edges, is colour independent):
//!
//! | plane | content |
//! |---|---|
//! | 0 | stones of the side to move |
//! | 1 | stones of the opponent |
//! | 2 | cell that is legal by capture/suicide rules but forbidden by ko |
//! | 3 | constant 1 when the swap move is available |
//! | 4 | constant 1 when Black is to move |
//! | 5 | cells the side to move may play |
//! | 6 | constant 1 (lets the net see the board edge past the zero padding) |
//!
//! No liberty, group or connection information is encoded: the net has to learn it.
//!
//! Policy actions are the cells (`row * size + col`), then swap, then no-move.

use crate::{Gonnect, Move, Player, State};
use mcts::game::Game;

pub const IN_PLANES: usize = 7;

pub const FLAG_BLACK_TO_MOVE: u8 = 1;
pub const FLAG_SWAP_LEGAL: u8 = 2;
pub const FLAG_NO_MOVE_LEGAL: u8 = 4;

pub fn num_actions(size: usize) -> usize {
    size * size + 2
}

pub fn swap_id(size: usize) -> u16 {
    (size * size) as u16
}

pub fn no_move_id(size: usize) -> u16 {
    (size * size + 1) as u16
}

pub fn action_id(mv: &Move, size: usize) -> u16 {
    if *mv == Move::SWAP {
        swap_id(size)
    } else if *mv == Move::NO_MOVE {
        no_move_id(size)
    } else {
        mv.index()
    }
}

/// Everything about a position the encoding and the shard records need, in fixed size (boards up
/// to 8x8 fit in a `u64` mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fields {
    pub black: u64,
    pub white: u64,
    pub ko: u64,
    pub legal: u64,
    pub flags: u8,
}

impl Fields {
    pub fn black_to_move(&self) -> bool {
        self.flags & FLAG_BLACK_TO_MOVE != 0
    }

    /// Legal action ids, in ascending order.
    pub fn legal_ids(&self, size: usize) -> Vec<u16> {
        let mut ids: Vec<u16> = (0..size * size)
            .filter(|&c| self.legal >> c & 1 == 1)
            .map(|c| c as u16)
            .collect();
        if self.flags & FLAG_SWAP_LEGAL != 0 {
            ids.push(swap_id(size));
        }
        if self.flags & FLAG_NO_MOVE_LEGAL != 0 {
            ids.push(no_move_id(size));
        }
        ids
    }

    /// The planes as `(row, col, plane)` floats, the layout `grid_cnn::Net::forward` takes.
    pub fn planes(&self, size: usize) -> Vec<f32> {
        let (own, opp) = if self.black_to_move() {
            (self.black, self.white)
        } else {
            (self.white, self.black)
        };
        let swap = f32::from(self.flags & FLAG_SWAP_LEGAL != 0);
        let black = f32::from(self.black_to_move());
        let mut out = vec![0.0f32; size * size * IN_PLANES];
        for cell in 0..size * size {
            let at = cell * IN_PLANES;
            let bit = |mask: u64| f32::from(mask >> cell & 1 == 1);
            out[at] = bit(own);
            out[at + 1] = bit(opp);
            out[at + 2] = bit(self.ko);
            out[at + 3] = swap;
            out[at + 4] = black;
            out[at + 5] = bit(self.legal);
            out[at + 6] = 1.0;
        }
        out
    }
}

/// The position's fields and its legal moves, in the order `Gonnect::generate_actions` gives them.
pub fn analyse(state: &State) -> (Fields, Vec<Move>) {
    let size = state.black().rows();
    assert!(size * size <= 64, "shard fields hold boards up to 8x8");
    let mask = |board: crate::Bits| board.iter_set().fold(0u64, |m, cell| m | 1 << cell);
    let occupied = state.occupied();
    let mut moves = Vec::new();
    let (mut legal, mut ko, mut flags) = (0u64, 0u64, 0u8);
    if state.can_swap && occupied.count_ones() == 1 {
        moves.push(Move::SWAP);
        flags |= FLAG_SWAP_LEGAL;
    }
    for index in !occupied {
        let (valid, captures) = state.valid(index);
        if !valid {
            continue;
        }
        if state.is_ko(index, captures) {
            ko |= 1 << index;
        } else {
            legal |= 1 << index;
            moves.push(Move::new(index as u16, captures));
        }
    }
    if moves.is_empty() {
        moves.push(Move::NO_MOVE);
        flags |= FLAG_NO_MOVE_LEGAL;
    }
    if state.turn == Player::Black {
        flags |= FLAG_BLACK_TO_MOVE;
    }
    (
        Fields {
            black: mask(state.black()),
            white: mask(state.white()),
            ko,
            legal,
            flags,
        },
        moves,
    )
}

/// The move with action id `id` in `state`, which must be legal there (the capture set is
/// recomputed rather than searched for in a legal-move list).
pub fn move_from_id(state: &State, id: u16) -> Move {
    let size = state.black().rows();
    if id == swap_id(size) {
        Move::SWAP
    } else if id == no_move_id(size) {
        Move::NO_MOVE
    } else {
        let (valid, captures) = state.valid(id as usize);
        assert!(valid, "action {id} is not a legal placement");
        Move::new(id, captures)
    }
}

/// `Gonnect`'s own view, for tests: the legal moves it generates.
pub fn generated_moves(state: &State) -> Vec<Move> {
    let mut moves = Vec::new();
    Gonnect::generate_actions(state, &mut moves);
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sized::{SizedGonnect, SizedState};
    use grid_cnn::{cell_map, SYMMETRIES};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    type G = SizedGonnect<7>;

    fn random_games(seed: u64, games: usize) -> Vec<Vec<State>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..games)
            .map(|_| {
                let mut s = SizedState::<7>::default();
                let mut trace = vec![s.0.clone()];
                while !G::is_terminal(&s) {
                    let mut actions = Vec::new();
                    G::generate_actions(&s, &mut actions);
                    s = G::apply(s, &actions[rng.gen_range(0..actions.len())]);
                    trace.push(s.0.clone());
                }
                trace
            })
            .collect()
    }

    #[test]
    fn analyse_agrees_with_generate_actions() {
        for game in random_games(3, 30) {
            for state in game.iter().filter(|s| !Gonnect::is_terminal(s)) {
                let (fields, moves) = analyse(state);
                assert_eq!(moves, generated_moves(state));
                let mut ids: Vec<u16> = moves.iter().map(|m| action_id(m, 7)).collect();
                ids.sort_unstable();
                assert_eq!(fields.legal_ids(7), ids);
                assert_eq!(fields.legal & fields.ko, 0);
                for m in &moves {
                    assert_eq!(move_from_id(state, action_id(m, 7)), *m);
                }
            }
        }
    }

    #[test]
    fn the_opening_offers_swap_only_after_the_first_stone() {
        let start = State::new(7);
        let (fields, _) = analyse(&start);
        assert_eq!(fields.flags, FLAG_BLACK_TO_MOVE);
        assert_eq!(fields.legal, (1u64 << 49) - 1);
        let first = Gonnect::apply(start, &move_from_id(&State::new(7), 24));
        let (fields, moves) = analyse(&first);
        assert_eq!(fields.flags, FLAG_SWAP_LEGAL);
        assert!(moves.contains(&Move::SWAP));
        assert_eq!(fields.black, 1 << 24);
        let planes = fields.planes(7);
        let at = |cell: usize, plane: usize| planes[cell * IN_PLANES + plane];
        assert_eq!(
            (at(24, 0), at(24, 1)),
            (0.0, 1.0),
            "white to move sees Black's stone as the opponent's"
        );
        assert_eq!((at(0, 3), at(0, 4), at(0, 6)), (1.0, 0.0, 1.0));
        assert_eq!(at(24, 5), 0.0, "an occupied cell is not legal");
    }

    /// A handful of real positions (with swap, ko and terminal-adjacent cases) written by Rust
    /// and read back by `research/az-train/tests/test_gonnect_records.py`, so the two encoders
    /// cannot drift. Regenerate with `UPDATE_FIXTURE=1 cargo test -p game-gonnect --lib fixture`.
    #[test]
    fn fixture_matches_the_encoding() {
        use crate::cnn::shard::{read_shard, write_shard, Record};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cnn/fixtures");
        let (shard_path, planes_path) =
            (dir.join("encode.shard.bin"), dir.join("encode.planes.bin"));

        let mut picked: Vec<State> = Vec::new();
        let mut with_ko = 0;
        for (g, game) in random_games(5, 300).into_iter().enumerate() {
            for (ply, state) in game.iter().enumerate() {
                if Gonnect::is_terminal(state) {
                    continue;
                }
                let (fields, _) = analyse(state);
                let interesting = (g < 3 && ply < 2) || (fields.ko != 0 && with_ko < 4);
                if interesting || (g < 6 && ply % 17 == 5) {
                    with_ko += usize::from(fields.ko != 0);
                    picked.push(state.clone());
                }
            }
        }
        assert!(with_ko >= 1, "no ko position found among the sampled games");
        let records: Vec<Record> = picked
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (fields, moves) = analyse(s);
                let mut policy = vec![0.0f32; 51];
                for m in &moves {
                    policy[action_id(m, 7) as usize] = 1.0 / moves.len() as f32;
                }
                Record {
                    fields,
                    value: if i.is_multiple_of(2) { 1.0 } else { -1.0 },
                    game: i as u32,
                    ply: (i % 50) as u8,
                    policy,
                }
            })
            .collect();
        let planes: Vec<u8> = records
            .iter()
            .flat_map(|r| r.fields.planes(7))
            .flat_map(f32::to_le_bytes)
            .collect();
        if std::env::var("UPDATE_FIXTURE").is_ok() {
            write_shard(&shard_path, 7, &records).unwrap();
            std::fs::write(&planes_path, &planes).unwrap();
        }
        let (size, stored) =
            read_shard(&shard_path).expect("fixture shard (run with UPDATE_FIXTURE=1)");
        assert_eq!(size, 7);
        assert_eq!(stored, records, "the stored shard fixture is stale");
        assert_eq!(
            std::fs::read(&planes_path).unwrap(),
            planes,
            "the stored planes fixture is stale"
        );
    }

    /// The game is invariant under the 8 board symmetries the CNN is trained and evaluated with:
    /// a game replayed through a symmetry has the same legal moves (mapped), the same ko cells
    /// and the same outcome at every ply.
    #[test]
    fn the_rules_are_d4_invariant_under_grid_cnns_cell_maps() {
        let n = 7;
        for (g, game) in random_games(11, 6).into_iter().enumerate() {
            for sym in 0..SYMMETRIES {
                let map = cell_map(n, sym);
                let mut s = State::new(n);
                let mut replay = State::new(n);
                for (ply, state) in game.iter().enumerate() {
                    if ply > 0 {
                        s = state.clone();
                    }
                    let (f, moves) = analyse(&s);
                    if Gonnect::is_terminal(&s) {
                        assert!(
                            Gonnect::is_terminal(&replay),
                            "game {g} sym {sym} ply {ply}: terminal only untransformed"
                        );
                        assert_eq!(Gonnect::winner(&s), Gonnect::winner(&replay));
                        break;
                    }
                    let (fr, _) = analyse(&replay);
                    let mapped = |mask: u64| {
                        (0..49)
                            .filter(|&c| mask >> c & 1 == 1)
                            .fold(0u64, |m, c| m | 1 << map[c])
                    };
                    assert_eq!(
                        fr.legal,
                        mapped(f.legal),
                        "game {g} sym {sym} ply {ply}: legal"
                    );
                    assert_eq!(fr.ko, mapped(f.ko), "game {g} sym {sym} ply {ply}: ko");
                    assert_eq!(fr.black, mapped(f.black));
                    assert_eq!(fr.white, mapped(f.white));
                    assert_eq!(fr.flags, f.flags);
                    // Play the next move of the recorded game on the transformed board.
                    let next = game[ply + 1].clone();
                    let played = moves
                        .iter()
                        .find(|m| Gonnect::apply(s.clone(), m) == next)
                        .expect("the recorded successor is the result of a legal move");
                    let id = action_id(played, n);
                    let mapped_id = if (id as usize) < 49 {
                        map[id as usize] as u16
                    } else {
                        id
                    };
                    let mv = move_from_id(&replay, mapped_id);
                    replay = Gonnect::apply(replay, &mv);
                }
            }
        }
    }
}
