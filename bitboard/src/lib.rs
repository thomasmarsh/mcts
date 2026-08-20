//! A generic bitboard, parameterized over storage backend (`Storage`) and
//! dimension kind (`Dim`). Replaces `game-core`'s hand-duplicated
//! `BitBoard<N, M>`/`BigBitBoard<N, M, WORDS>` with one implementation
//! shared by both, and adds a runtime-sized (`Dyn`) dimension kind so
//! a game can serve every board size from a single monomorphization instead
//! of a distinct compiled match arm per size.

mod adjacency;
mod board;
mod dim;
mod go;
mod storage;

pub use adjacency::{
    table_flood, table_neighbor_mask, Adjacency, NeighborList, RectAdjacency, MAX_NEIGHBORS,
};
pub use board::{Board, Direction};
pub use dim::{Const, Dim, Dyn};
pub use go::{check_go_move, GoEngine};
pub use storage::Storage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_get_set_count() {
        let mut b: Board<u64, Const<8>, Const<8>> = Board::new_const();
        assert_eq!(b.rows(), 8);
        assert_eq!(b.cols(), 8);
        assert_eq!(b.count_ones(), 0);
        assert!(!b.get(3, 4));

        b.set(3, 4);
        b.set(0, 0);
        b.set(7, 7);

        assert!(b.get(3, 4));
        assert_eq!(b.count_ones(), 3);
        assert_eq!(b.iter_set().collect::<Vec<_>>(), vec![0, 28, 63]);
    }

    #[test]
    fn dyn_get_set_count() {
        let mut b: Board<u64, Dyn, Dyn> = Board::new(Dyn(5), Dyn(5));
        assert_eq!(b.rows(), 5);
        assert_eq!(b.cols(), 5);

        b.set(2, 3);
        assert!(b.get(2, 3));
        assert!(!b.get(3, 2));
        assert_eq!(b.count_ones(), 1);
    }

    #[test]
    fn multi_word_storage() {
        // 6 words covers up to 19x19 = 361 bits, matching the plan's
        // Gonnect/AtariGo instantiation (`Board<[u64; 6], Dyn, Dyn>`).
        let mut b: Board<[u64; 6], Dyn, Dyn> = Board::new(Dyn(19), Dyn(19));
        assert_eq!(b.len(), 361);

        for row in 0..19 {
            b.set(row, 18);
        }
        assert_eq!(b.count_ones(), 19);
        assert_eq!(
            b.iter_set().collect::<Vec<_>>(),
            (0..19).map(|row| row * 19 + 18).collect::<Vec<_>>()
        );
    }

    #[test]
    fn const_vs_dyn_same_layout() {
        // Row-major indexing must match regardless of dimension kind, so
        // `Const`- and `Dyn`-backed boards of the same size are
        // interchangeable at the bit level.
        let mut a: Board<u64, Const<4>, Const<4>> = Board::new_const();
        let mut b: Board<u64, Dyn, Dyn> = Board::new(Dyn(4), Dyn(4));
        for (row, col) in [(0, 0), (1, 2), (3, 3)] {
            a.set(row, col);
            b.set(row, col);
        }
        assert_eq!(
            a.iter_set().collect::<Vec<_>>(),
            b.iter_set().collect::<Vec<_>>()
        );
    }
}
