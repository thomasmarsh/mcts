use super::{Board, Direction};
use crate::dim::{Const, Dim, Dyn};
use crate::storage::Storage;
use proptest::prelude::*;

// Array-backed oracle: an independent, obviously-correct `Vec<bool>`
// model checked against `Board` across the same representative sizes
// `bigbitboard.rs`'s oracle tests cover -- a sub-word board, an
// exact-word-boundary board, and every WORDS from 1..6 -- at *both*
// `Const` and `Dyn` dims, since both must agree bit-for-bit.

fn check_against_oracle<S: Storage, R: Dim, C: Dim>(
    rows: R,
    cols: C,
    sets: &[usize],
    clears: &[usize],
) {
    let (n, m) = (rows.get(), cols.get());
    let bits = n * m;
    let mut oracle = vec![false; bits];
    let mut board: Board<S, R, C> = Board::new(rows, cols);
    let to_coord = |i: usize| (i / m, i % m);

    for &i in sets {
        let i = i % bits;
        oracle[i] = true;
        let (r, c) = to_coord(i);
        board.set(r, c);
    }
    for &i in clears {
        let i = i % bits;
        oracle[i] = false;
        let (r, c) = to_coord(i);
        board.clear(r, c);
    }

    for (i, &expected) in oracle.iter().enumerate() {
        let (r, c) = to_coord(i);
        assert_eq!(board.get(r, c), expected, "get({i}) mismatch");
    }

    assert_eq!(
        board.count_ones() as usize,
        oracle.iter().filter(|&&b| b).count(),
        "count_ones mismatch"
    );

    let mut got: Vec<usize> = board.iter_set().collect();
    got.sort_unstable();
    let expected: Vec<usize> = (0..bits).filter(|&i| oracle[i]).collect();
    assert_eq!(got, expected, "iterated set bits mismatch");
}

fn check_binary_ops_against_oracle<S: Storage, R: Dim, C: Dim>(
    rows: R,
    cols: C,
    a_bits: &[usize],
    b_bits: &[usize],
) {
    let (n, m) = (rows.get(), cols.get());
    let bits = n * m;
    let mut oa = vec![false; bits];
    let mut ob = vec![false; bits];
    let mut a: Board<S, R, C> = Board::new(rows, cols);
    let mut b: Board<S, R, C> = Board::new(rows, cols);
    let to_coord = |i: usize| (i / m, i % m);

    for &i in a_bits {
        let i = i % bits;
        oa[i] = true;
        let (r, c) = to_coord(i);
        a.set(r, c);
    }
    for &i in b_bits {
        let i = i % bits;
        ob[i] = true;
        let (r, c) = to_coord(i);
        b.set(r, c);
    }

    let union = a | b;
    let inter = a & b;
    let xor = a ^ b;
    let not_a = !a;

    for i in 0..bits {
        let (r, c) = to_coord(i);
        assert_eq!(union.get(r, c), oa[i] || ob[i], "union mismatch at {i}");
        assert_eq!(inter.get(r, c), oa[i] && ob[i], "intersect mismatch at {i}");
        assert_eq!(xor.get(r, c), oa[i] ^ ob[i], "xor mismatch at {i}");
        assert_eq!(not_a.get(r, c), !oa[i], "not mismatch at {i}");
    }

    assert_eq!(
        a.intersects(b),
        (0..bits).any(|i| oa[i] && ob[i]),
        "intersects mismatch"
    );
    assert_eq!(
        a.is_subset(b),
        (0..bits).all(|i| !oa[i] || ob[i]),
        "is_subset mismatch"
    );
    assert_eq!(
        a.is_disjoint(b),
        (0..bits).all(|i| !(oa[i] && ob[i])),
        "is_disjoint mismatch"
    );
}

macro_rules! oracle_tests {
        ($mod_name:ident, $n:expr, $m:expr, $storage:ty, $max_index:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #[test]
                    fn const_get_set_clear_count_iter(
                        sets in proptest::collection::vec(0usize..$max_index, 0..200),
                        clears in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &sets, &clears);
                    }

                    #[test]
                    fn dyn_get_set_clear_count_iter(
                        sets in proptest::collection::vec(0usize..$max_index, 0..200),
                        clears in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &sets, &clears);
                    }

                    #[test]
                    fn const_binary_ops(
                        a_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        b_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_binary_ops_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &a_bits, &b_bits);
                    }

                    #[test]
                    fn dyn_binary_ops(
                        a_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        b_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_binary_ops_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &a_bits, &b_bits);
                    }
                }
            }
        };
    }

// Sub-word board.
oracle_tests!(oracle_3x3, 3, 3, u64, 9);
// Exact single-word boundary (64 bits, remainder == 0).
oracle_tests!(oracle_8x8, 8, 8, u64, 64);
// Multi-word sizes, matching `bigbitboard.rs`'s coverage.
oracle_tests!(oracle_9x9, 9, 9, [u64; 2], 81);
oracle_tests!(oracle_11x11, 11, 11, [u64; 2], 121);
oracle_tests!(oracle_13x13, 13, 13, [u64; 3], 169);
oracle_tests!(oracle_19x19, 19, 19, [u64; 6], 361);

#[test]
fn not_masks_padding_bits_in_last_word() {
    // A 9x9 board (81 bits) in 2 words leaves 47 padding bits past bit
    // 80 in word 1; complementing an empty board must not set them, or
    // count_ones would report 128 instead of 81.
    let empty: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
    let full = !empty;
    assert_eq!(full.count_ones(), 81);
}

#[test]
fn serde_round_trips_across_word_boundary() {
    let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
    board.set(0, 0);
    board.set(7, 0); // index 63, last bit of word 0
    board.set(7, 1); // index 64, first bit of word 1
    board.set(8, 8); // index 80, last valid bit

    let json = serde_json::to_string(&board).unwrap();
    let round_tripped: Board<[u64; 2], Const<9>, Const<9>> = serde_json::from_str(&json).unwrap();
    assert_eq!(round_tripped, board);
}

#[test]
fn serde_round_trips_dyn_dims() {
    let mut board: Board<[u64; 6], Dyn, Dyn> = Board::new(Dyn(13), Dyn(13));
    board.set(0, 0);
    board.set(12, 12);

    let json = serde_json::to_string(&board).unwrap();
    let round_tripped: Board<[u64; 6], Dyn, Dyn> = serde_json::from_str(&json).unwrap();
    assert_eq!(round_tripped, board);
    assert_eq!(round_tripped.rows(), 13);
    assert_eq!(round_tripped.cols(), 13);
}

#[test]
fn deserialize_rejects_const_length_mismatch() {
    let mut board: Board<[u64; 6], Dyn, Dyn> = Board::new(Dyn(9), Dyn(9));
    board.set(0, 0);
    let json = serde_json::to_string(&board).unwrap();

    let result: Result<Board<[u64; 6], Const<9>, Const<9>>, _> = serde_json::from_str(&json);
    // rows/cols (9, 9) match Const<9> here, so this should succeed --
    // the interesting negative case is a genuine size mismatch.
    assert!(result.is_ok());

    let mismatched_json = json.replace("\"rows\":9", "\"rows\":11");
    let result: Result<Board<[u64; 6], Const<9>, Const<9>>, _> =
        serde_json::from_str(&mismatched_json);
    assert!(result.is_err());
}

/////////////////////////////////////////////////////////////////////////////////////////////

// Shift/wall/adjacency/flood4/flood8/flood6/connectivity oracle tests,
// ported from `bigbitboard.rs`'s `check_shifts_against_oracle`/
// `check_flood4_against_oracle`/`check_flood6_against_oracle` to run
// against `Board` at both `Const` and `Dyn` dims.

/// Independent row/col-arithmetic oracle for the four cardinal shifts.
fn check_shifts_against_oracle<S: Storage, R: Dim, C: Dim>(rows: R, cols: C, bits: &[usize]) {
    let (n, m) = (rows.get(), cols.get());
    let total = n * m;
    let mut board: Board<S, R, C> = Board::new(rows, cols);
    let mut set: Vec<(usize, usize)> = Vec::new();
    for &i in bits {
        let i = i % total;
        let (r, c) = (i / m, i % m);
        board.set(r, c);
        set.push((r, c));
    }

    let check = |shifted: Board<S, R, C>, delta: (i64, i64), label: &str| {
        let expected: Vec<(usize, usize)> = set
            .iter()
            .filter_map(|&(r, c)| {
                let nr = r as i64 + delta.0;
                let nc = c as i64 + delta.1;
                if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                    Some((nr as usize, nc as usize))
                } else {
                    None
                }
            })
            .collect();
        for row in 0..n {
            for col in 0..m {
                let expect = expected.contains(&(row, col));
                assert_eq!(
                    shifted.get(row, col),
                    expect,
                    "{label}: mismatch at ({row},{col})"
                );
            }
        }
    };

    check(board.shift_north(), (1, 0), "shift_north");
    check(board.shift_south(), (-1, 0), "shift_south");
    check(board.shift_east(), (0, 1), "shift_east");
    check(board.shift_west(), (0, -1), "shift_west");
}

/// Independent BFS oracle for `flood4`.
fn check_flood4_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
    rows: R,
    cols: C,
    bits: &[usize],
    start_row: usize,
    start_col: usize,
) {
    let (n, m) = (rows.get(), cols.get());
    let total = n * m;
    let mut board: Board<S, R, C> = Board::new(rows, cols);
    for &i in bits {
        let i = i % total;
        board.set(i / m, i % m);
    }
    let start_row = start_row % n;
    let start_col = start_col % m;
    let start = start_row * m + start_col;

    let result = board.flood4(start);

    let mut visited: Board<S, R, C> = Board::new(rows, cols);
    let mut stack = vec![(start_row, start_col)];
    let ns: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    while let Some((row, col)) = stack.pop() {
        if !visited.get(row, col) && board.get(row, col) {
            visited.set(row, col);
            for &(dr, dc) in &ns {
                let nr = row as i64 + dr;
                let nc = col as i64 + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                    stack.push((nr as usize, nc as usize));
                }
            }
        }
    }

    assert_eq!(result, visited, "flood4 mismatch vs BFS oracle");
}

/// Independent BFS oracle for `flood6`, seeded from every set bit of
/// `seed_bits` -- mirrors `flood6`'s own multi-seed contract. The six
/// neighbor deltas are the four cardinals plus the northeast/southwest
/// diagonal (not northwest/southeast) -- see `shift_northeast`'s doc
/// comment.
fn check_flood6_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
    rows: R,
    cols: C,
    board_bits: &[usize],
    seed_bits: &[usize],
) {
    let (n, m) = (rows.get(), cols.get());
    let total = n * m;
    let mut board: Board<S, R, C> = Board::new(rows, cols);
    for &i in board_bits {
        let i = i % total;
        board.set(i / m, i % m);
    }
    let mut seed: Board<S, R, C> = Board::new(rows, cols);
    for &i in seed_bits {
        let i = i % total;
        seed.set(i / m, i % m);
    }

    let result = board.flood6(seed);

    let mut visited: Board<S, R, C> = Board::new(rows, cols);
    let mut stack: Vec<(usize, usize)> = seed.iter_set().map(|i| (i / m, i % m)).collect();
    let ns: [(i64, i64); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];
    while let Some((row, col)) = stack.pop() {
        if !visited.get(row, col) && board.get(row, col) {
            visited.set(row, col);
            for &(dr, dc) in &ns {
                let nr = row as i64 + dr;
                let nc = col as i64 + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                    stack.push((nr as usize, nc as usize));
                }
            }
        }
    }

    assert_eq!(result, visited, "flood6 mismatch vs BFS oracle");
}

macro_rules! shift_flood_oracle_tests {
        ($mod_name:ident, $n:expr, $m:expr, $storage:ty, $max_index:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #[test]
                    fn const_shifts(bits in proptest::collection::vec(0usize..$max_index, 0..200)) {
                        check_shifts_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &bits);
                    }

                    #[test]
                    fn dyn_shifts(bits in proptest::collection::vec(0usize..$max_index, 0..200)) {
                        check_shifts_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &bits);
                    }

                    #[test]
                    fn const_flood4(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        start_row in 0usize..$n,
                        start_col in 0usize..$m,
                    ) {
                        check_flood4_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &bits, start_row, start_col);
                    }

                    #[test]
                    fn dyn_flood4(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        start_row in 0usize..$n,
                        start_col in 0usize..$m,
                    ) {
                        check_flood4_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &bits, start_row, start_col);
                    }

                    #[test]
                    fn const_flood6(
                        board_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        seed_bits in proptest::collection::vec(0usize..$max_index, 0..10),
                    ) {
                        check_flood6_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &board_bits, &seed_bits);
                    }

                    #[test]
                    fn dyn_flood6(
                        board_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        seed_bits in proptest::collection::vec(0usize..$max_index, 0..10),
                    ) {
                        check_flood6_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &board_bits, &seed_bits);
                    }
                }
            }
        };
    }

// Sub-word board.
shift_flood_oracle_tests!(shift_flood_3x3, 3, 3, u64, 9);
// Exact single-word boundary (64 bits, remainder == 0).
shift_flood_oracle_tests!(shift_flood_8x8, 8, 8, u64, 64);
// Multi-word sizes.
shift_flood_oracle_tests!(shift_flood_9x9, 9, 9, [u64; 2], 81);
shift_flood_oracle_tests!(shift_flood_11x11, 11, 11, [u64; 2], 121);
shift_flood_oracle_tests!(shift_flood_13x13, 13, 13, [u64; 3], 169);
shift_flood_oracle_tests!(shift_flood_19x19, 19, 19, [u64; 6], 361);

/////////////////////////////////////////////////////////////////////////////////////////////

// Hand-verified regressions for wall masks, word-boundary shift carry,
// and the hex diagonal -- mirroring `bigbitboard.rs`'s equivalents.

#[test]
fn wall_masks_agree_between_const_and_dyn() {
    for direction in [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ] {
        let const_wall: Board<u64, Const<8>, Const<8>> = Board::new_const().wall(direction);
        let dyn_wall: Board<u64, Dyn, Dyn> = Board::new(Dyn(8), Dyn(8)).wall(direction);
        assert_eq!(
            const_wall.iter_set().collect::<Vec<_>>(),
            dyn_wall.iter_set().collect::<Vec<_>>(),
            "{direction:?} wall mismatch between Const and Dyn"
        );
    }
}

#[test]
fn shift_carries_across_word_boundary() {
    // 9x9: bit 63 is the last bit of word 0, bit 64 the first bit of
    // word 1. A horizontal chain straddling that boundary must shift
    // east/west as a unit, not lose the part that crosses words.
    let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
    board.set(7, 0); // index 63
    board.set(7, 1); // index 64

    let east = board.shift_east();
    assert!(east.get(7, 1));
    assert!(east.get(7, 2));
    assert_eq!(east.count_ones(), 2);

    let west = board.shift_west();
    assert_eq!(west.count_ones(), 1);
    assert!(west.get(7, 0));
}

#[test]
fn flood6_carries_the_hex_diagonal_across_a_word_boundary() {
    let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
    board.set(7, 0); // index 63, last bit of word 0
    board.set(8, 1); // index 73, in word 1
    let mut seed: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
    seed.set(7, 0);

    let flood = board.flood6(seed);
    assert!(flood.get(8, 1));
    assert_eq!(flood.count_ones(), 2);
}

#[test]
fn flood6_uses_northeast_southwest_diagonal_only() {
    let board: Board<u64, Const<3>, Const<3>> = !Board::new_const();
    let mut seed: Board<u64, Const<3>, Const<3>> = Board::new_const();
    seed.set(0, 0);
    let flood = board.flood6(seed);
    assert!(flood.get(1, 1));

    let mut isolated: Board<u64, Const<3>, Const<3>> = Board::new_const();
    isolated.set(0, 1);
    isolated.set(1, 0);
    let mut isolated_seed: Board<u64, Const<3>, Const<3>> = Board::new_const();
    isolated_seed.set(0, 1);
    let flood = isolated.flood6(isolated_seed);
    assert_eq!(flood, isolated_seed);
}

#[test]
fn adjacency_mask_excludes_self_and_off_board() {
    let mut board: Board<u64, Const<3>, Const<3>> = Board::new_const();
    board.set(1, 1);
    let adjacent = board.adjacency_mask();
    assert_eq!(adjacent.count_ones(), 4);
    assert!(adjacent.get(0, 1));
    assert!(adjacent.get(2, 1));
    assert!(adjacent.get(1, 0));
    assert!(adjacent.get(1, 2));
    assert!(!adjacent.get(1, 1));
}

#[test]
fn has_opposite_connection4_detects_a_spanning_group() {
    let mut spanning: Board<u64, Const<4>, Const<4>> = Board::new_const();
    for row in 0..4 {
        spanning.set(row, 1);
    }
    assert!(spanning.has_opposite_connection4(spanning.index_of(0, 1)));

    let mut isolated: Board<u64, Const<4>, Const<4>> = Board::new_const();
    isolated.set(1, 1);
    assert!(!isolated.has_opposite_connection4(isolated.index_of(1, 1)));
}

#[test]
fn has_opposite_connection8_matches_connection4_for_cardinal_only_groups() {
    let mut spanning: Board<u64, Const<4>, Const<4>> = Board::new_const();
    for col in 0..4 {
        spanning.set(2, col);
    }
    let start = spanning.index_of(2, 0);
    assert!(spanning.has_opposite_connection8(start));
    assert_eq!(
        spanning.has_opposite_connection8(start),
        spanning.has_opposite_connection4(start)
    );
}

#[test]
fn flood8_reaches_diagonal_only_neighbors() {
    // Unlike `flood4`, `flood8` reaches a purely diagonal neighbor with
    // no cardinal bridge between it and the seed -- see `flood8`'s doc
    // comment for how splitting the shifts across two statements
    // achieves this without a real diagonal shift primitive.
    let mut board: Board<u64, Const<4>, Const<4>> = Board::new_const();
    board.set(0, 0);
    board.set(1, 1); // diagonal-only neighbor of (0, 0); no cardinal bridge set

    let via4 = board.flood4(board.index_of(0, 0));
    let via8 = board.flood8(board.index_of(0, 0));
    assert!(!via4.get(1, 1));
    assert!(via8.get(1, 1));
    assert_eq!(via8.count_ones(), 2);
}

/// Independent BFS oracle for `flood8`, using the real 8-way (including
/// all four diagonals) neighbor set.
fn check_flood8_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
    rows: R,
    cols: C,
    bits: &[usize],
    start_row: usize,
    start_col: usize,
) {
    let (n, m) = (rows.get(), cols.get());
    let total = n * m;
    let mut board: Board<S, R, C> = Board::new(rows, cols);
    for &i in bits {
        let i = i % total;
        board.set(i / m, i % m);
    }
    let start_row = start_row % n;
    let start_col = start_col % m;
    let start = start_row * m + start_col;

    let result = board.flood8(start);

    let mut visited: Board<S, R, C> = Board::new(rows, cols);
    let mut stack = vec![(start_row, start_col)];
    let ns: [(i64, i64); 8] = [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, 1),
        (0, -1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];
    while let Some((row, col)) = stack.pop() {
        if !visited.get(row, col) && board.get(row, col) {
            visited.set(row, col);
            for &(dr, dc) in &ns {
                let nr = row as i64 + dr;
                let nc = col as i64 + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                    stack.push((nr as usize, nc as usize));
                }
            }
        }
    }

    assert_eq!(result, visited, "flood8 mismatch vs BFS oracle");
}

proptest! {
    #[test]
    fn flood8_connectivity_8x8(
        bits in proptest::collection::vec(0usize..64, 0..200),
        start_row in 0usize..8,
        start_col in 0usize..8,
    ) {
        check_flood8_against_oracle::<u64, Const<8>, Const<8>>(Const, Const, &bits, start_row, start_col);
    }

    #[test]
    fn flood8_connectivity_9x9_dyn(
        bits in proptest::collection::vec(0usize..81, 0..200),
        start_row in 0usize..9,
        start_col in 0usize..9,
    ) {
        check_flood8_against_oracle::<[u64; 2], Dyn, Dyn>(Dyn(9), Dyn(9), &bits, start_row, start_col);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

// O(1) symmetry transforms (`flip_cols`/`flip_rows`/`transpose`/`rot180`)
// vs. a brute-force per-bit permutation oracle -- independent of
// `game_core::symmetry::D4Symmetry`, since this crate doesn't depend on
// it (game-core depends on bitboard, not the reverse); the two are
// cross-checked against each other separately in game-core's own tests.

/// Applies a row/col index permutation to every set bit of `board`, one
/// bit at a time -- the "obviously correct" reference every O(1)
/// transform below is checked against.
fn permute_oracle<S: Storage, R: Dim, C: Dim>(
    board: Board<S, R, C>,
    f: impl Fn(usize, usize) -> (usize, usize),
) -> Board<S, R, C> {
    let mut out = board.empty_like();
    for i in board.iter_set() {
        let (r, c) = (i / board.cols(), i % board.cols());
        let (nr, nc) = f(r, c);
        out.set(nr, nc);
    }
    out
}

proptest! {
    #[test]
    fn rot180_matches_permutation_oracle_3x3(bits in proptest::collection::vec(0usize..9, 0..9)) {
        let mut board: Board<u64, Const<3>, Const<3>> = Board::new_const();
        for &i in &bits {
            board.set(i / 3, i % 3);
        }
        let expected = permute_oracle(board, |r, c| (2 - r, 2 - c));
        assert_eq!(board.rot180(), expected);
    }

    #[test]
    fn rot180_matches_permutation_oracle_8x8(bits in proptest::collection::vec(0usize..64, 0..64)) {
        let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
        for &i in &bits {
            board.set(i / 8, i % 8);
        }
        let expected = permute_oracle(board, |r, c| (7 - r, 7 - c));
        assert_eq!(board.rot180(), expected);
    }

    #[test]
    fn rot180_matches_permutation_oracle_dyn_non_square(bits in proptest::collection::vec(0usize..30, 0..30)) {
        let mut board: Board<u64, Dyn, Dyn> = Board::new(Dyn(5), Dyn(6));
        for &i in &bits {
            board.set(i / 6, i % 6);
        }
        let expected = permute_oracle(board, |r, c| (4 - r, 5 - c));
        assert_eq!(board.rot180(), expected);
    }

    #[test]
    fn flip_cols_matches_permutation_oracle_8x8(bits in proptest::collection::vec(0usize..64, 0..64)) {
        let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
        for &i in &bits {
            board.set(i / 8, i % 8);
        }
        let expected = permute_oracle(board, |r, c| (r, 7 - c));
        assert_eq!(board.flip_cols(), expected);
    }

    #[test]
    fn flip_rows_matches_permutation_oracle_8x8(bits in proptest::collection::vec(0usize..64, 0..64)) {
        let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
        for &i in &bits {
            board.set(i / 8, i % 8);
        }
        let expected = permute_oracle(board, |r, c| (7 - r, c));
        assert_eq!(board.flip_rows(), expected);
    }

    #[test]
    fn transpose_matches_permutation_oracle_8x8(bits in proptest::collection::vec(0usize..64, 0..64)) {
        let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
        for &i in &bits {
            board.set(i / 8, i % 8);
        }
        let expected = permute_oracle(board, |r, c| (c, r));
        assert_eq!(board.transpose(), expected);
    }

    #[test]
    fn flip_rows_then_flip_cols_matches_rot180_8x8(bits in proptest::collection::vec(0usize..64, 0..64)) {
        let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
        for &i in &bits {
            board.set(i / 8, i % 8);
        }
        assert_eq!(board.flip_rows().flip_cols(), board.rot180());
        assert_eq!(board.flip_cols().flip_rows(), board.rot180());
    }
}

#[test]
fn rot180_empty_board_is_a_noop() {
    let board: Board<u64, Const<8>, Const<8>> = Board::new_const();
    assert_eq!(board.rot180(), board);
}
