#![allow(unused)]

use bitboard::{Board, Const};
use game_core::display::RectangularBoard;
use game_core::display::RectangularBoardDisplay;
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::Serialize;
use std::fmt;

type BigBitBoard<const N: usize, const WORDS: usize> = Board<[u64; WORDS], Const<N>, Const<N>>;

#[derive(Copy, Clone, Serialize, Debug, Default, PartialEq, Eq)]
pub enum Player {
    #[default]
    Black,
    White,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct Move(pub u16);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Board cell: none (empty), Some(Black), or Some(White).
pub type Cell = Option<Player>;

/// Tanbo state. Board data lives in two `BigBitBoard<N, WORDS>`s (one per
/// colour) rather than a `Vec<Cell>`, so `State` is `Copy` and applying a
/// move never allocates.
///
/// `WORDS` must be `(N * N).div_ceil(64)` -- see [`BigBitBoard`]'s doc
/// comment for why this can't be derived automatically on stable Rust. A
/// wrong value fails to compile rather than silently truncating the board.
#[derive(Clone, Copy, Serialize, Debug, PartialEq, Eq)]
pub struct State<const N: usize, const WORDS: usize> {
    pub black: BigBitBoard<N, WORDS>,
    pub white: BigBitBoard<N, WORDS>,
    pub turn: Player,
    pub winner: Option<Player>,
}

impl<const N: usize, const WORDS: usize> Default for State<N, WORDS> {
    fn default() -> Self {
        // 2025 rules: denser starting position.
        Self::new_dense()
    }
}

impl<const N: usize, const WORDS: usize> State<N, WORDS> {
    /// 1993-style sparse initial position (as implemented in the original C++).
    ///
    /// Stones are placed on a grid with `step` spacing between them,
    /// alternating colors in a checkerboard-like pattern.  The specific
    /// `step` depends on board size: (N-1)/3 for general N, with special
    /// cases for 9 (4-stone corners), 13 (step=4), and 19 (step=6).
    pub fn new_sparse() -> Self {
        let mut black: BigBitBoard<N, WORDS> = BigBitBoard::new_const();
        let mut white: BigBitBoard<N, WORDS> = BigBitBoard::new_const();

        // N = 9 uses a specific 4-stone setup.
        if N == 9 {
            white.set_index(Self::index(1, 1));
            black.set_index(Self::index(7, 1));
            black.set_index(Self::index(1, 7));
            white.set_index(Self::index(7, 7));
            return Self {
                black,
                white,
                turn: Player::Black,
                winner: None,
            };
        }

        let step = match N {
            19 => 6,
            13 => 4,
            _ => (N - 1) / 3,
        };

        let mut c = Player::Black;
        let mut y = 0;
        while y < N {
            let mut x = 0;
            while x < N {
                match c {
                    Player::Black => black.set_index(Self::index(y, x)),
                    Player::White => white.set_index(Self::index(y, x)),
                }
                c = c.next();
                x += step;
            }
            c = c.next();
            y += step;
        }

        Self {
            black,
            white,
            turn: Player::Black,
            winner: None,
        }
    }

    /// 2025/2026 denser starting position.
    ///
    /// Every other row and every other column is filled with alternating
    /// black and white stones in a checkerboard pattern:
    ///
    /// ```text
    /// W . B . W . B . W
    /// . . . . . . . . .
    /// B . W . B . W . B
    /// . . . . . . . . .
    /// W . B . W . B . W
    /// ...
    /// ```
    pub fn new_dense() -> Self {
        let mut black: BigBitBoard<N, WORDS> = BigBitBoard::new_const();
        let mut white: BigBitBoard<N, WORDS> = BigBitBoard::new_const();

        for y in (0..N).step_by(2) {
            for x in (0..N).step_by(2) {
                // Checkerboard over the 2×2 blocks: each block gets one stone,
                // alternating White/Black diagonally.
                match ((y / 2) + (x / 2)) % 2 == 0 {
                    true => white.set_index(Self::index(y, x)),
                    false => black.set_index(Self::index(y, x)),
                }
            }
        }

        Self {
            black,
            white,
            turn: Player::Black,
            winner: None,
        }
    }

    #[inline(always)]
    fn index(row: usize, col: usize) -> usize {
        row * N + col
    }

    #[inline(always)]
    fn row_col(index: usize) -> (usize, usize) {
        (index / N, index % N)
    }

    /// Every occupied cell, of either colour.
    fn occupied(&self) -> BigBitBoard<N, WORDS> {
        self.black | self.white
    }

    /// This colour's board.
    fn of(&self, player: Player) -> BigBitBoard<N, WORDS> {
        match player {
            Player::Black => self.black,
            Player::White => self.white,
        }
    }

    pub fn color(&self, index: usize) -> Cell {
        if self.black.get_index(index) {
            Some(Player::Black)
        } else if self.white.get_index(index) {
            Some(Player::White)
        } else {
            None
        }
    }

    fn stone_count(&self, player: Player) -> usize {
        self.of(player).count_ones() as usize
    }

    /// List the four orthogonal neighbours of `index` that lie on the board.
    fn neighbours(index: usize) -> [Option<usize>; 4] {
        let (r, c) = Self::row_col(index);
        [
            if r > 0 {
                Some(Self::index(r - 1, c))
            } else {
                None
            },
            if r + 1 < N {
                Some(Self::index(r + 1, c))
            } else {
                None
            },
            if c > 0 {
                Some(Self::index(r, c - 1))
            } else {
                None
            },
            if c + 1 < N {
                Some(Self::index(r, c + 1))
            } else {
                None
            },
        ]
    }

    // ------------------------------------------------------------------
    // Move generation helpers
    // ------------------------------------------------------------------

    /// Is `candidate` a legal placement for a stone extending from
    /// `stone_index`, given `own` (that colour's board)? The empty point
    /// must be orthogonal-adjacent to *exactly one* stone of that colour
    /// (which is `stone_index`), and not adjacent to any other stone of the
    /// same color.
    fn valid_move(candidate: usize, own: BigBitBoard<N, WORDS>, stone_index: usize) -> bool {
        for nb in Self::neighbours(candidate).into_iter().flatten() {
            if nb != stone_index && own.get_index(nb) {
                return false;
            }
        }
        true
    }

    /// Trace a monochrome group starting at `start`, collecting all legal
    /// extension moves into `moves`. `own` is that colour's board and
    /// `occupied` is both colours' combined board. `visited` tracks which
    /// stones have been processed (shared across groups, when called from a
    /// loop over many groups). A local `group_moves` board prevents the same
    /// empty point being claimed twice within a single group.
    fn trace_group(
        player: Player,
        start: usize,
        own: BigBitBoard<N, WORDS>,
        occupied: BigBitBoard<N, WORDS>,
        visited: &mut BigBitBoard<N, WORDS>,
        moves: &mut Vec<Move>,
    ) {
        let mut stack = vec![start];
        let mut group_moves: BigBitBoard<N, WORDS> = BigBitBoard::new_const();

        while let Some(idx) = stack.pop() {
            if visited.get_index(idx) {
                continue;
            }
            visited.set_index(idx);

            // Empty neighbours of this stone are candidate placements.
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if occupied.get_index(nb) {
                    continue; // occupied
                }
                if group_moves.get_index(nb) {
                    continue; // already found for this group
                }
                if !Self::valid_move(nb, own, idx) {
                    continue;
                }
                group_moves.set_index(nb);
                moves.push(Move(nb as u16));
            }

            // Recurse to same-colour neighbours.
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if own.get_index(nb) && !visited.get_index(nb) {
                    stack.push(nb);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Bounded-root removal
    // ------------------------------------------------------------------

    /// Find every root (of either colour) that is bounded -- i.e. has no
    /// legal extension point (`trace_group` produces an empty move list).
    /// Cells already set in `visited` are skipped, so callers can exclude
    /// roots (such as the current root) that have already been accounted
    /// for.
    fn find_bounded_roots(
        black: BigBitBoard<N, WORDS>,
        white: BigBitBoard<N, WORDS>,
        visited: &mut BigBitBoard<N, WORDS>,
    ) -> Vec<(Player, usize)> {
        let occupied = black | white;
        let mut bounded = Vec::new();

        for i in 0..N * N {
            if visited.get_index(i) {
                continue;
            }
            let player = if black.get_index(i) {
                Player::Black
            } else if white.get_index(i) {
                Player::White
            } else {
                continue;
            };
            let own = match player {
                Player::Black => black,
                Player::White => white,
            };

            let mut group_moves = Vec::new();
            Self::trace_group(player, i, own, occupied, visited, &mut group_moves);

            if group_moves.is_empty() {
                bounded.push((player, i));
            }
        }

        bounded
    }

    /// Resolve captures after a stone has been placed at `current_root_stone`.
    ///
    /// The rules distinguish two cases: if the placement bounded the
    /// *current* root (the one just enlarged), only that root is removed --
    /// other roots that happen to also be bounded survive. Only when the
    /// current root is *not* bounded do other bounded roots, of either
    /// colour, get swept away.
    fn resolve_captures(
        current_root_stone: usize,
        black: &mut BigBitBoard<N, WORDS>,
        white: &mut BigBitBoard<N, WORDS>,
    ) {
        let current_player = if black.get_index(current_root_stone) {
            Player::Black
        } else {
            debug_assert!(white.get_index(current_root_stone));
            Player::White
        };
        let own = match current_player {
            Player::Black => *black,
            Player::White => *white,
        };
        let occupied = *black | *white;

        let mut visited: BigBitBoard<N, WORDS> = BigBitBoard::new_const();
        let mut current_root_moves = Vec::new();
        Self::trace_group(
            current_player,
            current_root_stone,
            own,
            occupied,
            &mut visited,
            &mut current_root_moves,
        );

        if current_root_moves.is_empty() {
            Self::remove_group(current_player, current_root_stone, black, white);
            return;
        }

        for (player, start) in Self::find_bounded_roots(*black, *white, &mut visited) {
            Self::remove_group(player, start, black, white);
        }
    }

    /// Flood-fill a monochrome group and clear all of its cells.
    fn remove_group(
        player: Player,
        start: usize,
        black: &mut BigBitBoard<N, WORDS>,
        white: &mut BigBitBoard<N, WORDS>,
    ) {
        let own = match player {
            Player::Black => &mut *black,
            Player::White => &mut *white,
        };

        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            if !own.get_index(idx) {
                continue; // already removed (duplicate stack entry)
            }
            own.clear_index(idx);
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if own.get_index(nb) {
                    stack.push(nb);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Apply
    // ------------------------------------------------------------------

    fn apply(&mut self, action: &Move) -> Self {
        let index = action.0 as usize;
        debug_assert!(self.color(index).is_none());

        // Place the stone.
        match self.turn {
            Player::Black => self.black.set_index(index),
            Player::White => self.white.set_index(index),
        }

        // Resolve captures per the current/non-current root distinction.
        Self::resolve_captures(index, &mut self.black, &mut self.white);

        // Check for game over: one colour eliminated.
        let black_count = self.stone_count(Player::Black);
        let white_count = self.stone_count(Player::White);

        // Current-root capture only ever removes the mover's own root (the
        // opponent's count is untouched), and non-current-root capture never
        // touches the just-placed root (so the mover's count stays >= 1).
        // Simultaneous elimination of both colours is therefore impossible.
        debug_assert!(black_count > 0 || white_count > 0);

        if black_count == 0 {
            self.winner = Some(Player::White);
        } else if white_count == 0 {
            self.winner = Some(Player::Black);
        } else {
            self.turn = self.turn.next();
        }

        *self
    }
}

// ---------------------------------------------------------------------------
// Game trait
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tanbo<const N: usize, const WORDS: usize>;

impl<const N: usize, const WORDS: usize> Game for Tanbo<N, WORDS> {
    type S = State<N, WORDS>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N, WORDS>, action: &Move) -> State<N, WORDS> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N, WORDS>, actions: &mut Vec<Move>) {
        if state.winner.is_some() {
            return;
        }

        let occupied = state.occupied();
        let own = state.of(state.turn);
        let mut visited: BigBitBoard<N, WORDS> = BigBitBoard::new_const();

        for i in 0..N * N {
            if !own.get_index(i) {
                continue;
            }
            if visited.get_index(i) {
                continue;
            }
            State::<N, WORDS>::trace_group(state.turn, i, own, occupied, &mut visited, actions);
        }
    }

    fn is_terminal(state: &State<N, WORDS>) -> bool {
        state.winner.is_some()
    }

    fn player_to_move(state: &State<N, WORDS>) -> Player {
        state.turn
    }

    fn winner(state: &State<N, WORDS>) -> Option<Player> {
        state.winner
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        const COL_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
        let (row, col) = State::<N, WORDS>::row_col(action.0 as usize);
        format!("{}{}", COL_NAMES[col] as char, row + 1)
    }

    fn parse_action(state: &Self::S, input: &str) -> Option<Self::A> {
        let mut chars = input.chars();
        if let Some(file) = chars.next() {
            let col = file.to_ascii_uppercase() as usize - b'A' as usize;
            if col < N {
                if let Ok(row) = chars
                    .collect::<String>()
                    .trim()
                    .parse::<usize>()
                    .map(|x| x - 1)
                {
                    if row < N {
                        let index = State::<N, WORDS>::index(row, col);
                        if state.color(index).is_none() {
                            return Some(Move(index as u16));
                        } else {
                            eprintln!("occupied: {index}");
                        }
                    } else {
                        eprintln!("row out of range: {row} must be >= 1 and <= {N}");
                    }
                }
            } else {
                eprintln!("col out of range: {col} must be >= 1 and <= {N}");
            }
        }
        None
    }

    fn num_players() -> usize {
        2
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<const N: usize, const WORDS: usize> RectangularBoard for State<N, WORDS> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        match self.color(Self::index(row, col)) {
            Some(Player::Black) => 'X',
            Some(Player::White) => 'O',
            None => '.',
        }
    }
}

impl<const N: usize, const WORDS: usize> fmt::Display for State<N, WORDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanbo_sparse_moves() {
        // Sparse init should produce valid moves.
        let state = State::<11, 2>::new_sparse();
        let mut actions = Vec::new();
        Tanbo::<11, 2>::generate_actions(&state, &mut actions);
        assert!(!actions.is_empty(), "sparse 11x11 should have moves");
    }

    #[test]
    fn test_tanbo_stone_counts() {
        // Dense: roughly N*N/4 stones total (every other row/col).
        let s13 = State::<13, 3>::new_dense();
        let total_dense = s13.stone_count(Player::Black) + s13.stone_count(Player::White);
        // 13x13: ceil(13/2)^2 = 7*7 = 49
        assert_eq!(total_dense, 49);

        // Sparse: much fewer stones (4 per dimension = 16 total for most sizes).
        let s13s = State::<13, 3>::new_sparse();
        let total_sparse = s13s.stone_count(Player::Black) + s13s.stone_count(Player::White);
        assert_eq!(total_sparse, 16);
        assert!(total_sparse < total_dense);
    }

    #[test]
    fn test_tanbo_shared_move_not_bounded() {
        // Regression test: a group's legal extension must not be considered
        // "used up" just because another group (of either colour) also claims
        // it.  White at (1,0) has two legal moves (0,0) and (1,1), but both
        // are ALSO legal moves for the Black group at (0,1).  The shared
        // `visited` array used to steal them from White, bounding White
        // incorrectly.
        //
        //   . B .
        //   W . .
        //   B . .
        let mut black: BigBitBoard<3, 1> = BigBitBoard::new_const();
        let mut white: BigBitBoard<3, 1> = BigBitBoard::new_const();
        black.set_index(1); // (0,1)
        white.set_index(3); // (1,0)
        black.set_index(6); // (2,0)

        let mut visited: BigBitBoard<3, 1> = BigBitBoard::new_const();
        let bounded = State::<3, 1>::find_bounded_roots(black, white, &mut visited);

        // White at (1,0) has legal moves and must survive.
        assert!(
            bounded.iter().all(|&(_, start)| start != 3),
            "White should not be bounded"
        );
    }

    #[test]
    fn test_tanbo_non_current_root_swept() {
        // Verify that bounded roots of BOTH colours are removed when the
        // current root (the one just extended) is *not* itself bounded.
        // Place one White stone, surround it completely with Black stones,
        // then let Black extend a separate, unbounded root elsewhere. The
        // isolated White group has no legal extension (every empty
        // neighbour is adjacent to 0 or 2+ White stones), so it should be
        // removed once Black's current root survives its own move.
        let mut black: BigBitBoard<9, 2> = BigBitBoard::new_const();
        let mut white: BigBitBoard<9, 2> = BigBitBoard::new_const();
        white.set_index(36); // (4,0)
                             // Surround with Black
        black.set_index(27); // (3,0)
        black.set_index(37); // (4,1)
        black.set_index(45); // (5,0)
                             // (4,0) is on the left edge, so no neighbour to the west.

        let state = State::<9, 2> {
            black,
            white,
            turn: Player::Black,
            winner: None,
        };

        // Extend the black root at (3,0) northward to (2,0): far from the
        // White-enclosing cluster, so this current root is not bounded by
        // its own move and the non-current-root-capture path triggers.
        let action = Move(State::<9, 2>::index(2, 0) as u16);
        let mut actions = Vec::new();
        Tanbo::<9, 2>::generate_actions(&state, &mut actions);
        assert!(actions.contains(&action), "expected move to be legal");

        let s1 = Tanbo::<9, 2>::apply(state, &action);
        assert!(
            Tanbo::<9, 2>::is_terminal(&s1),
            "White should be eliminated after Black's move"
        );
        assert_eq!(
            Tanbo::<9, 2>::winner(&s1),
            Some(Player::Black),
            "Black should win"
        );
    }

    #[test]
    fn test_tanbo_current_root_capture_only() {
        // If the mover's OWN current root becomes bounded, only that root is
        // removed -- even when another, unrelated root is also bounded. An
        // over-eager implementation that always sweeps every bounded root
        // would wrongly remove White's root here too.
        //
        //   B . W
        //   W W B
        //   . B W
        let mut black: BigBitBoard<3, 1> = BigBitBoard::new_const();
        let mut white: BigBitBoard<3, 1> = BigBitBoard::new_const();
        black.set_index(0); // (0,0) -- Black's current root (A)
        white.set_index(3); // (1,0) -- walls off A's only other exit
        white.set_index(2); // (0,2) -- walls off B's right exit
        white.set_index(4); // (1,1) -- walls off B's down exit
        black.set_index(5); // (1,2) -- unrelated Black stone, fences White's root
        black.set_index(7); // (2,1) -- unrelated Black stone, fences White's root
        white.set_index(8); // (2,2) -- separate, already-bounded White root

        let state = State::<3, 1> {
            black,
            white,
            turn: Player::Black,
            winner: None,
        };

        // Black's only legal move for root A is (0,1): its other neighbour,
        // (1,0), is already occupied.
        let action = Move(1);
        let mut actions = Vec::new();
        Tanbo::<3, 1>::generate_actions(&state, &mut actions);
        assert!(actions.contains(&action), "expected move to be legal");

        let s1 = Tanbo::<3, 1>::apply(state, &action);

        // After playing (0,1), root A = {(0,0),(0,1)} is walled in on every
        // side, so it alone is removed as a current-root capture.
        assert_eq!(s1.color(0), None, "Black's current root should be removed");
        assert_eq!(s1.color(1), None, "Black's current root should be removed");
        assert_eq!(
            s1.color(8),
            Some(Player::White),
            "White's unrelated bounded root must survive current-root-only capture"
        );
        assert!(!Tanbo::<3, 1>::is_terminal(&s1));
    }
}
