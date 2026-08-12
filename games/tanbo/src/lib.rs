#![allow(unused)]

use game_core::display::RectangularBoard;
use game_core::display::RectangularBoardDisplay;
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::Serialize;
use std::fmt;

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

/// Tanbo state.  Board data lives in a flat `Vec` indexed as `row * N + col`.
#[derive(Clone, Serialize, Debug, PartialEq, Eq)]
pub struct State<const N: usize> {
    /// Flat array of cells, length N*N.
    pub board: Vec<Cell>,
    pub turn: Player,
    pub winner: Option<Player>,
}

impl<const N: usize> Default for State<N> {
    fn default() -> Self {
        // 2025 rules: denser starting position.
        Self::new_dense()
    }
}

impl<const N: usize> State<N> {
    /// 1993-style sparse initial position (as implemented in the original C++).
    ///
    /// Stones are placed on a grid with `step` spacing between them,
    /// alternating colors in a checkerboard-like pattern.  The specific
    /// `step` depends on board size: (N-1)/3 for general N, with special
    /// cases for 9 (4-stone corners), 13 (step=4), and 19 (step=6).
    pub fn new_sparse() -> Self {
        let area = N * N;
        let mut board = vec![None; area];

        // N = 9 uses a specific 4-stone setup.
        if N == 9 {
            board[Self::index(1, 1)] = Some(Player::White);
            board[Self::index(7, 1)] = Some(Player::Black);
            board[Self::index(1, 7)] = Some(Player::Black);
            board[Self::index(7, 7)] = Some(Player::White);
            return Self {
                board,
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
                board[Self::index(y, x)] = Some(c);
                c = c.next();
                x += step;
            }
            c = c.next();
            y += step;
        }

        Self {
            board,
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
        let area = N * N;
        let mut board = vec![None; area];

        for y in (0..N).step_by(2) {
            for x in (0..N).step_by(2) {
                // Checkerboard over the 2×2 blocks: each block gets one stone,
                // alternating White/Black diagonally.
                let c = if ((y / 2) + (x / 2)) % 2 == 0 {
                    Player::White
                } else {
                    Player::Black
                };
                board[Self::index(y, x)] = Some(c);
            }
        }

        Self {
            board,
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

    fn occupied(&self) -> impl Iterator<Item = usize> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.map(|_| i))
    }

    fn color(&self, index: usize) -> Option<Player> {
        self.board[index]
    }

    fn stone_count(&self, player: Player) -> usize {
        self.board.iter().filter(|&&c| c == Some(player)).count()
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

    /// All orthogonal neighbours that contain a stone of `player`.
    fn neighbour_colors(index: usize, player: Player, board: &[Cell]) -> Vec<usize> {
        let mut out = Vec::with_capacity(4);
        for nb in Self::neighbours(index).into_iter().flatten() {
            if board[nb] == Some(player) {
                out.push(nb);
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Move generation helpers
    // ------------------------------------------------------------------

    /// Is `candidate` a legal placement for `player` when extending from the
    /// stone at `stone_index`?  The empty point must be orthogonal-adjacent to
    /// *exactly one* stone of `player` (which is `stone_index`), and not
    /// adjacent to any other stone of the same color.
    fn valid_move(candidate: usize, player: Player, stone_index: usize, board: &[Cell]) -> bool {
        for nb in Self::neighbours(candidate).into_iter().flatten() {
            if nb != stone_index && board[nb] == Some(player) {
                return false;
            }
        }
        true
    }

    /// Trace a monochrome group starting at `start`, collecting all legal
    /// extension moves into `moves`.  `visited` tracks which stones have been
    /// processed (shared across groups).  A local `group_moves` array prevents
    /// the same empty point being claimed twice within a single group.
    fn trace_group(
        player: Player,
        start: usize,
        board: &[Cell],
        visited: &mut [bool],
        moves: &mut Vec<Move>,
    ) {
        let mut stack = vec![start];
        let mut group_moves = vec![false; board.len()];

        while let Some(idx) = stack.pop() {
            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            // Empty neighbours of this stone are candidate placements.
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if board[nb].is_some() {
                    continue; // occupied
                }
                if group_moves[nb] {
                    continue; // already found for this group
                }
                if !Self::valid_move(nb, player, idx, board) {
                    continue;
                }
                group_moves[nb] = true;
                moves.push(Move(nb as u16));
            }

            // Recurse to same-colour neighbours.
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if board[nb] == Some(player) && !visited[nb] {
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
    fn find_bounded_roots(board: &[Cell], visited: &mut [bool]) -> Vec<(Player, usize)> {
        let mut bounded = Vec::new();

        for i in 0..board.len() {
            if visited[i] {
                continue;
            }
            let Some(player) = board[i] else { continue };

            let mut group_moves = Vec::new();
            Self::trace_group(player, i, board, visited, &mut group_moves);

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
    fn resolve_captures(current_root_stone: usize, board: &mut [Cell]) {
        let current_player =
            board[current_root_stone].expect("current root stone must be occupied");
        let mut visited = vec![false; board.len()];
        let mut current_root_moves = Vec::new();
        Self::trace_group(
            current_player,
            current_root_stone,
            board,
            &mut visited,
            &mut current_root_moves,
        );

        if current_root_moves.is_empty() {
            Self::remove_group(current_player, current_root_stone, board);
            return;
        }

        for (player, start) in Self::find_bounded_roots(board, &mut visited) {
            Self::remove_group(player, start, board);
        }
    }

    /// Flood-fill a monochrome group and clear all of its cells.
    fn remove_group(player: Player, start: usize, board: &mut [Cell]) {
        let mut stack = vec![start];
        let mut removed = vec![false; board.len()];
        while let Some(idx) = stack.pop() {
            if removed[idx] {
                continue;
            }
            removed[idx] = true;
            board[idx] = None;
            for nb in Self::neighbours(idx).into_iter().flatten() {
                if board[nb] == Some(player) && !removed[nb] {
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
        debug_assert!(self.board[index].is_none());

        // Place the stone.
        self.board[index] = Some(self.turn);

        // Resolve captures per the current/non-current root distinction.
        Self::resolve_captures(index, &mut self.board);

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

        self.clone()
    }
}

// ---------------------------------------------------------------------------
// Game trait
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tanbo<const N: usize>;

impl<const N: usize> Game for Tanbo<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N>, action: &Move) -> State<N> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        if state.winner.is_some() {
            return;
        }

        let mut visited = vec![false; N * N];

        for i in 0..state.board.len() {
            if state.board[i] != Some(state.turn) {
                continue;
            }
            if visited[i] {
                continue;
            }
            State::<N>::trace_group(state.turn, i, &state.board, &mut visited, actions);
        }
    }

    fn is_terminal(state: &State<N>) -> bool {
        state.winner.is_some()
    }

    fn player_to_move(state: &State<N>) -> Player {
        state.turn
    }

    fn winner(state: &State<N>) -> Option<Player> {
        state.winner
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        const COL_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
        let (row, col) = State::<N>::row_col(action.0 as usize);
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
                        let index = State::<N>::index(row, col);
                        if state.board[index].is_none() {
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

impl<const N: usize> RectangularBoard for State<N> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        match self.board[Self::index(row, col)] {
            Some(Player::Black) => 'X',
            Some(Player::White) => 'O',
            None => '.',
        }
    }
}

impl<const N: usize> fmt::Display for State<N> {
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
        let state = State::<11>::new_sparse();
        let mut actions = Vec::new();
        Tanbo::<11>::generate_actions(&state, &mut actions);
        assert!(!actions.is_empty(), "sparse 11x11 should have moves");
    }

    #[test]
    fn test_tanbo_stone_counts() {
        // Dense: roughly N*N/4 stones total (every other row/col).
        let s13 = State::<13>::new_dense();
        let total_dense = s13.stone_count(Player::Black) + s13.stone_count(Player::White);
        // 13x13: ceil(13/2)^2 = 7*7 = 49
        assert_eq!(total_dense, 49);

        // Sparse: much fewer stones (4 per dimension = 16 total for most sizes).
        let s13s = State::<13>::new_sparse();
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
        let b = vec![
            None,
            Some(Player::Black), // (0,1)
            None,
            Some(Player::White), // (1,0)
            None,
            None,
            Some(Player::Black), // (2,0)
            None,
            None,
        ];

        let mut visited = vec![false; b.len()];
        let bounded = State::<3>::find_bounded_roots(&b, &mut visited);

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
        let mut board = vec![None; 81];
        board[36] = Some(Player::White); // (4,0)
                                         // Surround with Black
        board[27] = Some(Player::Black); // (3,0)
        board[37] = Some(Player::Black); // (4,1)
        board[45] = Some(Player::Black); // (5,0)
                                         // (4,0) is on the left edge, so no neighbour to the west.

        let state = State::<9> {
            board,
            turn: Player::Black,
            winner: None,
        };

        // Extend the black root at (3,0) northward to (2,0): far from the
        // White-enclosing cluster, so this current root is not bounded by
        // its own move and the non-current-root-capture path triggers.
        let action = Move(State::<9>::index(2, 0) as u16);
        let mut actions = Vec::new();
        Tanbo::<9>::generate_actions(&state, &mut actions);
        assert!(actions.contains(&action), "expected move to be legal");

        let s1 = Tanbo::<9>::apply(state, &action);
        assert!(
            Tanbo::<9>::is_terminal(&s1),
            "White should be eliminated after Black's move"
        );
        assert_eq!(
            Tanbo::<9>::winner(&s1),
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
        let mut board = vec![None; 9]; // 3x3
        board[0] = Some(Player::Black); // (0,0) -- Black's current root (A)
        board[3] = Some(Player::White); // (1,0) -- walls off A's only other exit
        board[2] = Some(Player::White); // (0,2) -- walls off B's right exit
        board[4] = Some(Player::White); // (1,1) -- walls off B's down exit
        board[5] = Some(Player::Black); // (1,2) -- unrelated Black stone, fences White's root
        board[7] = Some(Player::Black); // (2,1) -- unrelated Black stone, fences White's root
        board[8] = Some(Player::White); // (2,2) -- separate, already-bounded White root

        let state = State::<3> {
            board,
            turn: Player::Black,
            winner: None,
        };

        // Black's only legal move for root A is (0,1): its other neighbour,
        // (1,0), is already occupied.
        let action = Move(1);
        let mut actions = Vec::new();
        Tanbo::<3>::generate_actions(&state, &mut actions);
        assert!(actions.contains(&action), "expected move to be legal");

        let s1 = Tanbo::<3>::apply(state, &action);

        // After playing (0,1), root A = {(0,0),(0,1)} is walled in on every
        // side, so it alone is removed as a current-root capture.
        assert_eq!(s1.color(0), None, "Black's current root should be removed");
        assert_eq!(s1.color(1), None, "Black's current root should be removed");
        assert_eq!(
            s1.color(8),
            Some(Player::White),
            "White's unrelated bounded root must survive current-root-only capture"
        );
        assert!(!Tanbo::<3>::is_terminal(&s1));
    }
}
