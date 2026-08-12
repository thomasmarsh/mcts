// Congo, designed by Demian Freeling.
//
// Rules summary: a 7x7 chess variant. Each side has a Lion, two Elephants, a
// Zebra, a Giraffe, a Crocodile, a Monkey, and seven Pawns. A river runs
// along the middle rank. The game is won the instant a Lion is captured;
// there is no stalemate -- a player with no legal move loses (zugzwang), and
// a bare-lion-vs-bare-lion position is a draw only if neither lion can
// immediately capture the other.
//
// Movement (see https://en.wikipedia.org/wiki/Congo_(chess_variant) and
// https://gambiter.com/chess/variants/Congo_chess_variant.html):
//   - Lion: one step orthogonally/diagonally, but confined to its own 3x3
//     castle (it can never step outside it). Its only way to act beyond the
//     castle is a special capture: if it has a clear file or diagonal line
//     to the enemy Lion, it may move there like a queen and capture it,
//     winning the game.
//   - Elephant: one or two squares orthogonally; the two-square move jumps
//     the intervening square regardless of what's on it.
//   - Zebra: moves like a chess knight.
//   - Giraffe: one step to an empty square (no capture), or a two-square
//     leap in any of the eight directions that both moves and captures,
//     jumping the intervening square regardless of occupancy.
//   - Crocodile: one step in any direction (move or capture), plus a rook
//     slide along its file toward the river (stopping at or before the
//     river row), plus -- once it is sitting in the river -- a rook slide
//     along the river rank in both directions.
//   - Pawn: one step straight or diagonally forward (move or capture). Once
//     past the river, it may also move (not capture) one or two squares
//     straight backward, blocked like a normal slide.
//   - Superpawn (promoted Pawn, reached on the back rank): keeps the Pawn's
//     forward move, adds a sideways move/capture, and adds a one- or
//     two-square straight-or-diagonal backward move (not capture), blocked
//     like a normal slide.
//   - Monkey: one step to an empty square (no capture), or a checkers-style
//     jump-capture over an adjacent enemy piece onto the empty square just
//     beyond it. Multiple jumps may chain in one move (changing direction is
//     allowed, a given enemy piece may only be jumped once, and squares may
//     be revisited); captured pieces are only removed once the whole chain
//     is chosen, and stopping after any prefix of the chain is legal since
//     capturing is never mandatory in Congo.
//
// River / drowning: except for the Crocodile, any piece that is still in
// the river on its owner's *next* turn after arriving drowns (removed from
// play). This implementation tracks that with a per-square counter
// (`river_since`) that starts at 1 the turn a piece lands in the river and
// is incremented each subsequent turn its owner ends with that same piece
// still there; reaching 2 drowns it. Wikipedia's account of the Monkey's
// river exception ("enters and leaves the river during a multi-capture
// without consequence, and drowns iff it ends two consecutive turns in the
// river") describes the same threshold, since only a move's *final* landing
// square is ever recorded here -- a Monkey's intermediate jump squares never
// touch `river_since` at all.
//
// # Representation
//
// `State` is a 49-cell mailbox (`squares: [Cell; 49]`, row-major, row 0 =
// Black's home rank) plus two occupancy `BitBoard<7, 7>`s (maintained
// incrementally, used for O(1) blocking/capture checks) and cached Lion
// squares (`black_lion`/`white_lion`, `None` once captured -- the fast path
// for both `terminal_status` and the Lion's ranged capture).
//
// `Move` is `{from, to, captures}`: a fixed `[u8; MAX_CAPTURES]` array plus a
// count, sorted ascending so two paths that capture the same set of pieces
// and land on the same square compare equal. Every piece other than the
// Monkey produces 0 or 1 captures; only the Monkey's jump chains produce
// more. Applying a move never needs the *order* of captures -- every jumped
// square was empty of anything but the captured piece the whole time (nothing
// is ever placed on an intermediate square) -- so an unordered, sorted set is
// sufficient to reproduce the resulting board.

use game_core::bitboard::BitBoard;
use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use mcts::game::{Game, PlayerIndex, TerminalStatus};

use serde::Serialize;
use std::fmt;

pub const SIZE: usize = 7;
pub const NUM_SQUARES: usize = SIZE * SIZE;
pub const RIVER_ROW: i32 = 3;
pub const MAX_CAPTURES: usize = 16;

type Board = BitBoard<7, 7>;

#[derive(Copy, Clone, Serialize, Debug, Default, PartialEq, Eq, Hash)]
pub enum Player {
    #[default]
    Black,
    White,
}

impl Player {
    #[inline(always)]
    pub fn next(self) -> Player {
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

#[derive(Copy, Clone, Serialize, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    Giraffe,
    Monkey,
    Elephant,
    Lion,
    Crocodile,
    Zebra,
    Pawn,
    Superpawn,
}

impl Piece {
    fn label(self) -> char {
        match self {
            Piece::Giraffe => 'g',
            Piece::Monkey => 'm',
            Piece::Elephant => 'e',
            Piece::Lion => 'l',
            Piece::Crocodile => 'c',
            Piece::Zebra => 'z',
            Piece::Pawn => 'p',
            Piece::Superpawn => 's',
        }
    }
}

type Cell = Option<(Player, Piece)>;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Coordinates
//////////////////////////////////////////////////////////////////////////////////////////////////

#[inline(always)]
fn idx(r: i32, c: i32) -> usize {
    (r * SIZE as i32 + c) as usize
}

#[inline(always)]
fn rc(i: usize) -> (i32, i32) {
    ((i / SIZE) as i32, (i % SIZE) as i32)
}

#[inline(always)]
fn try_pos(r: i32, c: i32) -> Option<usize> {
    if (0..SIZE as i32).contains(&r) && (0..SIZE as i32).contains(&c) {
        Some(idx(r, c))
    } else {
        None
    }
}

#[inline(always)]
fn in_castle(player: Player, r: i32, c: i32) -> bool {
    let rows = match player {
        Player::Black => 0..3,
        Player::White => 4..7,
    };
    rows.contains(&r) && (2..5).contains(&c)
}

const DIRS8: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
const ORTHO4: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const KNIGHT8: [(i32, i32); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

//////////////////////////////////////////////////////////////////////////////////////////////////
// Move
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub num_captures: u8,
    pub captures: [u8; MAX_CAPTURES],
}

impl Move {
    fn simple(from: usize, to: usize, capture: Option<usize>) -> Move {
        let mut captures = [0u8; MAX_CAPTURES];
        let num_captures = if let Some(sq) = capture {
            captures[0] = sq as u8;
            1
        } else {
            0
        };
        Move {
            from: from as u8,
            to: to as u8,
            num_captures,
            captures,
        }
    }

    fn chain(from: usize, to: usize, jumped: &[u8]) -> Move {
        let mut sorted = jumped.to_vec();
        sorted.sort_unstable();
        let mut captures = [0u8; MAX_CAPTURES];
        captures[..sorted.len()].copy_from_slice(&sorted);
        Move {
            from: from as u8,
            to: to as u8,
            num_captures: sorted.len() as u8,
            captures,
        }
    }

    pub fn captures(&self) -> &[u8] {
        &self.captures[..self.num_captures as usize]
    }

    pub fn is_capture(&self) -> bool {
        self.num_captures > 0
    }
}

fn square_name(i: usize) -> String {
    let (r, c) = rc(i);
    let file = (b'a' + c as u8) as char;
    let rank = SIZE as i32 - r;
    format!("{file}{rank}")
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// State
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    squares: [Cell; NUM_SQUARES],
    black_occ: Board,
    white_occ: Board,
    black_lion: Option<u8>,
    white_lion: Option<u8>,
    river_since: [u8; NUM_SQUARES],
    turn: Player,
}

impl Default for State {
    fn default() -> Self {
        State::initial()
    }
}

impl State {
    pub fn initial() -> State {
        let mut s = State {
            squares: [None; NUM_SQUARES],
            black_occ: Board::EMPTY,
            white_occ: Board::EMPTY,
            black_lion: None,
            white_lion: None,
            river_since: [0; NUM_SQUARES],
            turn: Player::Black,
        };

        const BACK_RANK: [Piece; 7] = [
            Piece::Giraffe,
            Piece::Monkey,
            Piece::Elephant,
            Piece::Lion,
            Piece::Elephant,
            Piece::Crocodile,
            Piece::Zebra,
        ];

        for (c, &piece) in BACK_RANK.iter().enumerate() {
            s.place(Player::Black, piece, idx(0, c as i32));
            s.place(Player::White, piece, idx(6, c as i32));
        }
        for c in 0..SIZE as i32 {
            s.place(Player::Black, Piece::Pawn, idx(1, c));
            s.place(Player::White, Piece::Pawn, idx(5, c));
        }

        s
    }

    pub fn turn(&self) -> Player {
        self.turn
    }

    pub fn get(&self, square: usize) -> Cell {
        self.squares[square]
    }

    pub fn river_since(&self, square: usize) -> u8 {
        self.river_since[square]
    }

    /// Rebuilds a `State` from its wire-serializable parts (used by the host
    /// adapter, which round-trips `squares` and `river_since` as plain JSON
    /// rather than this crate's internal bitboard/cached-lion-square form).
    pub fn from_parts(
        cells: [Cell; NUM_SQUARES],
        river_since: [u8; NUM_SQUARES],
        turn: Player,
    ) -> State {
        let mut black_occ = Board::EMPTY;
        let mut white_occ = Board::EMPTY;
        let mut black_lion = None;
        let mut white_lion = None;
        for (i, cell) in cells.iter().enumerate() {
            if let Some((color, piece)) = cell {
                match color {
                    Player::Black => black_occ.set(i),
                    Player::White => white_occ.set(i),
                }
                if *piece == Piece::Lion {
                    match color {
                        Player::Black => black_lion = Some(i as u8),
                        Player::White => white_lion = Some(i as u8),
                    }
                }
            }
        }
        State {
            squares: cells,
            black_occ,
            white_occ,
            black_lion,
            white_lion,
            river_since,
            turn,
        }
    }

    fn occ(&self, p: Player) -> Board {
        match p {
            Player::Black => self.black_occ,
            Player::White => self.white_occ,
        }
    }

    fn lion_square(&self, p: Player) -> Option<u8> {
        match p {
            Player::Black => self.black_lion,
            Player::White => self.white_lion,
        }
    }

    fn has_lion(&self, p: Player) -> bool {
        self.lion_square(p).is_some()
    }

    fn place(&mut self, color: Player, piece: Piece, sq: usize) {
        debug_assert!(self.squares[sq].is_none());
        self.squares[sq] = Some((color, piece));
        match color {
            Player::Black => self.black_occ.set(sq),
            Player::White => self.white_occ.set(sq),
        }
        if piece == Piece::Lion {
            match color {
                Player::Black => self.black_lion = Some(sq as u8),
                Player::White => self.white_lion = Some(sq as u8),
            }
        }
    }

    fn remove(&mut self, sq: usize) {
        if let Some((color, piece)) = self.squares[sq].take() {
            match color {
                Player::Black => self.black_occ &= !Board::from_index(sq),
                Player::White => self.white_occ &= !Board::from_index(sq),
            }
            if piece == Piece::Lion {
                match color {
                    Player::Black => self.black_lion = None,
                    Player::White => self.white_lion = None,
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Move application
    //////////////////////////////////////////////////////////////////////////////////////////

    pub fn apply(&self, mv: &Move) -> State {
        let mut s = *self;
        let mover = s.turn;
        let (_, piece) = s.squares[mv.from as usize].expect("move source must be occupied");

        for &sq in mv.captures() {
            s.remove(sq as usize);
        }
        s.remove(mv.from as usize);
        s.place(mover, piece, mv.to as usize);

        if piece == Piece::Pawn {
            let (r, _) = rc(mv.to as usize);
            let last_rank = match mover {
                Player::Black => 6,
                Player::White => 0,
            };
            if r == last_rank {
                s.remove(mv.to as usize);
                s.place(mover, Piece::Superpawn, mv.to as usize);
            }
        }

        s.update_river(mover, mv.to as usize);
        s.turn = mover.next();
        s
    }

    /// Advances each of `mover`'s river-square pieces' drowning counters and
    /// removes any that have now sat through a second consecutive turn.
    /// `destination` is this move's final landing square (for a Monkey chain,
    /// that means intermediate jump squares never touch the counter -- see
    /// the module doc comment).
    fn update_river(&mut self, mover: Player, destination: usize) {
        let river_squares: Vec<usize> = (0..SIZE as i32).map(|c| idx(RIVER_ROW, c)).collect();

        for &sq in &river_squares {
            if let Some((color, piece)) = self.squares[sq] {
                if color == mover {
                    if piece == Piece::Crocodile {
                        self.river_since[sq] = 0;
                    } else if sq == destination {
                        self.river_since[sq] = 1;
                    } else {
                        self.river_since[sq] += 1;
                    }
                }
            }
        }

        for &sq in &river_squares {
            if let Some((color, piece)) = self.squares[sq] {
                if color == mover && piece != Piece::Crocodile && self.river_since[sq] >= 2 {
                    self.remove(sq);
                    self.river_since[sq] = 0;
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Move generation
    //////////////////////////////////////////////////////////////////////////////////////////

    pub fn generate_moves(&self, actions: &mut Vec<Move>) {
        let mover = self.turn;
        for i in 0..NUM_SQUARES {
            if let Some((color, piece)) = self.squares[i] {
                if color == mover {
                    self.piece_moves(i, piece, mover, actions);
                }
            }
        }
    }

    fn piece_moves(&self, i: usize, piece: Piece, mover: Player, actions: &mut Vec<Move>) {
        match piece {
            Piece::Giraffe => self.giraffe_moves(i, mover, actions),
            Piece::Monkey => self.monkey_moves(i, mover, actions),
            Piece::Elephant => self.elephant_moves(i, mover, actions),
            Piece::Lion => self.lion_moves(i, mover, actions),
            Piece::Crocodile => self.crocodile_moves(i, mover, actions),
            Piece::Zebra => self.zebra_moves(i, mover, actions),
            Piece::Pawn => self.pawn_moves(i, mover, actions, false),
            Piece::Superpawn => self.pawn_moves(i, mover, actions, true),
        }
    }

    /// Pushes a move to `t` if it's empty (plain move) or holds an enemy
    /// piece (capture); does nothing if `t` holds a friendly piece.
    fn push_step(&self, actions: &mut Vec<Move>, from: usize, t: usize, mover: Player) {
        match self.squares[t] {
            None => actions.push(Move::simple(from, t, None)),
            Some((color, _)) if color != mover => actions.push(Move::simple(from, t, Some(t))),
            _ => {}
        }
    }

    fn giraffe_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        for (dr, dc) in DIRS8 {
            if let Some(t) = try_pos(r + dr, c + dc) {
                if self.squares[t].is_none() {
                    actions.push(Move::simple(i, t, None));
                }
            }
            if let Some(t2) = try_pos(r + 2 * dr, c + 2 * dc) {
                self.push_step(actions, i, t2, mover);
            }
        }
    }

    fn elephant_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        for (dr, dc) in ORTHO4 {
            for k in 1..=2 {
                if let Some(t) = try_pos(r + dr * k, c + dc * k) {
                    self.push_step(actions, i, t, mover);
                }
            }
        }
    }

    fn zebra_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        for (dr, dc) in KNIGHT8 {
            if let Some(t) = try_pos(r + dr, c + dc) {
                self.push_step(actions, i, t, mover);
            }
        }
    }

    fn lion_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        if in_castle(mover, r, c) {
            for (dr, dc) in DIRS8 {
                if let Some(t) = try_pos(r + dr, c + dc) {
                    let (tr, tc) = rc(t);
                    if in_castle(mover, tr, tc) {
                        self.push_step(actions, i, t, mover);
                    }
                }
            }
        }
        if let Some(target) = self.lion_sight_target(mover) {
            actions.push(Move::simple(i, target, Some(target)));
        }
    }

    /// If `mover`'s Lion has a clear file or diagonal line to the enemy
    /// Lion, returns the enemy Lion's square (the ranged capture target).
    fn lion_sight_target(&self, mover: Player) -> Option<usize> {
        let from = self.lion_square(mover)? as usize;
        let to = self.lion_square(mover.next())? as usize;
        let (r1, c1) = rc(from);
        let (r2, c2) = rc(to);
        let (dr, dc) = (r2 - r1, c2 - c1);
        let step = if dc == 0 {
            (dr.signum(), 0)
        } else if dr.abs() == dc.abs() {
            (dr.signum(), dc.signum())
        } else {
            return None;
        };
        let occupied = self.black_occ | self.white_occ;
        let mut sq = idx(r1 + step.0, c1 + step.1);
        while sq != to {
            if occupied.get(sq) {
                return None;
            }
            let (r, c) = rc(sq);
            sq = idx(r + step.0, c + step.1);
        }
        Some(to)
    }

    fn crocodile_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        for (dr, dc) in DIRS8 {
            if let Some(t) = try_pos(r + dr, c + dc) {
                self.push_step(actions, i, t, mover);
            }
        }

        if r != RIVER_ROW {
            let dir = if r > RIVER_ROW { -1 } else { 1 };
            let mut rr = r;
            while rr != RIVER_ROW {
                rr += dir;
                let t = idx(rr, c);
                match self.squares[t] {
                    None => actions.push(Move::simple(i, t, None)),
                    Some((color, _)) => {
                        if color != mover {
                            actions.push(Move::simple(i, t, Some(t)));
                        }
                        break;
                    }
                }
            }
        } else {
            for dc in [-1, 1] {
                let mut cc = c;
                loop {
                    cc += dc;
                    if !(0..SIZE as i32).contains(&cc) {
                        break;
                    }
                    let t = idx(RIVER_ROW, cc);
                    match self.squares[t] {
                        None => actions.push(Move::simple(i, t, None)),
                        Some((color, _)) => {
                            if color != mover {
                                actions.push(Move::simple(i, t, Some(t)));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    fn pawn_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>, superpawn: bool) {
        let (r, c) = rc(i);
        let fwd: i32 = match mover {
            Player::Black => 1,
            Player::White => -1,
        };

        for dc in [-1, 0, 1] {
            if let Some(t) = try_pos(r + fwd, c + dc) {
                self.push_step(actions, i, t, mover);
            }
        }

        if superpawn {
            for dc in [-1, 1] {
                if let Some(t) = try_pos(r, c + dc) {
                    self.push_step(actions, i, t, mover);
                }
            }
            for dcol in [-1i32, 0, 1] {
                self.retreat(actions, i, r, c, fwd, dcol);
            }
        } else {
            let past_river = match mover {
                Player::Black => r > RIVER_ROW,
                Player::White => r < RIVER_ROW,
            };
            if past_river {
                self.retreat(actions, i, r, c, fwd, 0);
            }
        }
    }

    /// One- and two-square backward (move-only) slide, `dcol` columns of
    /// drift per square (0 = straight, +-1 = diagonal). The two-square target
    /// requires the one-square square to also be clear ("without jumping").
    fn retreat(&self, actions: &mut Vec<Move>, from: usize, r: i32, c: i32, fwd: i32, dcol: i32) {
        if let Some(t1) = try_pos(r - fwd, c + dcol) {
            if self.squares[t1].is_none() {
                actions.push(Move::simple(from, t1, None));
                if let Some(t2) = try_pos(r - 2 * fwd, c + 2 * dcol) {
                    if self.squares[t2].is_none() {
                        actions.push(Move::simple(from, t2, None));
                    }
                }
            }
        }
    }

    fn monkey_moves(&self, i: usize, mover: Player, actions: &mut Vec<Move>) {
        let (r, c) = rc(i);
        for (dr, dc) in DIRS8 {
            if let Some(t) = try_pos(r + dr, c + dc) {
                if self.squares[t].is_none() {
                    actions.push(Move::simple(i, t, None));
                }
            }
        }

        // Piece the Monkey vacated is treated as empty for the whole chain;
        // nothing else moves during a chain, so this snapshot never changes.
        let occ_without_from = (self.black_occ | self.white_occ) & !Board::from_index(i);
        let enemy_occ = self.occ(mover.next());
        let mut path = Vec::new();
        self.monkey_jump_dfs(i, i, occ_without_from, enemy_occ, &mut path, actions);
    }

    fn monkey_jump_dfs(
        &self,
        from: usize,
        pos: usize,
        occ_without_from: Board,
        enemy_occ: Board,
        path: &mut Vec<u8>,
        actions: &mut Vec<Move>,
    ) {
        let (r, c) = rc(pos);
        for (dr, dc) in DIRS8 {
            let (Some(mid_rc), Some(land_rc)) = (
                try_pos_rc(r + dr, c + dc),
                try_pos_rc(r + 2 * dr, c + 2 * dc),
            ) else {
                continue;
            };
            let mid = idx(mid_rc.0, mid_rc.1);
            let land = idx(land_rc.0, land_rc.1);
            if enemy_occ.get(mid) && !path.contains(&(mid as u8)) && !occ_without_from.get(land) {
                path.push(mid as u8);
                actions.push(Move::chain(from, land, path));
                self.monkey_jump_dfs(from, land, occ_without_from, enemy_occ, path, actions);
                path.pop();
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Terminal status
    //////////////////////////////////////////////////////////////////////////////////////////

    fn bare_lion_draw(&self) -> bool {
        (self.black_occ | self.white_occ).count_ones() == 2
            && self.lion_sight_target(Player::Black).is_none()
            && self.lion_sight_target(Player::White).is_none()
    }

    pub fn terminal_status(&self) -> TerminalStatus<Player> {
        if !self.has_lion(Player::Black) {
            return TerminalStatus::Winner(Player::White);
        }
        if !self.has_lion(Player::White) {
            return TerminalStatus::Winner(Player::Black);
        }
        if self.bare_lion_draw() {
            return TerminalStatus::Draw;
        }
        let mut actions = Vec::new();
        self.generate_moves(&mut actions);
        if actions.is_empty() {
            TerminalStatus::Winner(self.turn.next())
        } else {
            TerminalStatus::NotTerminal
        }
    }
}

#[inline(always)]
fn try_pos_rc(r: i32, c: i32) -> Option<(i32, i32)> {
    if (0..SIZE as i32).contains(&r) && (0..SIZE as i32).contains(&c) {
        Some((r, c))
    } else {
        None
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Game
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Congo;

impl Game for Congo {
    type S = State;
    type A = Move;
    type P = Player;

    fn apply(state: State, action: &Move) -> State {
        state.apply(action)
    }

    fn generate_actions(state: &State, actions: &mut Vec<Move>) {
        state.generate_moves(actions);
    }

    fn is_terminal(state: &State) -> bool {
        !matches!(state.terminal_status(), TerminalStatus::NotTerminal)
    }

    fn terminal_status(state: &State) -> TerminalStatus<Player> {
        state.terminal_status()
    }

    fn winner(state: &State) -> Option<Player> {
        match state.terminal_status() {
            TerminalStatus::Winner(p) => Some(p),
            _ => None,
        }
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    fn num_players() -> usize {
        2
    }

    fn notation(state: &State, action: &Move) -> String {
        let (_, piece) = state.squares[action.from as usize].expect("action source occupied");
        let sep = if action.is_capture() { 'x' } else { '-' };
        format!(
            "{}{}{}{}",
            piece.label(),
            square_name(action.from as usize),
            sep,
            square_name(action.to as usize)
        )
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Display
//////////////////////////////////////////////////////////////////////////////////////////////////

impl RectangularBoard for State {
    const NUM_DISPLAY_ROWS: usize = SIZE;
    const NUM_DISPLAY_COLS: usize = SIZE;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let square = idx((SIZE as i32 - 1) - row as i32, col as i32);
        match self.squares[square] {
            Some((Player::Black, piece)) => piece.label().to_ascii_uppercase(),
            Some((Player::White, piece)) => piece.label(),
            None => {
                let (r, _) = rc(square);
                if r == RIVER_ROW {
                    '~'
                } else {
                    '.'
                }
            }
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::util::random_play;

    #[test]
    fn test_congo_random_play() {
        random_play::<Congo>();
    }

    fn custom(pieces: &[(usize, Player, Piece)], turn: Player) -> State {
        let mut cells = [None; NUM_SQUARES];
        for &(sq, p, pc) in pieces {
            cells[sq] = Some((p, pc));
        }
        State::from_parts(cells, [0; NUM_SQUARES], turn)
    }

    fn moves_from(state: &State, from: usize) -> Vec<Move> {
        let mut actions = Vec::new();
        state.generate_moves(&mut actions);
        actions
            .into_iter()
            .filter(|m| m.from as usize == from)
            .collect()
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Setup / display
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_initial_setup() {
        let s = State::initial();
        assert_eq!(s.get(idx(0, 3)), Some((Player::Black, Piece::Lion)));
        assert_eq!(s.get(idx(6, 3)), Some((Player::White, Piece::Lion)));
        assert_eq!(s.get(idx(0, 5)), Some((Player::Black, Piece::Crocodile)));
        assert_eq!(s.get(idx(1, 0)), Some((Player::Black, Piece::Pawn)));
        assert_eq!(s.get(idx(5, 6)), Some((Player::White, Piece::Pawn)));
        assert_eq!(s.get(idx(3, 0)), None);
        assert_eq!(s.turn(), Player::Black);
        assert_eq!(Congo::player_to_move(&s), Player::Black);
    }

    #[test]
    fn test_display_shows_river_and_black_on_top() {
        let s = State::initial();
        let text = s.to_string();
        let lines: Vec<&str> = text.lines().collect();
        // Row label "7" (rank 7, board row 0 = Black's home) should be the
        // first board row printed, holding the uppercase Black Lion label.
        assert!(lines[1].starts_with('7'));
        assert!(lines[1].contains('L'));
        // The river rank (rank 4, board row 3) must show '~' for its empty
        // squares.
        let river_line = lines.iter().find(|l| l.starts_with('4')).unwrap();
        assert!(river_line.contains('~'));
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Lion
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_lion_confined_to_castle() {
        let lion = idx(1, 3); // center of Black's castle
        let s = custom(
            &[
                (lion, Player::Black, Piece::Lion),
                (idx(6, 0), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, lion);
        assert_eq!(moves.len(), 8, "center castle square has 8 free neighbors");
        for m in &moves {
            let (r, c) = rc(m.to as usize);
            assert!(in_castle(Player::Black, r, c), "lion left its castle");
        }
    }

    #[test]
    fn test_lion_cannot_step_out_of_castle_edge() {
        let lion = idx(2, 3); // bottom-center edge of Black's castle (row 2 is the castle's last row)
        let s = custom(
            &[
                (lion, Player::Black, Piece::Lion),
                (idx(6, 0), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, lion);
        // Row 3 (the river) is outside the castle -- none of the three
        // southward neighbors should be reachable.
        for m in &moves {
            assert_ne!(rc(m.to as usize).0, 3);
        }
    }

    #[test]
    fn test_lion_ranged_capture_file() {
        let black_lion = idx(2, 3);
        let white_lion = idx(4, 3);
        let s = custom(
            &[
                (black_lion, Player::Black, Piece::Lion),
                (white_lion, Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, black_lion);
        assert!(moves
            .iter()
            .any(|m| m.to as usize == white_lion && m.captures() == [white_lion as u8]));
    }

    #[test]
    fn test_lion_ranged_capture_blocked_by_intervening_piece() {
        let black_lion = idx(2, 3);
        let white_lion = idx(4, 3);
        let s = custom(
            &[
                (black_lion, Player::Black, Piece::Lion),
                (white_lion, Player::White, Piece::Lion),
                (idx(3, 3), Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, black_lion);
        assert!(!moves.iter().any(|m| m.to as usize == white_lion));
    }

    #[test]
    fn test_lion_ranged_capture_diagonal() {
        let black_lion = idx(0, 2);
        let white_lion = idx(4, 6);
        let s = custom(
            &[
                (black_lion, Player::Black, Piece::Lion),
                (white_lion, Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, black_lion);
        assert!(moves
            .iter()
            .any(|m| m.to as usize == white_lion && m.captures() == [white_lion as u8]));
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Monkey
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_monkey_simple_step_only_to_empty() {
        let m = idx(3, 3);
        let s = custom(
            &[
                (m, Player::Black, Piece::Monkey),
                (idx(3, 4), Player::White, Piece::Pawn),
                (idx(0, 0), Player::Black, Piece::Lion),
                (idx(6, 0), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, m);
        // 7 empty neighbors (one occupied by the enemy pawn, handled by the
        // jump-capture path instead) plus the jump-capture over that pawn.
        let simple: Vec<_> = moves.iter().filter(|mv| !mv.is_capture()).collect();
        assert_eq!(simple.len(), 7);
        assert!(simple.iter().all(|mv| s.get(mv.to as usize).is_none()));
    }

    #[test]
    fn test_monkey_single_jump_capture() {
        let m = idx(3, 3);
        let enemy = idx(3, 4);
        let landing = idx(3, 5);
        let s = custom(
            &[
                (m, Player::Black, Piece::Monkey),
                (enemy, Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, m);
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == landing && mv.captures() == [enemy as u8]));
    }

    #[test]
    fn test_monkey_jump_blocked_by_occupied_landing() {
        let m = idx(3, 3);
        let enemy = idx(3, 4);
        let landing = idx(3, 5);
        let s = custom(
            &[
                (m, Player::Black, Piece::Monkey),
                (enemy, Player::White, Piece::Pawn),
                (landing, Player::White, Piece::Zebra),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, m);
        assert!(!moves
            .iter()
            .any(|mv| mv.captures().contains(&(enemy as u8))));
    }

    #[test]
    fn test_monkey_chain_capture_changes_direction() {
        let m = idx(0, 0);
        let e1 = idx(0, 1);
        let mid = idx(0, 2);
        let e2 = idx(1, 2);
        let end = idx(2, 2);
        let s = custom(
            &[
                (m, Player::Black, Piece::Monkey),
                (e1, Player::White, Piece::Pawn),
                (e2, Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, m);
        // Stopping after one jump is legal (capturing is never mandatory).
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == mid && mv.captures() == [e1 as u8]));
        // Continuing the chain in a new direction is also legal.
        let mut expected_full = [e1 as u8, e2 as u8];
        expected_full.sort_unstable();
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == end && mv.captures() == expected_full));
    }

    #[test]
    fn test_monkey_cannot_jump_same_piece_twice() {
        // Only one enemy piece (`e1`) sits between two landing squares on
        // either side of it; a would-be back-and-forth double jump over the
        // same piece must not appear as a two-capture move.
        let m = idx(3, 1);
        let e1 = idx(3, 2);
        let land = idx(3, 3);
        let s = custom(
            &[
                (m, Player::Black, Piece::Monkey),
                (e1, Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, m);
        assert!(moves.iter().all(|mv| mv.num_captures <= 1));
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == land && mv.captures() == [e1 as u8]));
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Crocodile
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_crocodile_slides_toward_river_and_stops_there() {
        let croc = idx(0, 3);
        let s = custom(&[(croc, Player::Black, Piece::Crocodile)], Player::Black);
        let moves = moves_from(&s, croc);
        let slide_targets: Vec<usize> = [1, 2, 3]
            .iter()
            .map(|&r| idx(r, 3))
            .filter(|&t| {
                moves
                    .iter()
                    .any(|mv| mv.to as usize == t && !mv.is_capture())
            })
            .collect();
        assert_eq!(
            slide_targets.len(),
            3,
            "should reach every empty square up to and including the river"
        );
        assert!(
            !moves.iter().any(|mv| mv.to as usize == idx(4, 3)),
            "must not slide past the river"
        );
    }

    #[test]
    fn test_crocodile_slide_blocked_and_captures() {
        let croc = idx(0, 3);
        let blocker = idx(2, 3);
        let s = custom(
            &[
                (croc, Player::Black, Piece::Crocodile),
                (blocker, Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, croc);
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == idx(1, 3) && !mv.is_capture()));
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == blocker && mv.captures() == [blocker as u8]));
        assert!(!moves.iter().any(|mv| mv.to as usize == idx(3, 3)));
    }

    #[test]
    fn test_crocodile_in_river_slides_along_rank() {
        let croc = idx(3, 3);
        let s = custom(&[(croc, Player::Black, Piece::Crocodile)], Player::Black);
        let moves = moves_from(&s, croc);
        for c in 0..SIZE as i32 {
            if c == 3 {
                continue;
            }
            assert!(
                moves.iter().any(|mv| mv.to as usize == idx(3, c)),
                "should slide the full river rank, including column {c}"
            );
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Pawn / Superpawn
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_pawn_no_retreat_before_river() {
        let p = idx(1, 3);
        let s = custom(&[(p, Player::Black, Piece::Pawn)], Player::Black);
        let moves = moves_from(&s, p);
        assert!(moves.iter().all(|mv| rc(mv.to as usize).0 == 2));
    }

    #[test]
    fn test_pawn_retreat_past_river_blocked_at_two_squares() {
        let p = idx(4, 3);
        let blocker = idx(2, 3); // the *second* retreat square is occupied
        let s = custom(
            &[
                (p, Player::Black, Piece::Pawn),
                (blocker, Player::Black, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, p);
        assert!(moves
            .iter()
            .any(|mv| mv.to as usize == idx(3, 3) && !mv.is_capture()));
        assert!(!moves.iter().any(|mv| mv.to as usize == idx(2, 3)));
    }

    #[test]
    fn test_pawn_retreat_past_river_both_squares_clear() {
        let p = idx(4, 3);
        let s = custom(&[(p, Player::Black, Piece::Pawn)], Player::Black);
        let moves = moves_from(&s, p);
        assert!(moves.iter().any(|mv| mv.to as usize == idx(3, 3)));
        assert!(moves.iter().any(|mv| mv.to as usize == idx(2, 3)));
    }

    #[test]
    fn test_promotion_to_superpawn() {
        let p = idx(5, 3);
        let s = custom(&[(p, Player::Black, Piece::Pawn)], Player::Black);
        let mv = Move::simple(p, idx(6, 3), None);
        let next = s.apply(&mv);
        assert_eq!(next.get(idx(6, 3)), Some((Player::Black, Piece::Superpawn)));
    }

    #[test]
    fn test_superpawn_sideways_and_diagonal_retreat() {
        let sp = idx(4, 3);
        let s = custom(&[(sp, Player::Black, Piece::Superpawn)], Player::Black);
        let moves = moves_from(&s, sp);
        assert!(moves.iter().any(|mv| mv.to as usize == idx(4, 2)));
        assert!(moves.iter().any(|mv| mv.to as usize == idx(4, 4)));
        assert!(moves.iter().any(|mv| mv.to as usize == idx(2, 1))); // 2-back diagonal
    }

    #[test]
    fn test_superpawn_diagonal_retreat_blocked_by_intervening_square() {
        let sp = idx(4, 3);
        let blocker = idx(3, 2); // the 1-back diagonal square
        let s = custom(
            &[
                (sp, Player::Black, Piece::Superpawn),
                (blocker, Player::Black, Piece::Pawn),
            ],
            Player::Black,
        );
        let moves = moves_from(&s, sp);
        assert!(!moves.iter().any(|mv| mv.to as usize == idx(2, 1)));
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // River / drowning
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_piece_freshly_entering_river_does_not_drown() {
        let p = idx(4, 3);
        let s = custom(
            &[
                (p, Player::Black, Piece::Pawn),
                (idx(0, 0), Player::Black, Piece::Lion),
                (idx(6, 0), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let mv = Move::simple(p, idx(3, 3), None);
        let next = s.apply(&mv);
        assert_eq!(next.get(idx(3, 3)), Some((Player::Black, Piece::Pawn)));
        assert_eq!(next.river_since(idx(3, 3)), 1);
    }

    #[test]
    fn test_piece_drowns_after_second_consecutive_turn_in_river() {
        // A pawn already sitting in the river (river_since = 1, as if it
        // arrived on Black's previous turn); Black now moves a different
        // piece (a harmless in-castle Lion shuffle), leaving the pawn
        // stranded through a second turn-end.
        let stuck = idx(3, 3);
        let lion = idx(1, 3);
        let mut cells = [None; NUM_SQUARES];
        cells[stuck] = Some((Player::Black, Piece::Pawn));
        cells[lion] = Some((Player::Black, Piece::Lion));
        cells[idx(6, 0)] = Some((Player::White, Piece::Lion));
        let mut river_since = [0u8; NUM_SQUARES];
        river_since[stuck] = 1;
        let s = State::from_parts(cells, river_since, Player::Black);

        let mv = Move::simple(lion, idx(1, 2), None);
        let next = s.apply(&mv);
        assert_eq!(next.get(stuck), None, "pawn should have drowned");
    }

    #[test]
    fn test_crocodile_never_drowns() {
        let stuck = idx(3, 3);
        let lion = idx(1, 3);
        let mut cells = [None; NUM_SQUARES];
        cells[stuck] = Some((Player::Black, Piece::Crocodile));
        cells[lion] = Some((Player::Black, Piece::Lion));
        cells[idx(6, 0)] = Some((Player::White, Piece::Lion));
        let mut river_since = [0u8; NUM_SQUARES];
        river_since[stuck] = 1;
        let s = State::from_parts(cells, river_since, Player::Black);
        let mv = Move::simple(lion, idx(1, 2), None);
        let next = s.apply(&mv);
        assert_eq!(next.get(stuck), Some((Player::Black, Piece::Crocodile)));
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Terminal status
    //////////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_lion_capture_ends_the_game() {
        let black_lion = idx(2, 3);
        let white_lion = idx(4, 3);
        let s = custom(
            &[
                (black_lion, Player::Black, Piece::Lion),
                (white_lion, Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        let mv = Move::simple(black_lion, white_lion, Some(white_lion));
        let next = s.apply(&mv);
        assert_eq!(
            next.terminal_status(),
            TerminalStatus::Winner(Player::Black)
        );
    }

    #[test]
    fn test_bare_lion_draw_when_neither_can_capture() {
        let s = custom(
            &[
                (idx(1, 2), Player::Black, Piece::Lion),
                (idx(5, 4), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        assert_eq!(s.terminal_status(), TerminalStatus::Draw);
    }

    #[test]
    fn test_bare_lion_not_a_draw_when_capture_available() {
        let s = custom(
            &[
                (idx(1, 3), Player::Black, Piece::Lion),
                (idx(5, 3), Player::White, Piece::Lion),
            ],
            Player::Black,
        );
        assert_eq!(s.terminal_status(), TerminalStatus::NotTerminal);
    }

    #[test]
    fn test_no_legal_moves_is_a_loss_not_a_draw() {
        // Constructed (not a naturally reachable) position purely to
        // exercise the "no stalemate: zero legal moves loses" rule: a Lion
        // sitting outside any castle has no castle-step moves, and with no
        // ranged capture available and no other pieces, Black has none at
        // all.
        let s = custom(
            &[
                (idx(3, 3), Player::Black, Piece::Lion),
                (idx(6, 1), Player::White, Piece::Lion),
                (idx(5, 0), Player::White, Piece::Pawn),
            ],
            Player::Black,
        );
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::White));
    }
}
