// Druid: http://cambolbro.com/games/druid/
//
// This game is hard for MCTS, and so probably a good benchmark.
//
// Implementation issues:
//
// - No tuning has been done yet.
// - MCTS-Solver might help in the more tactical situations
// - Board size is stored as a global const, but should be some game context
// - G::gen_moves can fail by producing an empty set when it has hit the ceiling
// - G::gen_moves and G::is_terminal are expensive
// - max_depth is helpful but I think reduces the quality of playouts
//
// When asked about MCTS issues, Cameron Browne (the game's designer) said the
// following. [Email correspondence, January 2013]
//
// > One approach is to use RAVE or other enhancements to improve the efficiency
// > of UCT, but as the paper shows even RAVE does not always work, and this could
// > take a lot of trial and error. Generally the better approach is to add some
// > heuristics to the playouts, to make each playout more realistic, i.e. more like
// > moves that people would actually make during a game. For example, adding forced
// > moves due to bridge intrusions solved the problem with Hex.
// >
// > Suitable heuristics for Druid might include:
// > 1. If the opponent's last move threatens to build on one of your pieces, make a
// >    blocking move with high probability.
// > 2. If the opponent's last move intrudes into one part of a fork virtually
// >    connecting two of your pieces, then make the corresponding fork move to save
// >    the connection with high probability.
// > 3. Make moves that threaten the opponent's best connection with high probability.
// > 4. Higher is better!
// >
// > Note that I say "with high probability" rather than applying that same move
// > every time, so there is still a bit of randomness in the playouts, otherwise
// > you could trick the AI into choosing the wrong move every time. Monte Carlo
// > search is all about playing the odds over large numbers of simulations, so
// > probabilistic approaches are generally best.
//
// When asked about an evaluation function for minimax, and difficultied on modeling
// connectedness, he said:
//
// > Do you mean the problem is that connections aren't permanent, i.e. they
// > can't be relied upon because they can be built over? If so, then a probabilistic
// > model might help: assign each adjacency a probability between 0 and 1 based
// > on how likely it is to survive. So if the opponent has no immediate chance of
// > breaking that connection in the next few moves its probability will be high (say
// > 0.95), but if the opponent can bridge over it next move then the probability
// > might be say 0.25, and if the opponent has a fork that guarantees them cutting
// > a connection regardless of what you do then its probability will be almost
// > 0 (maybe 0.05 to indicate that there still is a connection there, however
// > tenuous). Some connections might be guaranteed (probability 1) but proving this
// > could be a tricky problem in itself.
// >
// > Then when you have the probability for each adjacent step, the strength
// > of a connection from one side to the other is the product of the associated
// > probabilities for the steps along that path. This is the main difference between
// > Hex and Druid, apart from the hex/square topology: connections are permanent
// > (probability 1) in Hex but not in Druid.
// >
// > Another way to improve connection tests might be to identify virtual connections
// > (two nearby pieces that are not physically connected but which the opponent
// > can't block) and give then a high adjacency value, much like the good Hex
// > players count bridge connections and edge templates as "connected" for the sake
// > of their connectivity tests.
// >
// > [...]
// >
// > I'd start with the path probability mentioned above for an evaluation
// > function, i.e. fitness = your_best_path_prob / opponent's_best_path_prob.
// >
// > Then you could look at all of your best paths to connection and all of your
// > opponent's best paths to connection, and look for key cells that most of these
// > paths flow through.
// >
// > You could also incorporate some of the heuristics I mention above.
// >
// > As for UCT vs AB search, that's hard to say -- Druid is a difficult game!
// > But I've found that humans can't plan ahead reliably more than a few moves
// > due to the confusing 3D element, so perhaps a simple AB search could be quite
// > effective, assuming that your evaluation function is realistic.

use std::collections::VecDeque;

use rustc_hash::FxHashSet as HashSet;
use serde::{Deserialize, Serialize};

use crate::{
    game::{Game, PlayerIndex, TerminalStatus},
    zobrist::LazyZobristTable,
};

// NOTE: the standard game is 10x10 (and 9x9 for Trilith). Board size lives on
// `State` (see `Size::is_supported` below for the ceiling this is checked
// against) rather than here; this constant now only supplies the default
// size for `State::default()` / existing tests and demo binaries.
pub const DEFAULT_SIZE: Size = Size { w: 5, h: 5 };

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Hash, Eq)]
pub enum Player {
    Black,
    White,
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl Player {
    fn next(&mut self) {
        *self = match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Size {
    pub w: u8,
    pub h: u8,
}

impl Size {
    fn area(self) -> u16 {
        (self.w * self.h) as u16
    }

    /// Whether this size is safe to build a game on: big enough for a lintel
    /// to fit in either orientation, and small enough that the Zobrist hash
    /// (see `HASHES` below) can address every (position, color, height-bit)
    /// slot it needs without going out of bounds.
    pub fn is_supported(self) -> bool {
        if self.w < 3 || self.h < 3 {
            return false;
        }
        let area = self.area() as usize;
        area * 2 * zobrist_height_bits(self) <= HASHES_LEN
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Pos(pub u8, pub u8);

impl Pos {
    pub fn from(i: usize, size: Size) -> Pos {
        Pos(i as u8 % size.w, i as u8 / size.w)
    }

    pub fn index(self, width: u8) -> usize {
        (self.1 * width + self.0) as usize
    }

    fn adjacent(&self, size: Size) -> impl Iterator<Item = Pos> {
        let &Pos(x, y) = self;

        [(-1, 0), (1, 0), (0, -1), (0, 1)].into_iter().filter_map(move |(dx, dy)| {
            let nx = x as i8 + dx;
            let ny = y as i8 + dy;
            if (0..size.w as i8).contains(&nx) && (0..size.h as i8).contains(&ny) {
                Some(Pos(nx as u8, ny as u8))
            } else {
                None
            }
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

impl Orientation {
    fn delta(self) -> (u8, u8) {
        match self {
            Orientation::Horizontal => (1, 0),
            Orientation::Vertical => (0, 1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Piece {
    Sarsen,
    Lintel(Orientation),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize)]
pub struct Square {
    pub height: u16,
    pub piece: Option<Player>,
}

impl Square {
    fn matches(&self, color: Player) -> bool {
        self.piece.is_some_and(|p| p == color)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move(pub Piece, pub u8);

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize)]
pub struct Hand {
    pub sarsens: u8,
    pub lintels: u8,
}

impl Hand {
    fn new(size: Size) -> Hand {
        let n = size.w * size.h;
        // Trilith provides 48 sarsens and 20 lintels for a 9x9 board, which
        // is probably too few.
        //
        // Cameron Browne says:
        // > For a 10x10 board you'll need at least 100 cubes in
        // > total (enough to cover the board). A good distribution is 20x1 unit
        // > and 10x3 unit blocks per player.
        // >
        // > This will be sufficient for games that don't go on too long. If the
        // > games  get really involved, however, you'll run out of pieces in
        // > which case you  might:
        // > 1) Pick up a piece already on the board (provided that it's
        // >    reachable) and place it elsewhere, or
        // > 2) Use twice as many pieces :)
        // >
        // > If you're playing the "Druid's Walk" option each player will also
        // > require one pawn.
        //
        // For this game, for an NxM board we use N*M sarsens and half as
        // many lintels.
        Hand {
            sarsens: n * 2,
            lintels: n,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize)]
pub struct State {
    pub player: Player,
    pub board: Vec<Square>,
    pub hand_black: Hand,
    pub hand_white: Hand,
    pub size: Size,
}

// TODO:
//
// A move can be implemented as a u16 to support up to 128x128 board sizes:
//
// Move: u16
// - orientation: 1  bit
// - piece_type:  1  bit
// - location:    14 bits (up to 128 * 128 = 16384)
//
// State has some optimal packings depending on the board size. Note that
// above 9x9 the board state no longer fits in a 64 byte cache line. For
// purposes of board state packing, we have to assume a max height. We will
// take log2(N*M). For example, a 10x10 board would have a max height of 7.

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_SIZE)
    }
}

impl State {
    pub fn new(size: Size) -> Self {
        State {
            player: Player::Black,
            board: vec![
                Square {
                    height: 0,
                    piece: None,
                };
                size.area().into()
            ],
            hand_black: Hand::new(size),
            hand_white: Hand::new(size),
            size,
        }
    }

    pub fn at(&self, i: usize) -> Option<Player> {
        self.board[i].piece
    }

    pub fn current_hand(&self) -> &Hand {
        match self.player {
            Player::Black => &self.hand_black,
            Player::White => &self.hand_white,
        }
    }

    fn deplete(&mut self, piece: Piece) {
        match self.player {
            Player::Black => match piece {
                Piece::Sarsen => self.hand_black.sarsens -= 1,
                Piece::Lintel(_) => self.hand_black.lintels -= 1,
            },
            Player::White => match piece {
                Piece::Sarsen => self.hand_white.sarsens -= 1,
                Piece::Lintel(_) => self.hand_white.lintels -= 1,
            },
        }
    }

    pub fn moves(&self, moves: &mut Vec<Move>) {
        for i in 0..self.size.area() as usize {
            let Pos(x, y) = Pos::from(i, self.size);

            // Sarsen
            if self.current_hand().sarsens > 0 {
                if let Some(piece) = self.at(i) {
                    if self.player == piece {
                        moves.push(Move(Piece::Sarsen, i as u8));
                    }
                } else {
                    moves.push(Move(Piece::Sarsen, i as u8));
                }
            }

            // Lintel
            for orientation in [Orientation::Horizontal, Orientation::Vertical] {
                let (dx, dy) = orientation.delta();
                let c = [
                    Pos(x, y),
                    Pos(x + dx, y + dy),
                    Pos(x + dx + dx, y + dy + dy),
                ];
                if self.current_hand().lintels > 0 && c[2].0 < self.size.w && c[2].1 < self.size.h {
                    let h = c.map(|c| self.board[c.index(self.size.w)].height);
                    if h[0] == h[2] && h[1] <= h[0] {
                        if let Some(p0) = self.at(c[0].index(self.size.w)) {
                            if let Some(p2) = self.at(c[2].index(self.size.w)) {
                                let mut count = 0;
                                (p0 == self.player).then(|| count += 1);
                                (p2 == self.player).then(|| count += 1);
                                if let Some(p1) = self.at(c[1].index(self.size.w)) {
                                    if p1 == self.player && h[1] == h[0] {
                                        count += 1;
                                    }
                                }
                                if count == 2 {
                                    moves.push(Move(Piece::Lintel(orientation), i as u8));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Board cell indices touched by `m` -- 1 for a sarsen, 3 for a lintel.
    /// Depends only on `size`/`m`, not on current board contents, so it's
    /// safe to call before *or* after `apply` mutates the board (used by
    /// both `apply` itself and the incremental Zobrist update in
    /// `Game::apply`, which needs the same cells' pre- and post-move
    /// values). Unused slots beyond the returned length are filled with `0`,
    /// always a valid index.
    fn move_cells(&self, m: Move) -> ([usize; 3], usize) {
        match m.0 {
            Piece::Sarsen => ([m.1 as usize, 0, 0], 1),
            Piece::Lintel(orientation) => {
                let (dx, dy) = orientation.delta();
                let Pos(x, y) = Pos::from(m.1 as usize, self.size);
                let c = [
                    Pos(x, y),
                    Pos(x + dx, y + dy),
                    Pos(x + dx + dx, y + dy + dy),
                ];
                (c.map(|p| Pos::index(p, self.size.w)), 3)
            }
        }
    }

    pub fn apply(&mut self, m: Move) {
        self.deplete(m.0);
        let (cells, n) = self.move_cells(m);
        match m.0 {
            Piece::Sarsen => {
                let i = cells[0];
                let sq = &self.board[i];
                self.board[i] = Square {
                    height: sq.height + 1,
                    piece: Some(self.player),
                }
            }
            Piece::Lintel(_) => {
                let h = self.board[cells[0]].height + 1;
                cells[..n].iter().for_each(|&i| {
                    self.board[i] = Square {
                        height: h,
                        piece: Some(self.player),
                    }
                })
            }
        }
        self.player.next();
    }

    fn get_adjacent(&self, pos: Pos, seen: &HashSet<usize>, color: Player) -> Vec<usize> {
        pos.adjacent(self.size)
            .map(|x| Pos::index(x, self.size.w))
            .filter(|x| !seen.contains(x) && self.board[*x].matches(color))
            .collect()
    }

    fn bfs(
        &self,
        start: &Pos,
        goal: &HashSet<usize>,
        seen: &mut HashSet<usize>,
        color: Player,
    ) -> bool {
        if seen.contains(&start.index(self.size.w)) || !self.board[start.index(self.size.w)].matches(color) {
            return false;
        }

        let mut frontier = VecDeque::from(vec![start.index(self.size.w)]);

        while let Some(idx) = frontier.pop_front() {
            if goal.contains(&idx) {
                return true;
            }
            seen.insert(idx);

            frontier.extend(self.get_adjacent(Pos::from(idx, self.size), seen, color));
        }
        false
    }

    pub fn check_connection(&self, start: Vec<Pos>, end: Vec<Pos>, color: Player) -> bool {
        let goal = HashSet::from(end.into_iter().map(|x| Pos::index(x, self.size.w)).collect());
        let mut seen = HashSet::default();
        start
            .iter()
            .any(|pos| self.bfs(pos, &goal, &mut seen, color))
    }

    pub fn connection(&self) -> Option<Player> {
        let (top, bottom): (Vec<Pos>, Vec<Pos>) =
            (0..self.size.w).map(|x| (Pos(x, 0), Pos(x, self.size.h - 1))).unzip();
        if self.check_connection(top, bottom, Player::Black) {
            return Some(Player::Black);
        }

        let (left, right): (Vec<Pos>, Vec<Pos>) =
            (0..self.size.h).map(|y| (Pos(0, y), Pos(self.size.w - 1, y))).unzip();
        if self.check_connection(left, right, Player::White) {
            return Some(Player::White);
        }

        None
    }

    /// Shortest border-to-border path `color` still needs to build, counted
    /// in cells that aren't already `color`'s. Used only as a heuristic for
    /// non-terminal (depth-cutoff) playouts -- see `Druid::compute_utilities`
    /// -- so it deliberately approximates: it charges a flat cost of one per
    /// cell regardless of piece type (a lintel covers 3 cells per hand item,
    /// a sarsen covers one) and ignores height/legality entirely. A cell
    /// already owned by the *opponent* still costs only one, not infinity,
    /// since a lintel's legality only requires 2 of its 3 touched cells to
    /// already be the mover's color (`moves()` above), so the third can
    /// repaint an opponent's cell -- there's no such thing as a permanently
    /// blocked cell here.
    ///
    /// 0-1 BFS (a plain BFS `VecDeque`, front-pushing 0-cost relaxations and
    /// back-pushing 1-cost ones) rather than Dijkstra, since every edge cost
    /// is 0 or 1. Every cell has a finite cost (no impassable cells), so on
    /// a non-empty board this always finds a path -- `unwrap_or(u32::MAX)`
    /// is unreachable in practice, just a safe default.
    fn connect_distance(&self, color: Player) -> u32 {
        let cost = |i: usize| -> u32 {
            if self.board[i].matches(color) {
                0
            } else {
                1
            }
        };

        let area = self.size.area() as usize;
        let mut dist = vec![u32::MAX; area];
        let mut done = vec![false; area];
        let mut deque: VecDeque<usize> = VecDeque::new();

        let starts: Vec<Pos> = match color {
            Player::Black => (0..self.size.w).map(|x| Pos(x, 0)).collect(),
            Player::White => (0..self.size.h).map(|y| Pos(0, y)).collect(),
        };
        for pos in starts {
            let i = pos.index(self.size.w);
            let c = cost(i);
            if c < dist[i] {
                dist[i] = c;
                if c == 0 {
                    deque.push_front(i);
                } else {
                    deque.push_back(i);
                }
            }
        }

        while let Some(i) = deque.pop_front() {
            if done[i] {
                continue;
            }
            done[i] = true;
            let d = dist[i];
            for adj in Pos::from(i, self.size).adjacent(self.size) {
                let j = adj.index(self.size.w);
                if done[j] {
                    continue;
                }
                let step = cost(j);
                let nd = d + step;
                if nd < dist[j] {
                    dist[j] = nd;
                    if step == 0 {
                        deque.push_front(j);
                    } else {
                        deque.push_back(j);
                    }
                }
            }
        }

        let goals: Vec<Pos> = match color {
            Player::Black => (0..self.size.w).map(|x| Pos(x, self.size.h - 1)).collect(),
            Player::White => (0..self.size.h).map(|y| Pos(self.size.w - 1, y)).collect(),
        };
        goals
            .into_iter()
            .map(|pos| dist[pos.index(self.size.w)])
            .min()
            .unwrap_or(u32::MAX)
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let color_map = generate_map(self.size, |i| match self.board[i].piece {
            None => " .".into(),
            Some(Player::Black) => " X".into(),
            Some(Player::White) => " O".into(),
        });
        let height_map = generate_map(self.size, |i| match self.board[i].height {
            0 => " .".into(),
            n => format!(" {:x}", n),
        });

        // Combine color_map and height_map side by side
        writeln!(f)?;
        let color_lines: Vec<&str> = color_map.split('\n').collect();
        let height_lines: Vec<&str> = height_map.split('\n').collect();
        for (color_line, height_line) in color_lines.iter().zip(height_lines.iter()) {
            writeln!(f, "{}   {}", color_line, height_line,)?;
        }

        Ok(())
    }
}

fn generate_map<F>(size: Size, mut func: F) -> String
where
    F: FnMut(usize) -> String,
{
    let mut map = Vec::new();

    let column_labels = |map: &mut Vec<String>| {
        for c in ('A'..).take(size.w as usize) {
            map.push(format!(" {}", c));
        }
    };

    // Generate map
    map.push("   ".to_string());
    column_labels(&mut map);
    let mut row = size.h as usize;
    map.push(format!("   \n{:>3}", row));
    for i in 0..size.area() as usize {
        let c = func(i);
        map.push(c);
        if (i + 1) as u8 % size.w == 0 {
            map.push(format!(" {}", row));
            if row < 10 {
                map.push(" ".into());
            }
            row -= 1;
            if row != 0 {
                map.push(format!("\n{:>3}", row));
            }
        }
    }
    map.push("\n   ".into());
    column_labels(&mut map);
    map.push("   ".into());
    map.join("")
}

impl std::fmt::Display for HashedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// A naive Zobrist hash, will require a table of size:
//
//     size(N, M) = 2 * ceil(log2(N*M)) * (N*M + N*(M-1) + (N-1)*M)
//
// For a default 10x10 sized board that is 3920 entries. This, is not too high,
// but it is also not very efficient. In Druid, we only need to consider the
// top-down view. Occluded pieces do not need to contribute to the hash. The
// revised hash is better:
//
//     size(N, M) = 2 * N * M * bits_per_height(N, M)
//
// where `bits_per_height` is `ceil(log2(max_cell_height + 1))` -- see
// `zobrist_height_bits`/`max_cell_height`. A cell's height is bounded by a
// player's *hand* (`Hand::new` deals `N*M*2` sarsens), not by the board
// area, so for the standard 10x10 board that's 200 sarsens -> 8 bits ->
// size(10,10) = 2 * 100 * 8 = 1600. There is 8-way symmetry, but this is
// only useful in the early game.
//
// This bounds the largest board size we can support -- see `Size::is_supported`.
const HASHES_LEN: usize = 1600;
static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0xD401D);

/// Highest height a single cell can reach. A player can only raise a cell's
/// height with pieces from their own hand (repeated sarsens on one cell, or
/// lintels bridging out from it), and `Hand::new` hands out `n * 2` sarsens
/// per player -- so that's the ceiling for one cell, not the board area.
fn max_cell_height(size: Size) -> usize {
    Hand::new(size).sarsens as usize
}

// Number of bits used to encode a cell's height: each bit gets its own
// random table entry, XORed in when set, so a height in [0, 2^bits) maps to
// a distinct XOR combination (the entries are independent random u64s, so
// this is injective with overwhelming probability -- the standard trick for
// hashing bounded counters into a Zobrist scheme). `ceil(log2(n))` matches
// the sizing comment above, where `n` is the number of distinct heights a
// cell can take on (`max_cell_height(size) + 1`, since height ranges from 0
// up to and including the max).
fn zobrist_height_bits(size: Size) -> usize {
    let n = max_cell_height(size) + 1;
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// XOR contribution of a single cell to the board hash, for a given
/// (height, piece) at position `i`. Shared by the incremental update in
/// `Game::apply` and the from-scratch recompute used to validate it in
/// tests -- both need to agree on exactly which bits a cell contributes.
fn cell_zobrist(i: usize, height: u16, piece: Option<Player>, bits: usize) -> u64 {
    let h = height as usize;
    if h == 0 {
        return 0;
    }
    let c = piece.map(|p| p.to_index()).unwrap_or(0);
    let base = (i * 2 + c) * bits;
    (0..bits).fold(0, |hash, b| if h & (1 << b) != 0 { hash ^ HASHES.hash(base + b) } else { hash })
}

/// Full from-scratch board hash. `Game::apply` no longer uses this on the
/// hot path (see the incremental XOR-delta update there) -- kept for the
/// property test that checks the incremental update stays in sync with it.
#[cfg(test)]
fn recompute_hash(state: &State, bits: usize) -> u64 {
    state.board.iter().enumerate().fold(0, |hash, (i, square)| {
        hash ^ cell_zobrist(i, square.height, square.piece, bits)
    })
}

// A union-find over board cells, plus two virtual "border" nodes, used to
// answer `connection()` in ~O(1) instead of via a full BFS on every query.
// Deliberately *no* path compression: `find` needs to stay a pure `&self`
// read (union-by-rank alone keeps it at worst O(log n)) so `Connectivity`
// can answer queries from `Game::winner`/`terminal_status`, which only get
// `&State` -- see `Connectivity` below for why all the mutation happens in
// `Game::apply` instead.
#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet { parent: (0..n as u32).collect(), rank: vec![0; n] }
    }

    fn find(&self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] as usize != x {
            x = self.parent[x] as usize;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
    }

    fn connected(&self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    fn reset(&mut self) {
        for (i, p) in self.parent.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.rank.iter_mut().for_each(|r| *r = 0);
    }
}

/// Incremental replacement for `State::connection()`'s full BFS, maintained
/// alongside the Zobrist hash in `HashedState`. One union-find per color,
/// over board cells plus two virtual border nodes for that color's win axis
/// (Black: node `area` = top row, `area + 1` = bottom row; White: `area` =
/// left column, `area + 1` = right column) -- a color has won once its two
/// border nodes are in the same set.
///
/// Pieces never leave the board and height never decreases, but the piece
/// *color* on top of a cell isn't monotonic: a lintel's legality only
/// requires 2 of its 3 touched cells to already carry the mover's color
/// (`State::moves`), so the third can be a cell the opponent already built
/// on -- placing the lintel repaints all 3 touched cells to the mover's
/// color regardless, silently deleting a node from the losing color's
/// connectivity graph. A plain union-find can't retract a union for that, so
/// on a repaint like that we just rebuild the losing color's whole
/// union-find from the board (`rebuild`, O(area) -- the same cost
/// `connection()`'s BFS always paid, just now only on the moves that
/// actually need it instead of every ply). That rebuild has to happen
/// inside `Game::apply` (which has `&mut State`) rather than lazily on the
/// next query, because `Game::winner` only gets `&State`.
#[derive(Clone, Debug)]
struct Connectivity {
    black: DisjointSet,
    white: DisjointSet,
}

impl Connectivity {
    fn new(size: Size) -> Self {
        let n = size.area() as usize + 2;
        Connectivity { black: DisjointSet::new(n), white: DisjointSet::new(n) }
    }

    fn set_mut(&mut self, color: Player) -> &mut DisjointSet {
        match color {
            Player::Black => &mut self.black,
            Player::White => &mut self.white,
        }
    }

    fn set(&self, color: Player) -> &DisjointSet {
        match color {
            Player::Black => &self.black,
            Player::White => &self.white,
        }
    }

    /// Union cell `i` (already `color` in `board`) with its same-color
    /// neighbors and, if it's on `color`'s edge, that edge's border node.
    fn union_into(&mut self, size: Size, board: &[Square], i: usize, color: Player) {
        let area = size.area() as usize;
        let Pos(x, y) = Pos::from(i, size);
        let set = self.set_mut(color);
        match color {
            Player::Black => {
                if y == 0 {
                    set.union(i, area);
                }
                if y == size.h - 1 {
                    set.union(i, area + 1);
                }
            }
            Player::White => {
                if x == 0 {
                    set.union(i, area);
                }
                if x == size.w - 1 {
                    set.union(i, area + 1);
                }
            }
        }
        for adj in Pos(x, y).adjacent(size) {
            let j = adj.index(size.w);
            if board[j].matches(color) {
                self.set_mut(color).union(i, j);
            }
        }
    }

    /// Rebuild `color`'s union-find from scratch against the current board.
    fn rebuild(&mut self, size: Size, board: &[Square], color: Player) {
        self.set_mut(color).reset();
        for i in 0..size.area() as usize {
            if board[i].matches(color) {
                self.union_into(size, board, i, color);
            }
        }
    }

    /// Incorporate a move that just placed `mover`'s piece on `cells` (their
    /// post-move values already written into `board`), given each cell's
    /// pre-move `Square`.
    fn update(&mut self, size: Size, board: &[Square], cells: &[usize], old: &[Square], mover: Player) {
        let opponent = match mover {
            Player::Black => Player::White,
            Player::White => Player::Black,
        };
        if old.iter().any(|sq| sq.piece == Some(opponent)) {
            self.rebuild(size, board, opponent);
        }
        for &i in cells {
            self.union_into(size, board, i, mover);
        }
    }

    fn connected(&self, size: Size, color: Player) -> bool {
        let area = size.area() as usize;
        self.set(color).connected(area, area + 1)
    }

    fn winner(&self, size: Size) -> Option<Player> {
        if self.connected(size, Player::Black) {
            return Some(Player::Black);
        }
        if self.connected(size, Player::White) {
            return Some(Player::White);
        }
        None
    }
}

impl Default for Connectivity {
    fn default() -> Self {
        Connectivity::new(DEFAULT_SIZE)
    }
}

#[derive(Debug, Default, Clone)]
pub struct HashedState(State, u64, Connectivity);

// Deliberately excludes `Connectivity` (field 2): it's a pure cache derived
// from `(State, hash)` via `Game::apply`, but its internal union-find
// representation -- e.g. which node ends up as which set's root -- isn't
// canonical, so two logically-identical states reached via different move
// orders can carry different `Connectivity` bytes despite being equal.
// Comparing it would make this `PartialEq`/`Eq` impl unsound for the
// transposition-table dedupe check (`table.rs`'s `entry.state == state`)
// that relies on it.
impl PartialEq for HashedState {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for HashedState {}

impl HashedState {
    /// Panics if `size` isn't `Size::is_supported` -- callers that accept a
    /// size from outside this module (e.g. an API request) should check that
    /// first and reject unsupported sizes there instead of hitting this.
    pub fn new(size: Size) -> Self {
        assert!(size.is_supported(), "unsupported board size: {size:?}");
        // The all-zero hash is correct for any empty board under the scheme
        // below, regardless of size: no cell has a nonzero height yet, so no
        // bits get XORed in.
        HashedState(State::new(size), 0, Connectivity::new(size))
    }

    pub fn state(&self) -> &State {
        &self.0
    }

    /// Rebuild `Connectivity` from the current board. `Game::apply` is what
    /// normally keeps it in sync incrementally; this is only needed after
    /// mutating `.0.board` directly (bypassing `apply`), which only test
    /// code that hand-constructs a position should ever do.
    #[cfg(test)]
    fn resync_connectivity(&mut self) {
        self.2 = Connectivity::new(self.0.size);
        for color in [Player::Black, Player::White] {
            self.2.rebuild(self.0.size, &self.0.board, color);
        }
    }
}

#[derive(Clone)]
pub struct Druid;

impl Game for Druid {
    type S = HashedState;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &HashedState, actions: &mut Vec<Move>) {
        state.0.moves(actions);
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.1
    }

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {
        // Each (position, color) pair owns its own disjoint block of
        // `bits` table slots, so different cells/colors never collide;
        // within a block, each bit of the height is independently XORed
        // in (see `zobrist_height_bits`).
        let bits = zobrist_height_bits(state.0.size);
        debug_assert!(
            state.0.size.is_supported(),
            "HASHES table is too small for this board size; HashedState::new should have rejected it"
        );

        // A move only ever touches the 1 (sarsen) or 3 (lintel) cells
        // `move_cells` names -- snapshot their pre-move values so the hash
        // can be updated by XORing those cells' old contribution out and
        // their new contribution in, instead of recomputing the whole
        // board every ply.
        let (cells, n) = state.0.move_cells(*m);
        let old: [Square; 3] = std::array::from_fn(|i| state.0.board[cells[i]]);
        let mover = state.0.player;

        state.0.apply(*m);

        debug_assert!(
            state.0.board.iter().all(|square| (square.height as usize) < (1usize << bits)),
            "cell height exceeded the {bits}-bit Zobrist encoding for {:?}; max_cell_height's bound was wrong",
            state.0.size
        );

        let mut hash = state.1;
        for k in 0..n {
            let i = cells[k];
            hash ^= cell_zobrist(i, old[k].height, old[k].piece, bits);
            let sq = state.0.board[i];
            hash ^= cell_zobrist(i, sq.height, sq.piece, bits);
        }
        state.1 = hash;

        state.2.update(state.0.size, &state.0.board, &cells[..n], &old[..n], mover);

        state
    }

    fn is_terminal(state: &Self::S) -> bool {
        !matches!(Self::terminal_status(state), TerminalStatus::NotTerminal)
    }

    /// Single source of truth for both `is_terminal` and `winner`: both are
    /// answered by `Connectivity` (see above), so computing them separately
    /// (as the default `Game::terminal_status` does) means every caller that
    /// needs both -- e.g. the end of an MCTS rollout, which checks
    /// `is_terminal` to stop and then `winner`/`compute_utilities` to score
    /// it -- would otherwise redo the same connectivity read twice.
    /// Overriding this lets callers that go through `terminal_status` get
    /// both from one read; `is_terminal` and `winner` (below) still each do
    /// their own read when called alone, same as before.
    fn terminal_status(state: &Self::S) -> TerminalStatus<Player> {
        // Per the ruleset (http://cambolbro.com/games/druid/), the game is
        // won by completing a cross-board connection. That's the only real
        // win condition -- a depleted hand alone does *not* end the game,
        // since the other piece type may still have legal moves (that was
        // the bug: this used to trigger on either hand alone).
        if let Some(winner) = state.2.winner(state.0.size) {
            return TerminalStatus::Winner(winner);
        }

        // But the physical game's fallback for running out of pieces --
        // picking up and relocating a placed piece, or doubling the piece
        // count -- isn't implemented here, so this engine *can* reach a
        // true no-legal-moves state that the real game never would. Left
        // unterminated, that state feeds MCTS an empty action list (a
        // rollout crash) or lets a random playout burn its whole budget
        // re-stacking sarsens with no path to a connection. So: treat "no
        // legal moves" as a terminal draw, but only pay for the
        // `moves()` check once a hand is actually at zero for the mover --
        // that's the only situation where running dry is possible, so it's
        // a cheap, rare trigger rather than a call on every ply.
        let hand = state.0.current_hand();
        if hand.sarsens == 0 || hand.lintels == 0 {
            let mut actions = Vec::new();
            state.0.moves(&mut actions);
            if actions.is_empty() {
                return TerminalStatus::Draw;
            }
        }
        TerminalStatus::NotTerminal
    }

    fn notation(state: &Self::S, m: &Self::A) -> String {
        let Pos(x, y) = Pos::from(m.1 as usize, state.0.size);
        match m.0 {
            Piece::Sarsen => format!("S({},{})", x + 1, y + 1),
            Piece::Lintel(Orientation::Horizontal) => format!("L({},{},H)", x + 1, y + 1),
            Piece::Lintel(Orientation::Vertical) => format!("L({},{},V)", x + 1, y + 1),
        }
    }

    fn winner(state: &Self::S) -> Option<Player> {
        state.2.winner(state.0.size)
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.0.player
    }

    /// The default (`game.rs`) scores a non-terminal state as a flat 0. for
    /// both players. That default is only ever reached here via a playout
    /// hitting `max_playout_depth` before either side connects (a real
    /// winner is already handled by `terminal_status`/`trial.terminal` --
    /// see the backprop comment at
    /// `strategies/mcts/backprop.rs:95-103`, which only falls back to this
    /// function when there is genuinely nothing cached). Scoring every such
    /// cutoff as a draw throws away whatever progress either side has made
    /// -- this is the "max_depth ... reduces the quality of playouts" issue
    /// noted at the top of this file. Replace it with a cheap proxy for
    /// Cameron Browne's suggested fitness = your_best_path_prob /
    /// opponent's_best_path_prob: the difference in each color's shortest
    /// remaining border-to-border path (`State::connect_distance`),
    /// normalized to stay strictly inside (-1, 1) so it can never be
    /// confused with a real win/loss.
    fn compute_utilities(state: &Self::S) -> Vec<f64> {
        if let Some(winner) = Self::winner(state) {
            let wi = winner.to_index();
            return (0..Self::num_players()).map(|i| if i == wi { 1. } else { -1. }).collect();
        }

        // Neither color has connected (checked above), so both distances
        // are strictly positive: a distance of 0 would mean that color's
        // border-to-border path is already all their own cells, i.e. a
        // connection, which `Self::winner` would have already caught.
        let black_dist = state.0.connect_distance(Player::Black) as f64;
        let white_dist = state.0.connect_distance(Player::White) as f64;
        let black_score = (white_dist - black_dist) / (black_dist + white_dist);

        (0..Self::num_players())
            .map(|i| if i == Player::Black.to_index() { black_score } else { -black_score })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{
        mcts::{
            node::QInit,
            render::{self, NodeRender},
            strategy, SearchConfig, TreeSearch,
        },
        Search,
    };

    impl NodeRender for HashedState {}

    #[test]
    fn test_druid_render() {
        let mut search = TreeSearch::<Druid, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(20),
        );
        _ = search.choose_action(&HashedState::default());
        render::render(&search);
    }

    #[test]
    fn test_self_play_smoke_no_hash_collisions() {
        // A short self-play run with transpositions enabled, exercising the
        // real Zobrist hashing path end-to-end. With the corrected bit
        // width every state that lands in the same table bucket should
        // really be the same state -- so no bucket should ever need a
        // second `TableEntry` to disambiguate a collision.
        let mut search: TreeSearch<Druid, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(50),
        );

        let mut state = HashedState::default();
        for _ in 0..40 {
            if Druid::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);
            state = Druid::apply(state, &action);
        }

        for entries in search.table.table.0.values() {
            assert_eq!(
                entries.len(),
                1,
                "hash collision: {} distinct states shared one Zobrist hash",
                entries.len()
            );
        }
    }

    #[test]
    fn test_max_cell_height_matches_hand_sarsens() {
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            assert_eq!(max_cell_height(size), Hand::new(size).sarsens as usize);
        }
    }

    #[test]
    fn test_is_supported_accepts_default_and_common_sizes() {
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            assert!(size.is_supported(), "{size:?} should be supported under the corrected bit width");
        }
    }

    #[test]
    fn test_zobrist_height_encoding_is_injective_over_full_range() {
        // Confirms the per-cell height encoding is injective across the
        // *entire* representable range for a given bit width, not just the
        // heights a real game can reach -- the encoding scheme itself must
        // hold regardless of how the bound is derived.
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            let bits = zobrist_height_bits(size);
            // The bit width must be able to represent every height the game
            // can actually produce on one cell.
            assert!((1usize << bits) > max_cell_height(size));

            let mut seen = HashSet::default();
            for h in 0..(1usize << bits) {
                let mut hash = 0u64;
                for b in 0..bits {
                    if h & (1 << b) != 0 {
                        hash ^= HASHES.hash(b);
                    }
                }
                assert!(
                    seen.insert(hash),
                    "height {h} collided with an earlier height for size {size:?} (bits={bits})"
                );
            }
        }
    }

    #[test]
    fn test_zobrist_no_aliasing_past_old_area_sized_bit_width() {
        // Before the fix, `zobrist_height_bits` was sized off the board
        // area (25 for a 5x5 board), giving only `ceil(log2(25)) = 5` bits
        // -- a mod-32 ceiling. But a single hand can stack up to
        // `Hand::new(size).sarsens` (50) sarsens on one cell, so heights 1
        // and 33 used to alias. Drive a real cell up through `Game::apply`
        // and confirm every reachable height now hashes distinctly.
        let size = DEFAULT_SIZE;
        let cell = 0u8;
        let max_height = max_cell_height(size);
        assert_eq!(max_height, 50);

        let mut state = HashedState::new(size);
        let mut hashes_by_height = std::collections::HashMap::new();
        hashes_by_height.insert(0usize, state.1);

        for h in 1..=max_height {
            // Keep depleting the same hand so the cell keeps stacking
            // instead of running into the "only your own piece" rule that
            // real move generation would otherwise apply.
            state.0.player = Player::Black;
            state = Druid::apply(state, &Move(Piece::Sarsen, cell));
            assert_eq!(state.0.board[cell as usize].height, h as u16);
            hashes_by_height.insert(h, state.1);
        }

        let mut seen = HashSet::default();
        for (&h, &hash) in &hashes_by_height {
            assert!(seen.insert(hash), "height {h} collided with another reachable height's hash");
        }

        // The specific old (buggy) collision: height 1 vs height 1 + 32.
        let old_ceiling = 32usize;
        assert!(max_height >= 1 + old_ceiling);
        assert_ne!(
            hashes_by_height[&1],
            hashes_by_height[&(1 + old_ceiling)],
            "height 1 and height {} alias, matching the old area-sized bug",
            1 + old_ceiling
        );
    }

    #[test]
    fn test_is_terminal_false_when_one_hand_empty_but_other_can_move() {
        // The pre-fix bug: is_terminal ended the game as soon as *either*
        // hand (sarsens or lintels) hit zero, even if the other piece type
        // still had a legal move. Set up exactly that: no sarsens left in
        // hand, but a legal lintel exists anyway -- a lintel's support only
        // needs the *topmost* piece at each end cell to match the mover's
        // color, sarsen or not, so two Black-topped cells are enough.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        state.0.player = Player::Black;
        state.0.board[Pos(0, 0).index(size.w)] = Square { height: 1, piece: Some(Player::Black) };
        state.0.board[Pos(2, 0).index(size.w)] = Square { height: 1, piece: Some(Player::Black) };
        state.0.hand_black.sarsens = 0;
        state.0.hand_black.lintels = 1;
        // Poking `.0.board` directly bypasses `Game::apply`, which is what
        // normally keeps `Connectivity` in sync -- resync it so
        // `is_terminal`/`terminal_status` below read the position actually
        // set up here, not whatever `Connectivity` was at `new()`.
        state.resync_connectivity();

        assert!(state.0.connection().is_none());
        let mut actions = Vec::new();
        state.0.moves(&mut actions);
        assert!(!actions.is_empty(), "test setup should have produced a legal lintel move");

        assert!(
            !Druid::is_terminal(&state),
            "an empty sarsen hand must not end the game while a legal lintel move exists"
        );
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::NotTerminal,
            "terminal_status must agree with is_terminal"
        );
    }

    #[test]
    fn test_is_terminal_true_when_no_legal_moves_remain() {
        // Once *both* piece types are exhausted for the mover (and there's
        // no connection), there are no legal moves left -- this engine
        // doesn't implement the physical game's "pick up and relocate" or
        // "double the pieces" fallback, so treat that as a terminal draw
        // rather than feeding MCTS an empty action list.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        state.0.player = Player::Black;
        state.0.hand_black.sarsens = 0;
        state.0.hand_black.lintels = 0;

        assert!(state.0.connection().is_none());
        let mut actions = Vec::new();
        state.0.moves(&mut actions);
        assert!(actions.is_empty());

        assert!(Druid::is_terminal(&state), "no legal moves with no connection must be terminal");
        assert_eq!(Druid::winner(&state), None, "a no-legal-moves termination is a draw, not a win");
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::Draw,
            "terminal_status must agree with is_terminal/winner"
        );
    }

    #[test]
    fn test_incremental_hash_matches_full_recompute() {
        // `Game::apply` now updates the hash incrementally (XOR out the
        // touched cells' old contribution, XOR in the new) instead of
        // recomputing the whole board every ply. Confirm that stays
        // identical to a from-scratch recompute across many randomized
        // move sequences and board sizes, including games that run long
        // enough to restack cells past their original height.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let bits = zobrist_height_bits(size);
            let mut rng = SmallRng::seed_from_u64(size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = Druid::apply(state, &m);
                    actions.clear();

                    assert_eq!(
                        state.1,
                        recompute_hash(&state.0, bits),
                        "incremental hash diverged from full recompute at size={size:?} game={game} ply={ply}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_is_terminal_true_on_connection_regardless_of_hands() {
        // A completed connection ends the game even with plenty of pieces
        // left in hand.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        for x in 0..size.w {
            let i = Pos(x, 0).index(size.w);
            state.0.board[i] = Square { height: 1, piece: Some(Player::White) };
        }
        // See the comment on the equivalent call above: poking `.0.board`
        // directly bypasses the `Game::apply` path that normally keeps
        // `Connectivity` in sync.
        state.resync_connectivity();
        assert_eq!(state.0.connection(), Some(Player::White));
        assert!(Druid::is_terminal(&state), "a completed connection must be terminal");
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::Winner(Player::White),
            "terminal_status must agree with is_terminal/connection"
        );
    }

    #[test]
    fn test_connectivity_survives_a_move_that_flips_the_bridging_cell() {
        // `Connectivity`'s doc comment explains the subtlety this covers: a
        // lintel's legality only requires 2 of its 3 touched cells to
        // already match the mover's color, so a lintel can legally repaint
        // a cell the *opponent* owns -- silently deleting it from the
        // opponent's connectivity graph. A union-find that only ever unions
        // (never retracts) would keep reporting the old connection as
        // intact after that; this drives exactly that scenario through the
        // real `Game::apply` path and checks the incremental `winner()`
        // against a from-scratch `state.0.connection()` at every step.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        let col = 2u8;
        let mid = size.h / 2;

        let place = |state: HashedState, player: Player, pos: Pos| -> HashedState {
            let mut state = state;
            state.0.player = player;
            Druid::apply(state, &Move(Piece::Sarsen, pos.index(size.w) as u8))
        };

        // Black builds the column's top and bottom segments, leaving a gap
        // at the middle row -- not yet connected top-to-bottom.
        for y in [0, 1, size.h - 2, size.h - 1] {
            state = place(state, Player::Black, Pos(col, y));
        }
        assert_eq!(state.0.connection(), None, "gapped column must not be connected yet");
        assert_eq!(Druid::winner(&state), None, "incremental winner must agree");

        // Filling the gap connects the two segments into one continuous
        // top-to-bottom column: Black wins.
        state = place(state, Player::Black, Pos(col, mid));
        assert_eq!(state.0.connection(), Some(Player::Black), "filling the gap should complete the connection");
        assert_eq!(Druid::winner(&state), Some(Player::Black), "incremental winner must agree");

        // White builds sarsens flanking the bridge cell in the same row, at
        // the same height -- enough on their own (2 of 3 touched cells
        // already White) to legally place a horizontal lintel through the
        // bridge cell without it needing to match either White end.
        state = place(state, Player::White, Pos(col - 1, mid));
        state = place(state, Player::White, Pos(col + 1, mid));
        state.0.player = Player::White;
        state = Druid::apply(state, &Move(Piece::Lintel(Orientation::Horizontal), Pos(col - 1, mid).index(size.w) as u8));

        // The bridge cell is now White, splitting Black's column back into
        // two disconnected segments -- Black must no longer read as
        // connected, and the incremental path must agree with a from-scratch
        // recompute rather than keep reporting the connection that used to
        // exist through the now-repainted cell.
        assert_eq!(
            state.0.connection(),
            None,
            "the lintel should have broken Black's column by repainting the bridge cell"
        );
        assert_eq!(
            Druid::winner(&state),
            None,
            "incremental winner() must reflect the broken bridge, not the stale pre-flip connection"
        );
    }

    #[test]
    fn test_incremental_connectivity_matches_full_recompute() {
        // Randomized-game analogue of
        // `test_incremental_hash_matches_full_recompute`: after every move,
        // the incremental `Druid::winner` (backed by `Connectivity`) must
        // agree with a from-scratch `state.0.connection()` BFS.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng = SmallRng::seed_from_u64(0xC0117 + size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    if state.0.connection().is_some() {
                        break;
                    }
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = Druid::apply(state, &m);
                    actions.clear();

                    assert_eq!(
                        Druid::winner(&state),
                        state.0.connection(),
                        "incremental winner diverged from full recompute at size={size:?} game={game} ply={ply}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_utilities_still_scores_a_real_win_as_decisive() {
        // The heuristic branch must not shadow the real win/loss case: a
        // connected state still gets the exact +1./-1., not a value merely
        // close to it.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        for x in 0..size.w {
            let i = Pos(x, 0).index(size.w);
            state.0.board[i] = Square { height: 1, piece: Some(Player::White) };
        }
        state.resync_connectivity();
        assert_eq!(state.0.connection(), Some(Player::White));

        let utilities = Druid::compute_utilities(&state);
        assert_eq!(utilities[Player::White.to_index()], 1.);
        assert_eq!(utilities[Player::Black.to_index()], -1.);
    }

    #[test]
    fn test_compute_utilities_is_a_draw_on_the_symmetric_empty_board() {
        // On an empty square board, Black's top-bottom distance and White's
        // left-right distance are identical by symmetry, so the heuristic
        // should agree with the old flat-draw default here specifically.
        let state = HashedState::new(DEFAULT_SIZE);
        let utilities = Druid::compute_utilities(&state);
        assert_eq!(utilities, vec![0., 0.]);
    }

    #[test]
    fn test_compute_utilities_favors_the_color_closer_to_connecting() {
        // Black has built most of a top-to-bottom column (one cell short);
        // White hasn't built anything. A depth-cutoff playout landing here
        // should score this as good for Black, not a flat draw -- this is
        // the actual bug being fixed: the old default threw this signal
        // away entirely.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        let col = 2u8;
        for y in 0..size.h - 1 {
            let i = Pos(col, y).index(size.w);
            state.0.board[i] = Square { height: 1, piece: Some(Player::Black) };
        }
        state.resync_connectivity();
        assert_eq!(state.0.connection(), None, "one cell short of connecting");

        let utilities = Druid::compute_utilities(&state);
        let black = utilities[Player::Black.to_index()];
        let white = utilities[Player::White.to_index()];
        assert!(black > 0., "Black is one move from winning, should score above a draw: {black}");
        assert_eq!(black, -white, "zero-sum: the two utilities must be exact opposites");
        assert!(black < 1., "a non-terminal cutoff must never read as a real win");
    }

    #[test]
    fn test_connect_distance_zero_iff_connected() {
        let size = Size { w: 4, h: 4 };
        let mut state = HashedState::new(size);
        assert_eq!(
            state.0.connect_distance(Player::Black),
            size.h as u32,
            "empty board: every row costs 1, so the full height must be paid"
        );

        // Fill every row but the last: one cell short of a top-to-bottom
        // column.
        let col = 1u8;
        for y in 0..size.h - 1 {
            let i = Pos(col, y).index(size.w);
            state.0.board[i] = Square { height: 1, piece: Some(Player::Black) };
        }
        state.resync_connectivity();
        assert_eq!(state.0.connection(), None, "one cell short must not be connected yet");
        assert_eq!(
            state.0.connect_distance(Player::Black),
            1,
            "one cell short of a column should cost exactly 1"
        );

        // Fill the last cell: now a complete column, i.e. a win.
        let i = Pos(col, size.h - 1).index(size.w);
        state.0.board[i] = Square { height: 1, piece: Some(Player::Black) };
        state.resync_connectivity();
        assert_eq!(state.0.connection(), Some(Player::Black));
        assert_eq!(
            state.0.connect_distance(Player::Black),
            0,
            "a completed connection must cost exactly 0"
        );
    }
}
