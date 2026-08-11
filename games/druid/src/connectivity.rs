//! Incremental connectivity (`Connection`/`winner`) for a Druid position:
//! one union-find per color over the board cells plus two virtual border
//! nodes, maintained alongside the Zobrist hash in `HashedState`.

use crate::types::{Player, Pos, Size, Square};

// A union-find over board cells, plus two virtual "border" nodes, used to
// answer `connection()` in ~O(1) instead of via a full BFS on every query.
// Deliberately *no* path compression: `find` needs to stay a pure `&self`
// read (union-by-rank alone keeps it at worst O(log n)) so `Connectivity`
// can answer queries from `Game::winner`/`terminal_status`, which only get
// `&State` -- see `Connectivity` below for why all the mutation happens in
// `Game::apply` instead.
#[derive(Clone, Debug)]
pub(crate) struct DisjointSet {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    pub(crate) fn find(&self, x: usize) -> usize {
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
pub(crate) struct Connectivity {
    black: DisjointSet,
    white: DisjointSet,
}

impl Connectivity {
    pub(crate) fn new(size: Size) -> Self {
        let n = size.area() as usize + 2;
        Connectivity {
            black: DisjointSet::new(n),
            white: DisjointSet::new(n),
        }
    }

    fn set_mut(&mut self, color: Player) -> &mut DisjointSet {
        match color {
            Player::Black => &mut self.black,
            Player::White => &mut self.white,
        }
    }

    pub(crate) fn set(&self, color: Player) -> &DisjointSet {
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
    pub(crate) fn rebuild(&mut self, size: Size, board: &[Square], color: Player) {
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
    pub(crate) fn update(
        &mut self,
        size: Size,
        board: &[Square],
        cells: &[usize],
        old: &[Square],
        mover: Player,
    ) {
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

    pub(crate) fn winner(&self, size: Size) -> Option<Player> {
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
        Connectivity::new(crate::DEFAULT_SIZE)
    }
}