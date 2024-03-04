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
use serde::Serialize;

use crate::{
    game::{Game, PlayerIndex},
    zobrist::LazyZobristTable,
};

// TODO: trait Game should be implemented with a self parameter or some
// other way to maintain static context so we don't have to store this here.
// NOTE: the standard game is 10x10 (and 9x9 for Trilith). This can be set up to
// 11x11 before you trigger integer overflows (unless expanding some of the types).
pub const SIZE: Size = Size { w: 5, h: 5 };

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Hash)]
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

#[derive(Clone, Copy, Debug)]
pub struct Size {
    w: u8,
    h: u8,
}

impl Size {
    fn area(self) -> u16 {
        (self.w * self.h) as u16
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Pos(pub u8, pub u8);

impl Pos {
    pub fn from(i: usize, size: Size) -> Pos {
        Pos(i as u8 % size.w, i as u8 / size.h)
    }

    pub fn index(self, width: u8) -> usize {
        (self.1 * width + self.0) as usize
    }

    fn adjacent(&self, size: Size) -> Vec<Pos> {
        let &Pos(x, y) = self;

        [(-1, 0), (1, 0), (0, -1), (0, 1)]
            .iter()
            .filter_map(|&(dx, dy)| {
                let nx = x as i8 + dx;
                let ny = y as i8 + dy;
                if (0..size.w as i8).contains(&nx) && (0..size.h as i8).contains(&ny) {
                    Some(Pos(nx as u8, ny as u8))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Piece {
    Sarsen,
    Lintel(Orientation),
}

#[derive(Clone, Copy, Debug, Hash)]
pub struct Square {
    pub height: u16,
    pub piece: Option<Player>,
}

impl Square {
    fn matches(&self, color: Player) -> bool {
        self.piece.is_some_and(|p| p == color)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub Piece, pub u8);

#[derive(Clone, Debug, Hash)]
pub struct Hand {
    pub sarsens: u8,
    pub lintels: u8,
}

impl Hand {
    fn new() -> Hand {
        let n = SIZE.w * SIZE.h;
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

#[derive(Clone, Debug, Hash)]
pub struct State {
    pub player: Player,
    pub board: Vec<Square>,
    pub hand_black: Hand,
    pub hand_white: Hand,
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
        Self::new()
    }
}

impl State {
    pub fn new() -> Self {
        State {
            player: Player::Black,
            board: vec![
                Square {
                    height: 0,
                    piece: None,
                };
                SIZE.area().into()
            ],
            hand_black: Hand::new(),
            hand_white: Hand::new(),
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
        for i in 0..SIZE.area() as usize {
            let Pos(x, y) = Pos::from(i, SIZE);

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
                if self.current_hand().lintels > 0 && c[2].0 < SIZE.w && c[2].1 < SIZE.h {
                    let h = c.map(|c| self.board[c.index(SIZE.w)].height);
                    if h[0] == h[2] && h[1] <= h[0] {
                        if let Some(p0) = self.at(c[0].index(SIZE.w)) {
                            if let Some(p2) = self.at(c[2].index(SIZE.w)) {
                                let mut count = 0;
                                (p0 == self.player).then(|| count += 1);
                                (p2 == self.player).then(|| count += 1);
                                if let Some(p1) = self.at(c[1].index(SIZE.w)) {
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

    pub fn apply(&mut self, m: Move) {
        self.deplete(m.0);
        match m.0 {
            Piece::Sarsen => {
                let sq = &self.board[m.1 as usize];
                self.board[m.1 as usize] = Square {
                    height: sq.height + 1,
                    piece: Some(self.player),
                }
            }
            Piece::Lintel(orientation) => {
                let (dx, dy) = orientation.delta();
                let Pos(x, y) = Pos::from(m.1 as usize, SIZE);
                let c = [
                    Pos(x, y),
                    Pos(x + dx, y + dy),
                    Pos(x + dx + dx, y + dy + dy),
                ];
                let is = c.map(|x| Pos::index(x, SIZE.w));
                let h = self.board[m.1 as usize].height + 1;
                is.iter().for_each(|i| {
                    self.board[*i] = Square {
                        height: h,
                        piece: Some(self.player),
                    }
                })
            }
        }
        self.player.next();
    }

    fn get_adjacent(&self, pos: Pos, seen: &HashSet<usize>, color: Player) -> Vec<usize> {
        pos.adjacent(SIZE)
            .into_iter()
            .map(|x| Pos::index(x, SIZE.w))
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
        if seen.contains(&start.index(SIZE.w)) || !self.board[start.index(SIZE.w)].matches(color) {
            return false;
        }

        let mut frontier = VecDeque::from(vec![start.index(SIZE.w)]);

        while let Some(idx) = frontier.pop_front() {
            if goal.contains(&idx) {
                return true;
            }
            seen.insert(idx);

            frontier.extend(self.get_adjacent(Pos::from(idx, SIZE), seen, color));
        }
        false
    }

    pub fn check_connection(&self, start: Vec<Pos>, end: Vec<Pos>, color: Player) -> bool {
        let goal = HashSet::from(end.into_iter().map(|x| Pos::index(x, SIZE.w)).collect());
        let mut seen = HashSet::default();
        start
            .iter()
            .any(|pos| self.bfs(pos, &goal, &mut seen, color))
    }

    pub fn connection(&self) -> Option<Player> {
        let (top, bottom): (Vec<Pos>, Vec<Pos>) =
            (0..SIZE.w).map(|x| (Pos(x, 0), Pos(x, SIZE.h - 1))).unzip();
        if self.check_connection(top, bottom, Player::Black) {
            return Some(Player::Black);
        }

        let (left, right): (Vec<Pos>, Vec<Pos>) =
            (0..SIZE.h).map(|y| (Pos(0, y), Pos(SIZE.w - 1, y))).unzip();
        if self.check_connection(left, right, Player::White) {
            return Some(Player::White);
        }

        None
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let color_map = generate_map(|i| match self.board[i].piece {
            None => " .".into(),
            Some(Player::Black) => " X".into(),
            Some(Player::White) => " O".into(),
        });
        let height_map = generate_map(|i| match self.board[i].height {
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

fn generate_map<F>(mut func: F) -> String
where
    F: FnMut(usize) -> String,
{
    let mut map = Vec::new();

    let column_labels = |map: &mut Vec<String>| {
        for c in ('A'..).take(SIZE.w as usize) {
            map.push(format!(" {}", c));
        }
    };

    // Generate map
    map.push("   ".to_string());
    column_labels(&mut map);
    let mut row = SIZE.h as usize;
    map.push(format!("   \n{:>3}", row));
    for i in 0..SIZE.area() as usize {
        let c = func(i);
        map.push(c);
        if (i + 1) as u8 % SIZE.w == 0 {
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
//     size(N, M) = 2 * ceil(log2(N*M)) * N * M
//
// Then size(10,10) = 1400. There is 8-way symmetry, but this is only useful
// in the early game.
static HASHES: LazyZobristTable<1400> = LazyZobristTable::new(0xD401D);

#[derive(Debug, Default, Clone)]
pub struct HashedState(State, u64);

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
        state.0.apply(*m);

        let mut hash = 0;
        state.0.board.iter().enumerate().for_each(|(i, square)| {
            let h = square.height;
            if h > 0 {
                let c = square.piece.map(|x| x as usize).unwrap_or(0);
                let index = i * (h as usize + 7 * c);
                hash ^= HASHES.hash(index);
            }
        });
        state.1 = hash;

        state
    }

    fn is_terminal(state: &Self::S) -> bool {
        // This is not quite right - should be "no moves", but that's too expensive
        // to calculate.
        state.0.current_hand().sarsens == 0
            || state.0.current_hand().lintels == 0
            || state.0.connection().is_some()
        // || Druid::gen_moves(state).is_empty()
    }

    fn notation(_: &Self::S, m: &Self::A) -> String {
        let Pos(x, y) = Pos::from(m.1 as usize, SIZE);
        match m.0 {
            Piece::Sarsen => format!("S({},{})", x + 1, y + 1),
            Piece::Lintel(Orientation::Horizontal) => format!("L({},{},H)", x + 1, y + 1),
            Piece::Lintel(Orientation::Vertical) => format!("L({},{},V)", x + 1, y + 1),
        }
    }

    fn winner(state: &Self::S) -> Option<Player> {
        state.0.connection()
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.0.player
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
}
