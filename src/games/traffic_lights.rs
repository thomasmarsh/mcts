use crate::{
    display::{RectangularBoard, RectangularBoardDisplay},
    game::{Game, PlayerIndex},
    zobrist::LazyZobristTable,
};
use serde::Serialize;
use std::fmt::Display;

const USE_SYMMETRY: bool = false;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Player {
    First,
    Second,
}

impl Player {
    fn next(&self) -> Player {
        match self {
            Player::First => Player::Second,
            Player::Second => Player::First,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    R,
    Y,
    G,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

impl Move {
    fn new(piece: Piece, index: usize) -> Self {
        Move(((index as u8) << 2) | piece as u8)
    }

    fn _piece(self) -> Piece {
        match self.0 & 0b11 {
            0b01 => Piece::R,
            0b10 => Piece::Y,
            0b11 => Piece::G,
            _ => unreachable!(),
        }
    }

    fn index(self) -> usize {
        (self.0 >> 2) as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Position {
    pub turn: Player,
    pub winner: bool,
    pub board: u32,
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

impl Position {
    pub fn new() -> Self {
        Self {
            turn: Player::First,
            winner: false,
            board: 0,
        }
    }

    fn get(&self, index: usize) -> Option<Piece> {
        match ((self.board as usize) >> (index * 2)) & 0b11 {
            0b00 => None,
            0b01 => Some(Piece::R),
            0b10 => Some(Piece::Y),
            0b11 => Some(Piece::G),
            _ => unreachable!(),
        }
    }

    fn incr(&mut self, index: usize) {
        debug_assert_ne!(self.get(index), Some(Piece::G));
        let current = (self.board >> (index * 2)) & 0b11;
        debug_assert_ne!(current, 0b11);
        let clear = !(0b11 << (index * 2));
        self.board = (self.board & clear) | ((current + 1) << (index * 2));
    }

    pub fn has_winner(&mut self) -> bool {
        let check = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ];

        for (a, b, c) in check {
            let ax = self.get(a);
            let bx = self.get(b);
            let cx = self.get(c);

            if ax.is_some() && ax == bx && bx == cx {
                return true;
            }
        }
        false
    }

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {
        (0..9).for_each(|i| match self.get(i) {
            Some(Piece::Y) => actions.push(Move::new(Piece::G, i)),
            Some(Piece::R) => actions.push(Move::new(Piece::Y, i)),
            None => actions.push(Move::new(Piece::R, i)),
            _ => (),
        });
    }

    pub fn apply(&mut self, m: Move) {
        assert!(self.get(m.index()) != Some(Piece::G));
        self.incr(m.index());
        self.winner = self.has_winner();
        if !self.winner {
            self.turn = self.turn.next();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// 9 playable positions * 4 states * 2 players
const NUM_MOVES: usize = 72;

static HASHES: LazyZobristTable<NUM_MOVES> = LazyZobristTable::new(0x4);

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct HashedPosition {
    pub position: Position,
    pub(crate) hashes: [u64; 8],
}

impl HashedPosition {
    pub fn new() -> Self {
        Self {
            position: Position::new(),
            hashes: [0; 8],
        }
    }
}

impl Default for HashedPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl HashedPosition {
    #[inline]
    fn apply(&mut self, m: Move) {
        use super::ttt::sym;
        use super::ttt::NUM_SYMMETRIES;
        if USE_SYMMETRY {
            let mut symmetries = [0; NUM_SYMMETRIES];
            sym::index_symmetries(m.index(), &mut symmetries);
            // TODO: self.hashes[0] is producing bad values. The `else` branch below is working.
            for (i, index) in symmetries.iter().enumerate() {
                let value = ((self.position.board as usize) >> (index * 2)) & 0b11;
                let q = (index << 3) | (value << 1) | self.position.turn as usize;
                self.hashes[i] ^= HASHES.hash(q);
            }
        } else {
            let index = m.index();
            let value = ((self.position.board as usize) >> (index * 2)) & 0b11;
            let q = (index << 3) | (value << 1) | self.position.turn as usize;
            self.hashes[0] ^= HASHES.hash(q);
        }
        self.position.apply(m);
    }

    #[inline(always)]
    fn hash(&self) -> u64 {
        if USE_SYMMETRY {
            use super::ttt::sym;
            self.hashes[sym::canonical_symmetry(self.position.board)]
        } else {
            self.hashes[0]
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

impl RectangularBoard for HashedPosition {
    const NUM_DISPLAY_ROWS: usize = 3;
    const NUM_DISPLAY_COLS: usize = 3;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let index = row * 3 + col;
        match self.position.get(index) {
            Some(Piece::R) => 'R',
            Some(Piece::Y) => 'Y',
            Some(Piece::G) => 'G',
            None => '.',
        }
    }
}

impl Display for HashedPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct TrafficLights;

impl Game for TrafficLights {
    type S = HashedPosition;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        state.position.gen_moves(actions);
    }

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        let mut tmp = state;
        tmp.apply(*m);
        tmp
    }

    fn get_reward(init: &Self::S, term: &Self::S) -> f64 {
        let utility = Self::compute_utilities(term)[Self::player_to_move(init).to_index()];
        if utility < 0. {
            return utility * 100.;
        }
        utility
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let i = m.index();
        let x = i % 3;
        let y = i / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.position.winner
    }

    fn winner(state: &Self::S) -> Option<Player> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        Some(state.position.turn)
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.position.turn
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::strategies::mcts::render;
    use crate::util::random_play;

    #[test]
    fn test_tl_rand() {
        random_play::<TrafficLights>();
    }

    #[test]
    fn test_tl_symmetries() {
        if USE_SYMMETRY {
            let mut unhashed = FxHashSet::default();
            let mut hashed = FxHashSet::default();

            let mut stack = vec![HashedPosition::new()];
            let mut actions = Vec::new();
            while let Some(state) = stack.pop() {
                let k = state.position.board;
                if !unhashed.contains(&k) {
                    unhashed.insert(k);
                    hashed.insert(state.hash());

                    if !TrafficLights::is_terminal(&state) {
                        actions.clear();
                        TrafficLights::generate_actions(&state, &mut actions);
                        actions.iter().for_each(|action| {
                            stack.push(TrafficLights::apply(state, action));
                        });
                    }
                }
            }

            println!("distinct: {}", unhashed.len());
            println!("distinct w/symmetry: {}", hashed.len());

            // There are 36 bits of state in the board, counting illegal moves,
            // over 68 billion states. Only 256,208 states are legal given terminal
            // states with wins. Taking into account the eight-way symmetry, we get
            // a reduction in state space, but only a small reduction to 244,129
            // distinct states.
            assert_eq!(unhashed.len(), 256208);
            assert_eq!(hashed.len(), 244129);
        }
    }

    fn color_for(piece: Option<Piece>) -> String {
        match piece {
            None => "white",
            Some(Piece::R) => "red",
            Some(Piece::Y) => "yellow",
            Some(Piece::G) => "green",
        }
        .into()
    }

    impl render::NodeRender for HashedPosition {
        fn preamble() -> String {
            "  node [shape=plaintext];".into()
        }

        fn render(&self) -> String {
            format!(
                " [label=<
        <TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\">
            <TR>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
            </TR>
            <TR>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
            </TR>
            <TR>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
                <TD BGCOLOR=\"{}\" WIDTH=\"{width}\" HEIGHT=\"{width}\"></TD>
            </TR>
        </TABLE>
           >]",
                color_for(self.position.get(6)),
                color_for(self.position.get(7)),
                color_for(self.position.get(8)),
                color_for(self.position.get(3)),
                color_for(self.position.get(4)),
                color_for(self.position.get(5)),
                color_for(self.position.get(0)),
                color_for(self.position.get(1)),
                color_for(self.position.get(2)),
                width = 6
            )
        }
    }

    #[test]
    fn test_tl_render() {
        if USE_SYMMETRY {
            use crate::strategies::mcts::{render, strategy, SearchConfig, TreeSearch};
            use crate::strategies::Search;
            let mut search = TreeSearch::<TrafficLights, strategy::Ucb1>::default().config(
                SearchConfig::default()
                    .expand_threshold(0)
                    // .q_init(crate::strategies::mcts::node::UnvisitedValueEstimate::Draw)
                    .max_iterations(100)
                    .use_transpositions(true),
            );
            _ = search.choose_action(&HashedPosition::default());
            assert!(search.table.hits > 0);

            render::render_trans(&search, &HashedPosition::default());
        }
    }

    #[test]
    fn test_grave() {
        use crate::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
        use crate::strategies::Search;
        type TS = TreeSearch<TrafficLights, strategy::RaveMastDm>;
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(30)
                .q_init(QInit::Infinity)
                .select(
                    select::Rave::default()
                        .threshold(2)
                        .schedule(select::RaveSchedule::MinMSE { bias: 10e-6 })
                        .ucb(select::RaveUcb::Ucb1Tuned {
                            exploration_constant: 2f64.sqrt(),
                        }),
                )
                .use_transpositions(true)
                .seed(0),
        );
        let state = HashedPosition::default();
        _ = ts.choose_action(&state);
        println!("hits: {}", ts.table.hits);
        println!("table:");
        println!("{:#?}", ts.table);
        println!("grave:");
        println!("{:#?}", ts.stats.grave);

        assert!(ts.table.hits > 0);
        // render::render_trans(&ts);
    }
}
