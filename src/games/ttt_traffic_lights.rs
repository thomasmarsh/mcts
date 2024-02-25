use crate::game::{Game, PlayerIndex};
use serde::Serialize;
use std::fmt::Display;

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    R,
    Y,
    G,
}

const BOARD_LEN: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Position {
    pub turn: Player,
    pub winner: bool,
    pub board: [Option<Piece>; BOARD_LEN],
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

fn next(current: Option<Piece>) -> Option<Piece> {
    match current {
        None => Some(Piece::R),
        Some(Piece::R) => Some(Piece::Y),
        Some(Piece::Y) => Some(Piece::G),
        Some(Piece::G) => None,
    }
}

impl Position {
    pub fn new() -> Self {
        Self {
            turn: Player::First,
            winner: false,
            board: [None; BOARD_LEN],
        }
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
            let ax = self.board[a];
            let bx = self.board[b];
            let cx = self.board[c];

            if ax.is_some() && ax == bx && bx == cx {
                return true;
            }
        }
        false
    }

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {
        for (i, piece) in self.board.into_iter().enumerate() {
            if piece != Some(Piece::G) {
                actions.push(Move(i as u8))
            }
        }
    }

    pub fn apply(&mut self, m: Move) {
        assert!(self.board[m.0 as usize] != Some(Piece::G));
        self.board[m.0 as usize] = next(self.board[m.0 as usize]);
        self.winner = self.has_winner();
        if !self.winner {
            self.turn = self.turn.next();
        }
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..9 {
            match self.board[i] {
                Some(Piece::R) => write!(f, " R")?,
                Some(Piece::Y) => write!(f, " Y")?,
                Some(Piece::G) => write!(f, " G")?,
                None => write!(f, " .")?,
            }
            if (i + 1) % 3 == 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TttTrafficLights;

impl Game for TttTrafficLights {
    type S = Position;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        state.gen_moves(actions);
    }

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        let mut tmp = state;
        tmp.apply(*m);
        tmp
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let x = m.0 % 3;
        let y = m.0 / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.winner
    }

    fn winner(state: &Self::S) -> Option<Player> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        Some(state.turn)
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.turn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::mcts::render;
    use crate::util::random_play;

    #[test]
    fn test_ttt_traffic_lights() {
        random_play::<TttTrafficLights>();
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

    impl render::NodeRender for Position {
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
                color_for(self.board[6]),
                color_for(self.board[7]),
                color_for(self.board[8]),
                color_for(self.board[3]),
                color_for(self.board[4]),
                color_for(self.board[5]),
                color_for(self.board[0]),
                color_for(self.board[1]),
                color_for(self.board[2]),
                width = 2
            )
        }
    }

    #[test]
    fn test_render() {
        use crate::strategies::mcts::{render, util, SearchConfig, TreeSearch};
        use crate::strategies::Search;
        let mut search = TreeSearch::<TttTrafficLights, util::Ucb1>::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .q_init(crate::strategies::mcts::node::UnvisitedValueEstimate::Draw)
                .max_iterations(500),
        );
        _ = search.choose_action(&Position::default());
        render::render(&search);
    }
}
