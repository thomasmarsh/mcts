//! `battle_royale` must attribute a decisive game to the strategy that made
//! the winning move, regardless of whether the game advances the turn on that
//! move. Tic-tac-toe advances it (loser to move at a won terminal); Connect
//! Four does not (`turn` stays on the winner). A harness that compares
//! `winner()` to `player_to_move()` is only correct for the first convention
//! and silently inverts Connect Four results.

use mcts::algorithms::Search;
use mcts::game::Game;
use mcts::util::battle_royale;

/// Plays a fixed list of actions in order, ignoring the board state.
struct Scripted<G: Game> {
    moves: Vec<G::A>,
    next: usize,
    name: String,
}

impl<G: Game> Scripted<G> {
    fn new(moves: Vec<G::A>) -> Self {
        Self {
            moves,
            next: 0,
            name: "scripted".to_string(),
        }
    }
}

impl<G: Game> Search for Scripted<G>
where
    G::A: Clone,
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn choose_action(&mut self, _state: &G::S) -> G::A {
        let m = self.moves[self.next].clone();
        self.next += 1;
        m
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
}

mod connect4 {
    use super::*;
    use game_connect4::{Move, Standard};

    // First mover stacks column 0; the opponent cycles harmless columns and
    // never lines up four. The column-0 player drops their fourth disc and wins.
    fn stacker() -> Scripted<Standard> {
        Scripted::new(vec![Move(0), Move(0), Move(0), Move(0)])
    }
    fn cycler() -> Scripted<Standard> {
        Scripted::new(vec![Move(1), Move(2), Move(3), Move(1), Move(2)])
    }

    #[test]
    fn winner_is_first_strategy() {
        let mut s1 = stacker();
        let mut s2 = cycler();
        assert_eq!(battle_royale::<Standard, _, _>(&mut s1, &mut s2), Some(0));
    }

    #[test]
    fn winner_is_second_strategy() {
        // s1 moves first but only cycles; s2 stacks column 0 and wins on its
        // fourth move (ply 8).
        let mut s1 = cycler();
        let mut s2 = stacker();
        assert_eq!(battle_royale::<Standard, _, _>(&mut s1, &mut s2), Some(1));
    }
}

mod ttt {
    use super::*;
    use game_ttt::{Move, TicTacToe};

    #[test]
    fn winner_is_first_strategy() {
        // X (s1) takes the top row 0,1,2; O (s2) takes 3,4 and loses.
        let mut s1 = Scripted::<TicTacToe>::new(vec![Move(0), Move(1), Move(2)]);
        let mut s2 = Scripted::<TicTacToe>::new(vec![Move(3), Move(4)]);
        assert_eq!(battle_royale::<TicTacToe, _, _>(&mut s1, &mut s2), Some(0));
    }

    #[test]
    fn winner_is_second_strategy() {
        // X (s1) plays 6,7,8 order but O (s2) completes the middle row 3,4,5
        // first, on ply 6.
        let mut s1 = Scripted::<TicTacToe>::new(vec![Move(6), Move(7), Move(0)]);
        let mut s2 = Scripted::<TicTacToe>::new(vec![Move(3), Move(4), Move(5)]);
        assert_eq!(battle_royale::<TicTacToe, _, _>(&mut s1, &mut s2), Some(1));
    }
}
