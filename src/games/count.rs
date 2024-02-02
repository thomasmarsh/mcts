use crate::game::Game;

#[derive(Clone, Debug)]
pub struct Count(pub i32);

#[derive(Clone, Debug)]
pub enum Move {
    Add,
    Sub,
}

pub struct CountingGame;

impl Game for CountingGame {
    type S = Count;
    type P = ();
    type M = Move;

    fn apply(state: &Self::S, m: Self::M) -> Self::S {
        Count(match m {
            Move::Add => state.0 + 1,
            Move::Sub => state.0 - 1,
        })
    }

    fn gen_moves(state: &Self::S) -> Vec<Self::M> {
        if Self::is_terminal(state) {
            vec![]
        } else {
            vec![Move::Add, Move::Sub]
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.0 == 100
    }

    fn empty_move(_: &Self::S) -> Self::M {
        Move::Add
    }

    fn notation(_: &Self::S, m: &Self::M) -> String {
        format!("{:?}", m).to_string()
    }

    fn winner(_: &Self::S) -> Option<Self::P> {
        Some(())
    }

    fn player_to_move(_: &Self::S) -> Self::P {}
}
