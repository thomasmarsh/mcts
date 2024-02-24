use crate::game::*;
use serde::Serialize;

#[derive(Clone, Debug, Default)]
pub struct Count(pub i32);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Move {
    Add,
    Sub,
}

pub struct Unit;

impl PlayerIndex for Unit {
    fn to_index(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct CountingGame;

impl Game for CountingGame {
    type S = Count;
    type A = Move;
    type P = Unit;

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        Count(match m {
            Move::Add => state.0 + 1,
            Move::Sub => state.0 - 1,
        })
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        if !Self::is_terminal(state) {
            actions.extend(vec![Move::Add, Move::Sub]);
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.0 == 10
    }

    fn notation(_: &Self::S, m: &Self::A) -> String {
        format!("{:?}", m).to_string()
    }

    fn winner(_: &Self::S) -> Option<Unit> {
        Some(Unit)
    }

    fn player_to_move(_: &Self::S) -> Unit {
        Unit
    }

    fn num_players() -> usize {
        1
    }
}
