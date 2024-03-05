use crate::game::{Game, PlayerIndex};
use serde::Serialize;

// A trivial game with no moves

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
enum Never {}

#[derive(Clone)]
struct NullGame;

#[derive(PartialEq, Eq, Debug, Default, Clone, Copy)]
struct Unit;

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "()")
    }
}

impl PlayerIndex for Unit {
    fn to_index(&self) -> usize {
        0
    }
}

impl Game for NullGame {
    type S = Unit;
    type A = Option<Never>;
    type P = Unit;

    fn apply(_: Self::S, _: &Option<Never>) -> Self::S {
        unreachable!();
    }

    fn generate_actions(_: &Self::S, _: &mut Vec<Self::A>) {}

    fn is_terminal(_: &Self::S) -> bool {
        true
    }

    fn notation(_: &Self::S, _: &Option<Never>) -> String {
        unreachable!();
    }

    fn winner(_: &Self::S) -> Option<Unit> {
        None
    }

    fn player_to_move(_: &Self::S) -> Unit {
        Unit
    }
}
