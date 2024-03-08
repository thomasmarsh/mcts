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

impl Game for NullGame {
    type S = Unit;
    type A = Option<Never>;
    type K = ();

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

    fn winner(_: &Self::S) -> Option<PlayerIndex> {
        None
    }

    fn player_to_move(_: &Self::S) -> PlayerIndex {
        0.into()
    }
}
