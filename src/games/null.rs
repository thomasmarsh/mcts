use crate::game::{Game, PlayerIndex};
use serde::Serialize;

// A trivial game with no moves

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
enum Never {}

#[derive(Clone)]
struct NullGame;

struct Unit;

impl PlayerIndex for Unit {
    fn to_index(&self) -> usize {
        0
    }
}

impl Game for NullGame {
    type S = ();
    type A = Option<Never>;
    type P = Unit;

    fn apply(_: (), _: &Option<Never>) {
        unreachable!();
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {}

    fn is_terminal(_: &()) -> bool {
        true
    }

    fn notation(_: &(), _: &Option<Never>) -> String {
        unreachable!();
    }

    fn winner(_: &()) -> Option<Unit> {
        None
    }

    fn player_to_move(_: &()) -> Unit {
        Unit
    }
}
