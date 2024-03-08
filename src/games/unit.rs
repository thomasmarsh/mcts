use crate::game::{Game, PlayerIndex};

// A trivial game with one move
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Unit(pub bool);

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "()")
    }
}

#[derive(Clone)]
pub struct UnitGame;

pub struct Player;

impl Game for UnitGame {
    type S = Unit;
    type A = ();
    type K = ();

    fn apply(state: Self::S, _m: &Self::A) -> Self::S {
        assert!(!state.0);
        // assert!(m == ()); // always true
        Unit(true)
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        if !state.0 {
            actions.push(());
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.0
    }

    fn notation(_: &Self::S, _: &Self::A) -> String {
        "()".to_string()
    }

    fn winner(_: &Self::S) -> Option<PlayerIndex> {
        Some(0.into())
    }

    fn player_to_move(_: &Self::S) -> PlayerIndex {
        0.into()
    }

    fn num_players() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::mcts::{strategy, TreeSearch};
    use crate::strategies::Search;

    #[test]
    pub fn test_unit() {
        let mut search: TreeSearch<UnitGame, strategy::Ucb1> = TreeSearch::default();
        search.config.max_iterations = 10;
        let state = Unit::default();
        search.choose_action(&state);
        #[allow(clippy::unit_arg)]
        let new_state = UnitGame::apply(state, &());
        assert!(new_state.0);
        assert!(UnitGame::is_terminal(&new_state));
    }
}
