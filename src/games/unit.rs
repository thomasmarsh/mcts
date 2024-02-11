use crate::game::{Game, PlayerIndex};

// A trivial game with one move
#[derive(Default, Clone, Debug)]
pub struct Unit(pub bool);

pub struct UnitGame;

pub struct Player;

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        0
    }
}

impl Game for UnitGame {
    type S = Unit;
    type A = ();
    type P = Player;

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

    fn winner(_: &Self::S) -> Option<Player> {
        Some(Player)
    }

    fn player_to_move(_: &Self::S) -> Player {
        Player
    }

    fn num_players() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::mcts::{util, TreeSearch};
    use crate::strategies::Search;

    #[test]
    pub fn test_unit() {
        let mut search: TreeSearch<UnitGame, util::Ucb1> = TreeSearch::default();
        search.strategy.max_iterations = 10;
        let state = Unit::default();
        search.choose_action(&state);
        #[allow(clippy::unit_arg)]
        let new_state = UnitGame::apply(state, &());
        assert!(new_state.0);
        assert!(UnitGame::is_terminal(&new_state));
    }
}
