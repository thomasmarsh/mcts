use crate::game::Game;

// A trivial game with one move
#[derive(Default, Clone, Debug)]
struct Unit(pub bool);

struct UnitGame;

impl Game for UnitGame {
    type S = Unit;
    type M = ();
    type P = ();

    fn apply(state: &Self::S, _m: Self::M) -> Self::S {
        assert!(!state.0);
        // assert!(m == ()); // always true
        Unit(true)
    }

    fn gen_moves(state: &Self::S) -> Vec<Self::M> {
        if state.0 {
            vec![]
        } else {
            vec![()]
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.0
    }

    fn get_reward(_: &Self::S, state: &Self::S) -> i32 {
        assert!(state.0);
        1
    }

    fn empty_move(_state: &Self::S) -> Self::M {}

    fn notation(_: &Self::S, _: &Self::M) -> String {
        "()".to_string()
    }

    fn winner(_: &Self::S) -> Option<Self::P> {
        Some(())
    }

    fn player_to_move(_: &Self::S) -> Self::P {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::mcts::TreeSearch;

    #[test]
    pub fn test_unit() {
        let mut search: TreeSearch<UnitGame> = TreeSearch::new();
        search.config.max_rollouts = 10;
        let state = Unit::default();
        let m1 = search.choose_move(&state);
        assert!(m1 == Some(()));
        #[allow(clippy::unit_arg)]
        let new_state = UnitGame::apply(&state, m1.unwrap());
        assert!(new_state.0);
        let m2 = search.choose_move(&new_state);
        assert!(m2.is_none());
    }
}
