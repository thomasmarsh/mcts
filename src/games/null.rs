use crate::game::Game;

// A trivial game with no moves

#[derive(Clone, Debug, PartialEq)]
enum Never {}

struct NullGame;

impl Game for NullGame {
    type S = ();
    type M = Option<Never>;
    type P = ();

    fn apply(_: &(), _: Option<Never>) {
        unreachable!();
    }

    fn gen_moves(_: &()) -> Vec<Option<Never>> {
        vec![]
    }

    fn is_terminal(_: &()) -> bool {
        true
    }

    fn get_reward(_: &(), _: &()) -> i32 {
        0
    }

    fn notation(_: &(), _: &Option<Never>) -> String {
        unreachable!();
    }

    fn winner(_: &()) -> Option<()> {
        None
    }

    fn player_to_move(_: &()) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::mcts::TreeSearch;

    #[test]
    pub fn test_null() {
        println!("test_null");
        let mut search: TreeSearch<NullGame> = TreeSearch::new();
        search.config.max_rollouts = 10;
        let m = search.choose_move(&());
        assert!(m.is_none());
    }
}
