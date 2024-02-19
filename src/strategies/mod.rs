pub mod flat_mc;
pub mod mcts;
pub mod random;

use crate::game::Game;

pub trait Search: Sync + Send {
    type G: Game;

    fn friendly_name(&self) -> String;

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A;

    fn principle_variation(&self) -> Vec<<Self::G as Game>::A> {
        vec![]
    }

    fn estimated_depth(&self) -> usize {
        0
    }

    fn set_friendly_name(&mut self, name: &str);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerIndex;

    #[test]
    fn test_expand0() {
        use crate::games::ttt::*;
        type G = TicTacToe;
        let init_state = HashedPosition::new();

        for n in 0..3 {
            type TS = mcts::TreeSearch<G, mcts::util::Ucb1>;
            let mut ts = TS::default().config(
                mcts::SearchConfig::default()
                    .expand_threshold(n)
                    // NOTE: best_child will fail on final_action
                    // selection when we haven't expanded root.
                    .max_iterations(1 + n as usize),
            );

            ts.choose_action(&init_state);
            println!(
                "{n} [{}]: {:?}",
                ts.principle_variation().len(),
                ts.principle_variation()
            );
            if n == 0 {
                assert!(ts.principle_variation().len() > 1);
            } else {
                assert!(ts.principle_variation().len() == 1);
            }
        }
    }

    #[test]
    fn test_basics() {
        use crate::games::ttt::*;
        type G = TicTacToe;

        // Initial State
        // X O X
        // . O O
        // . X X
        // Turn: O
        //
        // for Move(3), score += 1
        // for Move(6), score += 0
        let init_state = HashedPosition {
            position: Position {
                turn: Piece::O,
                board: [
                    Some(Piece::X),
                    Some(Piece::O),
                    Some(Piece::X),
                    None,
                    Some(Piece::O),
                    Some(Piece::O),
                    None,
                    None,
                    // Some(Piece::X),
                    Some(Piece::X),
                ],
            },
            hash: 0,
        };

        // Configure new MCTS
        type TS = mcts::TreeSearch<G, mcts::util::ScalarAmaf>;
        let mut ts = TS::default().config(
            mcts::SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(100),
        );

        // Construct new root
        let root_id = ts.new_root();

        // Helper step function
        let step = |ts: &mut TS| {
            let mut ctx = mcts::SearchContext::new(root_id, init_state);
            ts.select(&mut ctx);
            let trial = ts.simulate(&ctx.state, G::player_to_move(&init_state).to_index());
            println!("trial actions: {:?}", trial.actions);
            println!("trial status: {:?}", trial.status);
            println!("utilites: {:?}", G::compute_utilities(&trial.state));
            println!(
                "relevant utility: {:?}",
                G::compute_utilities(&trial.state)[G::player_to_move(&init_state).to_index()]
            );
            ts.backprop(&mut ctx, trial, G::player_to_move(&init_state).to_index());

            ctx.current_id
        };

        // First pass: simulate over root node
        let child_id = step(&mut ts);

        assert_eq!(child_id, root_id);
        assert_eq!(ts.index.get(root_id).stats.num_visits, 1);

        // Second pass: expand child node
        let child_id = step(&mut ts);

        assert_ne!(child_id, root_id);
        assert_eq!(ts.index.get(root_id).stats.num_visits, 2);

        // Third pass: expand child node
        let _child_id = step(&mut ts);

        println!("{:#?}", ts.index);
    }
}
