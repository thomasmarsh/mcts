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
    fn test_parity() {
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
        //let init_state = HashedPosition::new();

        // Configure new MCTS
        type New = mcts::TreeSearch<G, mcts::util::ScalarAmaf>;
        let mut new: New = New::default().strategy(
            mcts::MctsStrategy::default()
                .playouts_before_expanding(1)
                .max_playout_depth(100),
        );

        // Construct new root
        let root_id_new = new.new_root();

        // Helper step function
        let step = |new: &mut New| {
            let mut ctx = mcts::SearchContext::new(root_id_new, init_state);
            new.select(&mut ctx);
            let trial = new.simulate(&ctx.state, G::player_to_move(&init_state).to_index());
            println!("trial actions: {:?}", trial.actions);
            println!("trial status: {:?}", trial.status);
            println!("utilites: {:?}", G::compute_utilities(&trial.state));
            println!(
                "relevant utility: {:?}",
                G::compute_utilities(&trial.state)[G::player_to_move(&init_state).to_index()]
            );
            new.backprop(&mut ctx, trial, G::player_to_move(&init_state).to_index());

            ctx.current_id
        };

        // First pass: simulate over root node
        let child_id = step(&mut new);

        assert_eq!(child_id, root_id_new);
        assert_eq!(new.index.get(root_id_new).stats.num_visits, 1);

        // Second pass: expand child node
        let child_id = step(&mut new);

        assert_ne!(child_id, root_id_new);
        assert_eq!(new.index.get(root_id_new).stats.num_visits, 2);

        // Third pass: expand child node
        let _child_id = step(&mut new);

        println!("{:#?}", new.index);
    }
}
