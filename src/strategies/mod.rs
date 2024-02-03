mod arena;
pub mod flat_mc;
pub mod mcts;
pub mod random;
pub mod sync_util;

use crate::game::Game;

pub trait Strategy<G: Game> {
    fn choose_move(&mut self, state: &G::S) -> Option<G::M>;

    fn set_verbose(&mut self) {}

    /// For strategies that can ponder indefinitely, set the timeout.
    /// This can be changed between calls to choose_move.
    fn set_timeout(&mut self, _timeout: std::time::Duration) {}

    /// Set the maximum depth to evaluate (instead of the timeout).
    /// This can be changed between calls to choose_move.
    fn set_max_depth(&mut self, _depth: u8) {}

    /// Set the maximum depth to evaluate (instead of the timeout).
    /// This can be changed between calls to choose_move.
    fn set_max_rollouts(&mut self, _max_rollouts: u32) {}

    /// From the last choose_move call, return the principal variation,
    /// i.e. the best sequence of moves for both players.
    fn principal_variation(&self) -> Vec<G::M> {
        Vec::new()
    }
}
