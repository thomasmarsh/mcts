//! The play-time search agent: one batch-1 tree per move over the CNN oracle.
//!
//! All randomness inside a move (Gumbel noise, board orientations) is seeded from the position
//! and the agent's fixed seed, never from the game, so an agent is a deterministic function of
//! the position: two copies of the same agent play the two seats of a paired opening as exact
//! mirror images, and a self-match scores exactly one half.

use std::sync::Arc;

use grid_cnn::{Net, Weights};
use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_batch::{explore, gumbel_explore_with_noise, gumbel_selected_action, Config};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::encode::{action_id, analyse, move_from_id};
use super::oracle::GonnectOracle;
use crate::sized::{SizedGonnect, SizedState};
use crate::Move;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Gumbel root with Sequential Halving; plays the surviving action.
    Gumbel,
    /// Noise-free completed-Q search, identity orientation; plays the most visited action.
    Deterministic,
}

pub struct CnnAgent<const N: usize> {
    name: String,
    oracle: GonnectOracle<N>,
    cfg: Config,
    kind: Kind,
    seed: u64,
}

impl<const N: usize> CnnAgent<N> {
    pub fn new(
        name: &str,
        weights: &Arc<Weights>,
        cfg: Config,
        kind: Kind,
        chunk_size: usize,
        seed: u64,
    ) -> Self {
        let orientation_seed = (kind == Kind::Gumbel).then_some(seed);
        CnnAgent {
            name: name.to_string(),
            oracle: GonnectOracle::new(Net::new(weights), chunk_size, orientation_seed),
            cfg,
            kind,
            seed,
        }
    }
}

impl<const N: usize> Search for CnnAgent<N> {
    type G = SizedGonnect<N>;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &SizedState<N>) -> Move {
        let (_, legal) = analyse(&state.0);
        if legal.len() == 1 {
            return legal[0];
        }
        let key = SizedGonnect::<N>::zobrist_hash(state) ^ self.seed;
        self.oracle.reseed(key);
        let id = match self.kind {
            Kind::Gumbel => {
                let mut rng = SmallRng::seed_from_u64(key.rotate_left(29));
                let (tree, noise) = gumbel_explore_with_noise(
                    &self.cfg,
                    &self.oracle,
                    std::slice::from_ref(state),
                    &mut rng,
                );
                gumbel_selected_action(&self.cfg, &tree, 0, &noise[0])
            }
            Kind::Deterministic => {
                let tree = explore(&self.cfg, &self.oracle, std::slice::from_ref(state));
                let visits = tree.child_visits(0, tree.root());
                let most = *visits.iter().max().unwrap();
                legal
                    .iter()
                    .map(|m| action_id(m, N))
                    .find(|&id| visits[id as usize] == most)
                    .unwrap()
            }
        };
        move_from_id(&state.0, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Gonnect, Player, State};
    use bitboard::Dyn;
    use grid_cnn::Geometry;

    fn zero_net_weights() -> Arc<Weights> {
        let g = Geometry {
            size: 7,
            in_planes: 7,
            channels: 8,
            blocks: 1,
            policy_planes: 2,
            policy_out: 51,
            value_planes: 1,
            value_hidden: 8,
        };
        Arc::new(Weights::zeros(g))
    }

    fn board(cells: &[usize]) -> Bits {
        let mut b = Bits::new(Dyn(7), Dyn(7));
        for &c in cells {
            b.set_index(c);
        }
        b
    }

    /// With a flat prior a search only finds a win by trying every root action, so these tests
    /// use more simulations than actions and consider them all.
    ///
    /// `mover` (to move) has a column of six stones at column 3 (rows 0..=5); the opponent has six
    /// far-away stones. The cell 45 (row 6, column 3) completes the mover's connection between the
    /// top and bottom edges.
    fn one_move_from_connection(mover: Player) -> SizedState<7> {
        let column = board(&[3, 10, 17, 24, 31, 38]);
        let other = board(&[0, 7, 14, 21, 28, 35]);
        let ones = !Bits::new(Dyn(7), Dyn(7));
        let (black, white) = if mover == Player::Black {
            (column, other)
        } else {
            (other, column)
        };
        SizedState(State::from_parts(
            black, white, ones, ones, mover, false, false,
        ))
    }

    fn agent(kind: Kind) -> CnnAgent<7> {
        let cfg = Config {
            num_simulations: 100,
            num_considered_actions: 51,
            value_scale: 0.1,
            max_visit_init: 50,
        };
        CnnAgent::<7>::new("test", &zero_net_weights(), cfg, kind, 16, 0x51)
    }

    #[test]
    fn a_search_takes_the_winning_connection_for_either_colour_and_kind() {
        for mover in [Player::Black, Player::White] {
            for kind in [Kind::Deterministic, Kind::Gumbel] {
                let state = one_move_from_connection(mover);
                let mv = agent(kind).choose_action(&state);
                assert_eq!(
                    mv.index(),
                    45,
                    "{mover:?} {kind:?} should complete the connection"
                );
                assert!(Gonnect::is_terminal(&Gonnect::apply(state.0, &mv)));
            }
        }
    }
}
