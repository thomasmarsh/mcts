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
use mcts_batch::{explore, gumbel_explore_with_noise, gumbel_selected_action, Config, Tree};
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

/// What one search left at the root, in the terms the play UI shows.
#[derive(Clone, Debug)]
pub struct RootSummary {
    /// Visits the root received, i.e. the simulations that ran (`0` when the move was forced).
    pub simulations: usize,
    /// Legal root actions that were visited, most visited first (ties by action id), each with
    /// its completed Q from the mover's point of view in `[-1, 1]`.
    pub actions: Vec<RootAction>,
    /// The most-visited line from the root, at most [`PV_LIMIT`] plies.
    pub principal_variation: Vec<Move>,
    /// The mean value of the root from the mover's point of view, in `[-1, 1]`.
    pub root_value: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RootAction {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
}

pub const PV_LIMIT: usize = 8;

impl<const N: usize> CnnAgent<N> {
    /// `None` for the tree when the position has a single legal move (no search is run).
    fn search(&self, state: &SizedState<N>) -> (u16, Option<Tree<SizedState<N>>>) {
        let (_, legal) = analyse(&state.0);
        if legal.len() == 1 {
            return (action_id(&legal[0], N), None);
        }
        let key = SizedGonnect::<N>::zobrist_hash(state) ^ self.seed;
        self.oracle.reseed(key);
        match self.kind {
            Kind::Gumbel => {
                let mut rng = SmallRng::seed_from_u64(key.rotate_left(29));
                let (tree, noise) = gumbel_explore_with_noise(
                    &self.cfg,
                    &self.oracle,
                    std::slice::from_ref(state),
                    &mut rng,
                );
                let id = gumbel_selected_action(&self.cfg, &tree, 0, &noise[0]);
                (id, Some(tree))
            }
            Kind::Deterministic => {
                let tree = explore(&self.cfg, &self.oracle, std::slice::from_ref(state));
                let visits = tree.child_visits(0, tree.root());
                let most = *visits.iter().max().unwrap();
                let id = legal
                    .iter()
                    .map(|m| action_id(m, N))
                    .find(|&id| visits[id as usize] == most)
                    .unwrap();
                (id, Some(tree))
            }
        }
    }

    /// [`Search::choose_action`], also summarising the search tree that chose the move.
    pub fn choose_with_summary(&mut self, state: &SizedState<N>) -> (Move, RootSummary) {
        let (id, tree) = self.search(state);
        let mv = move_from_id(&state.0, id);
        let summary = match tree {
            Some(tree) => summarise(&tree),
            None => RootSummary {
                simulations: 0,
                actions: vec![RootAction {
                    mv,
                    visits: 0,
                    q: 0.0,
                }],
                principal_variation: vec![mv],
                root_value: 0.0,
            },
        };
        (mv, summary)
    }
}

fn summarise<const N: usize>(tree: &Tree<SizedState<N>>) -> RootSummary {
    let root = tree.root();
    let visits = tree.child_visits(0, root);
    let q = tree.completed_qvalues(0, root);
    let mut actions: Vec<RootAction> = (0..tree.num_actions())
        .filter(|&id| visits[id] > 0)
        .map(|id| RootAction {
            mv: move_from_id(&tree.state(0, root).0, id as u16),
            visits: visits[id] as u32,
            q: q[id],
        })
        .collect();
    let id_of = |a: &RootAction| action_id(&a.mv, N);
    actions.sort_by(|a, b| b.visits.cmp(&a.visits).then(id_of(a).cmp(&id_of(b))));

    let mut principal_variation = Vec::new();
    let mut node = root;
    while principal_variation.len() < PV_LIMIT && !tree.is_terminal(0, node) {
        let visits = tree.child_visits(0, node);
        let Some((id, &most)) = visits.iter().enumerate().max_by_key(|&(id, &v)| (v, -(id as i64)))
        else {
            break;
        };
        if most == 0 {
            break;
        }
        principal_variation.push(move_from_id(&tree.state(0, node).0, id as u16));
        node = tree.child(0, node, id as u16);
    }

    RootSummary {
        simulations: tree.num_visits(0, root).max(0) as usize,
        actions,
        principal_variation,
        root_value: tree.value(0, root),
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
        let (id, _tree) = self.search(state);
        move_from_id(&state.0, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Gonnect, Player, State};
    use bitboard::Dyn;
    use grid_cnn::Geometry;

    fn zero_net_weights<const N: usize>() -> Arc<Weights> {
        let g = Geometry {
            size: N,
            in_planes: 7,
            channels: 8,
            blocks: 1,
            policy_planes: 2,
            policy_out: N * N + 2,
            value_planes: 1,
            value_hidden: 8,
        };
        Arc::new(Weights::zeros(g))
    }

    fn board<const N: usize>(cells: &[usize]) -> Bits {
        let mut b = Bits::new(Dyn(N), Dyn(N));
        for &c in cells {
            b.set_index(c);
        }
        b
    }

    /// With a flat prior a search only finds a win by trying every root action, so these tests
    /// use more simulations than actions and consider them all.
    ///
    /// `mover` (to move) has a column of `N - 1` stones at column 1 (rows `0..N-1`); the opponent
    /// has `N - 1` far-away stones in column 0. The cell in the last row of column 1 completes
    /// the mover's connection between the top and bottom edges.
    fn one_move_from_connection<const N: usize>(mover: Player) -> (SizedState<N>, usize) {
        let column: Vec<usize> = (0..N - 1).map(|r| r * N + 1).collect();
        let other: Vec<usize> = (0..N - 1).map(|r| r * N).collect();
        let ones = !Bits::new(Dyn(N), Dyn(N));
        let (black, white) = if mover == Player::Black {
            (board::<N>(&column), board::<N>(&other))
        } else {
            (board::<N>(&other), board::<N>(&column))
        };
        let state = SizedState(State::from_parts(
            black, white, ones, ones, mover, false, false,
        ));
        (state, (N - 1) * N + 1)
    }

    fn agent<const N: usize>(kind: Kind, simulations: usize) -> CnnAgent<N> {
        let cfg = Config {
            num_simulations: simulations,
            num_considered_actions: N * N + 2,
            value_scale: 0.1,
            max_visit_init: 50,
        };
        CnnAgent::<N>::new("test", &zero_net_weights::<N>(), cfg, kind, 16, 0x51)
    }

    fn takes_the_winning_connection<const N: usize>(simulations: usize, kinds: &[Kind]) {
        for mover in [Player::Black, Player::White] {
            for &kind in kinds {
                let (state, winning) = one_move_from_connection::<N>(mover);
                let mv = agent::<N>(kind, simulations).choose_action(&state);
                assert_eq!(
                    mv.index() as usize,
                    winning,
                    "{N}x{N} {mover:?} {kind:?} should complete the connection"
                );
                assert!(Gonnect::is_terminal(&Gonnect::apply(state.0, &mv)));
            }
        }
    }

    #[test]
    fn a_search_takes_the_winning_connection_for_either_colour_and_kind() {
        takes_the_winning_connection::<7>(100, &[Kind::Deterministic, Kind::Gumbel]);
    }

    #[test]
    fn a_9x9_search_takes_the_winning_connection_too() {
        takes_the_winning_connection::<9>(200, &[Kind::Deterministic]);
    }
}
