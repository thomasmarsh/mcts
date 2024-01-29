#![allow(dead_code)]

use crate::game::Game;
use crate::strategies::rollout::RolloutPolicy;
use crate::strategies::Strategy;
use crate::util::random_best;

use std::collections::HashMap;
use std::marker::PhantomData;

type Zobrist = u64; // hash
type PosHash<G> = (<G as Game>::M, Zobrist);

pub const WIN: i32 = i32::MAX;
pub const LOSS: i32 = -WIN;

#[derive(Clone)]
pub struct MCTSOptions {
    pub verbose: bool,
    pub max_rollout_depth: u32,
    pub rollouts_before_expanding: u32,
}

struct Node<G: Game> {
    children: Vec<PosHash<G>>,
    parents: Vec<Zobrist>,
    visits: u32,
    wins: u32,
    rave_visits: u32,
    rave_wins: u32,
}

impl<G: Game> Node<G> {
    fn new() -> Self {
        Self {
            children: Vec::new(),
            parents: Vec::new(),
            visits: 0,
            wins: 0,
            rave_visits: 0,
            rave_wins: 0,
        }
    }
}

struct Entry<G: Game> {
    node: Node<G>,
    state: G::S, // for collision check
}

impl<G: Game> Entry<G> {
    fn new(state: G::S) -> Self {
        Self {
            node: Node::new(),
            state,
        }
    }
}

// Transposition Table
struct Dag<G: Game>(HashMap<Zobrist, Entry<G>>);

impl<G: Game> Dag<G> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&self, pos_hash: &PosHash<G>) -> &Node<G> {
        // This unwrap shouldn't fail, because the hash must have been
        // inserted for us to have the hash in the first place.
        &self.0.get(&pos_hash.1).unwrap().node
    }

    fn add(&mut self, child_hash: u64, state: G::S) -> &mut Node<G>
    where
        <G as Game>::S: PartialEq,
    {
        &mut self
            .0
            .entry(child_hash)
            .and_modify(|entry| {
                if entry.state != state {
                    panic!("Collision detected: different state for the same key");
                }
                // Let's just keep the original data...
                // entry.node = Node::new();
            })
            .or_insert(Entry::new(state))
            .node
    }
}

const RAVE_EQUIV: f32 = 3500.;

/*
const
  NodePrior*: GoSint = 10
  ExpansionThreshold*: GoSint = 8 + NodePrior
  UctC*: float32 = 1.4
  RaveC*: float32 = 0
  RaveEquiv*: float32 = 3500
  MaxNbMoves* = 512
  PreallocatedSize* = 1 shl 17 # 2^16 is 65k elements*/

impl<G: Game> Node<G> {
    fn rave_urgency(&self) -> f32 {
        if self.rave_visits == 0 {
            return 0.;
        }

        let q = self.wins as f32;
        let n = self.visits as f32;
        let rave_q = self.rave_wins as f32;
        let rave_n = self.rave_visits as f32;

        let amaf = rave_q / rave_n;
        let beta = rave_n / (rave_n + n + (rave_n + n) / RAVE_EQUIV);
        (q / n) * (1. - beta) + beta * amaf
    }

    fn best_child(&self, dag: &Dag<G>) -> PosHash<G> {
        random_best(self.children.as_slice(), |pos_hash| {
            dag.get(pos_hash).rave_urgency()
        })
        // TODO: might fail; either logic bug, or weird race condition
        // when multithreading
        .unwrap()
        .clone()
    }

    fn best_move(&self, dag: &Dag<G>) -> PosHash<G> {
        // TODO: uct
        let mut max_visits: u32 = 0;
        let mut result = None;
        for child in &self.children {
            let visits = dag.get(child).visits;
            if visits > max_visits {
                result = Some(child);
                max_visits = visits;
            }
        }
        result.unwrap().clone()
    }

    fn expand(&mut self, dag: &mut Dag<G>, state: &G::S, parent_hash: Zobrist)
    where
        <G as Game>::S: Clone + PartialEq,
    {
        // parent = self
        // NOTE: would be nice to stream these rather than use the intermediate
        // vec, but an iterator requirement might be hard for some
        let mut moves = Vec::new();
        G::generate_moves(state, &mut moves);

        for m in moves {
            let mut state = state.clone();
            let tmp = G::apply(&mut state, m.clone()).unwrap();
            let child_hash = G::zobrist_hash(&tmp);
            dag.add(child_hash, tmp).parents.push(parent_hash);
            self.children.push((m, child_hash));

            // TODO: setting priors
        }
    }

    fn simulate(&self, node: &Node<G>, state: &mut G::S, mut for_rollout: bool) -> Option<i32> {
        //let winner = node.wins
        None
    }
}

/// A strategy that uses random playouts to explore the game tree to decide on the best move.
/// This can be used without an Evaluator, just using the rules of the game.
pub struct MonteCarloTreeSearch<G: Game> {
    options: MCTSOptions,
    max_rollouts: u32,
    // sqrt 2 is good for games with scores in range 0..1
    exploration: f32,
    // max_time: Duration,
    // timeout: Arc<AtomicBool>,
    rollout_policy: Option<Box<dyn RolloutPolicy<G = G> + Sync>>,
    pv: Vec<G::M>,
    game_type: PhantomData<G>,
}

impl<G: Game> MonteCarloTreeSearch<G> {
    pub fn new(options: MCTSOptions) -> Self {
        Self {
            options,
            max_rollouts: 0,
            exploration: 1.414,
            // max_time: Duration::from_secs(5),
            // timeout: Arc::new(AtomicBool::new(false)),
            rollout_policy: None,
            pv: Vec::new(),
            game_type: PhantomData,
        }
    }
}

impl<G: Game> Strategy<G> for MonteCarloTreeSearch<G> {
    fn choose_move(&mut self, _state: &<G as Game>::S) -> Option<<G as Game>::M> {
        unimplemented!()
    }

    fn set_timeout(&mut self, _timeout: std::time::Duration) {}

    fn set_max_depth(&mut self, _depth: u8) {}

    fn principal_variation(&self) -> Vec<<G as Game>::M> {
        unimplemented!()
    }
}
