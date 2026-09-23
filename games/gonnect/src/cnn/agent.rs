//! The play-time search agent: one batch-1 tree per move over the CNN oracle.
//!
//! All randomness inside a move (Gumbel noise, board orientations) is seeded from the position
//! and the agent's fixed seed, never from the game, so an agent is a deterministic function of
//! the position: two copies of the same agent play the two seats of a paired opening as exact
//! mirror images, and a self-match scores exactly one half.

use std::sync::Arc;

use grid_cnn::{Geometry, Net, Weights};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig};
use mcts::algorithms::mcts::node::{NodeState, QInit};
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};
use mcts::game::Game;
use mcts_batch::{explore, Config, Tree};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::encode::{action_id, analyse, move_from_id, num_actions, IN_PLANES};
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

/// A single-state adapter over [`GonnectOracle`] for `crates/mcts`'s `Evaluator`/`PolicyLogits`
/// contracts, which every ordinary (non-batched) `TreeSearch` consumes -- unlike `EnvOracle`,
/// which is shaped for `mcts_batch`'s flat-array multi-tree batching. `Arc`-wrapped so the same
/// oracle (and its orientation RNG stream) is shared by every clone `SearchConfig` needs.
#[derive(Clone)]
struct NetEvaluator<const N: usize>(Arc<GonnectOracle<N>>);

impl<const N: usize> Evaluator<SizedGonnect<N>> for NetEvaluator<N> {
    fn evaluate(&self, state: &SizedState<N>) -> Score {
        (self.0.single_value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

impl<const N: usize> PolicyLogits<SizedGonnect<N>> for NetEvaluator<N> {
    fn logits(&mut self, state: &SizedState<N>, actions: &[Move]) -> Vec<f64> {
        self.0.single_logits(state, actions)
    }
}

/// Only exists so `TreeSearch::default()` typechecks -- `.config(...)` always replaces this
/// placeholder (built over an otherwise-unused all-zero-weight net) before any real search
/// runs, exactly as `game_othello::convnet::CnnValueNet::default()` stands in for the same
/// structural reason.
impl<const N: usize> Default for NetEvaluator<N> {
    fn default() -> Self {
        let geometry = Geometry {
            size: N,
            in_planes: IN_PLANES,
            channels: 1,
            blocks: 1,
            policy_planes: 1,
            policy_out: num_actions(N),
            value_planes: 1,
            value_hidden: 1,
        };
        NetEvaluator(Arc::new(GonnectOracle::new(Net::new(&Weights::zeros(geometry)), 1, None)))
    }
}

type GumbelProfile<const N: usize> = Mcts<GumbelCompletedQ, EvaluatedCutoff<SizedGonnect<N>, NetEvaluator<N>>>;

pub struct CnnAgent<const N: usize> {
    name: String,
    oracle: Arc<GonnectOracle<N>>,
    cfg: Config,
    gcfg: GumbelConfig,
    kind: Kind,
    seed: u64,
    /// Only built and used for [`Kind::Gumbel`] -- [`Kind::Deterministic`] still runs its
    /// noise-free search through `mcts_batch::explore` (see `search`'s doc comment).
    search: TreeSearch<SizedGonnect<N>, GumbelProfile<N>>,
}

impl<const N: usize> CnnAgent<N> {
    pub fn new(
        name: &str,
        weights: &Arc<Weights>,
        cfg: Config,
        kind: Kind,
        chunk_size: usize,
        max_time: std::time::Duration,
        seed: u64,
    ) -> Self {
        let orientation_seed = (kind == Kind::Gumbel).then_some(seed);
        let oracle = Arc::new(GonnectOracle::new(Net::new(weights), chunk_size, orientation_seed));
        let gcfg = GumbelConfig {
            sims: cfg.num_simulations as u32,
            max_considered: cfg.num_considered_actions,
            c_visit: f64::from(cfg.max_visit_init),
            c_scale: f64::from(cfg.value_scale),
            ..GumbelConfig::default()
        };
        let net = NetEvaluator(oracle.clone());
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .select(GumbelCompletedQ::with_config(gcfg))
                .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                .with_policy_logits(net)
                .max_time(max_time)
                .seed(seed),
        );
        CnnAgent {
            name: name.to_string(),
            oracle,
            cfg,
            gcfg,
            kind,
            seed,
            search,
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
    /// [`Kind::Deterministic`] still runs through `mcts_batch`'s flat-array `explore`: it's a
    /// genuinely different root algorithm from [`Kind::Gumbel`] (deterministic candidate
    /// ranking with no Gumbel perturbation of the considered set at all, not just Gumbel search
    /// with less noise), so routing it through `crates/mcts`'s
    /// `RootMoveSelection::VisitCount` would inject randomness into root candidate selection
    /// that isn't there today. [`Kind::Gumbel`] runs through the ordinary
    /// `crates/mcts::TreeSearch`, so it honors `SearchConfig::max_iterations`/`max_time`
    /// alongside `Config::num_simulations`, and gains every other `TreeSearch` capability
    /// (select/backup strategy, opening book, etc.) for free.
    fn search(&mut self, state: &SizedState<N>) -> (u16, RootSummary) {
        let (_, legal) = analyse(&state.0);
        if legal.len() == 1 {
            let mv = legal[0];
            return (
                action_id(&mv, N),
                RootSummary {
                    simulations: 0,
                    actions: vec![RootAction { mv, visits: 0, q: 0.0 }],
                    principal_variation: vec![mv],
                    root_value: 0.0,
                },
            );
        }
        let key = SizedGonnect::<N>::zobrist_hash(state) ^ self.seed;
        self.oracle.reseed(key);
        match self.kind {
            Kind::Gumbel => {
                // Reseeded per move from the position, not carried over between moves, so the
                // agent stays a deterministic function of the position (see this module's doc
                // comment) rather than of its own search history.
                self.search.config.rng = SmallRng::seed_from_u64(key.rotate_left(29));
                let net = NetEvaluator(self.oracle.clone());
                let root_value = f64::from(net.evaluate(state)) / f64::from(EVAL_MAGNITUDE_LIMIT);
                let outcome =
                    gumbel_search_with_root_value(&mut self.search, state, &self.gcfg, root_value);
                let id = action_id(&outcome.action, N);
                (id, summarise_gumbel(&self.search, &outcome.completed_q, root_value as f32))
            }
            Kind::Deterministic => {
                let tree = explore(&self.cfg, self.oracle.as_ref(), std::slice::from_ref(state));
                let visits = tree.child_visits(0, tree.root());
                let most = *visits.iter().max().unwrap();
                let id = legal
                    .iter()
                    .map(|m| action_id(m, N))
                    .find(|&id| visits[id as usize] == most)
                    .unwrap();
                (id, summarise(&tree))
            }
        }
    }

    /// [`Search::choose_action`], also summarising the search that chose the move.
    pub fn choose_with_summary(&mut self, state: &SizedState<N>) -> (Move, RootSummary) {
        let (id, summary) = self.search(state);
        let mv = move_from_id(&state.0, id);
        (mv, summary)
    }
}

/// [`RootSummary`] for [`Kind::Gumbel`]'s `crates/mcts::TreeSearch` root. `completed_q` (from
/// the returned [`mcts::algorithms::mcts::gumbel::GumbelOutcome`]) already gives per-action Q in
/// the root mover's perspective; visit counts come straight from the root's own children, and
/// the principal variation is a direct most-visited-child walk of the searched tree.
fn summarise_gumbel<const N: usize>(
    search: &TreeSearch<SizedGonnect<N>, GumbelProfile<N>>,
    completed_q: &[(Move, f32)],
    root_value: f32,
) -> RootSummary {
    let root_id = search.root_id;
    let children = search.index.get(root_id).children();
    let q_of = |mv: Move| completed_q.iter().find(|&&(m, _)| m == mv).map_or(0.0, |&(_, q)| q);
    let mut actions: Vec<RootAction> = (0..children.len())
        .filter(|&i| children.num_visits(i) > 0)
        .map(|i| {
            let mv = children.action(i);
            RootAction {
                mv,
                visits: children.num_visits(i),
                q: q_of(mv),
            }
        })
        .collect();
    let id_of = |a: &RootAction| action_id(&a.mv, N);
    actions.sort_by(|a, b| b.visits.cmp(&a.visits).then(id_of(a).cmp(&id_of(b))));
    let simulations = (0..children.len()).map(|i| children.num_visits(i) as usize).sum();

    RootSummary {
        simulations,
        actions,
        principal_variation: gumbel_principal_variation(search, root_id),
        root_value,
    }
}

/// Most-visited-child walk from `root_id`, at most [`PV_LIMIT`] plies -- the
/// `crates/mcts::TreeSearch` analogue of `summarise`'s `mcts_batch::Tree` walk below.
fn gumbel_principal_variation<const N: usize>(
    search: &TreeSearch<SizedGonnect<N>, GumbelProfile<N>>,
    root_id: mcts::algorithms::mcts::index::Id,
) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut node_id = root_id;
    while pv.len() < PV_LIMIT {
        let Some(NodeState::Expanded(children)) = search.index.get(node_id).status() else {
            break;
        };
        let mut best: Option<(usize, u32)> = None;
        for idx in 0..children.len() {
            let v = children.num_visits(idx);
            if best.is_none_or(|(_, bv)| v > bv) {
                best = Some((idx, v));
            }
        }
        let Some((idx, visits)) = best else { break };
        if visits == 0 {
            break;
        }
        pv.push(children.action(idx));
        let Some(next_id) = children.node_id(idx) else { break };
        node_id = next_id;
    }
    pv
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
        let (id, _summary) = self.search(state);
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
        CnnAgent::<N>::new("test", &zero_net_weights::<N>(), cfg, kind, 16, std::time::Duration::default(), 0x51)
    }

    /// A large `num_simulations` (the common case once the tuner derives it from the general
    /// `max_iterations`/`MAX_ITER` default rather than a small per-preset count -- see
    /// `mcts_tune::search`'s `AlgorithmSpec::Net` arm) sizes Sequential Halving's first phase
    /// large too. A `max_time` deadline checked only between phases would then blow the deadline
    /// by however long that whole first phase takes; this asserts a short deadline actually cuts
    /// the search off within the phase, not just between phases.
    #[test]
    fn a_short_max_time_cuts_off_a_large_simulation_budget_within_the_first_phase() {
        let cfg = Config {
            num_simulations: 10_000,
            num_considered_actions: 16,
            value_scale: 0.1,
            max_visit_init: 50,
        };
        let mut agent = CnnAgent::<7>::new(
            "test",
            &zero_net_weights::<7>(),
            cfg,
            Kind::Gumbel,
            16,
            std::time::Duration::from_millis(50),
            0x51,
        );
        let start = std::time::Instant::now();
        let (_mv, summary) = agent.choose_with_summary(&SizedState::<7>::default());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "took {:?} against a 50ms deadline",
            start.elapsed()
        );
        // The first Sequential-Halving phase over 16 candidates and a 10,000-sim budget hands
        // out roughly 2,500 forced visits before the first phase boundary; a mid-phase cutoff
        // should land far short of that.
        assert!(summary.simulations < 1000, "ran {} simulations", summary.simulations);
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
