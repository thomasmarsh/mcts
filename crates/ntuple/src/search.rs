//! Play-time PUCT search over an n-tuple value model, as in Scheiermann and
//! Konen (arXiv 2204.13307): the trained agent is wrapped in a tree search only
//! when it plays, never during training.
//!
//! Every expanded node evaluates all of its successors once. That single pass
//! gives both the softmax prior over the node's actions and the value of each
//! child (the successor's exact result if the game ends there, else the model's
//! value), so a leaf costs one evaluation per legal action and nothing more.
//! Values are stored from the point of view of the player to move at the node
//! that owns the edge, and a value crossing an edge is negated exactly when the
//! player to move changes (passes are ordinary actions, so a pass edge keeps the
//! game's own turn bookkeeping rather than an assumed alternation).
//!
//! The tree persists between `choose_action` calls: when the next query state
//! is found inside the retained tree, the subtree below it is kept (and
//! everything else dropped); otherwise the search starts from a fresh root.
//! `iterations` counts new iterations per call, so a reused tree adds its
//! retained visits on top of the same compute budget.

use std::sync::Arc;

use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use serde::Deserialize;

use crate::geometry::CellFeatures;
use crate::selfplay::terminal_value;
use crate::weights::Model;

type State<F> = <<F as CellFeatures>::G as Game>::S;
type Action<F> = <<F as CellFeatures>::G as Game>::A;

/// How far below the last searched root a new query state is looked for: our
/// move, the reply, and room for forced passes in between.
const REUSE_DEPTH: usize = 8;
const NONE: u32 = u32::MAX;

/// Search settings. Every field is required: they live in a TOML file next to
/// the trainer's, never in code.
#[derive(Deserialize, Clone, Debug)]
pub struct PuctConfig {
    /// New search iterations per move.
    pub iterations: u32,
    /// Exploration constant in `Q + c_puct * P * sqrt(N) / (1 + n)`.
    pub c_puct: f32,
    /// Softmax temperature over the 1-ply successor values (values live in
    /// (-1, 1)): `P = softmax(value / prior_temperature)`.
    pub prior_temperature: f32,
    /// A node with at most this many empty cells is replaced by the game's exact
    /// result (see `CellFeatures::exact_value`) the first time the search
    /// reaches it. 0 switches the solver off. Priors are unaffected.
    pub empties_exact: u32,
}

struct Node<S, A> {
    state: S,
    /// Player to move at `state`.
    mover: usize,
    terminal: bool,
    /// The exact result for `mover` when the node was solved on creation. A
    /// solved node is a leaf: it has no actions and is never expanded.
    solved: Option<f32>,
    actions: Vec<A>,
    /// Value of each successor from this node's mover's point of view.
    edge_values: Vec<f32>,
    edge_terminal: Vec<bool>,
    priors: Vec<f32>,
    children: Vec<u32>,
    visits: Vec<u32>,
    /// Sum of backed-up values per edge, from this node's mover's view.
    sums: Vec<f64>,
    total: u32,
}

/// A PUCT searcher over the model's values: a [`Search`] that any harness can
/// gate. Fully deterministic (ties fall to the earlier action).
pub struct PuctPlayer<F: CellFeatures> {
    feats: F,
    model: Arc<Model>,
    cfg: PuctConfig,
    name: String,
    nodes: Vec<Node<State<F>, Action<F>>>,
    codes: Vec<u8>,
    actions: Vec<Action<F>>,
}

impl<F: CellFeatures> PuctPlayer<F> {
    pub fn new(feats: F, model: Arc<Model>, cfg: PuctConfig) -> PuctPlayer<F> {
        assert_eq!(F::G::num_players(), 2, "the search handles two-player games");
        assert!(cfg.iterations > 0, "iterations must be positive");
        assert!(cfg.prior_temperature > 0.0, "prior_temperature must be positive");
        let codes = vec![0u8; feats.num_cells()];
        PuctPlayer {
            feats,
            model,
            cfg,
            name: "ntuple-puct".to_string(),
            nodes: Vec::new(),
            codes,
            actions: Vec::new(),
        }
    }

    pub fn config(&self) -> &PuctConfig {
        &self.cfg
    }

    /// Nodes held by the retained tree.
    pub fn tree_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Visits at the retained root (0 with no tree).
    pub fn root_visits(&self) -> u32 {
        self.nodes.first().map_or(0, |n| n.total)
    }

    /// The retained root's actions with their visit counts and mean values from
    /// the root mover's point of view.
    pub fn root_edges(&self) -> Vec<(Action<F>, u32, f64)> {
        match self.nodes.first() {
            None => Vec::new(),
            Some(n) => (0..n.actions.len())
                .map(|e| {
                    let q = if n.visits[e] > 0 { n.sums[e] / n.visits[e] as f64 } else { 0.0 };
                    (n.actions[e].clone(), n.visits[e], q)
                })
                .collect(),
        }
    }

    /// `solve` lets the node be replaced by its exact result when it is within
    /// `empties_exact`; the search root is always expanded instead, since it
    /// needs its actions.
    fn new_node(&mut self, state: State<F>, terminal: bool, solve: bool) -> u32 {
        let mover = F::G::player_to_move(&state).to_index();
        let solved = if solve && !terminal && self.cfg.empties_exact > 0 {
            self.feats.exact_value(&state, self.cfg.empties_exact)
        } else {
            None
        };
        let mut node = Node {
            state,
            mover,
            terminal,
            solved,
            actions: Vec::new(),
            edge_values: Vec::new(),
            edge_terminal: Vec::new(),
            priors: Vec::new(),
            children: Vec::new(),
            visits: Vec::new(),
            sums: Vec::new(),
            total: 0,
        };
        if !terminal && solved.is_none() {
            F::G::generate_actions(&node.state, &mut node.actions);
            assert!(!node.actions.is_empty(), "a non-terminal state must have an action");
            let spc = self.model.geometry().states_per_cell();
            for a in &node.actions {
                let next = F::G::apply(node.state.clone(), a);
                let over = F::G::is_terminal(&next);
                let v = if over {
                    terminal_value::<F::G>(&next, mover)
                } else {
                    self.feats.cell_codes(&next, spc, &mut self.codes);
                    let v = self.model.value(&self.codes);
                    if F::G::player_to_move(&next).to_index() == mover {
                        v
                    } else {
                        -v
                    }
                };
                node.edge_values.push(v);
                node.edge_terminal.push(over);
            }
            node.priors = softmax(&node.edge_values, self.cfg.prior_temperature);
            let n = node.actions.len();
            node.children = vec![NONE; n];
            node.visits = vec![0; n];
            node.sums = vec![0.0; n];
        }
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    fn select(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        let explore = self.cfg.c_puct * (n.total.max(1) as f32).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for e in 0..n.actions.len() {
            let q = if n.visits[e] > 0 { (n.sums[e] / n.visits[e] as f64) as f32 } else { 0.0 };
            let score = q + explore * n.priors[e] / (1 + n.visits[e]) as f32;
            if score > best_score {
                best_score = score;
                best = e;
            }
        }
        best
    }

    /// One descent from the root, one expansion, one backup.
    fn iterate(&mut self, path: &mut Vec<(usize, usize)>) {
        path.clear();
        let mut node = 0usize;
        let value = loop {
            if self.nodes[node].terminal {
                break terminal_value::<F::G>(&self.nodes[node].state, self.nodes[node].mover);
            }
            if let Some(v) = self.nodes[node].solved {
                break v;
            }
            let e = self.select(node);
            path.push((node, e));
            let child = self.nodes[node].children[e];
            if child != NONE {
                node = child as usize;
                continue;
            }
            let parent = &self.nodes[node];
            let (edge_value, over) = (parent.edge_values[e], parent.edge_terminal[e]);
            let next = F::G::apply(parent.state.clone(), &parent.actions[e]);
            let parent_mover = parent.mover;
            let id = self.new_node(next, over, true);
            self.nodes[node].children[e] = id;
            let child = &self.nodes[id as usize];
            let (child_mover, solved) = (child.mover, child.solved);
            node = id as usize;
            if let Some(v) = solved {
                break v;
            }
            break if child_mover == parent_mover { edge_value } else { -edge_value };
        };

        let mut mover = self.nodes[node].mover;
        let mut v = value;
        for &(n, e) in path.iter().rev() {
            let m = self.nodes[n].mover;
            if m != mover {
                v = -v;
                mover = m;
            }
            let nd = &mut self.nodes[n];
            nd.visits[e] += 1;
            nd.sums[e] += v as f64;
            nd.total += 1;
        }
    }

    /// The node holding `state` within `REUSE_DEPTH` plies of the retained root.
    fn find(&self, state: &State<F>) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut frontier = vec![0usize];
        for depth in 0..=REUSE_DEPTH {
            let mut next = Vec::new();
            for &i in &frontier {
                if &self.nodes[i].state == state && !self.nodes[i].actions.is_empty() {
                    return Some(i);
                }
                if depth < REUSE_DEPTH {
                    next.extend(self.nodes[i].children.iter().filter(|&&c| c != NONE).map(|&c| c as usize));
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        None
    }

    /// Keep only the subtree under `new_root`, which becomes node 0.
    fn reroot(&mut self, new_root: usize) {
        let mut order = vec![new_root];
        let mut map = vec![NONE; self.nodes.len()];
        map[new_root] = 0;
        let mut i = 0;
        while i < order.len() {
            for &c in &self.nodes[order[i]].children {
                if c != NONE {
                    map[c as usize] = order.len() as u32;
                    order.push(c as usize);
                }
            }
            i += 1;
        }
        let mut old: Vec<Option<Node<_, _>>> = self.nodes.drain(..).map(Some).collect();
        self.nodes = order
            .iter()
            .map(|&o| {
                let mut n = old[o].take().unwrap();
                for c in n.children.iter_mut() {
                    if *c != NONE {
                        *c = map[*c as usize];
                    }
                }
                n
            })
            .collect();
    }

    /// Search `state` for `iterations` more iterations and return the most
    /// visited action (ties: higher mean value, then the earlier action).
    pub fn best_action(&mut self, state: &State<F>) -> Action<F> {
        self.actions.clear();
        F::G::generate_actions(state, &mut self.actions);
        if self.actions.len() == 1 {
            return self.actions[0].clone();
        }
        match self.find(state) {
            Some(i) => self.reroot(i),
            None => {
                self.nodes.clear();
                self.new_node(state.clone(), false, false);
            }
        }
        let mut path = Vec::new();
        for _ in 0..self.cfg.iterations {
            self.iterate(&mut path);
        }
        let root = &self.nodes[0];
        let mut best = 0;
        let mut best_key = (0u32, f64::NEG_INFINITY);
        for e in 0..root.actions.len() {
            let q = if root.visits[e] > 0 { root.sums[e] / root.visits[e] as f64 } else { f64::NEG_INFINITY };
            let key = (root.visits[e], q);
            if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
                best_key = key;
                best = e;
            }
        }
        root.actions[best].clone()
    }
}

fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = values.iter().map(|&v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

impl<F: CellFeatures> Search for PuctPlayer<F> {
    type G = F::G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State<F>) -> Action<F> {
        self.best_action(state)
    }

    fn arena_len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainConfig;
    use crate::selfplay::Trainer;
    use crate::test_support::TttCells;
    use game_ttt::{Move, TicTacToe};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    type Ttt = TicTacToe;
    type TttState = <TicTacToe as Game>::S;

    fn train_cfg(episodes: u64) -> TrainConfig {
        TrainConfig {
            seed: 3,
            episodes,
            n_tuples: 12,
            tuple_len: 4,
            states_per_cell: 3,
            alpha: 0.2,
            lambda: 0.5,
            gamma: 1.0,
            epsilon_start: 0.3,
            epsilon_end: 0.1,
            tcl: true,
            reset_traces_on_explore: true,
            trace_cutoff: 1e-4,
            log_every: 100,
            eval_games: 100,
            out_dir: String::new(),
        }
    }

    /// An untrained model (every value 0) when `episodes` is 0.
    fn model(episodes: u64) -> Arc<Model> {
        let mut t = Trainer::new(TttCells, train_cfg(episodes.max(1)));
        for _ in 0..episodes {
            t.run_episode();
        }
        Arc::new(t.model().clone())
    }

    fn puct(model: &Arc<Model>, iterations: u32) -> PuctPlayer<TttCells> {
        PuctPlayer::new(
            TttCells,
            model.clone(),
            PuctConfig { iterations, c_puct: 1.0, prior_temperature: 1.0, empties_exact: 0 },
        )
    }

    fn after(moves: &[u8]) -> TttState {
        moves.iter().fold(TttState::default(), |s, &m| Ttt::apply(s, &Move(m)))
    }

    /// Game-theoretic value of `s` for its player to move: +1, 0 or -1.
    fn exact(s: &TttState) -> f32 {
        let mover = Ttt::player_to_move(s).to_index();
        if Ttt::is_terminal(s) {
            return terminal_value::<Ttt>(s, mover);
        }
        let mut actions = Vec::new();
        Ttt::generate_actions(s, &mut actions);
        actions
            .iter()
            .map(|a| {
                let c = Ttt::apply(*s, a);
                let v = exact(&c);
                if Ttt::player_to_move(&c).to_index() == mover {
                    v
                } else {
                    -v
                }
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn action_value(s: &TttState, a: &Move) -> f32 {
        let c = Ttt::apply(*s, a);
        let v = exact(&c);
        if Ttt::player_to_move(&c).to_index() == Ttt::player_to_move(s).to_index() {
            v
        } else {
            -v
        }
    }

    #[test]
    fn takes_an_immediate_win_and_blocks_an_immediate_loss() {
        let m = model(0);
        // X holds 0 and 1, O holds 3 and 4, X to move: 2 wins.
        assert_eq!(puct(&m, 200).best_action(&after(&[0, 3, 1, 4])), Move(2));
        // X holds 0 and 1, O holds 3, O to move: 2 must be blocked.
        assert_eq!(puct(&m, 200).best_action(&after(&[0, 3, 1])), Move(2));
        // O to move with a win at 5 of its own (X 0, 1, 8; O 3, 4).
        assert_eq!(puct(&m, 200).best_action(&after(&[0, 3, 1, 4, 8])), Move(5));
    }

    #[test]
    fn terminal_successors_are_scored_exactly_from_the_movers_point_of_view() {
        let m = model(0);
        let root_edge = |s: TttState, mv: u8| {
            let mut p = puct(&m, 1);
            p.new_node(s, false, false);
            let n = &p.nodes[0];
            let e = n.actions.iter().position(|a| *a == Move(mv)).unwrap();
            (n.edge_values[e], n.edge_terminal[e])
        };
        // X wins by playing 2 (+1 for X), and O wins by playing 5 (+1 for O).
        assert_eq!(root_edge(after(&[0, 3, 1, 4]), 2), (1.0, true));
        assert_eq!(root_edge(after(&[0, 3, 1, 4, 8]), 5), (1.0, true));
        // X fills the last cell and the game is drawn.
        assert_eq!(root_edge(after(&[0, 1, 2, 4, 3, 5, 7, 6]), 8), (0.0, true));
        // Untrained model: every non-terminal successor is worth exactly 0.
        assert_eq!(root_edge(after(&[0, 3, 1, 4]), 5), (0.0, false));
    }

    #[test]
    fn priors_are_a_softmax_of_the_successor_values() {
        let m = model(0);
        let mut p = puct(&m, 1);
        p.new_node(after(&[0, 3, 1, 4]), false, false);
        let n = &p.nodes[0];
        let sum: f32 = n.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let win = n.actions.iter().position(|a| *a == Move(2)).unwrap();
        let other = (0..n.actions.len()).find(|&e| e != win).unwrap();
        let ratio = n.priors[win] / n.priors[other];
        assert!((ratio - (1.0f32 - 0.0).exp()).abs() < 1e-4, "ratio {ratio}");
    }

    #[test]
    fn plays_tic_tac_toe_without_losing_to_random_play_from_either_side() {
        let m = model(600);
        let mut rng = SmallRng::seed_from_u64(7);
        let mut points = 0.0;
        for game in 0..40 {
            let agent = game % 2;
            let mut p = puct(&m, 150);
            let mut s = TttState::default();
            while !Ttt::is_terminal(&s) {
                let a = if Ttt::player_to_move(&s).to_index() == agent {
                    p.best_action(&s)
                } else {
                    let mut acts = Vec::new();
                    Ttt::generate_actions(&s, &mut acts);
                    acts[rng.gen_range(0..acts.len())]
                };
                s = Ttt::apply(s, &a);
            }
            points += (terminal_value::<Ttt>(&s, agent) + 1.0) / 2.0;
        }
        assert!(points / 40.0 > 0.9, "score {}", points / 40.0);
    }

    #[test]
    fn an_agent_against_itself_scores_exactly_one_half_over_both_colours() {
        let m = model(600);
        let (mut a, mut b) = (puct(&m, 300), puct(&m, 300));
        let mut result_for_a = 0.0;
        for first_is_a in [true, false] {
            let mut s = TttState::default();
            while !Ttt::is_terminal(&s) {
                let a_moves = (Ttt::player_to_move(&s).to_index() == 0) == first_is_a;
                let mv = if a_moves { a.best_action(&s) } else { b.best_action(&s) };
                s = Ttt::apply(s, &mv);
            }
            let a_index = if first_is_a { 0 } else { 1 };
            result_for_a += (terminal_value::<Ttt>(&s, a_index) + 1.0) / 2.0;
        }
        assert_eq!(result_for_a / 2.0, 0.5);
    }

    #[test]
    fn the_same_configuration_plays_the_same_game() {
        let m = model(600);
        let play = || {
            let mut p = puct(&m, 250);
            let mut s = TttState::default();
            let mut line = Vec::new();
            while !Ttt::is_terminal(&s) {
                let a = p.best_action(&s);
                line.push(a);
                s = Ttt::apply(s, &a);
            }
            line
        };
        assert_eq!(play(), play());
    }

    #[test]
    fn a_reused_tree_keeps_its_statistics_and_still_chooses_a_game_theoretically_best_move() {
        let m = model(600);
        let mut rng = SmallRng::seed_from_u64(11);
        let iterations = 1500;
        let mut reuse_hits = 0;
        for game in 0..6 {
            let agent = game % 2;
            let mut kept = puct(&m, iterations);
            let mut s = TttState::default();
            while !Ttt::is_terminal(&s) {
                let mut acts = Vec::new();
                Ttt::generate_actions(&s, &mut acts);
                let mv = if Ttt::player_to_move(&s).to_index() == agent {
                    let before = kept.tree_nodes();
                    let mv = kept.best_action(&s);
                    let mut fresh = puct(&m, iterations);
                    let fresh_mv = fresh.best_action(&s);
                    if acts.len() > 1 {
                        assert_eq!(kept.nodes[0].state, s, "the retained root is the queried state");
                        if kept.root_visits() > iterations {
                            reuse_hits += 1;
                            assert!(before > 0);
                        }
                        let best = acts.iter().map(|a| action_value(&s, a)).fold(f32::NEG_INFINITY, f32::max);
                        assert_eq!(action_value(&s, &mv), best, "reused tree chose {mv:?} at {s}");
                        assert_eq!(action_value(&s, &fresh_mv), best, "fresh tree chose {fresh_mv:?}");
                    }
                    mv
                } else {
                    acts[rng.gen_range(0..acts.len())]
                };
                s = Ttt::apply(s, &mv);
            }
        }
        assert!(reuse_hits > 0, "no move ever reused a retained tree");
    }

    #[test]
    fn rerooting_keeps_exactly_the_subtree_below_the_new_root() {
        let m = model(600);
        let mut p = puct(&m, 300);
        let s0 = TttState::default();
        let mv = p.best_action(&s0);
        let (visits, nodes_before) = (p.root_edges(), p.tree_nodes());
        let e = visits.iter().position(|(a, _, _)| *a == mv).unwrap();
        let s1 = Ttt::apply(s0, &mv);
        let child = p.nodes[0].children[e] as usize;
        let child_visits: u32 = p.nodes[child].total;
        assert_eq!(child_visits + 1, visits[e].1, "an edge's visits are one more than its child's total");
        p.reroot(child);
        assert_eq!(p.nodes[0].state, s1);
        assert_eq!(p.root_visits(), child_visits);
        assert!(p.tree_nodes() < nodes_before);
    }
}
