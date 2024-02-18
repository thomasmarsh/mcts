use super::index::Arena;
use super::*;
use crate::game::Action;
use crate::strategies::Search;
use crate::util::random_best;
use rustc_hash::FxHashMap;
use serde::Serialize;

// Use epsilon-greed to alleviate starvation. Not mentioned in Chaslot paper. Also, doesn't really help.
const EPSILON: f64 = 0.1;

#[derive(Clone, Debug, Serialize)]
pub struct Entry<A: Action> {
    pub children: FxHashMap<A, index::Id>,
    pub utilities: Vec<f64>,
    pub num_visits: u64,
}

impl<A: Action> Entry<A> {
    fn update(&mut self, utilities: &[f64]) {
        assert_eq!(self.utilities.len(), utilities.len());
        self.utilities
            .iter_mut()
            .enumerate()
            .for_each(|(i, score)| {
                *score += utilities[i];
            });

        self.num_visits += 1;
    }

    fn score(&self, player: usize, parent_n: u64) -> Option<f64> {
        if self.num_visits == 0 {
            None
        } else {
            let ratio = self.num_visits as f64 / parent_n as f64;
            Some(ratio * (self.utilities[player] / self.num_visits as f64 + 1.) / 2.)
        }
    }

    fn new(num_players: usize) -> Self {
        Self {
            children: Default::default(),
            utilities: vec![0.; num_players],
            num_visits: 0,
        }
    }
}

impl<A: Action> Entry<A> {}

#[derive(Clone, Debug)]
pub struct OpeningBook<A: Action> {
    pub index: Arena<Entry<A>>,
    pub root_id: Id,
    pub num_players: usize,
}

impl<A: Action> OpeningBook<A> {
    fn new(num_players: usize) -> Self {
        let mut index = Arena::new();
        let root_id = index.insert(Entry::new(num_players));
        Self {
            index,
            root_id,
            num_players,
        }
    }

    fn get_mut(&mut self, id: Id) -> &mut Entry<A> {
        self.index.get_mut(id)
    }

    fn get(&self, id: Id) -> &Entry<A> {
        self.index.get(id)
    }

    fn insert(&mut self, value: Entry<A>) -> Id {
        self.index.insert(value)
    }
}

impl<A: Action> OpeningBook<A> {
    fn contains_action(&self, id: Id, action: &A) -> bool {
        self.index.get(id).children.contains_key(action)
    }

    // Get or insert a child for this id
    fn get_child(&mut self, id: Id, action: &A) -> Id {
        if !self.contains_action(id, action) {
            // Insert into index
            let child_id = self.insert(Entry::new(self.num_players));

            // Place index reference in hash map
            self.index
                .get_mut(id)
                .children
                .insert(action.clone(), child_id);
        }

        // Return the child id
        *self.index.get(id).children.get(action).unwrap()
    }

    pub fn add(&mut self, sequence: &[A], utilities: &[f64]) {
        let mut current_id = self.root_id;
        self.get_mut(current_id).update(utilities);

        sequence.iter().for_each(|action| {
            current_id = self.get_child(current_id, action);
            self.get_mut(current_id).update(utilities);
        });
    }

    pub fn score(&self, sequence: &[A], player: usize) -> Option<f64> {
        let mut current_id = self.root_id;
        let mut parent_n = self.get(current_id).num_visits;
        for action in sequence {
            if let Some(child_id) = self.get(current_id).children.get(action) {
                parent_n = self.get(current_id).num_visits;
                current_id = *child_id;
            } else {
                return None;
            }
        }
        self.get(current_id).score(player, parent_n)
    }
}

#[derive(Clone)]
pub struct QuasiBestFirst<G: Game, S: Strategy<G>> {
    pub k: Vec<f64>,
    pub book: OpeningBook<G::A>,
    pub search: TreeSearch<G, S>,
    pub rng: SmallRng,
}

/// I don't quite get this algorithm. I posted a question:
/// https://stackoverflow.com/questions/78017232/quasi-best-first-qbf-for-opening-book-generation-with-monte-carlo-tree-search
///
/// Algorithm 1 The “Quasi Best-First” (QBF) algorithm. λ is the number of machines
/// available. K is a constant. g is a game, defined as a sequence of game states.
/// The function “MoGoChoice” asks MOGO to choose a move.
///
/// ```ignore
/// QBF(K, λ)
/// while True do
///   for l = 1..λ, do
///     s =initial state; g = {s}.
///     while s is not a final state do
///       bestScore = K
///       bestMove = Null
///       for m in the set of possible moves in s do
///         score = percentage of won games by playing the move m in s
///         if score > bestScore then
///           bestScore = score
///           bestMove = m
///         end if
///       end for
///       if bestMove = Null then
///         bestMove = MoGoChoice(s) // lower level MCTS
///       end if
///       s = playMove(s, bestMove)
///       g = concat(g, s)
///     end while
///     Add g and the result of the game in the book.
///   end for
/// end while
/// ```
impl<G, S> QuasiBestFirst<G, S>
where
    G: Game,
    S: Strategy<G>,
    MctsStrategy<G, S>: Default,
{
    pub fn new(search: TreeSearch<G, S>, rng: SmallRng) -> Self {
        // The default value here is 0.5, but the Chaslot paper noted the difficulty
        // of elevating the black player in go when cold starting, prompting a lower
        // threshold for the initial player.
        // TODO: what about N-player games where N > 2
        let mut k = vec![0.005; G::num_players()];
        if k.len() == 2 {
            k[0] = 0.005;
        }
        Self {
            k,
            book: OpeningBook::new(G::num_players()),
            search,
            rng,
        }
    }

    /// Search is expected to be called multiple times to fill out the book.
    pub fn search(&mut self, init: &G::S) {
        let mut stack = Vec::new();
        let mut state = init.clone();
        while !G::is_terminal(&state) {
            let mut actions = Vec::new();
            G::generate_actions(&state, &mut actions);
            let player = G::player_to_move(&state).to_index();
            let index = self.best_child(player, stack.as_slice(), &state);
            state = G::apply(state, &actions[index]);
            stack.push(actions[index].clone());
        }

        let utilities = G::compute_utilities(&state);
        self.book.add(stack.as_slice(), utilities.as_slice());
        self.debug(init);
    }

    fn debug(&self, init: &G::S) {
        println!("book.len() = {}", self.book.index.len());
        let mut actions = Vec::new();
        G::generate_actions(init, &mut actions);

        self.search.index.get(self.search.root_id).actions();

        let root = self.book.index.get(self.book.root_id);
        println!("root: {}", root.num_visits);
        actions.iter().enumerate().for_each(|(i, action)| {
            let child_id_opt = root.children.get(action);
            let child = child_id_opt.map(|child_id| self.book.index.get(*child_id));
            let score = self.book.score(&[action.clone()], 0);
            println!(
                "- {i}: {:?}, {score:?} {action:?}",
                child.map_or(0, |c| c.num_visits),
            );
        });
    }

    fn best_child(&mut self, player: usize, stack: &[G::A], state: &G::S) -> usize {
        let k_score = self.k[player];

        // The child actions, enumerated since we plan to return an index.
        let mut available = Vec::new();
        G::generate_actions(state, &mut available);

        if self.rng.gen::<f64>() < EPSILON {
            return self.rng.gen_range(0..available.len());
        }

        // The prefix list of actions we use as a key
        let key_init = stack.to_vec();

        // TODO: a lot of the difficulty here is the handling of optionals. It would make
        // sense to have most of the SelectStrategy API return optionals, but it hasn't
        // been necessary until this point. Additionally, random_best and random_best_index
        // aren't great fits. We are misusing random_best here a bit w.r.t. neg infinity.
        let enumerated = available.iter().cloned().enumerate().collect::<Vec<_>>();
        let best = random_best(
            enumerated.as_slice(),
            &mut self.rng,
            |(_, action): &(usize, G::A)| {
                let mut key = key_init.clone();
                key.push(action.clone());

                let score = self
                    .book
                    .score(key.as_slice(), player)
                    .unwrap_or(f64::NEG_INFINITY);
                if score > k_score {
                    score
                } else {
                    // NOTE: we depend on random_best using this value internally
                    // as an equivalence for None types
                    f64::NEG_INFINITY
                }
            },
        );

        if let Some((best_index, _)) = best {
            *best_index
        } else {
            let action = self.search.choose_action(state);
            available.iter().position(|p| *p == action.clone()).unwrap()
        }
    }
}
