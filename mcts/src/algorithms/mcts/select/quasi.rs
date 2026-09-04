use super::super::book;
use super::super::config::SearchConfig;
use super::super::config::PolicyProfile;
use super::super::index::Id;
use super::super::node::real_action;
use super::super::node::ChildArray;
use super::super::search::TreeSearch;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Real;
use crate::algorithms::Search;
use crate::symmetry::incoming_sym;
use crate::util::random_best;

use rand::rngs::SmallRng;

/// Quasi Best-First comes from the Chaslot paper on Meta MCTS for opening book
/// generation. This is intended to be used differently than other strategies.
/// For opening book generation, we use the following settings for the higher
/// level MCTS config:
///
/// - expand_threshold: 0 (expand to terminal state during select)
/// - max_iterations: 1 (we only need one PV)
/// - simulate: n/a (ignored, due to max_iteration count)
/// - backprop: n/a (ignored, due to max_iteration count)
///
/// We add an epsilon-greedy parameter since this seems otherwise too greedy
/// a selection strategy and we don't see enough exploration.
///
///
/// > Algorithm 1 The "Quasi Best-First" (QBF) algorithm. λ is the number of machines
/// > available. K is a constant. g is a game, defined as a sequence of game states.
/// > The function "MoGoChoice" asks MOGO to choose a move.
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
#[derive(Clone)]
pub struct QuasiBestFirst<G: Game, S: PolicyProfile<G>> {
    pub book: book::OpeningBook<G::A>,
    pub search: TreeSearch<G, S>,
    pub epsilon: f64,
    pub k: Vec<f64>,
    pub key_init: Vec<G::A>,
}

impl<G, S> QuasiBestFirst<G, S>
where
    G: Game,
    S: PolicyProfile<G>,
    TreeSearch<G, S>: Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn book(mut self, book: book::OpeningBook<G::A>) -> Self {
        self.book = book;
        self
    }

    pub fn search(mut self, search: TreeSearch<G, S>) -> Self {
        self.search = search;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn k(mut self, k: Vec<f64>) -> Self {
        self.k = k;
        self
    }

    pub fn key_init(mut self, key_init: Vec<G::A>) -> Self {
        self.key_init = key_init;
        self
    }
}

impl<G, S> Default for QuasiBestFirst<G, S>
where
    G: Game,
    S: PolicyProfile<G>,
    TreeSearch<G, S>: Default,
{
    fn default() -> Self {
        // The default value here is 0.5, but the Chaslot paper noted the difficulty
        // of elevating the black player in go when cold starting, prompting a lower
        // threshold for the initial player.
        // TODO: what about N-player games where N > 2
        let mut k = vec![0.5; G::num_players()];
        if k.len() == 2 {
            k[0] = 0.1;
        }

        Self {
            book: book::OpeningBook::new(G::num_players()),
            search: TreeSearch::default(),
            epsilon: 0.3,
            k,
            key_init: vec![],
        }
    }
}

impl<G, S> SelectPolicy<G> for QuasiBestFirst<G, S>
where
    G: Game,
    S: PolicyProfile<G>,
    SearchConfig<G, S>: Default,
{
    type Score = f64;
    type Aux = ();

    fn label(&self) -> String {
        "quasi_best_first".into()
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let available = current.children();

        // The stack now contains the action path to the terminal state.
        // `ctx.stack.pairs()` walks root -> leaf, replaying real states from
        // `ctx.root_state` -- see `crate::symmetry::incoming_sym`'s doc
        // comment for why each parent's own incoming symmetry must come
        // from the real state in hand, not a cached edge value.
        // TODO: factor this pair iteration out of here
        let mut key_init = vec![];
        let mut replay_state = ctx.root_state.clone();
        for ((parent_id, _), (_, idx)) in ctx.stack.pairs() {
            let idx = *idx;
            let parent = ctx.index.get(*parent_id);
            let incoming_sym =
                incoming_sym::<G>(ctx.canonicalizes, parent.is_root(), Real(&replay_state));
            let action = real_action::<G>(parent.children(), idx, incoming_sym);
            replay_state = G::apply(replay_state, &action);
            key_init.push(action);
        }
        let player_to_move = G::player_to_move(ctx.state).to_index();
        let k_score = self.k[player_to_move];

        let enumerated: Vec<(usize, G::A)> = (0..available.len())
            .map(|i| (i, real_action::<G>(available, i, ctx.incoming_sym)))
            .collect();
        let best = random_best(enumerated.as_slice(), rng, |(_, action): &(usize, G::A)| {
            let mut key = key_init.clone();
            key.push(action.clone());

            let score = self
                .book
                .score(key.as_slice(), player_to_move)
                .unwrap_or(f64::NEG_INFINITY);
            if score > k_score {
                score
            } else {
                // NOTE: we depend on random_best using this value internally
                // as an equivalence for None types
                f64::NEG_INFINITY
            }
        });

        if let Some((best_index, _)) = best {
            *best_index
        } else {
            let action = self.search.choose_action(ctx.state);
            (0..available.len())
                .find(|&i| real_action::<G>(available, i, ctx.incoming_sym) == action)
                .unwrap()
        }
    }

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        _: &SelectContext<'_, G>,
        _: Id,
        _: &ChildArray<G::A>,
        _: usize,
        _: Self::Aux,
    ) -> f64 {
        0.
    }

    #[inline(always)]
    fn unvisited_value(&self, _: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        0.
    }
}
