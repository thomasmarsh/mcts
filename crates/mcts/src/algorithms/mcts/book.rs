use super::index;
use crate::game::Action;
use crate::game::Game;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct Entry<A: Action> {
    pub children: FxHashMap<A, index::Id>,
    pub utilities: Vec<f64>,
    pub num_visits: u64,
}

/// Wire shape for `Entry`: `children` as a `Vec` of pairs rather than
/// serde's native map encoding. serde_json's map support requires keys
/// that serialize as JSON strings, but an `Action`'s own `Serialize` impl
/// is free to emit anything (e.g. `games/gonnect`'s `Move` serializes as a
/// `(cell, capture-mask-hex)` tuple to dodge JS float precision loss on
/// wide bitboards) -- a plain pair list works for any `Action` regardless
/// of its own wire shape.
#[derive(Serialize, Deserialize)]
struct WireEntry<A: Action> {
    children: Vec<(A, index::Id)>,
    utilities: Vec<f64>,
    num_visits: u64,
}

impl<A: Action> Serialize for Entry<A> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        WireEntry {
            children: self
                .children
                .iter()
                .map(|(action, &id)| (action.clone(), id))
                .collect(),
            utilities: self.utilities.clone(),
            num_visits: self.num_visits,
        }
        .serialize(serializer)
    }
}

impl<'de, A: Action + Deserialize<'de>> Deserialize<'de> for Entry<A> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = WireEntry::<A>::deserialize(deserializer)?;
        Ok(Entry {
            children: wire.children.into_iter().collect(),
            utilities: wire.utilities,
            num_visits: wire.num_visits,
        })
    }
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

    fn score(&self, player: usize) -> Option<f64> {
        if self.num_visits == 0 {
            None
        } else {
            let q = self.utilities[player];
            let n = self.num_visits as f64;
            let avg_q = q / n; // -1..1
            Some((avg_q + 1.) / 2.)
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpeningBook<A: Action> {
    pub index: index::Arena<Entry<A>>,
    pub root_id: index::Id,
    pub num_players: usize,
}

impl<A: Action> OpeningBook<A> {
    pub fn new(num_players: usize) -> Self {
        let index = index::Arena::new();
        let root_id = index.insert(Entry::new(num_players));
        Self {
            index,
            root_id,
            num_players,
        }
    }

    fn get_mut(&mut self, id: index::Id) -> &mut Entry<A> {
        self.index.get_mut(id)
    }

    fn get(&self, id: index::Id) -> &Entry<A> {
        self.index.get(id)
    }

    fn insert(&mut self, value: Entry<A>) -> index::Id {
        self.index.insert(value)
    }
}

impl<A: Action> OpeningBook<A> {
    fn contains_action(&self, id: index::Id, action: &A) -> bool {
        self.index.get(id).children.contains_key(action)
    }

    // Get or insert a child for this id
    fn get_child(&mut self, id: index::Id, action: &A) -> index::Id {
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

    /// Walks `sequence` from the root, returning the `Id` it resolves to,
    /// or `None` if some prefix was never added to the book.
    fn resolve(&self, sequence: &[A]) -> Option<index::Id> {
        let mut current_id = self.root_id;
        for action in sequence {
            current_id = *self.get(current_id).children.get(action)?;
        }
        Some(current_id)
    }

    pub fn score(&self, sequence: &[A], player: usize) -> Option<f64> {
        self.score_at(self.resolve(sequence)?, player)
    }

    /// Every action explored from the position reached by `sequence`, with
    /// its book visit count and `player`'s empirical score, sorted by
    /// descending visits (the book's most-tested reply first). `None` if
    /// `sequence` itself was never added to the book. For a "top opening
    /// moves" report -- `score`/`add` alone can't enumerate a position's
    /// alternatives, only look up one action at a time.
    pub fn children(&self, sequence: &[A], player: usize) -> Option<Vec<(A, u64, Option<f64>)>> {
        Some(self.children_at(self.resolve(sequence)?, player))
    }

    /// `player`'s empirical score at a book node addressed directly by
    /// `Id`, skipping the root-to-node walk `score` does. For callers that
    /// already hold an `Id` -- e.g. one resolved once via
    /// `build_state_index` -- rather than an action sequence.
    pub fn score_at(&self, id: index::Id, player: usize) -> Option<f64> {
        self.get(id).score(player)
    }

    /// Every action explored from a book node addressed directly by `Id`.
    /// See `children`'s doc comment and `score_at`'s.
    pub fn children_at(&self, id: index::Id, player: usize) -> Vec<(A, u64, Option<f64>)> {
        let entry = self.get(id);
        let mut out: Vec<(A, u64, Option<f64>)> = entry
            .children
            .iter()
            .map(|(action, &child_id)| {
                let child = self.get(child_id);
                (action.clone(), child.num_visits, child.score(player))
            })
            .collect();
        out.sort_by_key(|(_, visits, _)| std::cmp::Reverse(*visits));
        out
    }

    pub fn num_visits_at(&self, id: index::Id) -> u64 {
        self.get(id).num_visits
    }

    /// Maps every book-tree position to its `Id`, keyed by the actual game
    /// state reached by replaying that position's action sequence from
    /// `initial` via `G::apply`. Lets a caller holding a live `G::S` -- but
    /// no record of the action sequence that reached it, e.g. a
    /// `GameAdapter::ai_move` request that only sees the current state --
    /// look its book entry up directly in O(1) instead of needing to
    /// reconstruct history. This walks the book's own tree (bounded by
    /// however many self-play games built it), not the game's full state
    /// graph, so it stays cheap even though the game itself may have far
    /// more reachable states than the book covers.
    pub fn build_state_index<G>(&self, initial: G::S) -> HashMap<G::S, index::Id>
    where
        G: Game<A = A>,
        G::S: Hash + Eq,
    {
        let mut out = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back((initial, self.root_id));
        while let Some((state, id)) = queue.pop_front() {
            for (action, &child_id) in &self.get(id).children {
                queue.push_back((G::apply(state.clone(), action), child_id));
            }
            out.insert(state, id);
        }
        out
    }

    /// Folds `other`'s statistics into `self`, matching nodes by the
    /// sequence of actions from each book's root -- `Id`s are arena-local
    /// and meaningless across two independently-built books, so nodes are
    /// found (or created) via `get_child` exactly as `add` does, rather
    /// than compared directly. Used to combine books built by independent
    /// self-play workers that all started from the same seed: each
    /// worker's own book only records its own new games (see
    /// `game_gonnect::book::build`), so summing them together, plus one
    /// copy of the shared seed, reproduces what a single sequential run
    /// covering every worker's games would have produced.
    pub fn merge(&mut self, other: &OpeningBook<A>) {
        assert_eq!(
            self.num_players, other.num_players,
            "cannot merge opening books built for different player counts"
        );
        self.merge_at(self.root_id, other, other.root_id);
    }

    fn merge_at(&mut self, self_id: index::Id, other: &OpeningBook<A>, other_id: index::Id) {
        let other_entry = other.get(other_id);
        self.get_mut(self_id).num_visits += other_entry.num_visits;
        for (i, u) in other_entry.utilities.iter().enumerate() {
            self.get_mut(self_id).utilities[i] += u;
        }
        for (action, &other_child_id) in &other_entry.children {
            let self_child_id = self.get_child(self_id, action);
            self.merge_at(self_child_id, other, other_child_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerIndex;

    /// Two players, actions are plain `u8`s. `add`'s `utilities` are always
    /// from player 0's perspective negated for player 1 (zero-sum), matching
    /// how a real `Game::compute_utilities` two-player result is shaped.
    fn book_with_two_games() -> OpeningBook<u8> {
        let mut book = OpeningBook::new(2);
        // Game 1: 1 -> 2, player 0 (mover at the root) wins.
        book.add(&[1, 2], &[1.0, -1.0]);
        // Game 2: 1 -> 3, player 0 loses.
        book.add(&[1, 3], &[-1.0, 1.0]);
        book
    }

    #[test]
    fn add_accumulates_visits_and_utilities_along_the_path() {
        let book = book_with_two_games();
        // Root and the shared prefix [1] both saw both games.
        assert_eq!(book.score(&[], 0), Some(0.5));
        assert_eq!(book.score(&[1], 0), Some(0.5));
        // Each leaf only saw its own game.
        assert_eq!(book.score(&[1, 2], 0), Some(1.0));
        assert_eq!(book.score(&[1, 3], 0), Some(0.0));
        // Player 1's perspective is the mirror image.
        assert_eq!(book.score(&[1, 2], 1), Some(0.0));
        assert_eq!(book.score(&[1, 3], 1), Some(1.0));
    }

    #[test]
    fn score_is_none_for_an_unexplored_sequence() {
        let book = book_with_two_games();
        assert_eq!(book.score(&[9], 0), None);
        assert_eq!(book.score(&[1, 9], 0), None);
    }

    #[test]
    fn children_ranks_by_visits_and_reports_each_score() {
        let book = book_with_two_games();
        let kids = book.children(&[1], 0).unwrap();
        assert_eq!(kids.len(), 2);
        // Both children of [1] have exactly one visit each -- verify the
        // full set rather than depending on tie-break order.
        let mut by_action: Vec<_> = kids.iter().map(|(a, n, s)| (*a, *n, *s)).collect();
        by_action.sort_by_key(|(a, _, _)| *a);
        assert_eq!(by_action, vec![(2, 1, Some(1.0)), (3, 1, Some(0.0))]);
    }

    #[test]
    fn children_is_none_for_an_unexplored_sequence() {
        let book = book_with_two_games();
        assert!(book.children(&[9], 0).is_none());
    }

    #[test]
    fn json_round_trip_preserves_scores_and_structure() {
        let book = book_with_two_games();
        let json = serde_json::to_string(&book).unwrap();
        let restored: OpeningBook<u8> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.num_players, book.num_players);
        assert_eq!(restored.score(&[], 0), book.score(&[], 0));
        assert_eq!(restored.score(&[1, 2], 0), book.score(&[1, 2], 0));
        assert_eq!(restored.score(&[1, 3], 1), book.score(&[1, 3], 1));

        let mut original = book.children(&[1], 0).unwrap();
        let mut round_tripped = restored.children(&[1], 0).unwrap();
        original.sort_by_key(|(a, _, _)| *a);
        round_tripped.sort_by_key(|(a, _, _)| *a);
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn score_at_and_children_at_match_the_sequence_walk() {
        let book = book_with_two_games();
        let id = book
            .index
            .get(book.root_id)
            .children
            .get(&1)
            .copied()
            .unwrap();
        assert_eq!(book.score_at(id, 0), book.score(&[1], 0));
        let mut via_id = book.children_at(id, 0);
        let mut via_sequence = book.children(&[1], 0).unwrap();
        via_id.sort_by_key(|(a, _, _)| *a);
        via_sequence.sort_by_key(|(a, _, _)| *a);
        assert_eq!(via_id, via_sequence);
        assert_eq!(book.num_visits_at(id), 2);
    }

    #[test]
    fn merge_combines_disjoint_and_shared_paths() {
        let mut a = OpeningBook::<u8>::new(2);
        a.add(&[1, 2], &[1.0, -1.0]);
        let mut b = OpeningBook::<u8>::new(2);
        b.add(&[1, 3], &[-1.0, 1.0]); // shares the [1] prefix with `a`
        b.add(&[5], &[1.0, -1.0]); // a path `a` never reached at all

        a.merge(&b);

        // The root sees all three games (`a`'s one, `b`'s two): two wins,
        // one loss for player 0.
        assert_eq!(a.score(&[], 0), Some(2.0 / 3.0));
        // [1] sees only the two games that passed through it -- one win,
        // one loss.
        assert_eq!(a.score(&[1], 0), Some(0.5));
        assert_eq!(a.score(&[1, 2], 0), Some(1.0));
        assert_eq!(a.score(&[1, 3], 0), Some(0.0));
        assert_eq!(a.score(&[5], 0), Some(1.0));
        assert_eq!(a.num_visits_at(a.root_id), 3);
    }

    #[test]
    #[should_panic(expected = "different player counts")]
    fn merge_rejects_mismatched_player_counts() {
        let mut a = OpeningBook::<u8>::new(2);
        let b = OpeningBook::<u8>::new(3);
        a.merge(&b);
    }

    /// A minimal `Game` whose state *is* the sequence of actions played so
    /// far, just enough to exercise `build_state_index`'s replay logic
    /// without pulling in a real game crate.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct FakeState(Vec<u8>);

    impl std::fmt::Display for FakeState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }

    impl PlayerIndex for u8 {
        fn to_index(&self) -> usize {
            *self as usize
        }
    }

    #[derive(Clone)]
    struct FakeGame;

    impl Game for FakeGame {
        type S = FakeState;
        type A = u8;
        type P = u8;

        fn apply(mut state: Self::S, action: &Self::A) -> Self::S {
            state.0.push(*action);
            state
        }

        fn generate_actions(_state: &Self::S, _actions: &mut Vec<Self::A>) {}

        fn winner(_state: &Self::S) -> Option<Self::P> {
            None
        }

        fn player_to_move(_state: &Self::S) -> Self::P {
            0
        }
    }

    #[test]
    fn build_state_index_maps_replayed_states_to_book_ids() {
        let book = book_with_two_games();
        let index = book.build_state_index::<FakeGame>(FakeState::default());

        // Every node in the two-game book (root, [1], [1,2], [1,3]) should
        // be present, keyed by the state reached by replaying its path.
        assert_eq!(index.len(), 4);

        let root_state = FakeState::default();
        assert_eq!(*index.get(&root_state).unwrap(), book.root_id);

        let after_1 = FakeState(vec![1]);
        let after_1_id = *index.get(&after_1).unwrap();
        assert_eq!(book.score_at(after_1_id, 0), book.score(&[1], 0));

        let after_1_2 = FakeState(vec![1, 2]);
        let after_1_2_id = *index.get(&after_1_2).unwrap();
        assert_eq!(book.score_at(after_1_2_id, 0), book.score(&[1, 2], 0));

        // A state the book never reached (action 9 is unexplored).
        assert!(!index.contains_key(&FakeState(vec![9])));

        // Sanity check that FakeGame::player_to_move at least type-checks
        // against PlayerIndex as a real Game's would.
        assert_eq!(FakeGame::player_to_move(&root_state).to_index(), 0);
    }
}
