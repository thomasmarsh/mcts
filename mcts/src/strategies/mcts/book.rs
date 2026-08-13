use super::index;
use crate::game::Action;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

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

    pub fn score(&self, sequence: &[A], player: usize) -> Option<f64> {
        let mut current_id = self.root_id;
        for action in sequence {
            let child_id = self.get(current_id).children.get(action)?;
            current_id = *child_id;
        }
        self.get(current_id).score(player)
    }

    /// Every action explored from the position reached by `sequence`, with
    /// its book visit count and `player`'s empirical score, sorted by
    /// descending visits (the book's most-tested reply first). `None` if
    /// `sequence` itself was never added to the book. For a "top opening
    /// moves" report -- `score`/`add` alone can't enumerate a position's
    /// alternatives, only look up one action at a time.
    pub fn children(&self, sequence: &[A], player: usize) -> Option<Vec<(A, u64, Option<f64>)>> {
        let mut current_id = self.root_id;
        for action in sequence {
            let child_id = self.get(current_id).children.get(action)?;
            current_id = *child_id;
        }
        let entry = self.get(current_id);
        let mut out: Vec<(A, u64, Option<f64>)> = entry
            .children
            .iter()
            .map(|(action, &child_id)| {
                let child = self.get(child_id);
                (action.clone(), child.num_visits, child.score(player))
            })
            .collect();
        out.sort_by_key(|(_, visits, _)| std::cmp::Reverse(*visits));
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
