use super::index;
use crate::game::{Action, PlayerIndex};

use rustc_hash::FxHashMap;
use serde::Serialize;

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

    fn score(&self, player: PlayerIndex) -> Option<f64> {
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

#[derive(Clone, Debug)]
pub struct OpeningBook<A: Action> {
    pub index: index::Arena<Entry<A>>,
    pub root_id: index::Id,
    pub num_players: usize,
}

impl<A: Action> OpeningBook<A> {
    pub fn new(num_players: usize) -> Self {
        let mut index = index::Arena::new();
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

    pub fn score(&self, sequence: &[A], player: PlayerIndex) -> Option<f64> {
        let mut current_id = self.root_id;
        for action in sequence {
            if let Some(child_id) = self.get(current_id).children.get(action) {
                current_id = *child_id;
            } else {
                return None;
            }
        }
        self.get(current_id).score(player)
    }
}
