//! Categorical policy logits used by policy-improvement search.
//!
//! This is deliberately separate from [`super::prior::PriorPolicy`]: logits
//! rank actions but are never converted into fictitious value visits.

use crate::game::Game;

/// Unbounded categorical logits for one state's legal actions.
pub trait PolicyLogits<G: Game>: Clone + Send + Sync {
    fn logits(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64>;
}

/// Object-safe storage for a [`PolicyLogits`] provider in [`super::SearchConfig`].
pub trait PolicyLogitsDyn<G: Game>: Send + Sync {
    fn logits(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64>;
    fn clone_box(&self) -> Box<dyn PolicyLogitsDyn<G>>;
}

impl<G, T> PolicyLogitsDyn<G> for T
where
    G: Game,
    T: PolicyLogits<G> + 'static,
{
    fn logits(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64> {
        PolicyLogits::logits(self, state, actions)
    }

    fn clone_box(&self) -> Box<dyn PolicyLogitsDyn<G>> {
        Box::new(self.clone())
    }
}

impl<G: Game> Clone for Box<dyn PolicyLogitsDyn<G>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
