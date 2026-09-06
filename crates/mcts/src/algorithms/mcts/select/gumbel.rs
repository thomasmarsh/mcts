//! Interior node selection for the Gumbel AlphaZero schedule (Danihelka et
//! al., ICLR 2022). Away from the root, Gumbel MCTS selects the child whose
//! visit count deviates most from the completed-Q-improved policy.
//!
//! This is a stub: it forwards every decision to PUCT-style [`Ucb1`] (plus
//! whatever `SearchConfig::prior` seeds), so the Gumbel root schedule can be
//! exercised end to end before the deterministic completed-Q rule is exact.
//! Only the label distinguishes it from `Ucb1` today.

use rand::rngs::SmallRng;

use super::{SelectContext, SelectPolicy, Ucb1};
use crate::algorithms::mcts::config;
use crate::algorithms::mcts::index::Id;
use crate::algorithms::mcts::node::ChildArray;
use crate::algorithms::mcts::BackpropFlags;
use crate::game::Game;

#[derive(Clone, Default)]
pub struct GumbelCompletedQ {
    inner: Ucb1,
}

impl<G: Game> SelectPolicy<G> for GumbelCompletedQ {
    type Score = <Ucb1 as SelectPolicy<G>>::Score;
    type Aux = <Ucb1 as SelectPolicy<G>>::Aux;

    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        <Ucb1 as SelectPolicy<G>>::setup(&mut self.inner, ctx)
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        <Ucb1 as SelectPolicy<G>>::best_child(&mut self.inner, ctx, rng)
    }

    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        aux: Self::Aux,
    ) -> Self::Score {
        <Ucb1 as SelectPolicy<G>>::score_child(&self.inner, ctx, child_id, children, idx, aux)
    }

    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: Self::Aux) -> Self::Score {
        <Ucb1 as SelectPolicy<G>>::unvisited_value(&self.inner, ctx, aux)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        <Ucb1 as SelectPolicy<G>>::backprop_flags(&self.inner)
    }

    fn supports_ismcts() -> bool {
        <Ucb1 as SelectPolicy<G>>::supports_ismcts()
    }

    fn requirements(&self) -> config::Requirements {
        SelectPolicy::<G>::requirements(&self.inner)
    }

    fn label(&self) -> String {
        "gumbel_completed_q".into()
    }
}
