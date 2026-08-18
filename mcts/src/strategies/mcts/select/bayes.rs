use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use crate::game::Game;

// Tesauro/Rajan/Segal 2010, "Bayesian Inference in Monte-Carlo Tree Search"
// (UAI). Both strategies here read `ChildSnapshot::posterior_mean`/
// `posterior_variance`, which only `backprop::BayesGaussian`/`BayesNumeric`
// populate -- pairing either of these with any other backprop strategy is
// rejected at `SearchConfig::validate()`-time (see `requirements()` below
// and `config::Requirements::needs_posterior`'s doc comment).

/// `B_i = mu_i + c * sqrt(2 ln N / n_i)` -- the paper's equation 3: replaces
/// UCB1's sample mean with the posterior mean, otherwise an unchanged UCB1
/// exploration term. `c` is not in the paper (which fixes the constant at
/// `1`) but is exposed here, defaulting to `1.0`, so it's tunable like every
/// other strategy's exploration constant in this codebase.
#[derive(Clone)]
pub struct BayesUct1 {
    pub c: f64,
}

impl Default for BayesUct1 {
    fn default() -> Self {
        Self { c: 1.0 }
    }
}

impl BayesUct1 {
    pub fn with_c(c: f64) -> Self {
        Self { c }
    }
}

impl<G: Game> SelectStrategy<G> for BayesUct1 {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        let stats = ctx.current_stats();
        2.0 * (stats.num_visits() as f64).max(1.).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        two_ln_n: f64,
    ) -> f64 {
        let snap = ctx.child_snapshot(child_id, children, idx);
        let n = snap.total_visits() as f64;
        snap.posterior_mean + self.c * (two_ln_n / n).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, two_ln_n: f64) -> f64 {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        unvisited_value + self.c * two_ln_n.sqrt()
    }

    fn requirements(&self) -> super::super::config::Requirements {
        super::super::config::Requirements {
            needs_posterior: true,
            ..super::super::config::Requirements::none()
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// `B_i = mu_i + c * sqrt(2 ln N) * sigma_i` -- the paper's equation 4:
/// additionally replaces UCB1's `1/sqrt(n_i)` exploration term with the
/// posterior standard deviation, motivated (per the paper) by
/// `sigma_i^2 ~ 1/n_i` in the simple-bandit case and the Interval-Estimation
/// intuition of "sample by expected value plus expected uncertainty". Same
/// `c` convention as `BayesUct1`.
#[derive(Clone)]
pub struct BayesUct2 {
    pub c: f64,
}

impl Default for BayesUct2 {
    fn default() -> Self {
        Self { c: 1.0 }
    }
}

impl BayesUct2 {
    pub fn with_c(c: f64) -> Self {
        Self { c }
    }
}

impl<G: Game> SelectStrategy<G> for BayesUct2 {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        let stats = ctx.current_stats();
        2.0 * (stats.num_visits() as f64).max(1.).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        two_ln_n: f64,
    ) -> f64 {
        let snap = ctx.child_snapshot(child_id, children, idx);
        let sigma = snap.posterior_variance.max(0.0).sqrt();
        snap.posterior_mean + self.c * two_ln_n.sqrt() * sigma
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, two_ln_n: f64) -> f64 {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        unvisited_value + self.c * two_ln_n.sqrt()
    }

    fn requirements(&self) -> super::super::config::Requirements {
        super::super::config::Requirements {
            needs_posterior: true,
            ..super::super::config::Requirements::none()
        }
    }
}
