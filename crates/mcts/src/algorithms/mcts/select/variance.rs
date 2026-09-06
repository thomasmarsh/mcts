use super::super::index::Id;
use super::super::node::{self, ChildArray};
use super::proven_exact_value;
use super::SelectContext;
use super::SelectPolicy;
use crate::game::Game;

////////////////////////////////////////////////////////////////////////////////
// Shared sample-variance helper (used by `Ucb1Tuned` and `UcbV`).

/// `max(0, Σq²/n − q̄²)` -- the sample-variance estimate UCB1-Tuned (Auer
/// 2002) and UCB-V (Audibert, Munos, Szepesvári 2009) share. `n` and `mean`
/// must be the *same* quantities the caller's exploration term uses -- here
/// `total_visits` and `exploitation_score`, matching `Ucb1Tuned`'s
/// pre-existing choice, so the in-flight (virtual-loss) adjustment is
/// consistent between the two halves of the score.
///
/// Note the asymmetry, inherited from `Ucb1Tuned`: `sum_squared_score` is
/// *not* virtual-loss-adjusted (backprop accumulates `reward²`
/// unconditionally) while `mean` and `n` are. An in-flight edge therefore
/// reads a slightly inflated variance -- marginally more exploration of
/// edges other workers are already on, which is mild and in the safe
/// direction (it does not concentrate workers).
#[inline]
pub(crate) fn sample_variance_raw(sum_squared_score: f64, n: f64, mean: f64) -> f64 {
    0f64.max(sum_squared_score / n - mean * mean)
}

#[inline]
pub(crate) fn sample_variance(snap: &node::ChildSnapshot) -> f64 {
    sample_variance_raw(
        snap.sum_squared_score,
        snap.total_visits() as f64,
        snap.exploitation_score(),
    )
}

////////////////////////////////////////////////////////////////////////////////
// UCB1-Tuned (Auer, Cesa-Bianchi, Fischer 2002). Relocated from `ucb.rs`
// verbatim -- it is a variance-aware bandit, cohesive with the family here.

#[derive(Clone)]
pub struct Ucb1Tuned {
    pub exploration_constant: f64,
}

impl Default for Ucb1Tuned {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl Ucb1Tuned {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

pub(crate) const VARIANCE_UPPER_BOUND: f64 = 1.;

#[inline(always)]
pub(crate) fn ucb1_tuned(
    exploration_constant: f64,
    exploit: f64,
    sample_variance: f64,
    visits_fraction: f64,
) -> f64 {
    exploit
        + (visits_fraction * VARIANCE_UPPER_BOUND.min(sample_variance)
            + exploration_constant * visits_fraction.sqrt())
}

impl<G: Game> SelectPolicy<G> for Ucb1Tuned {
    fn label(&self) -> String {
        "ucb1_tuned".into()
    }

    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits() as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        let snap = ctx.child_snapshot(child_id, children, idx);
        let exploit = snap.exploitation_score();
        let num_visits = snap.total_visits();
        let visits_fraction = parent_log / num_visits as f64;

        ucb1_tuned(
            self.exploration_constant,
            exploit,
            sample_variance(&snap),
            visits_fraction,
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> Self::Score {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        ucb1_tuned(
            self.exploration_constant,
            unvisited_value,
            VARIANCE_UPPER_BOUND,
            parent_log,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
// UCB-V (Audibert, Munos, Szepesvári, "Exploration-exploitation tradeoff
// using variance estimates in multi-armed bandits", TCS 2009).

/// UCB1 with the exploration term scaled by each child's observed sample
/// variance, so a low-variance child is trusted with fewer samples and a
/// high-variance one keeps getting explored. Reads
/// `ChildSnapshot::sum_squared_score`, populated unconditionally by every
/// backprop -- no backprop coupling, no new `Requirements` flag.
///
/// `c` scales the range/bias term (the paper's `3b`); default `√2`, the
/// shared exploration-constant default, so `c ≈ 1` is near the textbook
/// unit-range form. Utilities are on the `[-1, 1]` scale, treated as unit
/// range like `Ucb1Tuned`.
#[derive(Clone)]
pub struct UcbV {
    pub exploration_constant: f64,
}

impl Default for UcbV {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl UcbV {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

#[inline]
pub(crate) fn ucb_v(c: f64, exploit: f64, variance: f64, ln_n: f64, n: f64) -> f64 {
    exploit + (2.0 * variance * ln_n / n).sqrt() + c * 3.0 * ln_n / n
}

impl<G: Game> SelectPolicy<G> for UcbV {
    fn label(&self) -> String {
        "ucb_v".into()
    }

    type Score = f64;
    type Aux = f64;

    fn supports_ismcts() -> bool {
        true
    }

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits() as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        if let Some(v) = proven_exact_value(ctx, children, idx) {
            return v;
        }
        let snap = ctx.child_snapshot(child_id, children, idx);
        let n = snap.total_visits() as f64;
        let ln_n = if children.is_growable() {
            ((children.availability(idx) as f64).max(1.)).ln()
        } else {
            parent_log
        };
        ucb_v(
            self.exploration_constant,
            snap.exploitation_score(),
            sample_variance(&snap),
            ln_n,
            n,
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> f64 {
        let u = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        u + self.exploration_constant * parent_log.sqrt()
    }
}

////////////////////////////////////////////////////////////////////////////////
// KL-UCB (Garivier & Cappé, "The KL-UCB algorithm for bounded stochastic
// bandits and beyond", COLT 2011).

const KL_FLOOR: f64 = 1e-9;
const KL_BISECTION_ITERS: usize = 32;

/// Binary relative entropy `d(p‖q)` on `[0, 1]`, clamping both args off the
/// boundary so the logs stay finite. `d(p, p) = 0`; increasing in `|q − p|`.
#[inline]
fn bin_kl(p: f64, q: f64) -> f64 {
    let p = p.clamp(KL_FLOOR, 1.0 - KL_FLOOR);
    let q = q.clamp(KL_FLOOR, 1.0 - KL_FLOOR);
    p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
}

/// Largest `q ∈ [p_hat, 1]` with `n · d(p_hat ‖ q) ≤ budget`, by bisection.
/// `f(q) = n·d(p_hat, q) − budget` is `≤ 0` at `q = p_hat` and increasing on
/// `[p_hat, 1]`, so a fixed-iteration bisection converges to the upper root.
#[inline]
fn kl_ucb_upper(p_hat: f64, n: f64, budget: f64) -> f64 {
    debug_assert!(n > 0.0);
    if budget <= 0.0 {
        return p_hat;
    }
    let mut lo = p_hat;
    let mut hi = 1.0 - KL_FLOOR;
    for _ in 0..KL_BISECTION_ITERS {
        let mid = 0.5 * (lo + hi);
        if n * bin_kl(p_hat, mid) > budget {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// The tightest index-policy upper confidence bound for bounded rewards --
/// the largest value `u` whose binary-KL distance from the empirical mean is
/// within an `(ln N + c·ln ln N) / n` budget. The returned score IS `u`
/// (shifted back to `[-1, 1]`), not `mean + bonus`.
///
/// `c` scales the second-order `ln ln N` term (the paper fixes `c = 3` for
/// the proof, `c = 0` in practice); the tuner's `[0, 3]` range spans both.
/// Proven children bypass the KL solve (`proven_exact_value`) -- the bound is
/// meaningless at `q̄ ∈ {0, 1}`.
#[derive(Clone)]
pub struct KlUcb {
    pub exploration_constant: f64,
}

impl Default for KlUcb {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl KlUcb {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

impl<G: Game> SelectPolicy<G> for KlUcb {
    fn label(&self) -> String {
        "kl_ucb".into()
    }

    type Score = f64;
    type Aux = (f64, f64); // (ln_n, ln_ln_n)

    fn supports_ismcts() -> bool {
        true
    }

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> (f64, f64) {
        let n = (ctx.current_stats().num_visits() as f64).max(1.);
        let ln_n = n.ln();
        let ln_ln_n = n.max(3.0).ln().ln().max(0.0);
        (ln_n, ln_ln_n)
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        aux: (f64, f64),
    ) -> f64 {
        if let Some(v) = proven_exact_value(ctx, children, idx) {
            return v;
        }
        let (mut ln_n, ln_ln_n) = aux;
        let snap = ctx.child_snapshot(child_id, children, idx);
        let n = snap.total_visits() as f64;
        let p_hat = ((snap.exploitation_score() + 1.0) / 2.0).clamp(KL_FLOOR, 1.0 - KL_FLOOR);
        if children.is_growable() {
            ln_n = ((children.availability(idx) as f64).max(1.)).ln();
        }
        let rhs = ln_n + self.exploration_constant * ln_ln_n;
        kl_ucb_upper(p_hat, n, rhs) * 2.0 - 1.0
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: (f64, f64)) -> f64 {
        let (ln_n, _) = aux;
        let u = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        u + self.exploration_constant.max(1.0) * ln_n.sqrt()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::super::proven_to_utility;
    use super::node::Proven;
    use super::*;

    #[test]
    fn sample_variance_raw_matches_hand_computed() {
        assert!((sample_variance_raw(2.5, 4.0, 0.5) - 0.375).abs() < 1e-12);
        // A negative-would-be case clamps to zero.
        assert_eq!(sample_variance_raw(0.1, 4.0, 0.5), 0.0);
    }

    #[test]
    fn ucb_v_matches_hand_computed() {
        let ln_n = 100f64.ln();
        let got = ucb_v(1.0, 0.2, 0.09, ln_n, 10.0);
        assert!((got - 1.8694626).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn ucb_v_zero_variance_drops_sqrt_term() {
        let ln_n = 100f64.ln();
        assert_eq!(
            ucb_v(1.3, 0.2, 0.0, ln_n, 10.0),
            0.2 + 1.3 * 3.0 * ln_n / 10.0
        );
    }

    #[test]
    fn bin_kl_basics() {
        assert!(bin_kl(0.5, 0.5) < 1e-12);
        assert!(bin_kl(0.3, 0.3) < 1e-12);
        let q = 0.6;
        assert!((bin_kl(KL_FLOOR, q) - (-(1.0 - q).ln())).abs() < 1e-6);
        assert!(bin_kl(0.2, 0.8) > bin_kl(0.2, 0.4));
    }

    #[test]
    fn kl_ucb_upper_closed_form_at_zero() {
        let got = kl_ucb_upper(KL_FLOOR, 5.0, 2.0);
        let expected = 1.0 - (-2.0f64 / 5.0).exp();
        assert!((got - expected).abs() < 1e-4, "got {got}");
    }

    #[test]
    fn kl_ucb_upper_satisfies_defining_inequality() {
        for (p_hat, n, budget) in [(0.4, 8.0, 1.5), (0.6, 20.0, 0.8), (0.1, 3.0, 2.0)] {
            let u = kl_ucb_upper(p_hat, n, budget);
            assert!(p_hat <= u && u <= 1.0);
            if u < 1.0 - KL_FLOOR * 10.0 {
                assert!(
                    (n * bin_kl(p_hat, u) - budget).abs() < 1e-3,
                    "({p_hat},{n},{budget}) -> {u}"
                );
            } else {
                assert!(n * bin_kl(p_hat, u) <= budget + 1e-9);
            }
        }
    }

    #[test]
    fn kl_ucb_upper_huge_budget_saturates() {
        let u = kl_ucb_upper(0.5, 10.0, 1e6);
        assert!((1.0 - u).abs() < KL_FLOOR * 10.0);
    }

    #[test]
    fn kl_ucb_upper_zero_budget_is_p_hat() {
        assert_eq!(kl_ucb_upper(0.37, 4.0, 0.0), 0.37);
    }

    #[test]
    fn proven_to_utility_maps_per_d4() {
        assert_eq!(proven_to_utility(Proven::Win(1), 1), Some(1.0));
        assert_eq!(proven_to_utility(Proven::Win(0), 1), Some(-1.0));
        assert_eq!(proven_to_utility(Proven::Draw, 1), Some(0.0));
        assert_eq!(proven_to_utility(Proven::Unproven, 1), None);
    }
}
