use super::super::node::ChildArray;

pub(crate) fn conjugate_leaf_posterior<A: crate::game::Action>(
    slot: &PosteriorSlot<A>,
    player: usize,
    prior_variance: f64,
    obs_variance: f64,
) -> (f64, f64) {
    let (score, num_visits) = slot.own_observation(player);
    if num_visits == 0 {
        return (0.0, prior_variance);
    }
    let n = num_visits as f64;
    let sample_mean = score / n;
    let posterior_precision = 1.0 / prior_variance + n / obs_variance;
    let posterior_variance = 1.0 / posterior_precision;
    let posterior_mean = posterior_variance * (n * sample_mean / obs_variance);
    (posterior_mean, posterior_variance)
}

pub(crate) fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// Abramowitz & Stegun 7.1.26, max error ~1.5e-7 -- plenty for a UCB-style
// exploration bound, and avoids pulling in a stats crate for one function.
pub(crate) fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

pub(crate) fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Clark (1961)'s closed-form mean/variance of `max(X1, X2)` for two
/// independent Gaussians -- Tesauro/Rajan/Segal 2010 section 4.1, equations
/// following "we reduce the computation time by restructuring the equations
/// above to yield...". `BayesGaussian` folds this pairwise, left to right,
/// over however many children a node has (the paper's simpler O(K)
/// alternative to its O(K^2 log K) min-error pairing scheme).
pub(crate) fn clark_max_of_gaussians(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> (f64, f64) {
    let sigma_m = (sigma1 * sigma1 + sigma2 * sigma2).sqrt();
    if sigma_m == 0.0 {
        return (mu1.max(mu2), 0.0);
    }
    let alpha = (mu1 - mu2) / sigma_m;
    let phi = standard_normal_pdf(alpha);
    let big_phi = standard_normal_cdf(alpha);
    let f1 = alpha * big_phi + phi;
    let f2 =
        alpha * alpha * big_phi * (1.0 - big_phi) + (1.0 - 2.0 * big_phi) * alpha * phi - phi * phi;
    let mean = mu2 + sigma_m * f1;
    let variance =
        sigma2 * sigma2 + (sigma1 * sigma1 - sigma2 * sigma2) * big_phi + sigma_m * sigma_m * f2;
    (mean, variance.max(0.0))
}

/// `min(X1, X2)`'s mean/variance, via `min(X1, X2) = -max(-X1, -X2)` --
/// negating both means leaves the (order-independent) variance unchanged.
/// Needed alongside `clark_max_of_gaussians` because an interior node's
/// combined posterior for a player who is *not* that node's own mover must
/// track the mover's adversarial choice (the mover picks the child that
/// minimizes a non-mover's value, in this codebase's zero-sum two-player
/// convention -- see `update_posterior`'s `mover` parameter).
pub(crate) fn clark_min_of_gaussians(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> (f64, f64) {
    let (neg_mean, variance) = clark_max_of_gaussians(-mu1, sigma1, -mu2, sigma2);
    (-neg_mean, variance)
}

/// Left-folds `combine` (`clark_max_of_gaussians`/`clark_min_of_gaussians`)
/// pairwise over `posteriors`' `(mean, variance)` pairs, returning the
/// combined `(mean, variance)` -- or `None` for an empty input. `combine`
/// takes/returns `(mean, variance)`, but its own two-Gaussian formula is
/// written in terms of *standard deviation* (`sigma1`/`sigma2`), so each
/// fold step must convert the running variance to sigma before feeding it
/// back in, not thread it straight through as if it already were one: doing
/// that squares an already-squared quantity every step, collapsing the
/// combined variance toward zero well before the true combined spread is
/// reached once there are more than two inputs.
pub(crate) fn fold_gaussian_extremum(
    posteriors: impl Iterator<Item = (f64, f64)>,
    combine: fn(f64, f64, f64, f64) -> (f64, f64),
) -> Option<(f64, f64)> {
    posteriors
        .map(|(mu, var)| (mu, var.max(0.0).sqrt()))
        .reduce(|(mu0, sigma0), (mu1, sigma1)| {
            let (mean, variance) = combine(mu0, sigma0, mu1, sigma1);
            (mean, variance.max(0.0).sqrt())
        })
        .map(|(mean, sigma)| (mean, sigma * sigma))
}

/// Grid resolution `BayesNumeric` discretizes each node's posterior PDF
/// over -- also the fixed-size backing array for `PlayerStats::posterior_grid`.
pub const BAYES_GRID_SIZE: usize = 64;

pub(crate) fn grid_points(lo: f64, hi: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut points = [0.0; BAYES_GRID_SIZE];
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    for (i, p) in points.iter_mut().enumerate() {
        *p = lo + step * i as f64;
    }
    points
}

pub(crate) fn trapezoid_integral(values: &[f64; BAYES_GRID_SIZE], step: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..BAYES_GRID_SIZE - 1 {
        sum += (values[i] + values[i + 1]) * 0.5 * step;
    }
    sum
}

pub(crate) fn normal_pdf_grid(
    mean: f64,
    variance: f64,
    lo: f64,
    hi: f64,
) -> [f64; BAYES_GRID_SIZE] {
    let sigma = variance.max(1e-12).sqrt();
    let points = grid_points(lo, hi);
    let mut pdf = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        let z = (points[i] - mean) / sigma;
        pdf[i] = (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    }
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mass = trapezoid_integral(&pdf, step);
    if mass > 0.0 {
        for v in pdf.iter_mut() {
            *v /= mass;
        }
    }
    pdf
}

pub(crate) fn cdf_from_pdf(pdf: &[f64; BAYES_GRID_SIZE], step: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut cdf = [0.0; BAYES_GRID_SIZE];
    let mut acc = 0.0;
    for i in 0..BAYES_GRID_SIZE {
        if i > 0 {
            acc += (pdf[i - 1] + pdf[i]) * 0.5 * step;
        }
        cdf[i] = acc;
    }
    cdf
}

pub(crate) fn pdf_from_cdf(cdf: &[f64; BAYES_GRID_SIZE], step: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut pdf = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        pdf[i] = if i == 0 {
            (cdf[1] - cdf[0]) / step
        } else if i == BAYES_GRID_SIZE - 1 {
            (cdf[i] - cdf[i - 1]) / step
        } else {
            (cdf[i + 1] - cdf[i - 1]) / (2.0 * step)
        }
        .max(0.0);
    }
    pdf
}

pub(crate) fn mean_variance_from_pdf(pdf: &[f64; BAYES_GRID_SIZE], lo: f64, hi: f64) -> (f64, f64) {
    let points = grid_points(lo, hi);
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut weighted = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        weighted[i] = pdf[i] * points[i];
    }
    let mean = trapezoid_integral(&weighted, step);
    let mut sq = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        sq[i] = pdf[i] * (points[i] - mean) * (points[i] - mean);
    }
    let variance = trapezoid_integral(&sq, step).max(0.0);
    (mean, variance)
}

/// Numeric MAX distribution of `k` independent grid-discretized PDFs: a
/// value `v` is `<= max(X1..Xk)` iff every `Xi <= v`, so the parent's CDF is
/// the elementwise product of the children's CDFs; differentiating that
/// product back to a PDF (via central differences) gives the parent's
/// distribution (Tesauro/Rajan/Segal 2010 section 4, "a more convenient
/// calculation is to first compute the parent CDF... as the product of
/// child CDFs"). `BayesNumeric`'s exact (non-Gaussian-approximated)
/// counterpart to `clark_max_of_gaussians`.
pub(crate) fn numeric_max_of_pdfs(
    pdfs: &[[f64; BAYES_GRID_SIZE]],
    lo: f64,
    hi: f64,
) -> [f64; BAYES_GRID_SIZE] {
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut cdf_product = [1.0; BAYES_GRID_SIZE];
    for pdf in pdfs {
        let cdf = cdf_from_pdf(pdf, step);
        for i in 0..BAYES_GRID_SIZE {
            cdf_product[i] *= cdf[i];
        }
    }
    pdf_from_cdf(&cdf_product, step)
}

/// Numeric MIN distribution of `k` independent grid-discretized PDFs: a
/// value `v` is `>= min(X1..Xk)`'s complement iff every `Xi > v`, so
/// `P(min <= v) = 1 - product(1 - CDF_i(v))` (the survival-function dual of
/// `numeric_max_of_pdfs`'s CDF product). Needed for the same reason
/// `clark_min_of_gaussians` is: an interior node's posterior must be
/// combined via MIN, not MAX, for every player who isn't that node's own
/// mover (see `update_posterior`'s `mover` parameter).
pub(crate) fn numeric_min_of_pdfs(
    pdfs: &[[f64; BAYES_GRID_SIZE]],
    lo: f64,
    hi: f64,
) -> [f64; BAYES_GRID_SIZE] {
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut survival_product = [1.0; BAYES_GRID_SIZE];
    for pdf in pdfs {
        let cdf = cdf_from_pdf(pdf, step);
        for i in 0..BAYES_GRID_SIZE {
            survival_product[i] *= 1.0 - cdf[i];
        }
    }
    let mut cdf_min = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        cdf_min[i] = 1.0 - survival_product[i];
    }
    pdf_from_cdf(&cdf_min, step)
}

use super::*;

/// Tesauro/Rajan/Segal 2010's Bayesian backprop, Gaussian-approximation
/// variant ("_g" in the paper): each node's posterior is a Gaussian, and an
/// interior node's posterior is the MAX-of-children distribution computed
/// via `clark_max_of_gaussians`. Paired with `select::BayesUct1`/
/// `BayesUct2` (`select/bayes.rs`), which read the `posterior_mean`/
/// `posterior_variance` this writes.
#[derive(Debug, Clone, Copy)]
pub struct BayesGaussian {
    /// Prior variance on a node's true value before any observations.
    pub prior_variance: f64,
    /// Assumed variance of a single rollout's return, i.e. the conjugate
    /// update's observation noise.
    pub obs_variance: f64,
}

impl Default for BayesGaussian {
    fn default() -> Self {
        Self {
            prior_variance: 1.0,
            obs_variance: 1.0,
        }
    }
}

impl BayesGaussian {
    pub fn new(prior_variance: f64, obs_variance: f64) -> Self {
        Self {
            prior_variance,
            obs_variance,
        }
    }
}

impl BackpropPolicy for BayesGaussian {
    fn provides_posterior(&self) -> bool {
        true
    }

    fn update_posterior<A: crate::game::Action>(
        &self,
        player: usize,
        mover: usize,
        slot: &PosteriorSlot<A>,
        own_children: Option<&ChildArray<A>>,
    ) {
        let combine = if player == mover {
            clark_max_of_gaussians
        } else {
            clark_min_of_gaussians
        };
        let combined = own_children.and_then(|children| {
            fold_gaussian_extremum(
                (0..children.len())
                    .filter(|&i| children.is_explored(i))
                    .map(|i| children.posterior(i, player)),
                combine,
            )
        });
        let (mean, variance) = combined.unwrap_or_else(|| {
            conjugate_leaf_posterior(slot, player, self.prior_variance, self.obs_variance)
        });
        slot.set(player, mean, variance);
    }
}

/// Tesauro/Rajan/Segal 2010's Bayesian backprop, numeric-integration
/// variant ("_n" in the paper): each node's posterior is a discretized PDF
/// over `BAYES_GRID_SIZE` grid points spanning `[value_lo, value_hi]`, and
/// an interior node's posterior is the exact (to grid resolution) MAX-of-
/// children distribution computed via `numeric_max_of_pdfs`.
#[derive(Debug, Clone, Copy)]
pub struct BayesNumeric {
    pub prior_variance: f64,
    pub obs_variance: f64,
    /// Grid lower/upper bounds -- must cover the game's real utility range;
    /// defaults to this codebase's symmetric `[-1, 1]` convention.
    pub value_lo: f64,
    pub value_hi: f64,
}

impl Default for BayesNumeric {
    fn default() -> Self {
        Self {
            prior_variance: 1.0,
            obs_variance: 1.0,
            value_lo: -1.0,
            value_hi: 1.0,
        }
    }
}

impl BayesNumeric {
    pub fn new(prior_variance: f64, obs_variance: f64, value_lo: f64, value_hi: f64) -> Self {
        Self {
            prior_variance,
            obs_variance,
            value_lo,
            value_hi,
        }
    }

    fn leaf_pdf<A: crate::game::Action>(
        &self,
        slot: &PosteriorSlot<A>,
        player: usize,
    ) -> [f64; BAYES_GRID_SIZE] {
        let (mean, variance) =
            conjugate_leaf_posterior(slot, player, self.prior_variance, self.obs_variance);
        normal_pdf_grid(mean, variance, self.value_lo, self.value_hi)
    }
}

impl BackpropPolicy for BayesNumeric {
    fn provides_posterior(&self) -> bool {
        true
    }

    fn update_posterior<A: crate::game::Action>(
        &self,
        player: usize,
        mover: usize,
        slot: &PosteriorSlot<A>,
        own_children: Option<&ChildArray<A>>,
    ) {
        let child_pdfs: Vec<[f64; BAYES_GRID_SIZE]> = own_children
            .map(|children| {
                (0..children.len())
                    .filter(|&i| children.is_explored(i))
                    .map(|i| {
                        children.posterior_grid(i, player).unwrap_or_else(|| {
                            let (mu, var) = children.posterior(i, player);
                            normal_pdf_grid(mu, var, self.value_lo, self.value_hi)
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let pdf = if child_pdfs.is_empty() {
            self.leaf_pdf(slot, player)
        } else if player == mover {
            numeric_max_of_pdfs(&child_pdfs, self.value_lo, self.value_hi)
        } else {
            numeric_min_of_pdfs(&child_pdfs, self.value_lo, self.value_hi)
        };
        let (mean, variance) = mean_variance_from_pdf(&pdf, self.value_lo, self.value_hi);
        slot.set(player, mean, variance);
        slot.set_grid(player, pdf);
    }
}

#[cfg(test)]
mod bayes_tests {
    use super::*;
    use crate::algorithms::mcts::node::NodeStats;

    #[test]
    fn conjugate_posterior_matches_prior_when_unvisited() {
        let stats = NodeStats::new(1, false);
        let slot: PosteriorSlot<u8> = PosteriorSlot::Root(&stats);
        let (mean, variance) = conjugate_leaf_posterior(&slot, 0, 2.0, 1.0);
        assert_eq!(mean, 0.0);
        assert_eq!(variance, 2.0);
    }

    #[test]
    fn conjugate_posterior_shrinks_toward_sample_mean_with_visits() {
        let stats = NodeStats::new(1, false);
        // 10 observations averaging 0.5.
        for _ in 0..10 {
            stats.update(&[0.5]);
        }
        let slot: PosteriorSlot<u8> = PosteriorSlot::Root(&stats);
        let (mean, variance) = conjugate_leaf_posterior(&slot, 0, 1.0, 1.0);
        // precision = 1/1 + 10/1 = 11; mean = (10*0.5/1) / 11 = 5/11
        assert!((mean - 5.0 / 11.0).abs() < 1e-9);
        assert!((variance - 1.0 / 11.0).abs() < 1e-9);
        // More visits should pull the posterior mean closer to the sample
        // mean than a single observation would.
        assert!(mean > 0.4 && mean < 0.5);
    }

    #[test]
    fn clark_max_of_identical_gaussians_is_symmetric() {
        let (mean, variance) = clark_max_of_gaussians(0.0, 1.0, 0.0, 1.0);
        // max of two i.i.d. N(0,1): known closed form mean = 1/sqrt(pi).
        assert!(
            (mean - std::f64::consts::FRAC_1_SQRT_2 * (2.0 / std::f64::consts::PI).sqrt()).abs()
                < 1e-6
        );
        assert!(variance > 0.0 && variance < 1.0);
    }

    #[test]
    fn clark_max_dominated_by_higher_mean_when_variance_tiny() {
        // With near-zero variance, max(X1, X2) collapses to whichever has
        // the higher mean.
        let (mean, variance) = clark_max_of_gaussians(1.0, 1e-6, -1.0, 1e-6);
        assert!((mean - 1.0).abs() < 1e-3);
        assert!(variance < 1e-3);
    }

    #[test]
    fn numeric_max_of_pdfs_matches_clark_approximately() {
        let lo = -6.0;
        let hi = 6.0;
        let pdf1 = normal_pdf_grid(0.5, 1.0, lo, hi);
        let pdf2 = normal_pdf_grid(-0.5, 1.5, lo, hi);
        let combined = numeric_max_of_pdfs(&[pdf1, pdf2], lo, hi);
        let (numeric_mean, numeric_variance) = mean_variance_from_pdf(&combined, lo, hi);
        let (clark_mean, clark_variance) = clark_max_of_gaussians(0.5, 1.0, -0.5, 1.5f64.sqrt());
        assert!(
            (numeric_mean - clark_mean).abs() < 0.05,
            "numeric={numeric_mean} clark={clark_mean}"
        );
        assert!(
            (numeric_variance - clark_variance).abs() < 0.1,
            "numeric={numeric_variance} clark={clark_variance}"
        );
    }

    #[test]
    fn bayes_gaussian_leaf_then_max_combine() {
        let strategy = BayesGaussian::new(1.0, 1.0);
        assert!(strategy.provides_posterior());

        // Leaf node: no children, posterior comes from own observations.
        let leaf_stats = NodeStats::new(1, false);
        leaf_stats.update(&[1.0]);
        let leaf_slot = PosteriorSlot::Root(&leaf_stats);
        strategy.update_posterior::<u8>(0, 0, &leaf_slot, None);
        let (leaf_mean, _) = leaf_stats.posterior(0);
        assert!(leaf_mean > 0.0);
    }

    /// Regression test for a fold-wiring bug found via
    /// `examples/bayes_uct_bandit_tree.rs`: combining more than two
    /// children's posteriors collapsed the running variance toward zero
    /// (each fold step squared an already-squared quantity), which made
    /// `BayesUct1`/`BayesUct2`'s exploration term vanish and left search
    /// unable to correct an early wrong guess. Five children -- reproducing
    /// the exact case that surfaced it -- should combine to a variance in
    /// the same order of magnitude as its inputs, not many orders smaller.
    #[test]
    fn fold_gaussian_extremum_does_not_collapse_variance_past_two_children() {
        let arms: [(f64, f64); 5] = [
            (-0.020548, 0.006849),
            (0.171875, 0.015625),
            (0.076190, 0.009524),
            (0.750000, 0.083333),
            (0.343750, 0.031250),
        ];
        let (_, variance) =
            fold_gaussian_extremum(arms.into_iter(), clark_min_of_gaussians).unwrap();
        let min_input_variance = arms.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min);
        assert!(
            variance > min_input_variance * 0.1,
            "variance {variance:e} collapsed far below the smallest input variance {min_input_variance:e}"
        );
    }
}
