//! Temporal coherence learning (Beal and Smith): a per-weight learning-rate
//! multiplier `alpha_i = |N_i| / A_i`, where `N_i` accumulates the weight's
//! raw updates and `A_i` their absolute values. A weight whose updates keep
//! one sign has `alpha_i = 1`; one whose updates alternate is damped toward 0.

pub struct Tcl {
    enabled: bool,
    /// `[N_i, A_i]` per weight, `f64` because `A_i` grows for the whole run
    /// while the updates it absorbs stay small.
    counters: Vec<[f64; 2]>,
}

impl Tcl {
    pub fn new(n_weights: usize, enabled: bool) -> Tcl {
        Tcl {
            enabled,
            counters: if enabled {
                vec![[0.0; 2]; n_weights]
            } else {
                Vec::new()
            },
        }
    }

    /// Fold the raw update `u` (before any global learning rate) into weight
    /// `i`'s counters and return its rate multiplier in `[0, 1]`. With TCL
    /// disabled the multiplier is always 1.
    #[inline]
    pub fn rate(&mut self, i: usize, u: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let c = &mut self.counters[i];
        c[0] += u as f64;
        c[1] += (u as f64).abs();
        if c[1] > 0.0 {
            (c[0].abs() / c[1]) as f32
        } else {
            1.0
        }
    }

    /// `(N_i, A_i)` for weight `i`.
    pub fn counters(&self, i: usize) -> (f64, f64) {
        let c = self.counters[i];
        (c[0], c[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_sign_updates_keep_the_rate_at_one() {
        let mut t = Tcl::new(2, true);
        for u in [0.3, 0.05, 0.7, 0.01] {
            assert!((t.rate(0, u) - 1.0).abs() < 1e-6);
        }
        let (n, a) = t.counters(0);
        assert!((n - 1.06).abs() < 1e-9 && (a - 1.06).abs() < 1e-9);
        for u in [-0.3, -0.05] {
            assert!((t.rate(1, u) - 1.0).abs() < 1e-6, "all-negative is constant sign too");
        }
    }

    #[test]
    fn alternating_updates_shrink_the_rate() {
        let mut t = Tcl::new(1, true);
        let mut last = 1.0;
        for k in 0..20 {
            last = t.rate(0, if k % 2 == 0 { 0.4 } else { -0.4 });
        }
        // After 20 alternating +-0.4 updates N = 0 and A = 8.
        assert!(last < 1e-6, "rate {last}");
        // One more +0.4: N = 0.4, A = 8.4.
        assert!((t.rate(0, 0.4) - 0.4 / 8.4).abs() < 1e-6);
    }

    #[test]
    fn a_fresh_weight_has_rate_one_and_disabled_tcl_is_always_one() {
        let mut t = Tcl::new(1, true);
        assert_eq!(t.rate(0, 0.0), 1.0);
        let mut off = Tcl::new(1, false);
        assert_eq!(off.rate(0, 0.5), 1.0);
        assert_eq!(off.rate(0, -0.5), 1.0);
    }
}
