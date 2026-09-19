//! TD(lambda) learning over an n-tuple [`Model`], with eligibility traces and
//! temporal coherence.
//!
//! The value of a state is `V = tanh(sum of selected weights)` from the side to
//! move's point of view. Learning runs on *chains*: one chain per player,
//! holding the states where that player is to move, in order. A chain step
//! takes the previous state of the chain and a target (the discounted value of
//! the chain's next state, or the final result) and applies
//!
//! ```text
//! delta = target - V(prev)
//! e_i   = gamma * lambda * e_i + (1 - V(prev)^2) * count_i(prev)
//! w_i  += (alpha / n_images) * alpha_i * delta * e_i
//! ```
//!
//! where `count_i(prev)` is how often weight `i` is selected by `prev` and
//! `alpha_i` is the temporal-coherence multiplier from [`Tcl`]. Dividing
//! `alpha` by the number of selected weights makes `alpha` the step in value
//! space: with every image selecting one weight, `V` moves about `alpha * delta`.

use crate::tcl::Tcl;
use crate::weights::Model;

#[derive(Clone, Copy, Debug)]
pub struct TdParams {
    pub alpha: f32,
    pub lambda: f32,
    pub gamma: f32,
    pub tcl: bool,
    /// Traces below this magnitude are dropped after they have been applied.
    pub trace_cutoff: f32,
}

struct Chain {
    prev: Vec<u32>,
    has_prev: bool,
    /// Dense eligibility trace; `active` lists the indices where it is nonzero.
    e: Vec<f32>,
    active: Vec<u32>,
}

pub struct Learner {
    model: Model,
    tcl: Tcl,
    params: TdParams,
    chains: Vec<Chain>,
    step_scale: f32,
}

impl Learner {
    pub fn new(model: Model, params: TdParams, n_chains: usize) -> Learner {
        let n = model.geometry().n_weights();
        let step_scale = params.alpha / model.geometry().n_images() as f32;
        let chains = (0..n_chains)
            .map(|_| Chain {
                prev: Vec::new(),
                has_prev: false,
                e: vec![0.0; n],
                active: Vec::new(),
            })
            .collect();
        Learner {
            tcl: Tcl::new(n, params.tcl),
            model,
            params,
            chains,
            step_scale,
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn into_model(self) -> Model {
        self.model
    }

    pub fn tcl(&self) -> &Tcl {
        &self.tcl
    }

    /// Whether `chain` holds a state still waiting for its target.
    pub fn pending(&self, chain: usize) -> bool {
        self.chains[chain].has_prev
    }

    /// Number of weights carrying a live eligibility trace in `chain`.
    pub fn trace_len(&self, chain: usize) -> usize {
        self.chains[chain].active.len()
    }

    /// Record `indices` (the weights selected by the chain's newest state) as
    /// the state the next [`Learner::step`] will update.
    pub fn set_prev(&mut self, chain: usize, indices: &[u32]) {
        let ch = &mut self.chains[chain];
        ch.prev.clear();
        ch.prev.extend_from_slice(indices);
        ch.has_prev = true;
    }

    /// Drop the chain's eligibility trace, so later errors no longer reach the
    /// states before this point.
    pub fn clear_trace(&mut self, chain: usize) {
        let ch = &mut self.chains[chain];
        for &i in &ch.active {
            ch.e[i as usize] = 0.0;
        }
        ch.active.clear();
    }

    /// Apply one TD(lambda) update for the chain's pending state toward
    /// `target`, consuming it. Returns the TD error, or `None` if the chain had
    /// nothing pending.
    pub fn step(&mut self, chain: usize, target: f32) -> Option<f32> {
        let ch = &mut self.chains[chain];
        if !ch.has_prev {
            return None;
        }
        ch.has_prev = false;
        let w = self.model.weights_mut();

        let sum: f32 = ch.prev.iter().map(|&i| w[i as usize]).sum();
        let v = sum.tanh();
        let delta = target - v;
        let grad = 1.0 - v * v;

        let decay = self.params.gamma * self.params.lambda;
        for &i in &ch.active {
            ch.e[i as usize] *= decay;
        }
        if grad > 0.0 {
            for &i in &ch.prev {
                let e = &mut ch.e[i as usize];
                if *e == 0.0 {
                    ch.active.push(i);
                }
                *e += grad;
            }
        }

        let mut k = 0;
        while k < ch.active.len() {
            let i = ch.active[k] as usize;
            let e = ch.e[i];
            let u = delta * e;
            let rate = self.tcl.rate(i, u);
            w[i] += self.step_scale * rate * u;
            if e.abs() < self.params.trace_cutoff {
                ch.e[i] = 0.0;
                ch.active.swap_remove(k);
            } else {
                k += 1;
            }
        }
        Some(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Geometry, Tuple};

    /// One cell, three states: the code of that cell is the weight index, so a
    /// state is just "a", "b" or "c" = index 0, 1, 2.
    fn toy(w: [f32; 3], p: TdParams) -> Learner {
        let g = Geometry::from_tuples(
            3,
            vec![Tuple {
                name: "t".into(),
                squares: vec![0],
            }],
            &[vec![0u8]],
        );
        Learner::new(Model::from_weights(g, w.to_vec()), p, 2)
    }

    fn params(alpha: f32, lambda: f32, tcl: bool) -> TdParams {
        TdParams {
            alpha,
            lambda,
            gamma: 1.0,
            tcl,
            trace_cutoff: 1e-9,
        }
    }

    #[test]
    fn three_ply_td_lambda_matches_the_hand_computed_update() {
        let (alpha, lambda) = (0.5f64, 0.5f64);
        let w0 = [0.2f64, 0.4, -0.3];
        let mut l = toy(w0.map(|x| x as f32), params(alpha as f32, lambda as f32, false));

        // Chain of states a -> b -> c -> terminal win (+1), values from one
        // viewpoint. Expected values below are computed in f64 from the formulas.
        let (va, vb, vc) = (w0[0].tanh(), w0[1].tanh(), w0[2].tanh());
        let (ga, gb, gc) = (1.0 - va * va, 1.0 - vb * vb, 1.0 - vc * vc);

        // t=1: prev a, target V(b). e_a = ga.
        let d1 = vb - va;
        let mut wa = w0[0] + alpha * d1 * ga;
        // t=2: prev b, target V(c). e_a = lambda*ga, e_b = gb.
        let d2 = vc - vb;
        wa += alpha * d2 * lambda * ga;
        let mut wb = w0[1] + alpha * d2 * gb;
        // t=3: prev c, target +1. e_a = lambda^2*ga, e_b = lambda*gb, e_c = gc.
        let d3 = 1.0 - vc;
        wa += alpha * d3 * lambda * lambda * ga;
        wb += alpha * d3 * lambda * gb;
        let wc = w0[2] + alpha * d3 * gc;

        l.set_prev(0, &[0]);
        let got1 = l.step(0, vb as f32).unwrap();
        assert!((got1 as f64 - d1).abs() < 1e-6);
        l.set_prev(0, &[1]);
        let got2 = l.step(0, vc as f32).unwrap();
        assert!((got2 as f64 - d2).abs() < 1e-6);
        l.set_prev(0, &[2]);
        let got3 = l.step(0, 1.0).unwrap();
        assert!((got3 as f64 - d3).abs() < 1e-6);

        let got = l.model().weights();
        for (g, e) in got.iter().zip([wa, wb, wc]) {
            assert!((*g as f64 - e).abs() < 1e-5, "weights {got:?} vs {:?}", [wa, wb, wc]);
        }
        assert!(!l.pending(0));
        assert!(l.step(0, 0.0).is_none(), "a consumed chain has nothing to step");
    }

    #[test]
    fn chains_are_independent_and_clear_trace_cuts_credit() {
        let mut l = toy([0.0; 3], params(1.0, 0.9, false));
        // Chain 0 visits a then b; a clear_trace between them must stop b's
        // error from reaching a.
        l.set_prev(0, &[0]);
        l.step(0, 0.0).unwrap();
        l.clear_trace(0);
        l.set_prev(0, &[1]);
        l.step(0, 1.0).unwrap();
        let w = l.model().weights();
        assert_eq!(w[0], 0.0, "cleared trace: a untouched");
        assert!(w[1] > 0.0);
        // Chain 1 was never used.
        assert!(!l.pending(1));
        assert_eq!(l.trace_len(1), 0);
    }

    #[test]
    fn tcl_damps_a_weight_whose_updates_alternate() {
        let mut on = toy([0.0; 3], params(1.0, 0.0, true));
        let mut off = toy([0.0; 3], params(1.0, 0.0, false));
        for k in 0..40 {
            let target = if k % 2 == 0 { 0.5 } else { -0.5 };
            for l in [&mut on, &mut off] {
                l.set_prev(0, &[0]);
                l.step(0, target).unwrap();
            }
        }
        // Without TCL the weight chases the target back and forth; with it the
        // alternating errors shrink the rate, so the swing is smaller.
        let swing = |l: &Learner| l.model().weights()[0].abs();
        assert!(swing(&on) < swing(&off), "on {} off {}", swing(&on), swing(&off));
        let (n, a) = on.tcl().counters(0);
        assert!(n.abs() < a, "N {n} A {a}");
    }
}
