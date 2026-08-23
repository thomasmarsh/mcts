use mcts::backprop::{self, BackpropStrategy};
use serde::{Deserialize, Serialize};

/// Invokes a continuation after resolving a concrete backpropagation strategy.
pub trait BackpropCont {
    type Output;
    fn call<B: BackpropStrategy + 'static>(self, backprop: B) -> Self::Output;
}

/// `register_backprop!`'s table. `backprop::Classic` was, for a long time,
/// the *only* type in the whole workspace implementing `BackpropStrategy` --
/// a macro rather than a hand-written enum mainly so a second
/// `BackpropStrategy` impl slots in the same way a new `select`/`simulate`
/// family does, without inventing a new pattern. `BayesGaussian`/
/// `BayesNumeric` are the first strategies to actually exercise that: they
/// exist specifically to pair with `select::BayesUct1`/`BayesUct2`
/// (`config::Requirements::needs_posterior`), the first real select<->
/// backprop coupling this axis has ever had to carry.
///
/// There is no `Base.../...` recursive-wrapper split here (contrast
/// `register_select!`/`register_simulate!`): nothing wraps a
/// `BackpropStrategy` anywhere in the codebase.
///
/// There is also no `requirements_of_backprop` -- `BackpropStrategy` itself
/// declares no `requirements()`/`backprop_flags()` method
/// (`SearchConfig::requirements()` only unions `Select`/`Simulate`/
/// `FinalAction`, never `Backprop`), so there is nothing real for such a
/// function to call through to. The select<->backprop coupling
/// `needs_posterior` introduces is instead enforced directly in
/// `SearchConfig::validate()` against `S::Backprop::provides_posterior()`.
macro_rules! register_backprop {
    (
        $(
            $variant:ident { $($field:ident : $ty:ty),* $(,)? } => $ctor:expr
        ),+ $(,)?
    ) => {
        /// The config-IR node for the `backprop` axis.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum BackpropSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        /// Dispatches `spec` to the concrete `BackpropStrategy` it names by
        /// invoking `cont` with it -- see `with_base_select` above.
        pub fn with_backprop<C>(spec: &BackpropSpec, cont: C) -> C::Output
        where
            C: BackpropCont,
        {
            match spec.clone() {
                $(
                    BackpropSpec::$variant { $($field),* } => {
                        let backprop = $ctor;
                        cont.call(backprop)
                    }
                )+
            }
        }
    };
}

register_backprop! {
    Classic {} => backprop::Classic,
    BayesGaussian { prior_variance: f64, obs_variance: f64 } =>
        backprop::BayesGaussian::new(prior_variance, obs_variance),
    BayesNumeric { prior_variance: f64, obs_variance: f64, value_lo: f64, value_hi: f64 } =>
        backprop::BayesNumeric::new(prior_variance, obs_variance, value_lo, value_hi),
}
