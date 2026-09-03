use super::codec::{field, to_snake_case};
use mcts::backprop::{self, BackpropPolicy};
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Invokes a continuation after resolving a concrete backpropagation strategy.
pub trait BackpropCont {
    type Output;
    fn call<B: BackpropPolicy + 'static>(self, backprop: B) -> Self::Output;
}

/// `register_backprop!`'s table. `backprop::Classic` was, for a long time,
/// the *only* type in the whole workspace implementing `BackpropPolicy` --
/// a macro rather than a hand-written enum mainly so a second
/// `BackpropPolicy` impl slots in the same way a new `select`/`simulate`
/// family does, without inventing a new pattern. `BayesGaussian`/
/// `BayesNumeric` are the first strategies to actually exercise that: they
/// exist specifically to pair with `select::BayesUct1`/`BayesUct2`
/// (`config::Requirements::needs_posterior`), the first real select<->
/// backprop coupling this axis has ever had to carry.
///
/// There is no `Base.../...` recursive-wrapper split here (contrast
/// `register_select!`/`register_simulate!`): nothing wraps a
/// `BackpropPolicy` anywhere in the codebase.
///
/// There is also no `requirements_of_backprop` -- `BackpropPolicy` itself
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
        ///
        /// `Serialize`/`Deserialize` are hand-implemented below (routed
        /// through `serde_json::Value`, whose own impls are hand-written in
        /// `serde_json`, not derived) rather than `#[derive]`d: a
        /// `#[serde(tag = "kind", rename_all = "snake_case")]` derive here
        /// would expand into serde's generic `Visitor`/`Content`-buffering
        /// trait machinery, a compile-cost tax paid in full by every game
        /// binary that links this crate under LTO. The concrete match
        /// below produces the identical wire format from the same table
        /// with none of that generic machinery, while still implementing
        /// the real `serde::Serialize`/`Deserialize` traits --
        /// so `SearchSpec`/`CustomStrategySpec`'s own `#[derive(Serialize,
        /// Deserialize)]`, and every `serde_json::from_value::<
        /// CustomStrategySpec>` call site, keep working unmodified.
        #[derive(Debug, Clone, PartialEq)]
        pub enum BackpropSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        impl Serialize for BackpropSpec {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                match self {
                    $(
                        BackpropSpec::$variant { $($field),* } => {
                            #[allow(unused_mut)]
                            let mut map = serializer.serialize_map(None)?;
                            map.serialize_entry("kind", &to_snake_case(stringify!($variant)))?;
                            $(
                                map.serialize_entry(stringify!($field), $field)?;
                            )*
                            map.end()
                        }
                    )+
                }
            }
        }

        impl<'de> Deserialize<'de> for BackpropSpec {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let v = Value::deserialize(deserializer)?;
                let kind = v
                    .get("kind")
                    .and_then(Value::as_str)
                    .ok_or_else(|| D::Error::custom("missing `kind` field"))?;
                $(
                    if kind == to_snake_case(stringify!($variant)) {
                        return Ok(BackpropSpec::$variant {
                            $(
                                $field: field(&v, stringify!($field)).map_err(D::Error::custom)?,
                            )*
                        });
                    }
                )+
                Err(D::Error::custom(format!("unknown backprop kind: {kind:?}")))
            }
        }

        /// Dispatches `spec` to the concrete `BackpropPolicy` it names by
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
    // Power-UCT (Dam et al., IJCAI 2020). `depth == 0` means "every ancestor"
    // (the useful default); `depth > 0` limits the power-mean backup to the
    // nearest N plies above the leaf. `p == 1.0` with `alpha == 0.0` is
    // `Classic` (the strategy disables its own recompute pass). `alpha`
    // blends the power mean with the max over children (`alpha == 1.0` =
    // Full-Bellman max backup, Asai & Wissow AAAI 2025) at any `p`.
    PowerMean { p: f64, alpha: f64, depth: u32 } =>
        backprop::PowerMeanBackprop::new_mixed(
            p, alpha, if depth == 0 { None } else { Some(depth) }),
    // Sarsa-UCT(λ) / TD(λ) bootstrapped backup (Vodopivec et al., JAIR
    // 2017). `lambda == 1.0` is `Classic` (the strategy returns `None` from
    // `td_lambda`). `max_child != 0` is MaxMCTS(λ) (Khandelwal et al. ICML
    // 2016) -- bootstrap from `max` over children instead of the on-path
    // child. `u32` rather than `bool` for `max_child` since the schema has
    // no bool type (same as `depth` above).
    Td { lambda: f64, max_child: u32 } =>
        backprop::TdBackprop::new(lambda, max_child != 0),
    // MENTS soft value backup (Xiao et al., NeurIPS 2019), mellowmax form
    // (Asadi & Littman 2017 -- bounded, unlike literal log-sum-exp).
    // `tau -> 0` is the max backup, `tau -> inf` is `Classic`. Pairs with
    // `select::Ments` (`Requirements::needs_softmax_value`).
    Softmax { tau: f64 } => backprop::SoftmaxBackprop::new(tau),
}
