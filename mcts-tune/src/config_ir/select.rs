use mcts::game::Game;
use mcts::select::{self, SelectStrategy};
use mcts::Requirements;
use serde::{Deserialize, Serialize};

/// Invokes a continuation after resolving a concrete selection strategy.
pub trait SelectCont<G: Game> {
    type Output;
    fn call<S: SelectStrategy<G> + 'static>(self, select: S) -> Self::Output;
}

/// `register_select!`'s table, expanded into `BaseSelectSpec`/`SelectSpec`
/// and their dispatchers together so none of them can silently omit a
/// family the others still know about.
///
/// Each row is `Variant { field: ty, ... } => expr`, where `expr` is
/// evaluated with the row's fields bound by value (see `with_select`'s
/// `match spec.clone()`) and must produce a value implementing
/// `SelectStrategy<G>` for whatever `G` the call site is generic over.
///
/// `EpsilonGreedy` is not a row here -- it wraps an *inner* spec, and is
/// handled by hand below on both enums instead. Its inner spec is a
/// `BaseSelectSpec` (the table's variants only, no `EpsilonGreedy` of its
/// own) rather than a recursive `Box<SelectSpec>`: `with_select`'s
/// `EpsilonGreedy` arm has to call some dispatcher generically on `G`, and
/// if that dispatcher could itself recurse into another `EpsilonGreedy` arm,
/// rustc has to monomorphize `with_select<G, EpsilonGreedyCont<EpsilonGreedyCont<...>>>`
/// unboundedly at compile time (it can't see that real specs only ever
/// nest one level deep) and hits its instantiation recursion limit. Two
/// non-mutually-recursive enums route around that -- `EpsilonGreedy(EpsilonGreedy(x))`
/// stops being representable at all, matching the fact that it would be a
/// no-op composition anyway.
macro_rules! register_select {
    (
        $(
            $variant:ident { $($field:ident : $ty:ty),* $(,)? } => $ctor:expr
        ),+ $(,)?
    ) => {
        /// The inner spec an `EpsilonGreedy` wraps -- the table's families,
        /// with no `EpsilonGreedy` variant of its own (see
        /// `register_select!`'s doc comment on why).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum BaseSelectSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        /// Dispatches `spec` to the concrete `SelectStrategy<G>` it names by
        /// invoking `cont` with it -- see this module's doc comment for why
        /// this is a continuation call rather than a return value.
        pub fn with_base_select<G, C>(spec: &BaseSelectSpec, cont: C) -> C::Output
        where
            G: Game + 'static,
            C: SelectCont<G>,
        {
            match spec.clone() {
                $(
                    BaseSelectSpec::$variant { $($field),* } => {
                        let select = $ctor;
                        cont.call(select)
                    }
                )+
            }
        }

        /// The config-IR node for the `select` axis: every `BaseSelectSpec`
        /// family, plus `EpsilonGreedy` wrapping one of them.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum SelectSpec {
            $(
                $variant { $($field: $ty),* },
            )+
            EpsilonGreedy {
                epsilon: f64,
                inner: BaseSelectSpec,
            },
        }

        /// Dispatches `spec` the same way `with_base_select` does, plus the
        /// `EpsilonGreedy` wrapper.
        pub fn with_select<G, C>(spec: &SelectSpec, cont: C) -> C::Output
        where
            G: Game + 'static,
            C: SelectCont<G>,
        {
            match spec.clone() {
                $(
                    SelectSpec::$variant { $($field),* } => {
                        let select = $ctor;
                        cont.call(select)
                    }
                )+
                SelectSpec::EpsilonGreedy { epsilon, inner } => {
                    with_base_select::<G, _>(&inner, EpsilonGreedyCont { epsilon, cont })
                }
            }
        }
    };
}

register_select! {
    Ucb1 { c: f64 } => select::Ucb1::with_c(c),
    Ucb1Tuned { c: f64 } => select::Ucb1Tuned::with_c(c),
    Amaf { alpha: f64, c: f64 } => select::Amaf::new().alpha(alpha).exploration_constant(c),
    Rave { threshold: u32, schedule: select::RaveSchedule, ucb: select::RaveUcb } =>
        select::Rave::new(threshold, schedule, ucb),
    UctPn { c: f64, c_pn: f64 } => select::UctPn::with_c(c, c_pn),
    ProgressiveHistory { c: f64, ph_weight: f64 } =>
        select::ProgressiveHistory::new(select::Ucb1::with_c(c), ph_weight),
    BayesUct1 { c: f64 } => select::BayesUct1::with_c(c),
    BayesUct2 { c: f64 } => select::BayesUct2::with_c(c),
}

/// Forwards a resolved `S: SelectStrategy<G>` on to `cont`, wrapped in
/// `select::EpsilonGreedy` -- `with_select`'s handling of the recursive
/// `EpsilonGreedy` spec variant.
struct EpsilonGreedyCont<C> {
    epsilon: f64,
    cont: C,
}

impl<G, C> SelectCont<G> for EpsilonGreedyCont<C>
where
    G: Game + 'static,
    C: SelectCont<G>,
{
    type Output = C::Output;

    fn call<S: SelectStrategy<G> + 'static>(self, select: S) -> C::Output {
        let wrapped = select::EpsilonGreedy::<G, S>::new()
            .epsilon(self.epsilon)
            .inner(select);
        self.cont.call(wrapped)
    }
}

/// A `SelectCont` whose `Output` is just the resolved component's own
/// `Requirements` -- reusing `with_select`'s dispatch to compute this means
/// it's derived from the *actual* constructed component calling its own
/// `SelectStrategy::requirements()` (see `config::Requirements`'s doc
/// comment in `mcts`), not a second, independently-drifting description of
/// the same table.
pub(super) struct RequirementsCont<G>(pub(super) std::marker::PhantomData<G>);

impl<G: Game> SelectCont<G> for RequirementsCont<G> {
    type Output = Requirements;

    fn call<S: SelectStrategy<G>>(self, select: S) -> Requirements {
        select.requirements()
    }
}

pub fn requirements_of<G: Game + 'static>(spec: &SelectSpec) -> Requirements {
    with_select::<G, _>(spec, RequirementsCont(std::marker::PhantomData))
}
