//! Prototype: a JSON-serializable config IR for the `select` axis, plus a
//! `register_select!`-generated dispatcher that turns a runtime `SelectSpec`
//! into a compile-time-monomorphized `mcts::select::SelectStrategy<G>`.
//!
//! This is a proof of concept for one axis of the "config algebra" PLAN.md's
//! Composable Algebra section asks for -- not a replacement for
//! `make_candidate`'s existing four-axis dispatch (which this file doesn't
//! touch). Two things it demonstrates that `make_candidate`'s
//! `match p.family.as_str()` can't:
//!
//! - **A single source of truth.** `register_select!`'s table is the only
//!   place that names a `select::*` type; the `SelectSpec` enum, the runtime
//!   dispatcher, and the `Requirements` computation are all generated from
//!   it, so they can't drift apart the way `TrialParams`/the `match`
//!   arms/`strategy_tuner_info`'s conditions can today (three independently
//!   hand-maintained descriptions of the same thing).
//! - **Recursive composition.** `SelectSpec::EpsilonGreedy` wraps an
//!   arbitrary inner `SelectSpec`, matching `select::EpsilonGreedy<G, S>`'s
//!   own genericity -- `make_candidate` has no equivalent (its
//!   `TrialParams` is a single flat struct), which is exactly the gap that
//!   keeps e.g. nested-MCTS's inner strategy from being independently
//!   configurable.
//!
//! ## Why continuation-passing, not `Box<dyn SelectStrategy<G>>`
//!
//! `SelectStrategy::Score`/`Aux` are per-impl associated types (`Ucb1`'s is
//! `f64`, `UctPn`'s is also `f64` but for a different reason, a strategy
//! could in principle pick anything `PartialOrd + Copy`), so there's no
//! single object-safe trait to box: the concrete type has to be resolved at
//! compile time, exactly like `mcts-tune::make_candidate` already requires
//! (its whole reason for existing is picking `G`/`Sel`/`Sim`/`Bp`/`Fa` type
//! parameters at runtime). `with_select` resolves this the same way that
//! function does deeper in its own call graph: a generic continuation
//! (`SelectCont::call<S: SelectStrategy<G>>`) gets *invoked* once the
//! concrete type is known, rather than a value of that type being *returned*
//! -- higher-rank polymorphism via a trait with a generic method, standing
//! in for a boxed trait object.

use mcts::game::Game;
use mcts::select::{self, SelectStrategy};
use mcts::Requirements;
use serde::{Deserialize, Serialize};

/// A continuation that can be invoked with any concrete `S: SelectStrategy<G>`
/// -- the "callback" side of the CPS dispatch this module's doc comment
/// describes. `requirements_of` and any real caller building a `TreeSearch`
/// each implement this once, for their own `Output`.
pub trait SelectCont<G: Game> {
    type Output;
    fn call<S: SelectStrategy<G>>(self, select: S) -> Self::Output;
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
            G: Game,
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
            G: Game,
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
    Rave { threshold: u32, c: f64 } => select::Rave::default()
        .threshold(threshold)
        .ucb(select::RaveUcb::Ucb1 { exploration_constant: c }),
    UctPn { c: f64, c_pn: f64 } => select::UctPn::with_c(c, c_pn),
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
    G: Game,
    C: SelectCont<G>,
{
    type Output = C::Output;

    fn call<S: SelectStrategy<G>>(self, select: S) -> C::Output {
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
struct RequirementsCont<G>(std::marker::PhantomData<G>);

impl<G: Game> SelectCont<G> for RequirementsCont<G> {
    type Output = Requirements;

    fn call<S: SelectStrategy<G>>(self, select: S) -> Requirements {
        select.requirements()
    }
}

pub fn requirements_of<G: Game>(spec: &SelectSpec) -> Requirements {
    with_select::<G, _>(spec, RequirementsCont(std::marker::PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_nim::Nim;
    use mcts::strategies::mcts::strategy::Compose;
    use mcts::strategies::Search;
    use mcts::{simulate, SearchConfig, TreeSearch};

    /// Builds and runs a `TreeSearch<Nim, Compose<S, simulate::Uniform>>`
    /// for whatever concrete `S` `with_select` resolves -- the end-to-end
    /// proof that a `SelectSpec` parsed from JSON reaches an optimized,
    /// monomorphized search, not just a type-erased stand-in.
    struct RunCont<'a, G: Game> {
        state: &'a G::S,
    }

    impl<'a, G: Game> SelectCont<G> for RunCont<'a, G> {
        type Output = G::A;

        fn call<S: SelectStrategy<G>>(self, select: S) -> G::A {
            let mut ts = TreeSearch::<G, Compose<S, simulate::Uniform>>::default().config(
                SearchConfig::default()
                    .select(select)
                    .max_iterations(200)
                    .seed(1),
            );
            ts.choose_action(self.state)
        }
    }

    #[test]
    fn select_spec_round_trips_through_json() {
        let json = r#"{"kind":"rave","threshold":700,"c":1.5}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SelectSpec::Rave {
                threshold: 700,
                c: 1.5
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn epsilon_greedy_wraps_an_arbitrary_inner_spec() {
        let json = r#"{"kind":"epsilon_greedy","epsilon":0.2,"inner":{"kind":"uct_pn","c":1.4,"c_pn":1.0}}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        let SelectSpec::EpsilonGreedy { epsilon, inner } = &spec else {
            panic!("expected EpsilonGreedy");
        };
        assert_eq!(*epsilon, 0.2);
        assert_eq!(*inner, BaseSelectSpec::UctPn { c: 1.4, c_pn: 1.0 });
    }

    #[test]
    fn requirements_of_matches_the_real_components_own_answer() {
        // `UctPn` is the one hand-picked case in mcts/src that overrides
        // `requirements()` beyond `backprop_flags()` -- proving this table's
        // `requirements_of` reports the same thing the concrete component
        // does, with no second copy of "UctPn needs the solver / <=2
        // players" written here.
        let spec = SelectSpec::UctPn { c: 1.4, c_pn: 1.0 };
        let reqs = requirements_of::<Nim>(&spec);
        assert!(reqs.solver);
        assert_eq!(reqs.max_players, Some(2));

        // `select::Rave` reads its own ancestor-keyed GRAVE table
        // (`SelectContext::grave`), not the per-child AMAF field
        // `select::Amaf` uses -- so its real requirement is `grave`, not
        // `amaf` (see `Rave::backprop_flags`). Asserting the wrong one here
        // would have passed silently if this table just repeated a
        // hand-guessed answer instead of calling the real component.
        let rave = SelectSpec::Rave {
            threshold: 700,
            c: 1.4,
        };
        assert!(requirements_of::<Nim>(&rave).grave);

        let amaf = SelectSpec::Amaf { alpha: 1.0, c: 1.4 };
        assert!(requirements_of::<Nim>(&amaf).amaf);

        // Wrapping in EpsilonGreedy must not lose UctPn's requirements --
        // the same property `mcts-tests`' select-side test checks against
        // the real `select::EpsilonGreedy` type, exercised here through the
        // spec/dispatch layer instead.
        let wrapped = SelectSpec::EpsilonGreedy {
            epsilon: 0.1,
            inner: BaseSelectSpec::UctPn { c: 1.4, c_pn: 1.0 },
        };
        assert_eq!(requirements_of::<Nim>(&wrapped), reqs);
    }

    #[test]
    fn with_select_builds_a_working_tree_search_from_a_json_spec() {
        let spec: SelectSpec = serde_json::from_str(r#"{"kind":"ucb1","c":1.5}"#).unwrap();
        let state = <Nim as Game>::S::default();
        let action = with_select::<Nim, _>(&spec, RunCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured search must be legal"
        );
    }
}
