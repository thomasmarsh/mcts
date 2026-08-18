//! A JSON-serializable config IR covering every `Strategy<G>` axis
//! (`select`/`simulate`/`backprop`/`final_action`), plus `register_*!`-
//! generated dispatchers that turn a runtime spec into a
//! compile-time-monomorphized `mcts` strategy component. `mcts-tune::
//! make_candidate` converts its `TrialParams` into this IR via
//! `to_search_spec` and builds it with `build_search` below, for every
//! family except `"random"`/`"flat_mc"`. Two things this table-driven
//! approach gives that a hand-written `match p.family.as_str()` doesn't:
//!
//! - **A single source of truth.** `register_select!`'s table is the only
//!   place that names a `select::*` type; the `SelectSpec` enum, the runtime
//!   dispatcher, and the `Requirements` computation are all generated from
//!   it, so they can't drift apart the way three independently
//!   hand-maintained descriptions of the same thing could.
//! - **Recursive composition.** `SelectSpec::EpsilonGreedy` wraps an
//!   arbitrary inner `SelectSpec`, matching `select::EpsilonGreedy<G, S>`'s
//!   own genericity.
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

use mcts::backprop::{self, BackpropStrategy};
use mcts::game::Game;
use mcts::index::Id;
use mcts::node::{ChildArray, QInit};
use mcts::search::TreeStats;
use mcts::select::{self, SelectContext, SelectStrategy};
use mcts::simulate::{self, SimulateStrategy, Trial};
use mcts::strategies::mcts::config::BackpropFlags;
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{GraphSearch, Requirements, SearchConfig, TreeSearch};
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

/// A continuation that can be invoked with any concrete `S: SelectStrategy<G>`
/// -- the "callback" side of the CPS dispatch this module's doc comment
/// describes. `requirements_of` and any real caller building a `TreeSearch`
/// each implement this once, for their own `Output`.
/// `+ 'static` on `call`'s own `S` (not just on `G`) is what lets
/// `build_search` below box the eventually-fully-resolved `TreeSearch<G,
/// Compose<..>>` as `Box<dyn Search<G = G>>` -- `SelectStrategy` itself
/// declares no `'static` bound (nothing about "which move to explore" needs
/// one), but every real implementor (`Ucb1`, `Rave`, ...) is an owned type
/// with no borrowed fields anyway, so this costs nothing in practice.
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
struct RequirementsCont<G>(std::marker::PhantomData<G>);

impl<G: Game> SelectCont<G> for RequirementsCont<G> {
    type Output = Requirements;

    fn call<S: SelectStrategy<G>>(self, select: S) -> Requirements {
        select.requirements()
    }
}

pub fn requirements_of<G: Game + 'static>(spec: &SelectSpec) -> Requirements {
    with_select::<G, _>(spec, RequirementsCont(std::marker::PhantomData))
}

////////////////////////////////////////////////////////////////////////////////

/// The `simulate`-axis counterpart of `SelectCont` -- see this module's doc
/// comment for why continuation-passing rather than a boxed trait object.
/// See `SelectCont`'s doc comment for why `call`'s `S` carries `+ 'static`.
pub trait SimulateCont<G: Game> {
    type Output;
    fn call<S: SimulateStrategy<G> + 'static>(self, simulate: S) -> Self::Output;
}

/// `register_simulate!`'s table, expanded into `BaseSimulateSpec`/
/// `SimulateSpec` and their dispatchers, mirroring `register_select!` above.
///
/// `EpsilonGreedy` and `DecisiveMove` are not rows here -- both wrap an
/// *inner* spec (`simulate::EpsilonGreedy<G, S>`/`simulate::DecisiveMove<G,
/// S>` are generic over an arbitrary inner `SimulateStrategy`), and are
/// handled by hand below on both enums instead, the same way
/// `register_select!` special-cases its own `EpsilonGreedy`. Their inner
/// spec is a `BaseSimulateSpec` (the table's variants only, no wrapper
/// variants of its own) rather than a recursive `Box<SimulateSpec>`, for the
/// same unbounded-monomorphization reason `register_select!`'s doc comment
/// explains -- e.g. `EpsilonGreedy(DecisiveMove(EpsilonGreedy(...)))` stops
/// being representable, matching the fact that real configs only ever nest
/// one wrapper deep.
///
/// `MetaMcts` is also not a row here -- `simulate::MetaMcts<G, S: Strategy<G>>`
/// wraps a whole nested `TreeSearch`, not a `SimulateStrategy`. Its inner
/// search is *not* independently configurable: it's always `Compose<Ucb1,
/// Uniform>` with `Ucb1`'s default `c`, matching the one real caller
/// (`mcts-tune::make_candidate`'s `meta_mcts` arm, which has never varied
/// the inner strategy). Earlier revisions of this file let the inner
/// `select`/`simulate` be arbitrary specs, which multiplied the already
/// combinatorial `select` x `simulate` x `final_action` fan-out `with_select`/
/// `with_simulate` chase during monomorphization by another ~20x for no real
/// caller benefit (see `build_search`'s doc comment on the compile-time cost
/// this fan-out has) -- exactly the "not every axis benefits from being
/// truly recursive" case. If a real need for a configurable inner strategy
/// shows up, reintroduce it deliberately rather than by default.
macro_rules! register_simulate {
    (
        $(
            $variant:ident { $($field:ident : $ty:ty),* $(,)? } => $ctor:expr
        ),+ $(,)?
    ) => {
        /// The inner spec `EpsilonGreedy`/`DecisiveMove` wrap -- the table's
        /// families, with no wrapper variant of its own (see
        /// `register_simulate!`'s doc comment on why).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum BaseSimulateSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        /// Dispatches `spec` to the concrete `SimulateStrategy<G>` it names
        /// by invoking `cont` with it -- see `with_base_select` above.
        pub fn with_base_simulate<G, C>(spec: &BaseSimulateSpec, cont: C) -> C::Output
        where
            G: Game + 'static,
            C: SimulateCont<G>,
        {
            match spec.clone() {
                $(
                    BaseSimulateSpec::$variant { $($field),* } => {
                        let simulate = $ctor;
                        cont.call(simulate)
                    }
                )+
            }
        }

        /// The config-IR node for the `simulate` axis: every
        /// `BaseSimulateSpec` family, plus `EpsilonGreedy`/`DecisiveMove`
        /// each wrapping one of them.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum SimulateSpec {
            $(
                $variant { $($field: $ty),* },
            )+
            EpsilonGreedy {
                epsilon: f64,
                inner: BaseSimulateSpec,
            },
            DecisiveMove {
                mode: simulate::DecisiveMoveMode,
                inner: BaseSimulateSpec,
            },
            /// `simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>`
            /// -- a `DecisiveMove` wrapping an `EpsilonGreedy`, two wrapper
            /// levels deep. Not representable as `DecisiveMove { inner:
            /// BaseSimulateSpec }` (that only reaches one level), and not
            /// given a general recursive `inner: SimulateSpec` either, for
            /// the same unbounded-monomorphization reason `MetaMcts` above
            /// stays fixed-shape rather than reusing this table's general
            /// `EpsilonGreedy`/`DecisiveMove` machinery: `mcts-tune`'s
            /// `rave`/`ucb1_tuned_dm_mast` families are the only two real
            /// callers of this exact composition, and neither has ever
            /// varied the `Mast` leaf, so it's named and fixed here instead.
            DecisiveMoveMast {
                mode: simulate::DecisiveMoveMode,
                epsilon: f64,
            },
            /// `simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Nst>>`
            /// -- `DecisiveMoveMast`'s counterpart for an `Nst` leaf instead of
            /// `Mast`, same two-wrapper-levels-deep reasoning: Druid's
            /// `strong`/`master` presets (`games/druid/src/main.rs`'s `build_ai`)
            /// are the one real caller of this exact composition, and have
            /// never varied the `Nst` leaf either.
            DecisiveMoveNst {
                mode: simulate::DecisiveMoveMode,
                epsilon: f64,
                nst_backoff_threshold: u32,
            },
            MetaMcts {
                iterations: usize,
            },
        }

        /// Dispatches `spec` the same way `with_base_simulate` does, plus
        /// the `EpsilonGreedy`/`DecisiveMove`/`DecisiveMoveMast`/`MetaMcts`
        /// wrappers.
        pub fn with_simulate<G, C>(spec: &SimulateSpec, cont: C) -> C::Output
        where
            G: Game + 'static,
            C: SimulateCont<G>,
        {
            match spec.clone() {
                $(
                    SimulateSpec::$variant { $($field),* } => {
                        let simulate = $ctor;
                        cont.call(simulate)
                    }
                )+
                SimulateSpec::EpsilonGreedy { epsilon, inner } => {
                    with_base_simulate::<G, _>(&inner, EpsilonGreedySimulateCont { epsilon, cont })
                }
                SimulateSpec::DecisiveMove { mode, inner } => {
                    with_base_simulate::<G, _>(&inner, DecisiveMoveSimulateCont { mode, cont })
                }
                SimulateSpec::DecisiveMoveMast { mode, epsilon } => {
                    let simulate =
                        simulate::DecisiveMove::<G, simulate::EpsilonGreedy<G, simulate::Mast>>::new()
                            .mode(mode)
                            .inner(simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(
                                epsilon,
                            ));
                    cont.call(simulate)
                }
                SimulateSpec::DecisiveMoveNst { mode, epsilon, nst_backoff_threshold } => {
                    let simulate =
                        simulate::DecisiveMove::<G, simulate::EpsilonGreedy<G, simulate::Nst>>::new()
                            .mode(mode)
                            .inner(
                                simulate::EpsilonGreedy::<G, simulate::Nst>::with_epsilon(epsilon)
                                    .inner(
                                        simulate::Nst::new()
                                            .backoff_threshold(nst_backoff_threshold),
                                    ),
                            );
                    cont.call(simulate)
                }
                SimulateSpec::MetaMcts { iterations } => {
                    let inner = TreeSearch::<G, Compose<select::Ucb1, simulate::Uniform>>::default()
                        .config(
                            SearchConfig::<G, Compose<select::Ucb1, simulate::Uniform>>::new()
                                .max_iterations(iterations),
                        );
                    cont.call(simulate::MetaMcts { inner })
                }
            }
        }
    };
}

register_simulate! {
    Uniform {} => simulate::Uniform,
    Mast {} => simulate::Mast,
    Nst { backoff_threshold: u32 } => simulate::Nst::new().backoff_threshold(backoff_threshold),
}

/// Forwards a resolved `S: SimulateStrategy<G>` on to `cont`, wrapped in
/// `simulate::EpsilonGreedy` -- `with_simulate`'s handling of the recursive
/// `EpsilonGreedy` spec variant.
struct EpsilonGreedySimulateCont<C> {
    epsilon: f64,
    cont: C,
}

impl<G, C> SimulateCont<G> for EpsilonGreedySimulateCont<C>
where
    G: Game + 'static,
    C: SimulateCont<G>,
{
    type Output = C::Output;

    fn call<S: SimulateStrategy<G> + 'static>(self, simulate: S) -> C::Output {
        let wrapped = simulate::EpsilonGreedy::<G, S>::with_epsilon(self.epsilon).inner(simulate);
        self.cont.call(wrapped)
    }
}

/// Forwards a resolved `S: SimulateStrategy<G>` on to `cont`, wrapped in
/// `simulate::DecisiveMove` -- `with_simulate`'s handling of the recursive
/// `DecisiveMove` spec variant.
struct DecisiveMoveSimulateCont<C> {
    mode: simulate::DecisiveMoveMode,
    cont: C,
}

impl<G, C> SimulateCont<G> for DecisiveMoveSimulateCont<C>
where
    G: Game + 'static,
    C: SimulateCont<G>,
{
    type Output = C::Output;

    fn call<S: SimulateStrategy<G> + 'static>(self, simulate: S) -> C::Output {
        let wrapped = simulate::DecisiveMove::<G, S>::new()
            .mode(self.mode)
            .inner(simulate);
        self.cont.call(wrapped)
    }
}

/// A `SimulateCont` whose `Output` is just the resolved component's own
/// `Requirements` -- see `RequirementsCont` above for why this reuses the
/// real dispatch rather than a second, independently-drifting table.
struct SimulateRequirementsCont<G>(std::marker::PhantomData<G>);

impl<G: Game> SimulateCont<G> for SimulateRequirementsCont<G> {
    type Output = Requirements;

    fn call<S: SimulateStrategy<G>>(self, simulate: S) -> Requirements {
        simulate.requirements()
    }
}

pub fn requirements_of_simulate<G: Game + 'static>(spec: &SimulateSpec) -> Requirements {
    with_simulate::<G, _>(spec, SimulateRequirementsCont(std::marker::PhantomData))
}

/// A shadow of `SimulateStrategy<G>` covering only the methods anything
/// outside this module's own dispatch machinery actually calls on a
/// resolved simulate strategy (`playout`, `backprop_flags`, plus what
/// `Clone`/`Send`/`Sync` need) -- unlike `SimulateStrategy` itself, this one
/// is object-safe (no `Self`-by-value `Default`/`Clone` in its signature),
/// which is what lets `DynSimulate` below erase the ~10 concrete leaf types
/// `with_simulate` can produce (3 base families x wrapped in `EpsilonGreedy`/
/// `DecisiveMove`, plus `MetaMcts`) into one. Blanket-implemented over every
/// real `SimulateStrategy`, so nothing here can drift from
/// `register_simulate!`'s table -- there's no second by-hand list of
/// families to keep in sync.
trait ErasedSimulateStrategy<G: Game>: Send + Sync {
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G>;
    fn backprop_flags(&self) -> BackpropFlags;
    fn clone_box(&self) -> Box<dyn ErasedSimulateStrategy<G>>;
}

impl<G, S> ErasedSimulateStrategy<G> for S
where
    G: Game,
    S: SimulateStrategy<G> + 'static,
{
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        SimulateStrategy::playout(self, state, max_playout_depth, stats, prev_action, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        SimulateStrategy::backprop_flags(self)
    }

    fn clone_box(&self) -> Box<dyn ErasedSimulateStrategy<G>> {
        Box::new(self.clone())
    }
}

/// One `SimulateStrategy<G>` impl standing in for all of `with_simulate`'s
/// concrete leaf types, via a `Box<dyn ErasedSimulateStrategy<G>>` --
/// `build_search`'s way of stopping the `select` x `simulate` x
/// `final_action` monomorphization product from ever including `simulate`'s
/// share of the fan-out. `SimulateStrategy::playout` is called once per
/// search *iteration* (its own per-ply `select_move` calls happen inside
/// whichever concrete type's own `playout` body runs, fully statically
/// dispatched there), so the one indirect call this adds per iteration is
/// cheap relative to a whole rollout's game-state work -- unlike `select`,
/// which is deliberately left fully monomorphized in `SelectStage` below
/// because it's called once per *child* at every node on every
/// tree-descent step, a much hotter path.
pub struct DynSimulate<G: Game>(Box<dyn ErasedSimulateStrategy<G>>);

impl<G: Game> Clone for DynSimulate<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<G: Game> Default for DynSimulate<G> {
    fn default() -> Self {
        Self(Box::new(simulate::Uniform))
    }
}

impl<G: Game> SimulateStrategy<G> for DynSimulate<G> {
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        self.0
            .playout(state, max_playout_depth, stats, prev_action, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.0.backprop_flags()
    }
}

/// A `SimulateCont` that erases whatever concrete strategy `with_simulate`
/// resolves to into a `DynSimulate<G>` -- reuses `with_simulate`'s existing
/// dispatch (so `EpsilonGreedy`/`DecisiveMove`/`MetaMcts` wrapping all still
/// work) but stops that dispatch's fan-out from propagating any further:
/// everything downstream of `resolve_simulate` sees one fixed type.
struct EraseSimulateCont<G>(std::marker::PhantomData<G>);

impl<G: Game + 'static> SimulateCont<G> for EraseSimulateCont<G> {
    type Output = DynSimulate<G>;

    fn call<S: SimulateStrategy<G> + 'static>(self, simulate: S) -> DynSimulate<G> {
        DynSimulate(Box::new(simulate))
    }
}

/// Resolves `spec` to a single `DynSimulate<G>`, regardless of family --
/// see `DynSimulate`'s doc comment for why `build_search` uses this instead
/// of routing `S2` generically through its whole stage chain.
pub fn resolve_simulate<G: Game + 'static>(spec: &SimulateSpec) -> DynSimulate<G> {
    with_simulate::<G, _>(spec, EraseSimulateCont(std::marker::PhantomData))
}

////////////////////////////////////////////////////////////////////////////////

/// The `backprop`-axis counterpart of `SelectCont`/`SimulateCont`. Unlike
/// those two, `BackpropStrategy` (`mcts/src/strategies/mcts/backprop.rs`) is
/// not generic over `G: Game` -- its methods take `G` as a per-call type
/// parameter instead -- so this dispatcher needs no `G` of its own either.
/// See `SelectCont`'s doc comment for why `call`'s `B` carries `+ 'static`.
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

////////////////////////////////////////////////////////////////////////////////

/// The config-IR node for the `final_action` axis. `Strategy::FinalAction`
/// is bound by the exact same `select::SelectStrategy<G>` trait as `Select`
/// (`mcts/src/strategies/mcts/config.rs`'s `Strategy` trait), so this reuses
/// `SelectCont`'s dispatch machinery rather than inventing a parallel trait
/// -- but it is a deliberately *separate*, smaller table from `SelectSpec`/
/// `register_select!`, not new rows added to it: `final_action` only ever
/// picks a root child once search ends, and `mcts-tune`'s existing
/// `TrialParams::final_action` field only ever names one of
/// `RobustChild`/`MaxAvgScore`/`MaxRobustChild`/`SecureChild`
/// (`to_final_action_spec` in `mcts-tune/src/lib.rs`), never an in-tree
/// exploration strategy like
/// `Ucb1`/`Rave`/`UctPn`. None of the `select`-axis wrapper concerns
/// (recursive `EpsilonGreedy`, unbounded monomorphization) apply here since
/// none of these four types are themselves generic over an inner strategy,
/// so this needs no `Base.../...` split -- one flat enum, one dispatcher.
macro_rules! register_final_action {
    (
        $(
            $variant:ident { $($field:ident : $ty:ty),* $(,)? } => $ctor:expr
        ),+ $(,)?
    ) => {
        /// The config-IR node for the `final_action` axis.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        pub enum FinalActionSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        /// Dispatches `spec` to the concrete `SelectStrategy<G>` it names by
        /// invoking `cont` with it -- see `with_base_select` above.
        pub fn with_final_action<G, C>(spec: &FinalActionSpec, cont: C) -> C::Output
        where
            G: Game + 'static,
            C: SelectCont<G>,
        {
            match spec.clone() {
                $(
                    FinalActionSpec::$variant { $($field),* } => {
                        let final_action = $ctor;
                        cont.call(final_action)
                    }
                )+
            }
        }
    };
}

register_final_action! {
    RobustChild {} => select::RobustChild,
    MaxAvg {} => select::MaxAvgScore,
    MaxRobustChild {} => select::MaxRobustChild,
    SecureChild { a: f64 } => select::SecureChild { a },
}

/// Reuses `RequirementsCont` (defined above for the `select` axis) since
/// `final_action`'s dispatch resolves to the same `SelectStrategy<G>` trait.
pub fn requirements_of_final_action<G: Game + 'static>(spec: &FinalActionSpec) -> Requirements {
    with_final_action::<G, _>(spec, RequirementsCont(std::marker::PhantomData))
}

/// Whether `spec` resolves to a `BackpropStrategy` that populates
/// `posterior_mean`/`posterior_variance` (`BayesGaussian`/`BayesNumeric`) --
/// dispatched through `with_backprop`, same as every other spec->real-type
/// question in this file, rather than a second hand-matched list of names
/// that could drift from `register_backprop!`'s table.
pub fn provides_posterior(spec: &BackpropSpec) -> bool {
    struct ProvidesPosteriorCont;
    impl BackpropCont for ProvidesPosteriorCont {
        type Output = bool;
        fn call<B: BackpropStrategy + 'static>(self, backprop: B) -> bool {
            backprop.provides_posterior()
        }
    }
    with_backprop(spec, ProvidesPosteriorCont)
}

/// Validates a `SearchSpec`'s cross-axis coupling that `register_select!`/
/// `register_backprop!`'s dispatch alone can't catch: `BayesUct1`/
/// `BayesUct2` (`select`/`final_action`) set `Requirements::needs_posterior`,
/// which only `BayesGaussian`/`BayesNumeric` (`backprop`) satisfy. Neither
/// `config_ir::build_search` nor `TreeSearch` itself calls this
/// automatically (mirroring `SearchConfig::validate`'s own opt-in status,
/// see its doc comment) -- callers that build a search from a
/// caller-supplied `SearchSpec` (`build_custom`, `mcts_tune::build_search`)
/// call this first so a bad pairing is a rejected config, not a search that
/// silently runs `BayesUct1`/`BayesUct2` against zeroed posterior fields.
pub fn validate_search_spec<G: Game + 'static>(spec: &SearchSpec) -> Result<(), String> {
    let reqs = requirements_of::<G>(&spec.select)
        .union(requirements_of_final_action::<G>(&spec.final_action));
    if reqs.needs_posterior && !provides_posterior(&spec.backprop) {
        return Err(
            "select/final_action strategy requires a Bayesian backprop strategy \
             (bayes_gaussian/bayes_numeric) that provides posterior mean/variance estimates"
                .to_string(),
        );
    }
    Ok(())
}

/// The `final_action`-axis counterpart of `ErasedSimulateStrategy` -- a
/// shadow of `SelectStrategy<G>` covering only `best_child` (the one method
/// `TreeSearch` actually calls on `config.final_action`, once per move, at
/// the very end of a search) plus `backprop_flags`/`Clone`/`Send`/`Sync`.
/// Blanket-implemented the same way, for the same single-source-of-truth
/// reason.
trait ErasedFinalActionStrategy<G: Game>: Send + Sync {
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize;
    fn backprop_flags(&self) -> BackpropFlags;
    fn clone_box(&self) -> Box<dyn ErasedFinalActionStrategy<G>>;
}

impl<G, S> ErasedFinalActionStrategy<G> for S
where
    G: Game,
    S: SelectStrategy<G> + 'static,
{
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        SelectStrategy::best_child(self, ctx, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        SelectStrategy::backprop_flags(self)
    }

    fn clone_box(&self) -> Box<dyn ErasedFinalActionStrategy<G>> {
        Box::new(self.clone())
    }
}

/// One `SelectStrategy<G>` impl standing in for all four `final_action`
/// families, via `Box<dyn ErasedFinalActionStrategy<G>>` -- the
/// `final_action`-axis half of collapsing `build_search`'s monomorphization
/// product (see `DynSimulate`'s doc comment). `final_action` is called once
/// per move, the coldest of the four axes, so this is the cheapest of the
/// two erasures to add. `Score`/`Aux` are fixed to `()`: nothing outside
/// `best_child`'s own delegated call ever reads them, since `best_child` is
/// always overridden here rather than falling back to the trait's default
/// (which is the only thing that would call `score_child`/`unvisited_value`/
/// `setup` on `Self`).
pub struct DynFinalAction<G: Game>(Box<dyn ErasedFinalActionStrategy<G>>);

impl<G: Game> Clone for DynFinalAction<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<G: Game> Default for DynFinalAction<G> {
    fn default() -> Self {
        Self(Box::new(select::RobustChild))
    }
}

impl<G: Game> SelectStrategy<G> for DynFinalAction<G> {
    type Score = ();
    type Aux = ();

    fn setup(&mut self, _ctx: &SelectContext<'_, G>) -> Self::Aux {}

    fn score_child(
        &self,
        _ctx: &SelectContext<'_, G>,
        _child_id: Id,
        _children: &ChildArray<G::A>,
        _idx: usize,
        _aux: Self::Aux,
    ) -> Self::Score {
        unreachable!("DynFinalAction always overrides best_child directly")
    }

    fn unvisited_value(&self, _ctx: &SelectContext<'_, G>, _aux: Self::Aux) -> Self::Score {
        unreachable!("DynFinalAction always overrides best_child directly")
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        self.0.best_child(ctx, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.0.backprop_flags()
    }
}

/// A `SelectCont` that erases whatever concrete `final_action` strategy
/// `with_final_action` resolves to into a `DynFinalAction<G>` -- the
/// `final_action`-axis counterpart of `EraseSimulateCont`.
struct EraseFinalActionCont<G>(std::marker::PhantomData<G>);

impl<G: Game + 'static> SelectCont<G> for EraseFinalActionCont<G> {
    type Output = DynFinalAction<G>;

    fn call<S: SelectStrategy<G> + 'static>(self, final_action: S) -> DynFinalAction<G> {
        DynFinalAction(Box::new(final_action))
    }
}

/// Resolves `spec` to a single `DynFinalAction<G>`, regardless of family --
/// see `DynFinalAction`'s doc comment for why `build_search` uses this
/// instead of routing `FA` generically through its whole stage chain.
pub fn resolve_final_action<G: Game + 'static>(spec: &FinalActionSpec) -> DynFinalAction<G> {
    with_final_action::<G, _>(spec, EraseFinalActionCont(std::marker::PhantomData))
}

////////////////////////////////////////////////////////////////////////////////

/// The full four-axis config-IR node: one spec per `Strategy<G>` axis, the
/// same four names `mcts::strategy::Compose<Sel, Sim, Bp, FA>` takes as type
/// parameters. `mcts-tune::make_candidate` converts a `TrialParams` into one
/// of these via `to_search_spec` and builds it with `build_search` below, for
/// every family except `"random"`/`"flat_mc"` (not a `Compose<..>`
/// `Strategy`, so those two stay direct arms in `make_candidate` instead).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpec {
    pub select: SelectSpec,
    pub simulate: SimulateSpec,
    pub backprop: BackpropSpec,
    pub final_action: FinalActionSpec,
}

/// The non-strategy `SearchConfig` knobs a `SearchSpec` needs alongside
/// itself -- this module's counterpart of `mcts-tune::to_search_spec`'s
/// generic settings (iteration/time budget, `q_init`, threading, ...), factored out
/// because they're orthogonal to which axis families are chosen and don't
/// belong inside any one `*Spec` enum.
#[derive(Clone)]
pub struct SearchSettings {
    pub max_iterations: usize,
    pub max_playout_depth: usize,
    pub expand_threshold: u32,
    pub q_init: QInit,
    pub use_transpositions: bool,
    pub use_mcts_solver: bool,
    pub reuse_tree: bool,
    pub num_tree_threads: usize,
    pub seed: u64,
    pub max_time: Option<std::time::Duration>,
    pub graph_search: Option<GraphSearch>,
    pub solver_loss_threshold: Option<u32>,
    pub contempt_factor: Option<f64>,
}

/// Resolves `spec.select`, then hands off to `BackpropStage` -- the one
/// remaining generic link in `build_search`'s dispatch chain.
///
/// `simulate` and `final_action` are deliberately *not* separate generic
/// stages here (contrast an earlier version of this file, and the four-stage
/// shape `register_select!`'s `MetaMctsCont`/`MetaMctsInnerCont` comment used
/// to reference): chaining all four axes through fully generic continuations
/// makes every one of them a multiplicative factor in what rustc has to
/// monomorphize -- `select` (~7 variants) x `simulate` (~10 concrete leaf
/// types once `EpsilonGreedy`/`DecisiveMove` wrapping is counted) x
/// `final_action` (4), each instantiating the *entire* `TreeSearch` engine,
/// which is what turned a clean `cargo test --no-run -p mcts-tune` into a
/// 7+-minute, `__eh_frame`-overflowing build (see `plan/
/// config-algebra-step4.md`). `select` stays fully monomorphized here since
/// it's the hottest of the four axes (`SelectStrategy::best_child`/
/// `score_child` run once per *child*, at every node, every tree-descent
/// step); `simulate` and `final_action` are resolved eagerly via
/// `resolve_simulate`/`resolve_final_action` into the fixed `DynSimulate<G>`/
/// `DynFinalAction<G>` types instead, collapsing their ~40x combined
/// contribution to the monomorphization product down to 1x. `backprop`
/// stays a real generic type parameter too (see `BackpropStage` below) --
/// with `register_backprop!`'s table now at 3 rows (`Classic`,
/// `BayesGaussian`, `BayesNumeric`), the full product is `select` (~9) x
/// `backprop` (3) = ~27 `TreeSearch` monomorphizations, well under the
/// ~170-280 that caused the original blowup, so no further erasure is
/// needed at this scale -- revisit if `backprop`'s table grows enough to
/// make that product a problem again.
struct SelectStage<'a, G: Game> {
    spec: &'a SearchSpec,
    settings: &'a SearchSettings,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G> SelectCont<G> for SelectStage<'a, G>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<S1: SelectStrategy<G> + 'static>(self, select: S1) -> Self::Output {
        let simulate = resolve_simulate::<G>(&self.spec.simulate);
        let final_action = resolve_final_action::<G>(&self.spec.final_action);
        with_backprop(
            &self.spec.backprop,
            BackpropStage {
                settings: self.settings,
                select,
                simulate,
                final_action,
                marker: std::marker::PhantomData,
            },
        )
    }
}

/// Second and final link: resolves `spec.backprop` and, with all four axes
/// now concrete (`simulate`/`final_action` already erased to `DynSimulate<G>`/
/// `DynFinalAction<G>` by `SelectStage`), builds the actual
/// `TreeSearch<G, Compose<S1, DynSimulate<G>, B, DynFinalAction<G>>>` and
/// boxes it -- the only place in this chain that touches `SearchConfig`
/// directly, since it's the first point every type parameter `Compose` needs
/// is known.
struct BackpropStage<'a, G: Game, S1> {
    settings: &'a SearchSettings,
    select: S1,
    simulate: DynSimulate<G>,
    final_action: DynFinalAction<G>,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G, S1> BackpropCont for BackpropStage<'a, G, S1>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<B: BackpropStrategy + 'static>(self, backprop: B) -> Self::Output {
        type S<G, S1, B> = Compose<S1, DynSimulate<G>, B, DynFinalAction<G>>;
        let mut config = SearchConfig::<G, S<G, S1, B>>::new()
            .max_iterations(self.settings.max_iterations)
            .max_playout_depth(self.settings.max_playout_depth)
            .expand_threshold(self.settings.expand_threshold)
            .q_init(self.settings.q_init)
            .use_transpositions(self.settings.use_transpositions)
            .use_mcts_solver(self.settings.use_mcts_solver)
            .reuse_tree(self.settings.reuse_tree)
            .num_tree_threads(self.settings.num_tree_threads)
            .seed(self.settings.seed)
            .select(self.select)
            .simulate(self.simulate)
            .backprop(backprop)
            .final_action(self.final_action);
        if let Some(max_time) = self.settings.max_time {
            config = config.max_time(max_time);
        }
        if let Some(graph_search) = self.settings.graph_search {
            config = config.graph_search(graph_search);
        }
        if let Some(solver_loss_threshold) = self.settings.solver_loss_threshold {
            config = config.solver_loss_threshold(solver_loss_threshold);
        }
        config = config.contempt_factor(self.settings.contempt_factor);
        Box::new(TreeSearch::<G, S<G, S1, B>>::new().config(config))
    }
}

/// Builds a runnable `Box<dyn Search<G = G>>` from a `SearchSpec` --
/// `mcts-tune::make_candidate`'s dispatch for every family except
/// `"random"`/`"flat_mc"`: convert `TrialParams` via `to_search_spec`, then
/// call this, driven entirely by this file's registry-generated dispatch
/// rather than a hand-written `match p.family.as_str()`.
pub fn build_search<G>(spec: &SearchSpec, settings: &SearchSettings) -> Box<dyn Search<G = G>>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
{
    with_select::<G, _>(
        &spec.select,
        SelectStage {
            spec,
            settings,
            marker: std::marker::PhantomData,
        },
    )
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
        let json = r#"{"kind":"rave","threshold":700,"schedule":{"kind":"threshold","rave":700},"ucb":{"kind":"ucb1","exploration_constant":1.5}}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SelectSpec::Rave {
                threshold: 700,
                schedule: select::RaveSchedule::Threshold { rave: 700 },
                ucb: select::RaveUcb::Ucb1 {
                    exploration_constant: 1.5
                },
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn progressive_history_spec_round_trips_through_json() {
        let json = r#"{"kind":"progressive_history","c":1.4,"ph_weight":2.5}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SelectSpec::ProgressiveHistory {
                c: 1.4,
                ph_weight: 2.5
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn bayes_uct_spec_round_trips_through_json() {
        let json = r#"{"kind":"bayes_uct1","c":1.0}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec, SelectSpec::BayesUct1 { c: 1.0 });
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);

        let json = r#"{"kind":"bayes_uct2","c":1.0}"#;
        let spec: SelectSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec, SelectSpec::BayesUct2 { c: 1.0 });
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);
    }

    #[test]
    fn bayes_backprop_spec_round_trips_through_json() {
        let json = r#"{"kind":"bayes_gaussian","prior_variance":1.0,"obs_variance":1.0}"#;
        let spec: BackpropSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            BackpropSpec::BayesGaussian {
                prior_variance: 1.0,
                obs_variance: 1.0,
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);

        let json = r#"{"kind":"bayes_numeric","prior_variance":1.0,"obs_variance":1.0,"value_lo":-1.0,"value_hi":1.0}"#;
        let spec: BackpropSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            BackpropSpec::BayesNumeric {
                prior_variance: 1.0,
                obs_variance: 1.0,
                value_lo: -1.0,
                value_hi: 1.0,
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);
    }

    /// The select<->backprop coupling this whole feature exists to exercise:
    /// `BayesUct1`/`BayesUct2` set `Requirements::needs_posterior`, which
    /// only `BayesGaussian`/`BayesNumeric` satisfy -- `Classic` (or any other
    /// backprop) must be rejected, not silently read zeroed posterior
    /// fields.
    #[test]
    fn validate_search_spec_rejects_bayes_select_paired_with_classic_backprop() {
        let spec = SearchSpec {
            select: SelectSpec::BayesUct1 { c: 1.0 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::RobustChild {},
        };
        let err = validate_search_spec::<Nim>(&spec).unwrap_err();
        assert!(err.contains("Bayesian backprop"), "{err}");
    }

    #[test]
    fn validate_search_spec_accepts_bayes_select_paired_with_bayes_backprop() {
        let spec = SearchSpec {
            select: SelectSpec::BayesUct2 { c: 1.0 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::BayesGaussian {
                prior_variance: 1.0,
                obs_variance: 1.0,
            },
            final_action: FinalActionSpec::RobustChild {},
        };
        assert!(validate_search_spec::<Nim>(&spec).is_ok());

        let spec = SearchSpec {
            backprop: BackpropSpec::BayesNumeric {
                prior_variance: 1.0,
                obs_variance: 1.0,
                value_lo: -1.0,
                value_hi: 1.0,
            },
            ..spec
        };
        assert!(validate_search_spec::<Nim>(&spec).is_ok());
    }

    /// A non-Bayes select paired with a Bayes backprop is fine -- the
    /// backprop just does extra work nothing reads, no different from any
    /// other over-provisioned `Requirements`.
    #[test]
    fn validate_search_spec_accepts_classic_select_paired_with_bayes_backprop() {
        let spec = SearchSpec {
            select: SelectSpec::Ucb1 { c: 1.4 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::BayesGaussian {
                prior_variance: 1.0,
                obs_variance: 1.0,
            },
            final_action: FinalActionSpec::RobustChild {},
        };
        assert!(validate_search_spec::<Nim>(&spec).is_ok());
    }

    /// End-to-end proof that a `BayesUct1` select paired with a
    /// `BayesGaussian` backprop, both parsed from JSON via `build_search`,
    /// runs a real search rather than tripping the `needs_posterior`
    /// rejection or panicking on the posterior fields it reads.
    #[test]
    fn build_search_runs_bayes_uct_paired_with_bayes_backprop() {
        let spec = SearchSpec {
            select: SelectSpec::BayesUct1 { c: 1.0 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::BayesGaussian {
                prior_variance: 1.0,
                obs_variance: 1.0,
            },
            final_action: FinalActionSpec::RobustChild {},
        };
        validate_search_spec::<Nim>(&spec).unwrap();
        let mut search = build_search::<Nim>(&spec, &nim_search_settings());
        let state = <Nim as Game>::S::default();
        let action = search.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(legal.contains(&action));
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
            schedule: select::RaveSchedule::Threshold { rave: 700 },
            ucb: select::RaveUcb::Ucb1 {
                exploration_constant: 1.4,
            },
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

    #[test]
    fn with_select_builds_a_working_tree_search_for_progressive_history() {
        let spec: SelectSpec =
            serde_json::from_str(r#"{"kind":"progressive_history","c":1.4,"ph_weight":2.5}"#)
                .unwrap();
        let state = <Nim as Game>::S::default();
        let action = with_select::<Nim, _>(&spec, RunCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured search must be legal"
        );
    }

    /// Builds and runs a `TreeSearch<Nim, Compose<select::Ucb1, S>>` for
    /// whatever concrete `S` `with_simulate` resolves -- the `simulate`-axis
    /// counterpart of `RunCont` above.
    struct RunSimulateCont<'a, G: Game> {
        state: &'a G::S,
    }

    impl<'a, G: Game> SimulateCont<G> for RunSimulateCont<'a, G> {
        type Output = G::A;

        fn call<S: SimulateStrategy<G>>(self, simulate: S) -> G::A {
            let mut ts = TreeSearch::<G, Compose<select::Ucb1, S>>::default().config(
                SearchConfig::default()
                    .simulate(simulate)
                    .max_iterations(200)
                    .seed(1),
            );
            ts.choose_action(self.state)
        }
    }

    #[test]
    fn simulate_spec_round_trips_through_json() {
        let json = r#"{"kind":"nst","backoff_threshold":10}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SimulateSpec::Nst {
                backoff_threshold: 10
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn simulate_epsilon_greedy_and_decisive_move_wrap_an_arbitrary_inner_spec() {
        let json = r#"{"kind":"epsilon_greedy","epsilon":0.2,"inner":{"kind":"mast"}}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        let SimulateSpec::EpsilonGreedy { epsilon, inner } = &spec else {
            panic!("expected EpsilonGreedy");
        };
        assert_eq!(*epsilon, 0.2);
        assert_eq!(*inner, BaseSimulateSpec::Mast {});

        let json = r#"{"kind":"decisive_move","mode":"win_loss","inner":{"kind":"nst","backoff_threshold":5}}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        let SimulateSpec::DecisiveMove { mode, inner } = &spec else {
            panic!("expected DecisiveMove");
        };
        assert_eq!(*mode, simulate::DecisiveMoveMode::WinLoss);
        assert_eq!(
            *inner,
            BaseSimulateSpec::Nst {
                backoff_threshold: 5
            }
        );
    }

    #[test]
    fn requirements_of_simulate_matches_the_real_components_own_answer() {
        // `Nst` sets both `global` and `nst` (see `simulate::Nst`'s doc
        // comment on why it needs the unigram table on top of its own
        // bigram one) -- asserting only `nst` here would have passed
        // silently if this table dropped `global` from the real
        // `backprop_flags()` answer.
        let nst = SimulateSpec::Nst {
            backoff_threshold: 5,
        };
        let reqs = requirements_of_simulate::<Nim>(&nst);
        assert!(reqs.global);
        assert!(reqs.nst);

        let mast = SimulateSpec::Mast {};
        assert!(requirements_of_simulate::<Nim>(&mast).global);
        assert!(!requirements_of_simulate::<Nim>(&mast).nst);

        let uniform = SimulateSpec::Uniform {};
        assert_eq!(
            requirements_of_simulate::<Nim>(&uniform),
            Requirements::default()
        );

        // Wrapping in EpsilonGreedy/DecisiveMove must not lose Nst's
        // requirements -- both wrappers delegate `requirements()` straight
        // to `inner` (see `simulate::EpsilonGreedy`/`DecisiveMove`'s own
        // doc comments), so this checks that survives the spec/dispatch
        // layer too.
        let wrapped_eg = SimulateSpec::EpsilonGreedy {
            epsilon: 0.1,
            inner: BaseSimulateSpec::Nst {
                backoff_threshold: 5,
            },
        };
        assert_eq!(requirements_of_simulate::<Nim>(&wrapped_eg), reqs);

        let wrapped_dm = SimulateSpec::DecisiveMove {
            mode: simulate::DecisiveMoveMode::Win,
            inner: BaseSimulateSpec::Nst {
                backoff_threshold: 5,
            },
        };
        assert_eq!(requirements_of_simulate::<Nim>(&wrapped_dm), reqs);
    }

    #[test]
    fn with_simulate_builds_a_working_tree_search_from_a_json_spec() {
        let spec: SimulateSpec = serde_json::from_str(r#"{"kind":"uniform"}"#).unwrap();
        let state = <Nim as Game>::S::default();
        let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured search must be legal"
        );
    }

    #[test]
    fn meta_mcts_spec_round_trips_through_json() {
        let json = r#"{"kind":"meta_mcts","iterations":50}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec, SimulateSpec::MetaMcts { iterations: 50 });
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn with_simulate_builds_a_working_nested_search_for_meta_mcts() {
        // The inner search is always `Compose<Ucb1, Uniform>` -- see
        // `register_simulate!`'s doc comment on why `MetaMcts`'s inner
        // strategy isn't independently configurable.
        let spec = SimulateSpec::MetaMcts { iterations: 25 };
        let state = <Nim as Game>::S::default();
        let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured MetaMcts search must be legal"
        );
    }

    #[test]
    fn decisive_move_mast_spec_round_trips_through_json() {
        let json = r#"{"kind":"decisive_move_mast","mode":"win_loss","epsilon":0.2}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SimulateSpec::DecisiveMoveMast {
                mode: simulate::DecisiveMoveMode::WinLoss,
                epsilon: 0.2,
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn with_simulate_builds_a_working_search_for_decisive_move_mast() {
        let spec = SimulateSpec::DecisiveMoveMast {
            mode: simulate::DecisiveMoveMode::WinLoss,
            epsilon: 0.2,
        };
        let state = <Nim as Game>::S::default();
        let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured DecisiveMoveMast search must be legal"
        );
    }

    #[test]
    fn backprop_spec_round_trips_through_json() {
        let json = r#"{"kind":"classic"}"#;
        let spec: BackpropSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec, BackpropSpec::Classic {});
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);
    }

    /// A `BackpropCont` whose `Output` is just a marker proving `with_backprop`
    /// actually resolved to a real `BackpropStrategy` -- there's no
    /// `requirements()` to check (see `register_backprop!`'s doc comment on
    /// why), so this is the `backprop`-axis analogue of the `select`/
    /// `simulate` "build a working search" tests, minus the search: any
    /// `BackpropStrategy` is usable in a `Compose<..>` without further
    /// per-type configuration.
    struct ResolvedCont;

    impl BackpropCont for ResolvedCont {
        type Output = &'static str;

        fn call<B: BackpropStrategy>(self, _backprop: B) -> &'static str {
            "resolved"
        }
    }

    #[test]
    fn with_backprop_resolves_a_real_backprop_strategy() {
        let spec = BackpropSpec::Classic {};
        assert_eq!(with_backprop(&spec, ResolvedCont), "resolved");
    }

    #[test]
    fn final_action_spec_round_trips_through_json() {
        let json = r#"{"kind":"secure_child","a":2.5}"#;
        let spec: FinalActionSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec, FinalActionSpec::SecureChild { a: 2.5 });
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);
    }

    #[test]
    fn requirements_of_final_action_matches_the_real_components_own_answer() {
        // None of the four `final_action` families override `requirements()`
        // beyond the `SelectStrategy` default, unlike `UctPn` on the `select`
        // axis -- so this just pins that they all resolve to
        // `Requirements::none()` rather than silently picking up some future
        // override without a test noticing.
        for spec in [
            FinalActionSpec::RobustChild {},
            FinalActionSpec::MaxAvg {},
            FinalActionSpec::MaxRobustChild {},
            FinalActionSpec::SecureChild { a: 4.0 },
        ] {
            assert_eq!(
                requirements_of_final_action::<Nim>(&spec),
                Requirements::default()
            );
        }
    }

    /// Builds and runs a `TreeSearch<Nim, Compose<select::Ucb1, simulate::Uniform,
    /// backprop::Classic, S>>` for whatever concrete `S` `with_final_action`
    /// resolves -- the `final_action`-axis counterpart of `RunCont`/
    /// `RunSimulateCont` above.
    struct RunFinalActionCont<'a, G: Game> {
        state: &'a G::S,
    }

    impl<'a, G: Game> SelectCont<G> for RunFinalActionCont<'a, G> {
        type Output = G::A;

        fn call<S: SelectStrategy<G>>(self, final_action: S) -> G::A {
            let mut ts = TreeSearch::<
                G,
                Compose<select::Ucb1, simulate::Uniform, backprop::Classic, S>,
            >::default()
            .config(
                SearchConfig::default()
                    .final_action(final_action)
                    .max_iterations(200)
                    .seed(1),
            );
            ts.choose_action(self.state)
        }
    }

    #[test]
    fn with_final_action_builds_a_working_tree_search_from_a_json_spec() {
        let spec: FinalActionSpec = serde_json::from_str(r#"{"kind":"robust_child"}"#).unwrap();
        let state = <Nim as Game>::S::default();
        let action = with_final_action::<Nim, _>(&spec, RunFinalActionCont { state: &state });
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured search must be legal"
        );
    }

    fn nim_search_settings() -> SearchSettings {
        SearchSettings {
            max_iterations: 200,
            max_playout_depth: 200,
            expand_threshold: 1,
            q_init: QInit::Parent,
            use_transpositions: false,
            use_mcts_solver: false,
            reuse_tree: false,
            num_tree_threads: 1,
            seed: 1,
            max_time: None,
            graph_search: None,
            solver_loss_threshold: None,
            contempt_factor: None,
        }
    }

    #[test]
    fn search_spec_round_trips_through_json() {
        let json = r#"{
            "select": {"kind": "ucb1", "c": 1.4},
            "simulate": {"kind": "uniform"},
            "backprop": {"kind": "classic"},
            "final_action": {"kind": "robust_child"}
        }"#;
        let spec: SearchSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SearchSpec {
                select: SelectSpec::Ucb1 { c: 1.4 },
                simulate: SimulateSpec::Uniform {},
                backprop: BackpropSpec::Classic {},
                final_action: FinalActionSpec::RobustChild {},
            }
        );
    }

    #[test]
    fn build_search_builds_a_working_tree_search_from_a_full_json_spec() {
        // Unlike every other test in this file, this drives all four axes'
        // specs through one call, proving they compose into a real, runnable
        // `Box<dyn Search<G>>` -- not just that each axis resolves on its
        // own.
        let spec: SearchSpec = serde_json::from_str(
            r#"{
                "select": {"kind": "epsilon_greedy", "epsilon": 0.1, "inner": {"kind": "ucb1", "c": 1.4}},
                "simulate": {"kind": "mast"},
                "backprop": {"kind": "classic"},
                "final_action": {"kind": "secure_child", "a": 2.0}
            }"#,
        )
        .unwrap();
        let mut search = build_search::<Nim>(&spec, &nim_search_settings());
        let state = <Nim as Game>::S::default();
        let action = search.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured search must be legal"
        );
    }

    #[test]
    fn build_search_wires_meta_mcts_through_the_full_spec() {
        let spec = SearchSpec {
            select: SelectSpec::Ucb1 { c: 1.4 },
            simulate: SimulateSpec::MetaMcts { iterations: 25 },
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::RobustChild {},
        };
        let mut search = build_search::<Nim>(&spec, &nim_search_settings());
        let state = <Nim as Game>::S::default();
        let action = search.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a JSON-configured MetaMcts search must be legal"
        );
    }

    #[test]
    fn build_search_applies_graph_search_setting() {
        // `Nim` has no real `zobrist_hash` (defaults to a constant `0`),
        // which collapses every position into one graph node -- fine for a
        // single-iteration root expansion (only one node is ever visited),
        // but running deeper would corrupt move legality across positions
        // that fold into the same hash. `mcts-tune::lib.rs`'s
        // `mcgs_trial_selects_combined_graph_statistics` test uses the same
        // one-iteration trick for the same reason. Real `mcgs`-enabled
        // callers are guarded by the "mcgs requires a game with a zobrist
        // hash" check (step 4c), which this config-IR layer intentionally
        // doesn't duplicate (see this file's `SearchSettings` doc comment).
        let spec = SearchSpec {
            select: SelectSpec::Ucb1 { c: 1.4 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::RobustChild {},
        };
        let mut settings = nim_search_settings();
        settings.max_iterations = 1;
        settings.expand_threshold = 0;
        settings.graph_search = Some(GraphSearch::Dag(mcts::GraphStats::Both));
        let mut search = build_search::<Nim>(&spec, &settings);
        let state = <Nim as Game>::S::default();
        let action = search.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a graph-search-configured search must be legal"
        );
    }

    #[test]
    fn build_search_applies_solver_settings() {
        let spec = SearchSpec {
            select: SelectSpec::UctPn { c: 1.4, c_pn: 1.4 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::RobustChild {},
        };
        let mut settings = nim_search_settings();
        settings.solver_loss_threshold = Some(1);
        settings.contempt_factor = Some(0.1);
        let mut search = build_search::<Nim>(&spec, &settings);
        let state = <Nim as Game>::S::default();
        let action = search.choose_action(&state);
        let mut legal = Vec::new();
        Nim::generate_actions(&state, &mut legal);
        assert!(
            legal.contains(&action),
            "the action chosen by a solver-configured search must be legal"
        );
    }
}
