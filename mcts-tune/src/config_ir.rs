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

use mcts::backprop::{self, BackpropStrategy};
use mcts::game::Game;
use mcts::node::QInit;
use mcts::select::{self, SelectStrategy};
use mcts::simulate::{self, SimulateStrategy};
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{Requirements, SearchConfig, TreeSearch};
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
    Rave { threshold: u32, c: f64 } => select::Rave::default()
        .threshold(threshold)
        .ucb(select::RaveUcb::Ucb1 { exploration_constant: c }),
    UctPn { c: f64, c_pn: f64 } => select::UctPn::with_c(c, c_pn),
    ProgressiveHistory { c: f64, ph_weight: f64 } =>
        select::ProgressiveHistory::new(select::Ucb1::with_c(c), ph_weight),
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
/// wraps a whole nested `TreeSearch`, not a `SimulateStrategy`, so its inner
/// spec is a *pair* of specs (one per axis of that nested search's
/// `Compose<InnerSel, InnerSim>`) rather than a `BaseSimulateSpec`. The inner
/// `select` field is a full `SelectSpec` (select has no `MetaMcts` variant of
/// its own, so nothing there can recurse); the inner `simulate` field is a
/// `BaseSimulateSpec`, not a full `SimulateSpec`, specifically so it *can't*
/// name another `MetaMcts` -- both because that would hit the same
/// unbounded-monomorphization trap and because nested-nested MCTS isn't a
/// realistic configuration anyway.
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
            MetaMcts {
                iterations: usize,
                select: SelectSpec,
                simulate: BaseSimulateSpec,
            },
        }

        /// Dispatches `spec` the same way `with_base_simulate` does, plus
        /// the `EpsilonGreedy`/`DecisiveMove`/`MetaMcts` wrappers.
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
                SimulateSpec::MetaMcts { iterations, select, simulate } => {
                    with_select::<G, _>(
                        &select,
                        MetaMctsCont { iterations, simulate_spec: simulate, cont },
                    )
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

/// The second stage of `SimulateSpec::MetaMcts`'s two-stage CPS dispatch:
/// once the nested search's `select` spec has resolved to a concrete `S1`,
/// this resolves its `simulate` spec, builds the actual nested
/// `TreeSearch<G, Compose<S1, S2>>`, and wraps it in `simulate::MetaMcts`
/// before forwarding to the outer `cont`.
struct MetaMctsInnerCont<G, S1, C> {
    iterations: usize,
    select: S1,
    cont: C,
    marker: std::marker::PhantomData<G>,
}

impl<G, S1, C> SimulateCont<G> for MetaMctsInnerCont<G, S1, C>
where
    G: Game + 'static,
    S1: SelectStrategy<G> + 'static,
    C: SimulateCont<G>,
{
    type Output = C::Output;

    fn call<S2: SimulateStrategy<G> + 'static>(self, simulate: S2) -> C::Output {
        let inner = TreeSearch::<G, Compose<S1, S2>>::default().config(
            SearchConfig::<G, Compose<S1, S2>>::new()
                .select(self.select)
                .simulate(simulate)
                .max_iterations(self.iterations),
        );
        self.cont.call(simulate::MetaMcts { inner })
    }
}

/// The first stage of `SimulateSpec::MetaMcts`'s CPS dispatch -- resolves the
/// nested search's `select` spec (a full `SelectSpec`, not
/// `BaseSelectSpec`: select has no `MetaMcts` variant of its own, so nothing
/// here can recurse), then hands off to `MetaMctsInnerCont` for the
/// `simulate` spec.
struct MetaMctsCont<C> {
    iterations: usize,
    simulate_spec: BaseSimulateSpec,
    cont: C,
}

impl<G, C> SelectCont<G> for MetaMctsCont<C>
where
    G: Game + 'static,
    C: SimulateCont<G>,
{
    type Output = C::Output;

    fn call<S1: SelectStrategy<G> + 'static>(self, select: S1) -> C::Output {
        with_base_simulate::<G, _>(
            &self.simulate_spec,
            MetaMctsInnerCont {
                iterations: self.iterations,
                select,
                cont: self.cont,
                marker: std::marker::PhantomData,
            },
        )
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

/// `register_backprop!`'s table. `backprop::Classic` is, as of this writing,
/// the *only* type in the whole workspace implementing `BackpropStrategy`
/// (`grep -rn "impl.*BackpropStrategy for"` turns up exactly one hit), so
/// this table starts with one row -- a macro rather than a hand-written enum
/// mainly so a second `BackpropStrategy` impl slots in the same way a new
/// `select`/`simulate` family does, without inventing a new pattern.
///
/// There is no `Base.../...` recursive-wrapper split here (contrast
/// `register_select!`/`register_simulate!`): nothing wraps a
/// `BackpropStrategy` anywhere in the codebase.
///
/// There is also no `requirements_of_backprop` -- `BackpropStrategy` itself
/// declares no `requirements()`/`backprop_flags()` method
/// (`SearchConfig::requirements()` only unions `Select`/`Simulate`/
/// `FinalAction`, never `Backprop`), so there is nothing real for such a
/// function to call through to.
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
/// (`build_with_final_action`/`build_pn_with_final_action` in
/// `mcts-tune/src/lib.rs`), never an in-tree exploration strategy like
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

////////////////////////////////////////////////////////////////////////////////

/// The full four-axis config-IR node: one spec per `Strategy<G>` axis, the
/// same four names `mcts::strategy::Compose<Sel, Sim, Bp, FA>` takes as type
/// parameters. This is a second, independent entry point alongside
/// `mcts-tune::build_search`/`make_candidate`'s `match p.family.as_str()`
/// dispatch -- it doesn't touch or replace that function, which stays the
/// dispatch path every existing caller (SMAC3, the ladder driver,
/// `strategy_tune_eval`) actually uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpec {
    pub select: SelectSpec,
    pub simulate: SimulateSpec,
    pub backprop: BackpropSpec,
    pub final_action: FinalActionSpec,
}

/// The non-strategy `SearchConfig` knobs a `SearchSpec` needs alongside
/// itself -- this module's counterpart of `mcts-tune::base_config`'s generic
/// settings (iteration/time budget, `q_init`, threading, ...), factored out
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
}

/// Resolves `spec.select`, then hands off to `SimulateStage` -- the first
/// link of the four-stage CPS chain `build_search` drives, mirroring
/// `MetaMctsCont`/`MetaMctsInnerCont`'s two-stage chain above but one axis
/// longer.
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
        with_simulate::<G, _>(
            &self.spec.simulate,
            SimulateStage {
                spec: self.spec,
                settings: self.settings,
                select,
                marker: std::marker::PhantomData,
            },
        )
    }
}

/// Second link: resolves `spec.simulate` once `select` is already resolved,
/// then hands off to `BackpropStage`.
struct SimulateStage<'a, G: Game, S1> {
    spec: &'a SearchSpec,
    settings: &'a SearchSettings,
    select: S1,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G, S1> SimulateCont<G> for SimulateStage<'a, G, S1>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<S2: SimulateStrategy<G> + 'static>(self, simulate: S2) -> Self::Output {
        with_backprop(
            &self.spec.backprop,
            BackpropStage {
                spec: self.spec,
                settings: self.settings,
                select: self.select,
                simulate,
                marker: std::marker::PhantomData,
            },
        )
    }
}

/// Third link: resolves `spec.backprop`, then hands off to `FinalActionStage`.
struct BackpropStage<'a, G: Game, S1, S2> {
    spec: &'a SearchSpec,
    settings: &'a SearchSettings,
    select: S1,
    simulate: S2,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G, S1, S2> BackpropCont for BackpropStage<'a, G, S1, S2>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
    S2: SimulateStrategy<G> + 'static,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<B: BackpropStrategy + 'static>(self, backprop: B) -> Self::Output {
        with_final_action::<G, _>(
            &self.spec.final_action,
            FinalActionStage {
                settings: self.settings,
                select: self.select,
                simulate: self.simulate,
                backprop,
                marker: std::marker::PhantomData,
            },
        )
    }
}

/// Final link: resolves `spec.final_action` and, with all four axes now
/// concrete, builds the actual `TreeSearch<G, Compose<S1, S2, B, FA>>` and
/// boxes it -- the only place in this chain that touches `SearchConfig`
/// directly, since it's the first point every type parameter `Compose` needs
/// is known.
struct FinalActionStage<'a, G: Game, S1, S2, B> {
    settings: &'a SearchSettings,
    select: S1,
    simulate: S2,
    backprop: B,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G, S1, S2, B> SelectCont<G> for FinalActionStage<'a, G, S1, S2, B>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
    S1: SelectStrategy<G> + 'static,
    S2: SimulateStrategy<G> + 'static,
    B: BackpropStrategy + 'static,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<FA: SelectStrategy<G> + 'static>(self, final_action: FA) -> Self::Output {
        let mut config = SearchConfig::<G, Compose<S1, S2, B, FA>>::new()
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
            .backprop(self.backprop)
            .final_action(final_action);
        if let Some(max_time) = self.settings.max_time {
            config = config.max_time(max_time);
        }
        Box::new(TreeSearch::<G, Compose<S1, S2, B, FA>>::new().config(config))
    }
}

/// Builds a runnable `Box<dyn Search<G = G>>` from a `SearchSpec` -- the
/// config-IR equivalent of `mcts-tune::build_search`, but driven entirely by
/// this file's registry-generated dispatch instead of `TrialParams`/
/// `make_candidate`'s hand-written `match p.family.as_str()`. Nothing calls
/// this yet: it's a second, independently-tested entry point sitting next to
/// the existing one, not a replacement for it.
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
        let json = r#"{"kind":"meta_mcts","iterations":50,"select":{"kind":"ucb1","c":1.4},"simulate":{"kind":"uniform"}}"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            SimulateSpec::MetaMcts {
                iterations: 50,
                select: SelectSpec::Ucb1 { c: 1.4 },
                simulate: BaseSimulateSpec::Uniform {},
            }
        );
        assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
    }

    #[test]
    fn meta_mcts_inner_select_can_itself_be_wrapped_in_epsilon_greedy() {
        // The inner `select` field is a full `SelectSpec`, not a
        // `BaseSelectSpec` -- unlike `simulate`, nothing on the select side
        // can recurse into `MetaMcts`, so there's no reason to forbid its
        // own `EpsilonGreedy` wrapper here.
        let json = r#"{
            "kind": "meta_mcts",
            "iterations": 25,
            "select": {"kind": "epsilon_greedy", "epsilon": 0.1, "inner": {"kind": "ucb1", "c": 1.4}},
            "simulate": {"kind": "mast"}
        }"#;
        let spec: SimulateSpec = serde_json::from_str(json).unwrap();
        let SimulateSpec::MetaMcts { select, .. } = &spec else {
            panic!("expected MetaMcts");
        };
        assert!(matches!(select, SelectSpec::EpsilonGreedy { .. }));
    }

    #[test]
    fn with_simulate_builds_a_working_nested_search_for_meta_mcts() {
        let spec = SimulateSpec::MetaMcts {
            iterations: 25,
            select: SelectSpec::Ucb1 { c: 1.4 },
            simulate: BaseSimulateSpec::Uniform {},
        };
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
            simulate: SimulateSpec::MetaMcts {
                iterations: 25,
                select: SelectSpec::Ucb1 { c: 1.4 },
                simulate: BaseSimulateSpec::Uniform {},
            },
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
}
