use super::codec::{field, to_snake_case};
use mcts::game::Game;
use mcts::search::TreeStats;
use mcts::select;
use mcts::simulate::{self, SimulatePolicy, Trial};
use mcts::algorithms::mcts::config::BackpropFlags;
use mcts::algorithms::mcts::strategy::Compose;
use mcts::{Requirements, SearchConfig, TreeSearch};
use rand::rngs::SmallRng;
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Invokes a continuation after resolving a concrete simulation strategy.
pub trait SimulateCont<G: Game> {
    type Output;
    fn call<S: SimulatePolicy<G> + 'static>(self, simulate: S) -> Self::Output;
}

/// `register_simulate!`'s table, expanded into `BaseSimulateSpec`/
/// `SimulateSpec` and their dispatchers, mirroring `register_select!` above.
///
/// `EpsilonGreedy` and `DecisiveMove` are not rows here -- both wrap an
/// *inner* spec (`simulate::EpsilonGreedy<G, S>`/`simulate::DecisiveMove<G,
/// S>` are generic over an arbitrary inner `SimulatePolicy`), and are
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
/// wraps a whole nested `TreeSearch`, not a `SimulatePolicy`. Its inner
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
        ///
        /// `Serialize`/`Deserialize` are hand-implemented below rather than
        /// `#[derive]`d -- see `register_backprop!`'s doc comment in
        /// `backprop.rs` for why.
        #[derive(Debug, Clone, PartialEq)]
        pub enum BaseSimulateSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        impl Serialize for BaseSimulateSpec {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                match self {
                    $(
                        BaseSimulateSpec::$variant { $($field),* } => {
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

        impl<'de> Deserialize<'de> for BaseSimulateSpec {
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
                        return Ok(BaseSimulateSpec::$variant {
                            $(
                                $field: field(&v, stringify!($field)).map_err(D::Error::custom)?,
                            )*
                        });
                    }
                )+
                Err(D::Error::custom(format!("unknown simulate kind: {kind:?}")))
            }
        }

        /// Dispatches `spec` to the concrete `SimulatePolicy<G>` it names
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
        /// `BaseSimulateSpec` variant, plus `EpsilonGreedy`/`DecisiveMove`
        /// each wrapping one of them.
        ///
        /// `Serialize`/`Deserialize` are hand-implemented below rather than
        /// `#[derive]`d -- see `register_backprop!`'s doc comment in
        /// `backprop.rs` for why. The five wrapper variants below aren't
        /// part of `$variant`'s table (see this macro's doc comment on why
        /// `EpsilonGreedy`/`DecisiveMove`/`DecisiveMoveMast`/
        /// `DecisiveMoveNst`/`MetaMcts` are hand-written here rather than
        /// table rows), so their match arms are hand-written too, the same
        /// way their `with_simulate` dispatch arms already are below --
        /// adding a new one still means touching this macro body in one
        /// place, not a second file.
        #[derive(Debug, Clone, PartialEq)]
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

        impl Serialize for SimulateSpec {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                match self {
                    $(
                        SimulateSpec::$variant { $($field),* } => {
                            #[allow(unused_mut)]
                            let mut map = serializer.serialize_map(None)?;
                            map.serialize_entry("kind", &to_snake_case(stringify!($variant)))?;
                            $(
                                map.serialize_entry(stringify!($field), $field)?;
                            )*
                            map.end()
                        }
                    )+
                    SimulateSpec::EpsilonGreedy { epsilon, inner } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(EpsilonGreedy)))?;
                        map.serialize_entry("epsilon", epsilon)?;
                        map.serialize_entry("inner", inner)?;
                        map.end()
                    }
                    SimulateSpec::DecisiveMove { mode, inner } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(DecisiveMove)))?;
                        map.serialize_entry("mode", mode)?;
                        map.serialize_entry("inner", inner)?;
                        map.end()
                    }
                    SimulateSpec::DecisiveMoveMast { mode, epsilon } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(DecisiveMoveMast)))?;
                        map.serialize_entry("mode", mode)?;
                        map.serialize_entry("epsilon", epsilon)?;
                        map.end()
                    }
                    SimulateSpec::DecisiveMoveNst {
                        mode,
                        epsilon,
                        nst_backoff_threshold,
                    } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(DecisiveMoveNst)))?;
                        map.serialize_entry("mode", mode)?;
                        map.serialize_entry("epsilon", epsilon)?;
                        map.serialize_entry("nst_backoff_threshold", nst_backoff_threshold)?;
                        map.end()
                    }
                    SimulateSpec::MetaMcts { iterations } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(MetaMcts)))?;
                        map.serialize_entry("iterations", iterations)?;
                        map.end()
                    }
                }
            }
        }

        impl<'de> Deserialize<'de> for SimulateSpec {
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
                        return Ok(SimulateSpec::$variant {
                            $(
                                $field: field(&v, stringify!($field)).map_err(D::Error::custom)?,
                            )*
                        });
                    }
                )+
                if kind == to_snake_case(stringify!(EpsilonGreedy)) {
                    return Ok(SimulateSpec::EpsilonGreedy {
                        epsilon: field(&v, "epsilon").map_err(D::Error::custom)?,
                        inner: field(&v, "inner").map_err(D::Error::custom)?,
                    });
                }
                if kind == to_snake_case(stringify!(DecisiveMove)) {
                    return Ok(SimulateSpec::DecisiveMove {
                        mode: field(&v, "mode").map_err(D::Error::custom)?,
                        inner: field(&v, "inner").map_err(D::Error::custom)?,
                    });
                }
                if kind == to_snake_case(stringify!(DecisiveMoveMast)) {
                    return Ok(SimulateSpec::DecisiveMoveMast {
                        mode: field(&v, "mode").map_err(D::Error::custom)?,
                        epsilon: field(&v, "epsilon").map_err(D::Error::custom)?,
                    });
                }
                if kind == to_snake_case(stringify!(DecisiveMoveNst)) {
                    return Ok(SimulateSpec::DecisiveMoveNst {
                        mode: field(&v, "mode").map_err(D::Error::custom)?,
                        epsilon: field(&v, "epsilon").map_err(D::Error::custom)?,
                        nst_backoff_threshold: field(&v, "nst_backoff_threshold")
                            .map_err(D::Error::custom)?,
                    });
                }
                if kind == to_snake_case(stringify!(MetaMcts)) {
                    return Ok(SimulateSpec::MetaMcts {
                        iterations: field(&v, "iterations").map_err(D::Error::custom)?,
                    });
                }
                Err(D::Error::custom(format!("unknown simulate kind: {kind:?}")))
            }
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
    Lgr {} => simulate::Lgr::<G>::new(),
    Lgr2 {} => simulate::Lgr2::<G>::new(),
    // Lgr2Mast: simulate::Lgr2<G, simulate::Lgr<G, simulate::Mast>> --
    // LGRF-2's usual pairing with MAST (Baier & Drake 2010; Tak, Winands &
    // Björnsson 2012) as the fallback once both reply tables miss, instead
    // of Lgr2's plain uniform-random bottom. A fixed composition (like
    // DecisiveMoveMast above) rather than a generically configurable inner,
    // for the same reason register_simulate!'s doc comment gives for not
    // making wrappers arbitrarily recursive.
    Lgr2Mast {} => simulate::Lgr2::<G, simulate::Lgr<G, simulate::Mast>>::new(),
}

/// Forwards a resolved `S: SimulatePolicy<G>` on to `cont`, wrapped in
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

    fn call<S: SimulatePolicy<G> + 'static>(self, simulate: S) -> C::Output {
        let wrapped = simulate::EpsilonGreedy::<G, S>::with_epsilon(self.epsilon).inner(simulate);
        self.cont.call(wrapped)
    }
}

/// Forwards a resolved `S: SimulatePolicy<G>` on to `cont`, wrapped in
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

    fn call<S: SimulatePolicy<G> + 'static>(self, simulate: S) -> C::Output {
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

    fn call<S: SimulatePolicy<G>>(self, simulate: S) -> Requirements {
        simulate.requirements()
    }
}

pub fn requirements_of_simulate<G: Game + 'static>(spec: &SimulateSpec) -> Requirements {
    with_simulate::<G, _>(spec, SimulateRequirementsCont(std::marker::PhantomData))
}

/// A shadow of `SimulatePolicy<G>` covering only the methods anything
/// outside this module's own dispatch machinery actually calls on a
/// resolved simulate strategy (`playout`, `backprop_flags`, plus what
/// `Clone`/`Send`/`Sync` need) -- unlike `SimulatePolicy` itself, this one
/// is object-safe (no `Self`-by-value `Default`/`Clone` in its signature),
/// which is what lets `DynSimulate` below erase the ~10 concrete leaf types
/// `with_simulate` can produce (3 base families x wrapped in `EpsilonGreedy`/
/// `DecisiveMove`, plus `MetaMcts`) into one. Blanket-implemented over every
/// real `SimulatePolicy`, so nothing here can drift from
/// `register_simulate!`'s table -- there's no second by-hand list of
/// families to keep in sync.
trait ErasedSimulatePolicy<G: Game>: Send + Sync {
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G>;
    fn backprop_flags(&self) -> BackpropFlags;
    fn clone_box(&self) -> Box<dyn ErasedSimulatePolicy<G>>;
}

impl<G, S> ErasedSimulatePolicy<G> for S
where
    G: Game,
    S: SimulatePolicy<G> + 'static,
{
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        SimulatePolicy::playout(self, state, max_playout_depth, stats, prev_action, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        SimulatePolicy::backprop_flags(self)
    }

    fn clone_box(&self) -> Box<dyn ErasedSimulatePolicy<G>> {
        Box::new(self.clone())
    }
}

/// One `SimulatePolicy<G>` impl standing in for all of `with_simulate`'s
/// concrete leaf types, via a `Box<dyn ErasedSimulatePolicy<G>>` --
/// `build_search`'s way of stopping the `select` x `simulate` x
/// `final_action` monomorphization product from ever including `simulate`'s
/// share of the fan-out. `SimulatePolicy::playout` is called once per
/// search *iteration* (its own per-ply `select_move` calls happen inside
/// whichever concrete type's own `playout` body runs, fully statically
/// dispatched there), so the one indirect call this adds per iteration is
/// cheap relative to a whole rollout's game-state work -- `select`'s own,
/// hotter per-child dispatch (once per child, at every node, every
/// tree-descent step) is erased the same way, via `DynSelect`.
pub struct DynSimulate<G: Game>(Box<dyn ErasedSimulatePolicy<G>>);

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

impl<G: Game> SimulatePolicy<G> for DynSimulate<G> {
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

    fn call<S: SimulatePolicy<G> + 'static>(self, simulate: S) -> DynSimulate<G> {
        DynSimulate(Box::new(simulate))
    }
}

/// Resolves `spec` to a single `DynSimulate<G>`, regardless of variant --
/// see `DynSimulate`'s doc comment for why `build_search` uses this instead
/// of routing `S2` generically through its whole stage chain.
pub fn resolve_simulate<G: Game + 'static>(spec: &SimulateSpec) -> DynSimulate<G> {
    with_simulate::<G, _>(spec, EraseSimulateCont(std::marker::PhantomData))
}
