use super::codec::{field, to_snake_case};
use mcts::game::Game;
use mcts::index::Id;
use mcts::node::ChildArray;
use mcts::select::{self, SelectContext, SelectStrategy};
use mcts::strategies::mcts::config::BackpropFlags;
use mcts::Requirements;
use rand::rngs::SmallRng;
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

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
        ///
        /// `Serialize`/`Deserialize` are hand-implemented below rather than
        /// `#[derive]`d -- see `register_backprop!`'s doc comment in
        /// `backprop.rs` for why (identical wire format, none of serde's
        /// generic derive machinery).
        #[derive(Debug, Clone, PartialEq)]
        pub enum BaseSelectSpec {
            $(
                $variant { $($field: $ty),* },
            )+
        }

        impl Serialize for BaseSelectSpec {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                match self {
                    $(
                        BaseSelectSpec::$variant { $($field),* } => {
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

        impl<'de> Deserialize<'de> for BaseSelectSpec {
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
                        return Ok(BaseSelectSpec::$variant {
                            $(
                                $field: field(&v, stringify!($field)).map_err(D::Error::custom)?,
                            )*
                        });
                    }
                )+
                Err(D::Error::custom(format!("unknown select kind: {kind:?}")))
            }
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
        ///
        /// `Serialize`/`Deserialize` are hand-implemented below rather than
        /// `#[derive]`d -- see `register_backprop!`'s doc comment in
        /// `backprop.rs` for why. `EpsilonGreedy`'s `inner: BaseSelectSpec`
        /// field routes through `BaseSelectSpec`'s own hand-implemented
        /// impls above via `field::<BaseSelectSpec>`/`serialize_entry`,
        /// exactly like any other field type -- no recursion-depth concern
        /// the way a derive-based `Content`-buffering implementation would
        /// have had, since this isn't a self-referential type (`SelectSpec`
        /// wraps `BaseSelectSpec`, which has no `SelectSpec`/`EpsilonGreedy`
        /// variant of its own).
        #[derive(Debug, Clone, PartialEq)]
        pub enum SelectSpec {
            $(
                $variant { $($field: $ty),* },
            )+
            EpsilonGreedy {
                epsilon: f64,
                inner: BaseSelectSpec,
            },
        }

        impl Serialize for SelectSpec {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                match self {
                    $(
                        SelectSpec::$variant { $($field),* } => {
                            #[allow(unused_mut)]
                            let mut map = serializer.serialize_map(None)?;
                            map.serialize_entry("kind", &to_snake_case(stringify!($variant)))?;
                            $(
                                map.serialize_entry(stringify!($field), $field)?;
                            )*
                            map.end()
                        }
                    )+
                    SelectSpec::EpsilonGreedy { epsilon, inner } => {
                        let mut map = serializer.serialize_map(None)?;
                        map.serialize_entry("kind", &to_snake_case(stringify!(EpsilonGreedy)))?;
                        map.serialize_entry("epsilon", epsilon)?;
                        map.serialize_entry("inner", inner)?;
                        map.end()
                    }
                }
            }
        }

        impl<'de> Deserialize<'de> for SelectSpec {
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
                        return Ok(SelectSpec::$variant {
                            $(
                                $field: field(&v, stringify!($field)).map_err(D::Error::custom)?,
                            )*
                        });
                    }
                )+
                if kind == to_snake_case(stringify!(EpsilonGreedy)) {
                    return Ok(SelectSpec::EpsilonGreedy {
                        epsilon: field(&v, "epsilon").map_err(D::Error::custom)?,
                        inner: field(&v, "inner").map_err(D::Error::custom)?,
                    });
                }
                Err(D::Error::custom(format!("unknown select kind: {kind:?}")))
            }
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
    UcbV { c: f64 } => select::UcbV::with_c(c),
    KlUcb { c: f64 } => select::KlUcb::with_c(c),
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

/// A shadow of `SelectStrategy<G>` covering only `best_child` (the one
/// method `select_step` calls on a search's `select` component, once per
/// tree-descent step) plus `backprop_flags`/`Clone`/`Send`/`Sync` -- the
/// `select`-axis counterpart of `ErasedSimulateStrategy` in `simulate.rs`.
/// `final_action.rs`'s `resolve_final_action` reuses this same shadow trait
/// and `DynSelect` rather than defining its own copy, since both axes erase
/// the identical `SelectStrategy<G>` trait.
/// `score_child`/`unvisited_value` aren't part of this shadow: whichever
/// concrete family a `DynSelect` box holds still runs its own per-child
/// scoring loop inside its own `best_child` (the default implementation in
/// `mcts::select::SelectStrategy`), fully statically dispatched there --
/// only the one per-node call into the box is erased, not the per-child
/// comparisons inside it. Unlike `SelectStrategy` itself, this shadow is
/// object-safe (no `Self`-by-value `Default`/`Clone`, no associated `Score`/
/// `Aux` types in its signature), which is what lets `DynSelect` below erase
/// every concrete family `with_select` can produce into one type.
/// Blanket-implemented over every real `SelectStrategy`, so nothing here can
/// drift from `register_select!`'s table.
trait ErasedSelectStrategy<G: Game>: Send + Sync {
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize;
    fn backprop_flags(&self) -> BackpropFlags;
    fn clone_box(&self) -> Box<dyn ErasedSelectStrategy<G>>;
}

impl<G, S> ErasedSelectStrategy<G> for S
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

    fn clone_box(&self) -> Box<dyn ErasedSelectStrategy<G>> {
        Box::new(self.clone())
    }
}

/// One `SelectStrategy<G>` impl standing in for all of `with_select`'s
/// concrete leaf types (the `register_select!` table, each optionally
/// wrapped in `EpsilonGreedy`), via a `Box<dyn ErasedSelectStrategy<G>>` --
/// mirrors `DynSimulate` in `simulate.rs`, and is also what
/// `final_action.rs`'s `resolve_final_action` returns directly (`select` and
/// `final_action` are two independently-configured axes that both happen to
/// erase through this one type). This is what `config_ir::build_search`
/// composes `select` from: `select_step` runs a
/// `dyn` call once per tree-descent step rather than one of ~16 statically
/// monomorphized bodies, collapsing that axis's contribution to
/// `build_search`'s output to a single `TreeSearch` shape per game
/// regardless of which `select` family a config names. `Score`/`Aux` are
/// fixed to `()`: nothing outside `best_child`'s own delegated call ever
/// reads them, since `best_child` is always overridden here rather than
/// falling back to the trait's default (which is the only thing that would
/// call `score_child`/`unvisited_value`/`setup` on `Self`).
pub struct DynSelect<G: Game>(Box<dyn ErasedSelectStrategy<G>>);

impl<G: Game> Clone for DynSelect<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<G: Game> Default for DynSelect<G> {
    fn default() -> Self {
        Self(Box::new(select::Ucb1::default()))
    }
}

impl<G: Game> SelectStrategy<G> for DynSelect<G> {
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
        unreachable!("DynSelect always overrides best_child directly")
    }

    fn unvisited_value(&self, _ctx: &SelectContext<'_, G>, _aux: Self::Aux) -> Self::Score {
        unreachable!("DynSelect always overrides best_child directly")
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        self.0.best_child(ctx, rng)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.0.backprop_flags()
    }
}

/// A `SelectCont` that erases whatever concrete strategy `with_select`
/// resolves to into a `DynSelect<G>` -- reuses `with_select`'s existing
/// dispatch (so `EpsilonGreedy` wrapping still works) but stops that
/// dispatch's fan-out from propagating any further: everything downstream
/// of `resolve_select` sees one fixed type.
pub(super) struct EraseSelectCont<G>(pub(super) std::marker::PhantomData<G>);

impl<G: Game + 'static> SelectCont<G> for EraseSelectCont<G> {
    type Output = DynSelect<G>;

    fn call<S: SelectStrategy<G> + 'static>(self, select: S) -> DynSelect<G> {
        DynSelect(Box::new(select))
    }
}

/// Resolves `spec` to a single `DynSelect<G>`, regardless of family -- see
/// `DynSelect`'s doc comment for why `build_search` uses this instead of
/// routing `S1` generically through its whole stage chain.
pub fn resolve_select<G: Game + 'static>(spec: &SelectSpec) -> DynSelect<G> {
    with_select::<G, _>(spec, EraseSelectCont(std::marker::PhantomData))
}
