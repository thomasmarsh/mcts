use super::backprop::{with_backprop, BackpropCont, BackpropSpec};
use super::search::SearchSpec;
use super::select::{requirements_of, RequirementsCont, SelectCont};
use mcts::backprop::BackpropStrategy;
use mcts::game::Game;
use mcts::index::Id;
use mcts::node::ChildArray;
use mcts::select::{self, SelectContext, SelectStrategy};
use mcts::strategies::mcts::config::BackpropFlags;
use mcts::Requirements;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

/// The config-IR node for the `final_action` axis. It is deliberately a
/// separate, smaller table from `SelectSpec`/`register_select!`: `final_action` only ever
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
