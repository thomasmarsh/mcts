use super::backprop::{with_backprop, BackpropCont, BackpropSpec};
use super::final_action::{resolve_final_action, DynFinalAction, FinalActionSpec};
use super::select::{with_select, SelectCont, SelectSpec};
use super::simulate::{resolve_simulate, DynSimulate, SimulateSpec};
use mcts::backprop::BackpropStrategy;
use mcts::game::Game;
use mcts::node::QInit;
use mcts::select::SelectStrategy;
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{GraphSearch, SearchConfig, TreeSearch};
use serde::{Deserialize, Serialize};

/// One configuration node for each axis of a composed MCTS strategy.
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
