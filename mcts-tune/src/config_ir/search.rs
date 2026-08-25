use super::backprop::{with_backprop, BackpropCont, BackpropSpec};
use super::final_action::{resolve_final_action, FinalActionSpec};
use super::select::{resolve_select, DynSelect, SelectSpec};
use super::simulate::{resolve_simulate, DynSimulate, SimulateSpec};
use mcts::backprop::BackpropStrategy;
use mcts::game::Game;
use mcts::node::QInit;
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{GraphSearch, SearchConfig, TranspositionKeying, TreeSearch};
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
    pub transposition_keying: TranspositionKeying,
    pub solver_loss_threshold: Option<u32>,
    pub contempt_factor: Option<f64>,
}

/// Resolves `spec.backprop` -- the one remaining generic link in
/// `build_search`'s dispatch chain. `select`, `simulate`, and `final_action`
/// are all resolved eagerly, before this stage, into fixed types: `select`
/// and `final_action` both resolve to `DynSelect<G>` (two independently
/// configured specs that happen to erase through the same type -- see
/// `select.rs`'s `DynSelect` doc comment), and `simulate` resolves to
/// `DynSimulate<G>`. Chaining every axis through a fully generic continuation
/// (contrast an earlier version of this file)
/// makes each one a multiplicative factor in what rustc has to
/// monomorphize -- `select` (~9 variants once `EpsilonGreedy` wrapping is
/// counted) x `simulate` (~10) x `final_action` (4) x `backprop` (3), each
/// instantiating the *entire* `TreeSearch` engine, which is what once turned
/// a clean `cargo test --no-run -p mcts-tune` into a 7+-minute,
/// `__eh_frame`-overflowing build. `backprop` is the sole survivor: with
/// `register_backprop!`'s table at 3 rows (`Classic`, `BayesGaussian`,
/// `BayesNumeric`), `with_backprop` only ever instantiates `TreeSearch` 3
/// times per game binary, cheap enough that erasing it too buys nothing
/// worth its own indirect call.
struct BackpropStage<'a, G: Game> {
    settings: &'a SearchSettings,
    select: DynSelect<G>,
    simulate: DynSimulate<G>,
    final_action: DynSelect<G>,
    marker: std::marker::PhantomData<G>,
}

impl<'a, G> BackpropCont for BackpropStage<'a, G>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
{
    type Output = Box<dyn Search<G = G>>;

    fn call<B: BackpropStrategy + 'static>(self, backprop: B) -> Self::Output {
        type S<G, B> = Compose<DynSelect<G>, DynSimulate<G>, B, DynSelect<G>>;
        let mut config = SearchConfig::<G, S<G, B>>::new()
            .max_iterations(self.settings.max_iterations)
            .max_playout_depth(self.settings.max_playout_depth)
            .expand_threshold(self.settings.expand_threshold)
            .q_init(self.settings.q_init)
            .use_transpositions(self.settings.use_transpositions)
            .use_mcts_solver(self.settings.use_mcts_solver)
            .reuse_tree(self.settings.reuse_tree)
            .transposition_keying(self.settings.transposition_keying)
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
        Box::new(TreeSearch::<G, S<G, B>>::new().config(config))
    }
}

/// Builds a runnable `Box<dyn Search<G = G>>` from a `SearchSpec` --
/// `mcts-tune::make_candidate`'s dispatch for every family except
/// `"random"`/`"flat_mc"`: convert `TrialParams` via `to_search_spec`, then
/// call this, driven entirely by this file's registry-generated dispatch
/// rather than a hand-written `match p.family.as_str()`. `select`, `simulate`,
/// and `final_action` are resolved up front into their `Dyn*<G>` forms;
/// `backprop` is resolved last, through `BackpropStage`, since it's the only
/// axis this function still dispatches by monomorphizing over the real
/// strategy type.
pub fn build_search<G>(spec: &SearchSpec, settings: &SearchSettings) -> Box<dyn Search<G = G>>
where
    G: Game + 'static,
    G::S: std::fmt::Display,
{
    with_backprop(
        &spec.backprop,
        BackpropStage {
            settings,
            select: resolve_select::<G>(&spec.select),
            simulate: resolve_simulate::<G>(&spec.simulate),
            final_action: resolve_final_action::<G>(&spec.final_action),
            marker: std::marker::PhantomData,
        },
    )
}
