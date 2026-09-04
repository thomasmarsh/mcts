use super::*;
use crate::game::Game;
use std::marker::PhantomData;

/// Name a composed search by its four `Strategy` axes instead of a bespoke
/// marker type. `TreeSearch<G, Compose<select::Ucb1, simulate::Mast>>` is the
/// fully static equivalent of a hand-written `struct` + `impl Strategy<G>`
/// (still monomorphized -- `Compose` carries no runtime state, just the four
/// type parameters). `Backprop`/`FinalAction` default to the common
/// `Classic`/`RobustChild` pair; name them explicitly to override either.
///
/// `Compose` (or, on the tune path, a `config_ir::SearchSpec` built from it)
/// is *the* way to spell a composed search. The only named `Strategy` impl
/// the core library still ships is `Ucb1`; convenience bundles for a specific
/// experiment belong next to that experiment as a local `type` alias, not a
/// `pub struct` in `mcts`.
///
/// # `SearchConfig::name`
///
/// A composed search has no bespoke name, so `SearchConfig::default()`
/// builds one from the four axes' `label()`s (`compose_search_name` in
/// `config.rs`):
///
/// - always `mcts[<select>]` -- e.g. `mcts[ucb1]`;
/// - `+<simulate>` is appended when the simulate label isn't the default
///   `"uniform"` -- e.g. `mcts[ucb1+mast]`, `mcts[ucb1+eps_greedy(mast)]`
///   (wrapper policies fold their inner label in);
/// - once `backprop` or `final_action` is non-default the name switches to
///   the positional form `mcts[<select>/<simulate>/<backprop>]` --
///   e.g. `mcts[ments/uniform/softmax]` -- with `/<final_action>` appended
///   only when that label isn't the default `"robust_child"`.
///
/// The builder setters (`.select()` / `.simulate()` / `.backprop()` /
/// `.final_action()`) recompute this name in place; an explicit `.name(..)`
/// freezes it.
#[derive(Clone, Copy, Default)]
pub struct Compose<Sel, Sim, Bp = backprop::Classic, FA = select::RobustChild>(
    PhantomData<(Sel, Sim, Bp, FA)>,
);

impl<G, Sel, Sim, Bp, FA> Strategy<G> for Compose<Sel, Sim, Bp, FA>
where
    G: Game,
    Sel: select::SelectPolicy<G>,
    Sim: simulate::SimulatePolicy<G>,
    Bp: backprop::BackpropPolicy,
    FA: select::SelectPolicy<G>,
{
    type Select = Sel;
    type Simulate = Sim;
    type Backprop = Bp;
    type FinalAction = FA;
}

/// Plain UCT with nothing added: UCB1 selection, uniform-random playouts,
/// classic backprop, robust-child final move. The do-nothing baseline and the
/// default `S` type parameter of `SearchConfig<G, S>` / `TreeSearch<G, S>`.
#[derive(Clone, Default)]
pub struct Ucb1;

impl<G: Game> Strategy<G> for Ucb1 {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;
}
