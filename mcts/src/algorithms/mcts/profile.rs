use super::*;
use crate::game::Game;
use std::marker::PhantomData;

/// Name a composed search by its four `PolicyProfile` axes instead of a bespoke
/// marker type. `TreeSearch<G, Mcts<select::Ucb1, simulate::Mast>>` is the
/// fully static equivalent of a hand-written `struct` + `impl PolicyProfile<G>`
/// (still monomorphized -- `Mcts` carries no runtime state, just the four
/// type parameters). Every axis defaults: `Mcts::default()` (no type params
/// named) is plain UCT -- UCB1 selection, uniform-random playouts, classic
/// backprop, robust-child final move -- and each parameter named explicitly
/// overrides just that axis.
///
/// `Mcts` (or, on the tune path, a `config_ir::SearchSpec` built from it)
/// is *the* way to spell a composed search. Convenience bundles for a specific
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
pub struct Mcts<
    Sel = select::Ucb1,
    Sim = simulate::Uniform,
    Bp = backprop::Classic,
    FA = select::RobustChild,
>(PhantomData<(Sel, Sim, Bp, FA)>);

impl<G, Sel, Sim, Bp, FA> PolicyProfile<G> for Mcts<Sel, Sim, Bp, FA>
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

/// Transitional alias for the pre-`profile` name. Migrating.
pub use Mcts as Compose;

/// Transitional alias: plain UCT is now `Mcts` with all default type params.
pub type Ucb1 = Mcts;
