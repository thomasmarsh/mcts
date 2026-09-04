use super::SearchConfig;

use super::*;
use crate::game::Game;
use std::marker::PhantomData;

/// Ad hoc composition of the four `Strategy` axes without declaring a new
/// marker type: `TreeSearch<G, Compose<select::Ucb1, simulate::Uniform>>` is
/// the fully static equivalent of a bespoke `struct` + `impl Strategy<G>`
/// (still monomorphized -- `Compose` carries no runtime state, just the four
/// type parameters). `Backprop`/`FinalAction` default to the common
/// `Classic`/`RobustChild` pair; name them explicitly to override either.
/// Reach for a named marker type (like the ones below) instead of `Compose`
/// only when the strategy needs its own `friendly_name()` or a `config()`
/// override with non-default settings.
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

// Vanilla UCT
#[derive(Clone, Default)]
pub struct Ucb1;

impl<G: Game> Strategy<G> for Ucb1 {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

// Vanilla UCT + decisive move
#[derive(Clone, Default)]
pub struct Ucb1DM;

impl<G: Game> Strategy<G> for Ucb1DM {
    type Select = select::Ucb1;
    type Simulate = simulate::DecisiveMove<G>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

// Vanilla UCT + PN-MCTS's UCT-PN selection formula (Kowalski et al. 2023).
// Only meaningful with `use_mcts_solver` on -- see `select::UctPn`'s doc
// comment.
#[derive(Clone, Default)]
pub struct Ucb1Pn;

impl<G: Game> Strategy<G> for Ucb1Pn {
    type Select = select::UctPn;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_pn".into()
    }
}

// Vanilla UCT + Generalized Proof-Number MCTS selection (Kowalski, Soemers,
// Kosakowski & Winands, arXiv:2506.13249): UCB1 plus a per-player
// proof-number bias (PNMax by default). Only meaningful with
// `use_mcts_solver` on -- see `select::GpnUct`'s doc comment. Unlike
// `Ucb1Pn`, works at any player count.
#[derive(Clone, Default)]
pub struct Ucb1Gpn;

impl<G: Game> Strategy<G> for Ucb1Gpn {
    type Select = select::GpnUct;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_gpn".into()
    }
}

// Vanilla UCT + Score-Bounded MCTS selection (Cazenave & Saffidine, CG
// 2010): alpha-beta-style pruning + bound-induced value bias from each
// node's graded-score interval. Only meaningful with `use_mcts_solver` on
// and a two-player game that overrides `Game::score_bounds()` -- see
// `select::ScoreBoundedUct`'s doc comment.
#[derive(Clone, Default)]
pub struct Ucb1ScoreBounded;

impl<G: Game> Strategy<G> for Ucb1ScoreBounded {
    type Select = select::ScoreBoundedUct;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_score_bounded".into()
    }
}

// UCT-PN + Mast
#[derive(Clone, Default)]
pub struct Ucb1PnMast;

impl<G: Game> Strategy<G> for Ucb1PnMast {
    type Select = select::UctPn;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_pn_mast".into()
    }
}

// Vanilla UCT + Mast
#[derive(Clone, Default)]
pub struct Ucb1Mast;

impl<G: Game> Strategy<G> for Ucb1Mast {
    type Select = select::Ucb1;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_mast".into()
    }
}

// Vanilla UCT + NST
#[derive(Clone, Default)]
pub struct Ucb1Nst;

impl<G: Game> Strategy<G> for Ucb1Nst {
    type Select = select::Ucb1;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Nst>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_nst".into()
    }
}

// Vanilla UCT + Progressive History
#[derive(Clone, Default)]
pub struct Ucb1ProgressiveHistory;

impl<G: Game> Strategy<G> for Ucb1ProgressiveHistory {
    type Select = select::ProgressiveHistory;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_progressive_history".into()
    }
}

// Vanilla UCT, but the final move choice requires visits and average score
// to agree (falling back to average score alone when they don't).
#[derive(Clone, Default)]
pub struct Ucb1MaxRobust;

impl<G: Game> Strategy<G> for Ucb1MaxRobust {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxRobustChild;

    fn friendly_name() -> String {
        "ucb1_max_robust".into()
    }
}

#[derive(Clone, Default)]
pub struct Amaf;

impl<G: Game> Strategy<G> for Amaf {
    type Select = select::Amaf;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "amaf".into()
    }
}

#[derive(Clone, Default)]
pub struct QuasiBestFirst;

impl<G: Game> Strategy<G> for QuasiBestFirst {
    type Select = select::EpsilonGreedy<G, select::QuasiBestFirst<G, Ucb1Mast>>;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "qbf/ucb1+mast".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().select(select::EpsilonGreedy::new().epsilon(0.3))
    }
}

/// Rave select + DecisiveMove<EpsilonGreedy<Mast>> simulate, the shape every
/// non-Druid demo/test call site tunes hyperparameters around. Generic over
/// `G` (unlike the other named strategies above) because its `Simulate` type
/// embeds `G` itself.
pub type RaveMastDm<G> =
    Compose<select::Rave, simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>>;
