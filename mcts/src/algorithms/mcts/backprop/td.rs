use super::*;

/// Sarsa-UCT(λ) / TD(λ) bootstrapped backup (Vodopivec, Samothrakis, Šter,
/// "On Monte Carlo Tree Search and Reinforcement Learning", JAIR 2017): after
/// a playout, each ancestor accumulates a truncated λ-return
/// `G_t = (1 − λ)·V(s_{t+1}) + λ·G_{t+1}` (γ = 1, terminal-only reward, base
/// case `G_L = z`) instead of the shared terminal return `z`. Bootstrapping
/// from the child's *current* estimate counters the stale-cumulative-mean
/// bias an upper tree node's plain average carries once its subtree's policy
/// has moved on. The step size is the sample average
/// `α = 1/N` -- exactly the existing score-sum / visit-count machinery, so
/// this is the textbook offline λ-return update
/// `V(s_t) ← V(s_t) + (1/N)(G_t − V(s_t))`.
/// TODO: a constant step size `α` would need a separate stored value field
/// on `PlayerStats` (read by `expected_score` instead of `score / visits`)
/// and is deferred.
///
/// `lambda == 1.0` recovers plain UCT exactly -- every node accumulates `z`,
/// bit-identical to `Classic` (the strategy returns `None` from `td_lambda`,
/// so `update` never clones the running return or runs the recursion).
/// `lambda == 0.0` is a pure one-step bootstrap. Vodopivec's guidance for
/// adversarial games: the useful band is `[0.8, 1.0]`; larger gains from
/// small `λ` are a single-agent / dense-reward phenomenon, so the default is
/// `1.0` and the tuner sweeps down.
///
/// `max_child` switches the bootstrap from the on-path child (Sarsa /
/// on-policy -- Khandelwal et al. ICML 2016's `MCTS(λ)`) to `max` over the
/// node's children (Q-learning / off-policy -- their `MaxMCTS(λ)`). `false`
/// (Sarsa) is the default; `MaxMCTS(λ)` composes with a max-ward backup bias
/// and can compound over-optimism. Complex backups help most with
/// sparse/delayed reward and low branching, and can hurt dense-reward
/// high-branching games (same caveat `PowerMeanBackprop`'s doc comment
/// carries).
///
/// A distinct strategy from `PowerMeanBackprop`, not composable with it: this
/// changes the *input* each node accumulates; the power-mean operator
/// *overwrites* a node's value from its children afterward. `MaxMCTS(λ)` is
/// `max_child: true` here, not `TdBackprop` combined with `PowerMeanBackprop`.
///
/// AMAF / GRAVE / GLOBAL / NST / LGR side tables and the MCTS-Solver /
/// score-bound / proof-number passes are untouched -- they keep reading the
/// raw terminal `z`. Only `PlayerStats::score` / `sum_squared_score` change,
/// so a variance-based selector (UCB1-Tuned, RAVE) sees the λ-return spread.
#[derive(Debug, Clone, Copy)]
pub struct TdBackprop {
    /// λ-return decay. `1.0` = plain Monte-Carlo mean backup (== `Classic`);
    /// `0.0` = one-step bootstrap. Tuner bounds `[0.0, 1.0]`, default `1.0`.
    pub lambda: f64,
    /// Bootstrap from `max` over children (MaxMCTS(λ)) instead of the on-path
    /// child (Sarsa-UCT(λ)). Default `false`.
    pub max_child: bool,
}

impl Default for TdBackprop {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            max_child: false,
        }
    }
}

impl TdBackprop {
    pub fn new(lambda: f64, max_child: bool) -> Self {
        Self { lambda, max_child }
    }
}

impl BackpropPolicy for TdBackprop {
    fn label(&self) -> String {
        "td".into()
    }

    fn td_lambda(&self) -> Option<(f64, bool)> {
        // lambda == 1 is exactly the mean backup; return None so `update`
        // takes the untouched `Classic` path and stays bit-identical.
        if self.lambda == 1.0 {
            None
        } else {
            Some((self.lambda, self.max_child))
        }
    }
}
