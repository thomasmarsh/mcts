use super::*;

use crate::game::Game;
use node::QInit;
use rand::rngs::SmallRng;
use rand_core::SeedableRng;

////////////////////////////////////////////////////////////////////////////////

pub const GRAVE: usize = 0b001;
pub const GLOBAL: usize = 0b010;
pub const AMAF: usize = 0b100;
/// NST (Tak & Winands): bigram extension of the `GLOBAL`/MAST table. Its own
/// bit rather than folding into `GLOBAL` -- `Nst::backprop_flags()` sets both
/// (NST's hard-cutover backoff still needs the unigram table `GLOBAL` writes),
/// but a plain `Mast` user shouldn't pay for the extra chronological-order
/// reconstruction and bigram-table write NST needs (see `backprop.rs`'s
/// `flags.nst()` block).
pub const NST: usize = 0b1000;

pub struct BackpropFlags(pub usize);

impl BackpropFlags {
    pub fn grave(&self) -> bool {
        self.0 & GRAVE == GRAVE
    }

    pub fn global(&self) -> bool {
        self.0 & GLOBAL == GLOBAL
    }

    pub fn amaf(&self) -> bool {
        self.0 & AMAF == AMAF
    }

    pub fn nst(&self) -> bool {
        self.0 & NST == NST
    }
}

impl std::ops::BitOr for BackpropFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait Strategy<G: Game>: Clone + Sync + Send + Default {
    type Select: select::SelectStrategy<G>;
    type Simulate: simulate::SimulateStrategy<G>;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<G>;

    fn friendly_name() -> String {
        "unknown".into()
    }

    // Override new to provide strategy specific defaults
    fn config() -> SearchConfig<G, Self> {
        SearchConfig::default()
    }
}

#[derive(Clone)]
pub struct SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    pub select: S::Select,
    pub simulate: S::Simulate,
    pub backprop: S::Backprop,
    pub final_action: S::FinalAction,
    pub q_init: QInit,
    pub expand_threshold: u32,
    pub max_playout_depth: usize,
    pub max_iterations: usize,
    pub max_time: std::time::Duration,
    pub use_transpositions: bool,

    /// MCTS-Solver (Winands et al.): backprop derives and propagates proven
    /// win/loss/draw status alongside the usual visit/score stats, selection
    /// short-circuits onto a proven-winning child and avoids proven-losing
    /// ones, and `choose_action` stops early once the root itself is
    /// resolved. `false` (the default) keeps the untouched plain-UCT
    /// behavior. Scoped to `G::num_players() <= 2` -- see
    /// `debug_assert!`s at the call sites that derive `Proven` values.
    pub use_mcts_solver: bool,

    /// Final-move-selection "contempt factor" (Kowalski et al. 2023, Section
    /// VII.C): when the root's own expected score for the player to move
    /// falls below this threshold, `select_final_action` prefers any root
    /// child already proven a draw over whatever `final_action` would
    /// otherwise pick -- accepting a known draw rather than gambling on an
    /// unresolved line that looks promising in the averages but might
    /// secretly be a loss. Checked only after the proven-win short-circuit
    /// finds no win, and only when `use_mcts_solver` is on (draws are only
    /// ever provably known with the solver enabled). `None` (the default)
    /// disables this -- the paper's own baseline, a contempt factor below
    /// every possible outcome (`< -1`), "behaves exactly as a single-layer
    /// PN-MCTS final move selection".
    pub contempt_factor: Option<f64>,

    /// MCTS-Solver's proven-loss selection threshold `T` (Kowalski et al.
    /// 2023, Section III.B): a child already proven a loss for the mover is
    /// only excluded from selection once its own visit count exceeds this,
    /// rather than the instant it's proven. Guards against the "narrow
    /// paths" bias the paper describes -- hard-excluding a proven-loss
    /// sibling immediately can over-concentrate search onto whatever
    /// children remain before their own stats are trustworthy. The paper
    /// uses `T = 5` throughout its experiments. `0` (the default) means
    /// every proven-loss child is already excluded on its very first visit
    /// (see `select::is_proven_loss`'s doc comment for why that's exactly
    /// the prior unconditional-exclusion behavior, not merely close to it) --
    /// i.e. this knob is additive, not a behavior change by default.
    pub solver_loss_threshold: u32,

    pub rng: SmallRng,
    pub verbose: bool,
    pub name: String,

    /// Number of independent trees to search in parallel ("root
    /// parallelism"): each thread runs its own full `TreeSearch` to
    /// completion, and the root-level visit counts/scores are summed across
    /// trees to pick the final action. `1` (the default) keeps the
    /// single-threaded path with unchanged behavior/output.
    ///
    /// Composes with `num_tree_threads`: when both are `> 1`, this many
    /// trees are searched in parallel and each of *those* is itself
    /// tree-parallel across `num_tree_threads` threads -- a hybrid split
    /// (e.g. 4 trees x 2 threads each on 8 cores), the standard way to
    /// balance shared-tree lock contention against duplicated search effort.
    pub num_threads: usize,

    /// Number of rollouts to run per selected leaf ("leaf parallelism"):
    /// after `select` walks the tree single-threaded and picks one leaf, up
    /// to `num_rollouts_per_leaf` playouts are fired from that leaf's state
    /// on separate threads and each backpropagated in turn, instead of just
    /// one. Cheaper than tree parallelism since it never touches node/edge
    /// structure concurrently -- simulate strategies already operate on a
    /// state cloned for the rollout -- but strategies that embed their own
    /// nested search (`simulate::MetaMcts`) should leave this at `1`, since
    /// cloning a full nested `TreeSearch` per rollout isn't worth it there.
    /// `1` (the default) keeps the untouched single-rollout-per-leaf path.
    pub num_rollouts_per_leaf: usize,

    /// Number of worker threads that descend *one shared* tree concurrently
    /// ("tree parallelism"), using virtual loss so they spread out across
    /// the tree instead of piling onto the same path. Unlike `num_threads`
    /// (root parallelism, independent trees merged at the end), this shares
    /// search effort across threads -- the bigger win, at the cost of the
    /// arena/stats needing to be concurrent-safe. `1` (the default) keeps
    /// the untouched single-threaded path.
    ///
    /// Composes with `num_threads` for a hybrid split -- see its doc comment.
    pub num_tree_threads: usize,

    /// Tree reuse across moves ("re-rooting"): instead of discarding the
    /// whole search tree at the start of every `choose_action` call, try to
    /// find the node matching the incoming state somewhere in the tree left
    /// over from the previous call (by Zobrist hash, bounded to a few plies
    /// of search -- see `TreeSearch::find_reachable`) and promote it to be
    /// the new root, keeping its accumulated visit/score stats instead of
    /// starting over. Falls back to the untouched full-reset behavior
    /// whenever no match is found (first move of a game, or the actual play
    /// went somewhere this side's own search never reached). `false` (the
    /// default) keeps every prior determinism/regression test's assumption
    /// that two `choose_action` calls with the same config never share state.
    pub reuse_tree: bool,

    /// Bounded pruning after re-rooting: once `reuse_or_reset` promotes a
    /// child to root and the arena's total node count exceeds this, compact
    /// the arena down to just the subtree reachable from the new root
    /// (`TreeSearch::compact`), discarding every unreachable sibling
    /// `reuse_tree` would otherwise leave as garbage forever. `None` (the
    /// default) never compacts -- unbounded growth, byte-identical to every
    /// prior session's behavior. Only meaningful when `reuse_tree` is also
    /// `true`; a plain `reset()` already starts from a single-node arena, so
    /// there's nothing to compact on that path.
    pub max_arena_len: Option<usize>,
}

impl<G, S> Default for SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: QInit::default(),
            expand_threshold: 1,
            max_playout_depth: usize::MAX,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
            use_mcts_solver: false,
            contempt_factor: None,
            solver_loss_threshold: 0,
            rng: SmallRng::from_entropy(),
            verbose: false,
            name: format!("mcts[{}]", S::friendly_name()),
            num_threads: 1,
            num_rollouts_per_leaf: 1,
            num_tree_threads: 1,
            reuse_tree: false,
            max_arena_len: None,
        }
    }
}

impl<G, S> SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select(mut self, select: S::Select) -> Self {
        self.select = select;
        self
    }

    pub fn simulate(mut self, simulate: S::Simulate) -> Self {
        self.simulate = simulate;
        self
    }

    pub fn backprop(mut self, backprop: S::Backprop) -> Self {
        self.backprop = backprop;
        self
    }

    pub fn final_action(mut self, final_action: S::FinalAction) -> Self {
        self.final_action = final_action;
        self
    }

    pub fn q_init(mut self, q_init: QInit) -> Self {
        self.q_init = q_init;
        self
    }

    pub fn expand_threshold(mut self, expand_threshold: u32) -> Self {
        self.expand_threshold = expand_threshold;
        self
    }

    pub fn max_playout_depth(mut self, max_playout_depth: usize) -> Self {
        self.max_playout_depth = max_playout_depth;
        self
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    // NOTE: special logic here
    pub fn max_time(mut self, max_time: std::time::Duration) -> Self {
        self.max_time = max_time;
        if self.max_time != std::time::Duration::default() {
            self.max_iterations(usize::MAX)
        } else {
            self
        }
    }

    pub fn use_transpositions(mut self, use_transpositions: bool) -> Self {
        self.use_transpositions = use_transpositions;
        self
    }

    pub fn use_mcts_solver(mut self, use_mcts_solver: bool) -> Self {
        self.use_mcts_solver = use_mcts_solver;
        self
    }

    pub fn contempt_factor(mut self, contempt_factor: Option<f64>) -> Self {
        self.contempt_factor = contempt_factor;
        self
    }

    pub fn solver_loss_threshold(mut self, solver_loss_threshold: u32) -> Self {
        self.solver_loss_threshold = solver_loss_threshold;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn rng(mut self, rng: SmallRng) -> Self {
        self.rng = rng;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads.max(1);
        self
    }

    pub fn num_rollouts_per_leaf(mut self, num_rollouts_per_leaf: usize) -> Self {
        self.num_rollouts_per_leaf = num_rollouts_per_leaf.max(1);
        self
    }

    pub fn num_tree_threads(mut self, num_tree_threads: usize) -> Self {
        self.num_tree_threads = num_tree_threads.max(1);
        self
    }

    pub fn reuse_tree(mut self, reuse_tree: bool) -> Self {
        self.reuse_tree = reuse_tree;
        self
    }

    pub fn max_arena_len(mut self, max_arena_len: Option<usize>) -> Self {
        self.max_arena_len = max_arena_len;
        self
    }
}
