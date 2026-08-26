use super::*;

use crate::game::Game;
use backprop::BackpropStrategy;
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
/// LGR (Last Good Reply, Baier & Drake): per-player reply table, keyed by
/// the opponent's preceding move rather than the mover's own action like
/// `GLOBAL`/`NST` -- its own bit since it neither reads nor writes either
/// of those tables (see `simulate::Lgr`'s doc comment).
pub const LGR: usize = 0b10000;
/// LGRF-2 (`simulate::Lgr2`): the 2-ply reply table, keyed by (this
/// player's own previous move, the opponent's reply to it) -- independent
/// of `LGR`'s 1-ply table since `Lgr2` reads/writes both (its own table
/// plus, via its `inner: Lgr`, `LGR`'s) rather than replacing it.
pub const LGR2: usize = 0b100000;

/// Controls whether a search owns a tree or shares positions reached by
/// distinct move orders. `Dag` uses a root-relative ply in addition to the
/// position hash, so its transposition graph cannot contain a cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphSearch {
    #[default]
    Tree,
    Dag(GraphStats),
}

/// How `GraphSearch::Dag` derives a `table::TranspositionKey` from a
/// resolved successor state. `PerPly` (the default) pairs the position hash
/// with the node's root-relative ply, so two histories only ever merge when
/// they reach the same state at the same depth -- the graph this produces is
/// necessarily acyclic, since a node's ply strictly increases along any
/// path. `StateOnly` drops ply from the key, merging on position alone
/// regardless of depth: this is what lets a genuine transposition from
/// reversible or capturing moves (the same state reached after a different
/// number of plies) share one graph node, at the cost that the resulting
/// graph can contain real cycles -- a hot loop descending it needs its own
/// depth bound rather than relying on ply's strict increase to terminate
/// (see `SearchConfig::max_playout_depth`).
///
/// `StateOnly` also raises the bar on `Game::zobrist_hash`/`Game::S`
/// equality beyond what `PerPly` already required. Merging two histories
/// into one node is only correct if the position's true value cannot depend
/// on which history reached it -- the Graph History Interaction problem
/// (Kishimoto & Müller, AAAI 2004). `PerPly` narrows this to "two histories
/// agreeing on `(hash, ply)`"; `StateOnly` widens it to "two histories
/// agreeing on `hash` alone, at any two plies", which a hash that omits a
/// history-relevant fact (a repetition counter, ko state, anything not
/// visible on the raw board) can satisfy while the true values still
/// differ. Games with such rules (Gonnect's ko rule is the concrete example
/// in this codebase) must not enable `StateOnly` until their hash is
/// audited to fully capture history. See `Game::zobrist_hash`'s doc comment
/// for the per-game contract this implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranspositionKeying {
    #[default]
    PerPly,
    StateOnly,
}

/// The owner of MCTS visit and value statistics in graph search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphStats {
    /// Each parent action has independent statistics. This is the historic
    /// transposition-table behavior.
    Edges,
    /// A shared position owns its statistics, regardless of its parent.
    Nodes,
    /// Keep local action statistics as well as the shared position estimate.
    #[default]
    Both,
}

impl GraphStats {
    pub(crate) fn uses_edges(self) -> bool {
        matches!(self, Self::Edges | Self::Both)
    }

    pub(crate) fn uses_nodes(self) -> bool {
        matches!(self, Self::Nodes | Self::Both)
    }
}

/// The paper's residual information-leak correction (arXiv 2012.11045v1),
/// only meaningful under `GraphStats::Both` (it compares an edge's local
/// estimate against its target node's shared estimate, so it has nothing to
/// read in `Edges`/`Nodes` mode where only one of the two exists). At an
/// edge into a node reached by more than one parent, the local edge
/// estimate can disagree with the node's shared estimate -- the node has
/// learned from paths this edge never took. `Disabled` (the default) never
/// checks, preserving today's behavior. `Residual { epsilon }` is the
/// paper's bounded correction: descent stops and a correction trial is
/// backpropagated only through the saved path whenever the two estimates
/// diverge by more than `epsilon`. See `correction::residual_correction` for
/// the pure algebra this drives.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum McgsCorrection {
    #[default]
    Disabled,
    Residual {
        epsilon: f64,
    },
}

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

    pub fn lgr(&self) -> bool {
        self.0 & LGR == LGR
    }

    pub fn lgr2(&self) -> bool {
        self.0 & LGR2 == LGR2
    }
}

impl std::ops::BitOr for BackpropFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Declarative summary of what one component (a `Select`/`Simulate`/
/// `Backprop` instance) needs from shared tree storage, and what hard
/// constraints it places on the game it's paired with. This generalizes
/// `BackpropFlags` (which only ever covered the four backprop bit flags) to
/// also cover the "do these two choices even make sense together" axis --
/// PLAN.md's Composable Algebra section calls this "resolution of how
/// different algorithms interact". Composing two components is `union`;
/// the union *is* the resolved interaction (storage requirements only ever
/// grow, never conflict), which is why this needs no per-pair-of-features
/// match arm the way a naive interaction table would.
///
/// Deliberately flat and `Copy`: every field here is either "some shared
/// table must exist" (safe to over-provision -- an unused table costs
/// nothing but its own allocation) or "this component only makes sense
/// under constraint X" (`max_players`, checked by `validate`). Nothing here
/// yet changes actual `Node`/`ChildArray` layout -- see this struct's use
/// as the seed of a future storage-quantization pass, not a claim that one
/// already exists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Requirements {
    /// GRAVE's ancestor-lookup table (`select::Rave`'s `get_ref` walk).
    pub grave: bool,
    /// The MAST unigram action-value table (`simulate::Mast`,
    /// `select::ProgressiveHistory`).
    pub global: bool,
    /// Per-child AMAF stats (`node::PlayerStats::amaf`), read by
    /// `select::Amaf`/`select::Rave`.
    pub amaf: bool,
    /// NST's bigram table, on top of `global` (see `simulate::Nst`'s doc
    /// comment on why it needs its own bit).
    pub nst: bool,
    /// LGR's per-player reply table (`simulate::Lgr`), independent of
    /// `global`/`nst` -- see `LGR`'s doc comment.
    pub lgr: bool,
    /// LGRF-2's 2-ply reply table (`simulate::Lgr2`), on top of `lgr` --
    /// see `LGR2`'s doc comment.
    pub lgr2: bool,
    /// This component's own scoring only means something once
    /// `use_mcts_solver` is on -- e.g. `select::UctPn`'s proof/disproof rank
    /// bonus degenerates to a harmless constant with the solver off (see its
    /// doc comment), so this is advisory, not enforced by `validate`.
    pub solver: bool,
    /// Upper bound this component places on `Game::num_players()` -- e.g.
    /// MCTS-Solver's `Proven` representation (`node::Proven`'s doc comment)
    /// is only sound for <= 2 players. `None` means unconstrained.
    pub max_players: Option<usize>,
    /// This component reads `node::PlayerStats::posterior_mean`/
    /// `posterior_variance` (`select::BayesUct1`/`BayesUct2`), which only a
    /// `backprop::BayesGaussian`/`BayesNumeric` strategy populates. Unlike
    /// `solver` above, this *is* enforced by `SearchConfig::validate` --
    /// reading zeroed posterior fields wouldn't degenerate to a harmless
    /// no-op the way an unused `solver` bit does, it would silently run a
    /// select strategy that always sees `(0.0, 0.0)`.
    pub needs_posterior: bool,
}

impl Requirements {
    pub const fn none() -> Self {
        Self {
            grave: false,
            global: false,
            amaf: false,
            nst: false,
            lgr: false,
            lgr2: false,
            solver: false,
            max_players: None,
            needs_posterior: false,
        }
    }

    /// Combines two components' requirements. Associative and commutative,
    /// so a composition of N components can fold this over all of them in
    /// any order -- what makes union the whole "interaction resolution"
    /// mechanism instead of a per-pair table.
    pub const fn union(self, other: Self) -> Self {
        Self {
            grave: self.grave || other.grave,
            global: self.global || other.global,
            amaf: self.amaf || other.amaf,
            nst: self.nst || other.nst,
            lgr: self.lgr || other.lgr,
            lgr2: self.lgr2 || other.lgr2,
            solver: self.solver || other.solver,
            max_players: match (self.max_players, other.max_players) {
                (None, x) | (x, None) => x,
                (Some(a), Some(b)) => Some(if a < b { a } else { b }),
            },
            needs_posterior: self.needs_posterior || other.needs_posterior,
        }
    }

    /// Lifts the existing `BackpropFlags` bitset into `Requirements` --
    /// every `SelectStrategy`/`SimulateStrategy`'s default `requirements()`
    /// is defined in terms of this, so components that only ever needed
    /// `backprop_flags()` (the large majority) get a correct `requirements()`
    /// for free with no code change.
    pub fn from_backprop_flags(flags: BackpropFlags) -> Self {
        Self {
            grave: flags.grave(),
            global: flags.global(),
            amaf: flags.amaf(),
            nst: flags.nst(),
            lgr: flags.lgr(),
            lgr2: flags.lgr2(),
            ..Self::none()
        }
    }

    pub fn backprop_flags(&self) -> BackpropFlags {
        let mut bits = 0;
        if self.grave {
            bits |= GRAVE;
        }
        if self.global {
            bits |= GLOBAL;
        }
        if self.amaf {
            bits |= AMAF;
        }
        if self.nst {
            bits |= NST;
        }
        if self.lgr {
            bits |= LGR;
        }
        if self.lgr2 {
            bits |= LGR2;
        }
        BackpropFlags(bits)
    }

    /// Checks this composition's hard constraints against a game's player
    /// count -- the one interaction that has to actually be rejected, as
    /// opposed to merely unioned into a shared table. Everything else two
    /// components need from each other is either a storage union (harmless
    /// to over-provision) or advisory (`solver`).
    pub fn validate(&self, num_players: usize) -> Result<(), String> {
        if let Some(max) = self.max_players {
            if num_players > max {
                return Err(format!(
                    "composed strategy requires num_players() <= {max}, got {num_players}"
                ));
            }
        }
        Ok(())
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
    /// Explicit graph-search mode. `use_transpositions(true)` remains a
    /// compatibility alias for the old edge-statistics table behavior.
    pub graph_search: GraphSearch,
    pub use_transpositions: bool,

    /// Only meaningful under `GraphSearch::Dag` -- see
    /// `TranspositionKeying`'s doc comment. `PerPly` (the default) matches
    /// every existing `GraphSearch::Dag` behavior unchanged.
    pub transposition_keying: TranspositionKeying,

    /// Residual information-leak correction, only meaningful under
    /// `GraphSearch::Dag(GraphStats::Both)` -- see `McgsCorrection`'s doc
    /// comment. `Disabled` (the default) never checks, so this is a no-op
    /// for every other `graph_search`/`use_transpositions` configuration.
    pub mcgs_correction: McgsCorrection,

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

    /// MCTS-IP/MS-d-Visit-0 (Baier & Winands): a per-action prior computed
    /// at expansion time (`prior::PriorStrategy`), seeded into each freshly-
    /// expanded node's children before any of them is visited. `None` (the
    /// default) keeps every existing search's untouched `QInit`-driven
    /// `unvisited_value` behavior. See `prior`'s module doc comment for why
    /// this is a boxed trait object rather than a third generic parameter on
    /// `Strategy<G>`/`TreeSearch<G, S>`.
    pub prior: Option<Box<dyn prior::PriorStrategyDyn<G>>>,
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
            graph_search: GraphSearch::Tree,
            use_transpositions: false,
            transposition_keying: TranspositionKeying::PerPly,
            mcgs_correction: McgsCorrection::Disabled,
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
            prior: None,
        }
    }
}

impl<G, S> SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    pub(crate) fn graph_stats(&self) -> Option<GraphStats> {
        match self.graph_search {
            GraphSearch::Tree if self.use_transpositions => Some(GraphStats::Edges),
            GraphSearch::Tree => None,
            GraphSearch::Dag(stats) => Some(stats),
        }
    }

    pub(crate) fn uses_transpositions(&self) -> bool {
        self.graph_stats().is_some()
    }

    /// This configuration's resolved `Requirements`: the union of every
    /// component's own `requirements()`. `select`/`simulate` carry the
    /// interesting cases today (`final_action`/`backprop` are `SelectStrategy`/
    /// `BackpropStrategy` too, so they're included for completeness, not
    /// because any current impl needs it).
    pub fn requirements(&self) -> Requirements {
        <S::Select as select::SelectStrategy<G>>::requirements(&self.select)
            .union(<S::Simulate as simulate::SimulateStrategy<G>>::requirements(&self.simulate))
            .union(<S::FinalAction as select::SelectStrategy<G>>::requirements(
                &self.final_action,
            ))
    }

    /// Validates this configuration's resolved `Requirements` against `G`,
    /// e.g. rejecting `select::UctPn` (MCTS-Solver's `max_players: Some(2)`)
    /// paired with a >2-player game, and (see below) `use_mcts_solver` or
    /// `prior` paired with a >2-player game.
    pub fn validate(&self) -> Result<(), String> {
        self.requirements().validate(G::num_players())?;
        // Unlike a component's advisory `requirements().solver` bit (which
        // degenerates to a no-op with the solver off, e.g. `UctPn`'s doc
        // comment), `use_mcts_solver` itself directly gates whether
        // `backprop.rs` derives `node::Proven` values -- a representation
        // only sound for `num_players() <= 2` (`node::Proven`'s doc
        // comment). Checked here, not just via the `debug_assert!` at the
        // derivation site, so a release build rejects this instead of
        // silently deriving nonsense.
        if self.use_mcts_solver && G::num_players() > 2 {
            return Err(format!(
                "use_mcts_solver requires num_players() <= 2, got {}",
                G::num_players()
            ));
        }
        // `prior::PriorStrategy` isn't one of `Select`/`Simulate`/
        // `FinalAction`, so its own `requirements()` doesn't participate in
        // the union above -- checked separately here instead.
        if let Some(prior) = &self.prior {
            prior.requirements().validate(G::num_players())?;
        }
        if self.transposition_keying == TranspositionKeying::StateOnly {
            // `StateOnly` lets the graph contain real cycles (a shared node
            // reachable from itself via a reversible/capturing move
            // sequence), so `select_step`'s descent needs a real depth
            // bound to guarantee termination.
            if self.max_playout_depth == usize::MAX {
                return Err(
                    "TranspositionKeying::StateOnly requires a finite max_playout_depth \
                     -- it gates the descent-depth guard against unbounded cycles in the \
                     merged graph"
                        .to_string(),
                );
            }
        }
        if self.requirements().needs_posterior && !self.backprop.provides_posterior() {
            return Err(
                "select/final_action strategy requires a Bayesian backprop strategy \
                 (BayesGaussian/BayesNumeric) that provides posterior mean/variance estimates"
                    .to_string(),
            );
        }
        if self.prior.is_some() {
            // `prior::PriorStrategy` seeds a not-yet-created child's stats
            // directly into `ChildArray`'s edge-owned rows and relies on
            // `select::random_best_index`'s unvisited branch reading them
            // back via `SelectContext::child_snapshot` without a live `Id` --
            // sound whenever `child_snapshot` ignores its `child_id`
            // argument (every `GraphStats` mode except `Nodes`, which
            // dereferences it directly), and only whenever no active
            // component also needs a genuine `Id` for that same
            // not-yet-created child the way `select::Rave`'s GRAVE
            // ancestor lookup does (keyed by the child's own hash, which a
            // placeholder `Id` can't stand in for).
            if matches!(self.graph_stats(), Some(GraphStats::Nodes)) {
                return Err(
                    "prior strategy is incompatible with GraphStats::Nodes -- child stats \
                     it seeds live in ChildArray (edge-owned), which Nodes mode never reads"
                        .to_string(),
                );
            }
            if self.requirements().grave {
                return Err(
                    "prior strategy is incompatible with GRAVE (select::Rave) -- GRAVE's \
                     ancestor lookup needs a real child Id, which an unvisited prior-seeded \
                     child doesn't have yet"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

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

    pub fn graph_search(mut self, graph_search: GraphSearch) -> Self {
        self.graph_search = graph_search;
        self
    }

    pub fn transposition_keying(mut self, transposition_keying: TranspositionKeying) -> Self {
        self.transposition_keying = transposition_keying;
        self
    }

    pub fn mcgs_correction(mut self, mcgs_correction: McgsCorrection) -> Self {
        self.mcgs_correction = mcgs_correction;
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

    pub fn with_prior(mut self, prior: impl prior::PriorStrategy<G> + 'static) -> Self {
        self.prior = Some(Box::new(prior));
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

#[cfg(test)]
mod requirements_tests {
    use super::*;

    #[test]
    fn union_is_a_bitwise_or_over_flags_and_a_min_over_max_players() {
        let a = Requirements {
            amaf: true,
            max_players: Some(4),
            ..Requirements::none()
        };
        let b = Requirements {
            global: true,
            max_players: Some(2),
            ..Requirements::none()
        };
        let combined = a.union(b);
        assert!(combined.amaf && combined.global);
        assert!(
            !combined.grave && !combined.nst && !combined.lgr && !combined.lgr2 && !combined.solver
        );
        assert_eq!(
            combined.max_players,
            Some(2),
            "the tighter of two player-count bounds wins"
        );
    }

    #[test]
    fn union_with_unconstrained_max_players_keeps_the_other_side() {
        let unconstrained = Requirements::none();
        let bounded = Requirements {
            max_players: Some(2),
            ..Requirements::none()
        };
        assert_eq!(unconstrained.union(bounded).max_players, Some(2));
        assert_eq!(bounded.union(unconstrained).max_players, Some(2));
    }

    #[test]
    fn backprop_flags_round_trip_through_requirements() {
        let flags = BackpropFlags(GRAVE | GLOBAL | AMAF | NST | LGR | LGR2);
        let reqs = Requirements::from_backprop_flags(flags);
        assert!(reqs.grave && reqs.global && reqs.amaf && reqs.nst && reqs.lgr && reqs.lgr2);
        let round_tripped = reqs.backprop_flags();
        assert!(round_tripped.grave() && round_tripped.global());
        assert!(
            round_tripped.amaf()
                && round_tripped.nst()
                && round_tripped.lgr()
                && round_tripped.lgr2()
        );
    }

    #[test]
    fn validate_rejects_exceeding_max_players_and_accepts_within_bound() {
        let reqs = Requirements {
            max_players: Some(2),
            ..Requirements::none()
        };
        assert!(reqs.validate(2).is_ok());
        assert!(reqs.validate(1).is_ok());
        assert!(reqs.validate(3).is_err());
    }
}

#[cfg(test)]
mod search_config_validate_tests {
    use super::*;
    use crate::game::PlayerIndex;

    #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
    struct Player(usize);

    impl PlayerIndex for Player {
        fn to_index(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    struct State;

    impl std::fmt::Display for State {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "State")
        }
    }

    /// A minimal three-player game, existing purely to exercise
    /// `SearchConfig::validate`'s player-count checks -- every ply is a
    /// forced pass and the game never actually terminates in these tests,
    /// since no test here runs a search to completion.
    #[derive(Clone)]
    struct ThreePlayerGame;

    impl Game for ThreePlayerGame {
        type S = State;
        type A = u8;
        type P = Player;

        fn apply(state: Self::S, _action: &Self::A) -> Self::S {
            state
        }

        fn generate_actions(_state: &Self::S, actions: &mut Vec<Self::A>) {
            actions.push(0);
        }

        fn winner(_state: &Self::S) -> Option<Self::P> {
            None
        }

        fn player_to_move(_state: &Self::S) -> Self::P {
            Player(0)
        }

        fn num_players() -> usize {
            3
        }
    }

    #[test]
    fn validate_rejects_use_mcts_solver_for_more_than_two_players() {
        let config =
            SearchConfig::<ThreePlayerGame, strategy::Ucb1>::default().use_mcts_solver(true);
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_use_mcts_solver_off_for_more_than_two_players() {
        let config = SearchConfig::<ThreePlayerGame, strategy::Ucb1>::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_two_player_only_prior_for_more_than_two_players() {
        let config = SearchConfig::<ThreePlayerGame, strategy::Ucb1>::default()
            .with_prior(prior::EvaluatorPrior::<ThreePlayerGame>::new());
        assert!(config.validate().is_err());
    }

    #[test]
    fn evaluated_cutoff_reports_a_two_player_max_players_requirement() {
        use simulate::SimulateStrategy;
        let inner = simulate::EvaluatedCutoff::<
            ThreePlayerGame,
            crate::evaluator::MaterialBlind,
            simulate::Uniform,
        >::new();
        assert_eq!(
            SimulateStrategy::<ThreePlayerGame>::requirements(&inner).max_players,
            Some(2)
        );
    }
}
