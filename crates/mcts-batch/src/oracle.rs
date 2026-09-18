//! The seam between a batched MCTS search and a particular environment.
//!
//! Mirrors AlphaZero.jl's `BatchedMcts.EnvOracle` (`init_fn`/`transition_fn`)
//! and DeepMind's `mctx` `RecurrentFn`: everything the search needs from the
//! environment -- state transitions and value/policy evaluation -- goes
//! through two batched calls instead of one call per leaf. `transition`
//! receives every leaf due for expansion in one simulation round across the
//! *whole* batch of trees, so a single evaluator call can score all of them
//! together -- the actual point of this crate (see `mcts-gpu.md`): the
//! current per-node `crates/mcts` engine can't batch evaluator calls this
//! way because it advances one tree to completion before starting the next.

/// The result of evaluating a fresh batch of states: everything a newly
/// created tree node needs cached, per state.
pub struct StepOutput<Env> {
    /// The internal state after this step, one per input state.
    pub states: Vec<Env>,
    /// Whether each resulting state is terminal.
    pub terminal: Vec<bool>,
    /// Flattened `(count, num_actions)` legality mask, row-major (state,
    /// action) -- `valid_actions[i * num_actions + a]`.
    pub valid_actions: Vec<bool>,
    /// Flattened `(count, num_actions)` policy prior, same layout as
    /// `valid_actions`. Need not be pre-masked or normalized -- the search
    /// does that (`validate_prior`-equivalent) once per node.
    pub policy_prior: Vec<f32>,
    /// Value prior for each state, from the perspective of the player about
    /// to move in that state (matches `mcts::evaluator::Evaluator`'s
    /// "nega" convention) -- in `[-1, 1]`.
    pub value_prior: Vec<f32>,
}

/// The additional per-transition fields `transition` reports on top of
/// [`StepOutput`] -- the reward/perspective bookkeeping a fresh `init` has
/// no analogue for (there is no "previous player" at the root).
pub struct TransitionOutput<Env> {
    pub step: StepOutput<Env>,
    /// Immediate reward for the transition, from the mover's perspective.
    /// Always `0.0` for games (like Othello) with no intermediate reward.
    pub rewards: Vec<f32>,
    /// Whether the mover changed across this transition. Almost always
    /// `true` for two-player games without passes-that-skip-a-turn in a
    /// different sense than Othello's (Othello's own pass still flips the
    /// mover, so this is unconditionally `true` there too).
    pub player_switched: Vec<bool>,
}

/// Batched environment oracle: everything [`crate::search`] needs to grow
/// and score a batch of trees, decoupled from any specific game so the same
/// search code drives Othello today and another game later without
/// changes (`mcts-gpu.md`'s explicit design goal).
pub trait EnvOracle<Env>: Sync {
    /// Number of actions `A` -- the fixed width of every `valid_actions`/
    /// `policy_prior` row this oracle ever returns.
    fn num_actions(&self) -> usize;

    /// Evaluate a fresh batch of root states (one call, whole batch).
    fn init(&self, envs: &[Env]) -> StepOutput<Env>;

    /// Apply one action per state and evaluate the results (one call, the
    /// whole batch of leaves due for expansion this simulation round).
    /// `states[i]`/`actions[i]` are the parent and chosen action for output
    /// row `i`; `actions[i]` is a valid index into that parent's own
    /// `valid_actions` row (guaranteed by the caller).
    fn transition(&self, states: &[Env], actions: &[u16]) -> TransitionOutput<Env>;
}
