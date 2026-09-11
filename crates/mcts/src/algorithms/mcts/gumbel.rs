//! The Gumbel AlphaZero root schedule (Danihelka et al., ICLR 2022).
//!
//! Policy improvement is guaranteed with a small fixed simulation budget by
//! combining three pieces:
//!
//! 1. **Gumbel-top-k** over the prior logits picks `m` candidate actions at
//!    the root: draw `g_a ~ Gumbel(0, 1)` per legal action and keep the `m`
//!    largest `g_a + logit_a`.
//! 2. **Sequential Halving** spends the budget `n` over those candidates,
//!    running a batch of forced-root descents (`TreeSearch::descend_from`)
//!    for each surviving candidate per phase and dropping the weaker half
//!    between phases, ranked by `g_a + logit_a + sigma(q_a)`.
//! 3. The **final action** is the argmax of that same quantity over the last
//!    surviving set, with `q_a` the completed-Q estimate and
//!    `sigma(q) = (c_visit + max_b N_b) * c_scale * q`.
//!
//! The improved-policy training target is the completed-Q policy distribution.
//!
//! This module is orchestration only: it drives the existing
//! `select`/`simulate`/`backprop` primitives plus `descend_from`, and owns
//! no search state of its own. It re-roots the tree (`TreeSearch::reset`)
//! once per move.

use rand::Rng;

use crate::algorithms::mcts::config::{PolicyProfile, SearchConfig};
use crate::algorithms::mcts::index::Id;
use crate::algorithms::mcts::search::shared::expand;
use crate::algorithms::mcts::search::{SearchContext, TreeSearch};
use crate::algorithms::mcts::simulate::SimulatePolicy;
use crate::game::{Game, PlayerIndex};

/// How the Sequential-Halving ranking `sigma` term scales with visits.
///
/// `sigma_a = scale(a) * c_scale * transform(Q_a)`. `NodeFloor` is the
/// Mctx-verbatim rule: `scale(a)` is the per-node `(c_visit + max_visit)`
/// floor for every action regardless of how many of its own visits landed.
/// The other modes make `scale(a)` depend on action `a`'s realized visit
/// count so a barely-visited action's noisy completed-Q cannot dominate the
/// Gumbel exploration budget at a small simulation budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SigmaMode {
    /// Mctx-verbatim: `scale(a) = c_visit + max_visit` for every action.
    #[default]
    NodeFloor,
    /// `scale(a) = min(realized_visits_a, c_visit + max_visit)`.
    Smooth,
    /// `scale(a) = 0` while `realized_visits_a < k`, else `c_visit + max_visit`.
    HardGate(u32),
    /// `scale(a) = realized_visits_a`, dropping the `max_visit` term entirely.
    RealizedOnly,
}

/// The per-action multiplier on `c_scale * transform(Q_a)` in the
/// Sequential-Halving ranking key. Interior selection can reuse this once
/// visit-aware interior sigma is in scope.
pub fn sigma_visit_scale(
    mode: SigmaMode,
    realized_visits: u32,
    max_visits: u32,
    c_visit: f64,
) -> f64 {
    let floor = c_visit + max_visits as f64;
    match mode {
        SigmaMode::NodeFloor => floor,
        SigmaMode::Smooth => (realized_visits as f64).min(floor),
        SigmaMode::HardGate(k) => {
            if realized_visits < k {
                0.0
            } else {
                floor
            }
        }
        SigmaMode::RealizedOnly => realized_visits as f64,
    }
}

/// How the final root move is chosen once the Sequential-Halving schedule
/// finishes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RootMoveSelection {
    /// Full Gumbel: return the surviving candidate with the largest
    /// `g_a + logit_a + sigma(completed_q_a)` ranking key.
    #[default]
    CompletedQ,
    /// Return the most-visited root action (`argmax_a N(a)`, ties broken
    /// toward the lowest action index), ignoring the completed-Q ranking key.
    /// This also forces plain visit-count / PUCT interior selection with no
    /// completed-Q override. The Gumbel top-`m` sampling and the
    /// Sequential-Halving visit allocation are unchanged, and the recorded
    /// improved-policy training target is unaffected.
    VisitCount,
}

/// The most-visited action, ties broken toward the lowest index. A hand-built
/// root reduces to its child visit vector for this decision.
pub(crate) fn most_visited_action(visits: &[u32]) -> usize {
    (0..visits.len())
        .max_by(|&a, &b| visits[a].cmp(&visits[b]).then(b.cmp(&a)))
        .expect("root has at least one child")
}

/// Knobs for one Gumbel move. All of these belong in `config.toml` for a
/// real run (`feedback_config_as_data`); the defaults match the paper's
/// small-budget regime.
#[derive(Clone, Copy, Debug)]
pub struct GumbelConfig {
    /// Total simulation budget `n` spent per move across all candidates.
    pub sims: u32,
    /// Maximum number of root candidates `m` considered before Sequential
    /// Halving begins (clamped to the number of legal moves).
    pub max_considered: usize,
    /// `sigma`'s visit offset -- how many visits a candidate needs before
    /// its completed-Q starts to outweigh its Gumbel draw.
    pub c_visit: f64,
    /// `sigma`'s value scale.
    pub c_scale: f64,
    /// Normalize completed Q values to their observed min-max range before
    /// applying the visit scale. Equal values remain unchanged.
    pub rescale_q: bool,
    /// Fill unvisited actions with the mixed root value before ranking and
    /// policy improvement. `false` retains the pre-completion diagnostic.
    pub use_completed_q: bool,
    /// Interior (non-root) selection rule. `true` (Full Gumbel) runs the
    /// deterministic completed-Q visit-matching rule at every interior node.
    /// `false` selects "root-only" Gumbel: the root Sequential-Halving
    /// schedule is unchanged, but interior nodes fall back to PUCT with no
    /// completed-Q override.
    pub interior_completed_q: bool,
    /// PUCT exploration constant used at interior nodes only when
    /// `interior_completed_q` is `false`.
    pub interior_c_puct: f64,
    /// How the root Sequential-Halving `sigma` term scales with visits.
    /// `NodeFloor` is the Mctx-verbatim default; the visit-aware modes stop a
    /// one-visit noisy completed-Q from swamping exploration at small budgets.
    pub sigma_mode: SigmaMode,
    /// How the final root move is chosen. `CompletedQ` (default) is Full
    /// Gumbel; `VisitCount` returns `argmax_a N(a)` and forces PUCT interior
    /// selection.
    pub root_move_selection: RootMoveSelection,
    /// Recording-only override for the improved-policy target's `c_scale`.
    /// `None` (default) reuses the played-move `c_scale`, so the recorded
    /// target is byte-identical to current behaviour. A larger value sharpens
    /// the recorded target toward the completed-Q argmax without touching the
    /// played move, the Sequential-Halving survivor ranking, or the returned
    /// action -- those stay Mctx-verbatim.
    pub target_c_scale: Option<f64>,
    /// Recording-only override for the improved-policy target's `rescale_q`.
    /// `None` (default) reuses the played-move `rescale_q`. Setting it to
    /// `false` stops the completed-Q spread being compressed into `[0, 1]`
    /// before the visit scale, which at a small simulation budget is the
    /// dominant reason the recorded target collapses back onto the raw prior.
    pub target_rescale_q: Option<bool>,
}

impl GumbelConfig {
    /// The config [`improved_policy`] sees when building the *recorded*
    /// training target: `target_c_scale` / `target_rescale_q` applied over the
    /// played-move values. Every other field, including the played-move
    /// `c_scale` / `rescale_q` used by [`candidate_score`] and the returned
    /// action, is left untouched.
    fn recorded_target_config(&self) -> GumbelConfig {
        GumbelConfig {
            c_scale: self.target_c_scale.unwrap_or(self.c_scale),
            rescale_q: self.target_rescale_q.unwrap_or(self.rescale_q),
            ..*self
        }
    }
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            sims: 32,
            max_considered: 8,
            c_visit: 50.0,
            c_scale: 0.1,
            rescale_q: true,
            use_completed_q: true,
            interior_completed_q: true,
            interior_c_puct: 1.25,
            sigma_mode: SigmaMode::NodeFloor,
            root_move_selection: RootMoveSelection::CompletedQ,
            target_c_scale: None,
            target_rescale_q: None,
        }
    }
}

/// One Gumbel move: the action to play, plus the improved-policy target.
#[derive(Clone, Debug)]
pub struct GumbelOutcome<A> {
    pub action: A,
    /// `(action, probability)` over the Sequential Halving visit
    /// distribution -- only children that received a visit appear.
    pub visit_distribution: Vec<(A, f32)>,
    /// `(action, probability)` from the completed-Q policy improvement rule.
    pub improved_policy: Vec<(A, f32)>,
    /// `(action, completed_q)` in the same order as `improved_policy`, in the
    /// root mover's perspective. This is the raw completed-Q vector the
    /// improved policy is built from -- roughly on a `[-1, 1]` scale (leaf
    /// `tanh` value or terminal `+/-1`), *before* any `rescale_q` `[0, 1]`
    /// normalization. Exposed so a self-play move sampler can apply a value
    /// filter over the children.
    pub completed_q: Vec<(A, f32)>,
}

/// One Sequential-Halving phase: the top `num_considered` candidates (ranked
/// by the survivor key carried over from the previous phase) each receive
/// `visits[rank]` additional forced root visits this phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShPhase {
    /// How many of the ranked candidates are visited this phase.
    pub num_considered: usize,
    /// Forced visits for the candidate at each rank `0..num_considered`. The
    /// final phase can be ragged when the budget runs out part-way through a
    /// round, so higher-ranked candidates may get one more visit than the rest.
    pub visits: Vec<u32>,
}

/// The DeepMind Mctx Sequential-Halving visit schedule, expressed as a list of
/// phases. This is a transcription of `get_sequence_of_considered_visits` in
/// `mctx/_src/seq_halving.py`: `log2max = ceil(log2 m)` rounds of budget, each
/// round giving `max(1, n / (log2max * num_considered))` visits to every one of
/// the current `num_considered` top candidates, then halving
/// `num_considered <- max(2, num_considered / 2)`, stopping once the budget `n`
/// is spent. The total visits issued equal `n` exactly (Mctx truncates its
/// per-simulation sequence to `n`); the schedule never overspends and never
/// leaves budget unused.
pub(crate) fn mctx_sh_schedule(m: usize, n: u32) -> Vec<ShPhase> {
    if n == 0 {
        return Vec::new();
    }
    if m <= 1 {
        return vec![ShPhase {
            num_considered: 1,
            visits: vec![n],
        }];
    }
    let log2max = u32::BITS - (m as u32 - 1).leading_zeros();
    let mut phases = Vec::new();
    let mut num_considered = m;
    let mut spent = 0u32;
    while spent < n {
        let per_round = (n / (log2max * num_considered as u32)).max(1);
        let mut visits = vec![0u32; num_considered];
        'rounds: for _ in 0..per_round {
            for slot in visits.iter_mut() {
                if spent == n {
                    break 'rounds;
                }
                *slot += 1;
                spent += 1;
            }
        }
        phases.push(ShPhase {
            num_considered,
            visits,
        });
        num_considered = (num_considered / 2).max(2);
    }
    phases
}

/// A standard Gumbel(0, 1) draw, `-ln(-ln u)` for `u` uniform on `(0, 1]`.
fn sample_gumbel(rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.gen_range(f64::MIN_POSITIVE..=1.0);
    -(-(u.ln())).ln()
}

fn root_child_visits<G, S>(search: &TreeSearch<G, S>, root_id: Id) -> Vec<u32>
where
    G: Game,
    S: PolicyProfile<G>,
    G::S: std::fmt::Display,
{
    let children = search.index.get(root_id).children();
    (0..children.len())
        .map(|i| children.num_visits(i))
        .collect()
}

/// Complete unvisited action values with the prior-weighted mixed value from
/// Mctx. All values are from the root player's perspective.
pub fn completed_q(root_value: f64, logits: &[f64], visits: &[u32], q_values: &[f64]) -> Vec<f64> {
    assert_eq!(logits.len(), visits.len());
    assert_eq!(visits.len(), q_values.len());
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let priors: Vec<f64> = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let visited_prior: f64 = priors
        .iter()
        .zip(visits)
        .filter_map(|(&prior, &visits)| (visits > 0).then_some(prior))
        .sum();
    let weighted_q = priors
        .iter()
        .zip(visits)
        .zip(q_values)
        .filter_map(|((&prior, &visits), &q)| (visits > 0).then_some(prior * q))
        .sum::<f64>()
        / visited_prior.max(f64::MIN_POSITIVE);
    let total_visits: u32 = visits.iter().sum();
    let mixed_value = (root_value + total_visits as f64 * weighted_q) / (1 + total_visits) as f64;
    visits
        .iter()
        .zip(q_values)
        .map(|(&visits, &q)| if visits == 0 { mixed_value } else { q })
        .collect()
}

/// Optionally normalize completed Q values to `[0, 1]`. An equal-Q vector
/// is left alone so it cannot perturb the prior distribution.
pub fn transform_completed_q(values: &[f64], rescale_q: bool) -> Vec<f64> {
    if !rescale_q || values.is_empty() {
        return values.to_vec();
    }
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = hi - lo;
    if range <= f64::EPSILON {
        return values.to_vec();
    }
    values.iter().map(|&value| (value - lo) / range).collect()
}

/// The completed-Q policy target. `logits` and `visits` follow `actions`'
/// order; illegal actions are therefore not represented at all.
pub fn improved_policy(
    logits: &[f64],
    visits: &[u32],
    completed_q: &[f64],
    cfg: &GumbelConfig,
) -> Vec<f32> {
    assert_eq!(logits.len(), visits.len());
    assert_eq!(logits.len(), completed_q.len());
    if logits.is_empty() {
        return Vec::new();
    }
    let max_visit = visits.iter().copied().max().unwrap_or(0) as f64;
    let scale = (cfg.c_visit + max_visit) * cfg.c_scale;
    let transformed = transform_completed_q(completed_q, cfg.rescale_q);
    let scores: Vec<f64> = logits
        .iter()
        .zip(transformed)
        .map(|(&logit, q)| logit + scale * q)
        .collect();
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| (w / total) as f32).collect()
}

fn root_completed_q<G, S>(
    search: &TreeSearch<G, S>,
    root_id: Id,
    player: usize,
    root_value: f64,
    logits: &[f64],
    cfg: &GumbelConfig,
) -> Vec<f64>
where
    G: Game,
    S: PolicyProfile<G>,
    G::S: std::fmt::Display,
{
    let children = search.index.get(root_id).children();
    let visits: Vec<_> = (0..children.len())
        .map(|i| children.num_visits(i))
        .collect();
    let q_values: Vec<_> = (0..children.len())
        .map(|i| children.expected_score(i, player))
        .collect();
    if cfg.use_completed_q {
        completed_q(root_value, logits, &visits, &q_values)
    } else {
        q_values
            .into_iter()
            .zip(visits)
            .map(|(q, visits)| if visits == 0 { 0.0 } else { q })
            .collect()
    }
}

/// `g_a + logit_a + sigma(q_a)` -- the Sequential Halving ranking key and the
/// final-selection score, using a completed root Q value.
fn candidate_score(
    idx: usize,
    gumbel: &[f64],
    logits: &[f64],
    completed_q: &[f64],
    cfg: &GumbelConfig,
    visits: &[u32],
    max_visits: u32,
) -> f64 {
    let transformed = transform_completed_q(completed_q, cfg.rescale_q);
    let scale = sigma_visit_scale(cfg.sigma_mode, visits[idx], max_visits, cfg.c_visit);
    let sigma = scale * cfg.c_scale * transformed[idx];
    gumbel[idx] + logits[idx] + sigma
}

fn run_forced_iteration<G, S>(
    search: &mut TreeSearch<G, S>,
    root_id: Id,
    state: &G::S,
    forced: &G::A,
) where
    G: Game,
    S: PolicyProfile<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    search.reset_iter();
    let mut ctx = SearchContext::new(root_id, state.clone());
    if let Some(utilities) = search.descend_from(forced, &mut ctx) {
        search.backprop_correction(&utilities);
        return;
    }
    let trial = search.simulate(&ctx.state);
    search.trial = Some(trial);
    search.backprop();
}

/// Run one Gumbel move from `state`, re-rooting `search` onto it first.
pub fn gumbel_search<G, S>(
    search: &mut TreeSearch<G, S>,
    state: &G::S,
    cfg: &GumbelConfig,
) -> GumbelOutcome<G::A>
where
    G: Game,
    S: PolicyProfile<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    gumbel_search_with_root_value(search, state, cfg, 0.0)
}

/// As [`gumbel_search`], with the root evaluation supplied explicitly in the
/// root mover's perspective.  The evaluator is intentionally outside the
/// search tree so root completion never depends on a warm-up simulation.
pub fn gumbel_search_with_root_value<G, S>(
    search: &mut TreeSearch<G, S>,
    state: &G::S,
    cfg: &GumbelConfig,
    root_value: f64,
) -> GumbelOutcome<G::A>
where
    G: Game,
    S: PolicyProfile<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    let player = G::player_to_move(state).to_index();
    let root_id = search.reset(player, G::zobrist_hash(state));

    // Expand the root against the literal caller state so its child list is
    // the real, directly-playable move set (`expand`'s own doc comment).
    let amaf = search.config.requirements().amaf;
    let canon = search.config.canonicalizes();
    let use_solver = search.config.use_mcts_solver;
    expand::<G>(
        &search.index,
        root_id,
        state,
        use_solver,
        amaf,
        canon,
        false,
        search.config.prior.as_deref_mut(),
        search.config.policy_logits.as_deref_mut(),
        |expanded_state| search.config.simulate.raw_evaluator_value(expanded_state),
    );

    // With a non-zero `expand_threshold`, `select_step` bails out at any
    // node whose visit count hasn't reached it yet -- including the root, on
    // the first descents of the move. Spend `expand_threshold` un-forced
    // warm-up simulations on the root first so every forced descent below
    // actually takes its root edge.
    for _ in 0..search.config.expand_threshold {
        search.reset_iter();
        let mut ctx = SearchContext::new(root_id, state.clone());
        if search.select(&mut ctx).is_none() {
            let trial = search.simulate(&ctx.state);
            search.trial = Some(trial);
            search.backprop();
        }
    }

    let (k, actions) = {
        let children = search.index.get(root_id).children();
        let k = children.len();
        let actions: Vec<G::A> = (0..k).map(|i| children.action(i)).collect();
        (k, actions)
    };
    debug_assert!(k > 0);

    let logits = search
        .index
        .get(root_id)
        .children()
        .policy_logits()
        .to_vec();
    let gumbel: Vec<f64> = (0..k)
        .map(|_| sample_gumbel(&mut search.config.rng))
        .collect();

    let m = cfg.max_considered.clamp(1, k);
    let mut considered: Vec<usize> = (0..k).collect();
    considered.sort_by(|&a, &b| {
        (gumbel[b] + logits[b])
            .partial_cmp(&(gumbel[a] + logits[a]))
            .unwrap()
    });
    considered.truncate(m);

    let schedule = mctx_sh_schedule(m, cfg.sims);
    for (phase_idx, sh_phase) in schedule.iter().enumerate() {
        debug_assert_eq!(considered.len(), sh_phase.num_considered);
        for (rank, &a) in considered.iter().enumerate() {
            for _ in 0..sh_phase.visits[rank] {
                run_forced_iteration(search, root_id, state, &actions[a]);
            }
        }
        let Some(next_phase) = schedule.get(phase_idx + 1) else {
            break;
        };
        let visits = root_child_visits(search, root_id);
        let max_visits = visits.iter().copied().max().unwrap_or(0);
        let completed_q = root_completed_q(search, root_id, player, root_value, &logits, cfg);
        considered.sort_by(|&a, &b| {
            let sb = candidate_score(b, &gumbel, &logits, &completed_q, cfg, &visits, max_visits);
            let sa = candidate_score(a, &gumbel, &logits, &completed_q, cfg, &visits, max_visits);
            sb.partial_cmp(&sa).unwrap()
        });
        considered.truncate(next_phase.num_considered);
    }

    let visits = root_child_visits(search, root_id);
    let max_visits = visits.iter().copied().max().unwrap_or(0);
    let completed_q = root_completed_q(search, root_id, player, root_value, &logits, cfg);
    let best = match cfg.root_move_selection {
        RootMoveSelection::VisitCount => most_visited_action(&visits),
        RootMoveSelection::CompletedQ => *considered
            .iter()
            .max_by(|&&a, &&b| {
                let sa =
                    candidate_score(a, &gumbel, &logits, &completed_q, cfg, &visits, max_visits);
                let sb =
                    candidate_score(b, &gumbel, &logits, &completed_q, cfg, &visits, max_visits);
                sa.partial_cmp(&sb).unwrap()
            })
            .expect("Gumbel keeps at least one candidate"),
    };

    let visit_distribution = {
        let children = search.index.get(root_id).children();
        let total: u32 = (0..k).map(|i| children.num_visits(i)).sum();
        if total == 0 {
            vec![(actions[best].clone(), 1.0)]
        } else {
            (0..k)
                .filter_map(|i| {
                    let v = children.num_visits(i);
                    (v > 0).then(|| (actions[i].clone(), v as f32 / total as f32))
                })
                .collect()
        }
    };
    // The recorded training target may be sharpened independently of the
    // played move: only this call sees the `target_*` overrides.
    let improved = improved_policy(&logits, &visits, &completed_q, &cfg.recorded_target_config());

    let action = actions[best].clone();
    let child_completed_q: Vec<(G::A, f32)> = actions
        .iter()
        .cloned()
        .zip(completed_q.iter().map(|&q| q as f32))
        .collect();
    GumbelOutcome {
        action,
        visit_distribution,
        improved_policy: actions.into_iter().zip(improved).collect(),
        completed_q: child_completed_q,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        candidate_score, completed_q, improved_policy, mctx_sh_schedule, most_visited_action,
        sigma_visit_scale, transform_completed_q, GumbelConfig, RootMoveSelection, SigmaMode,
    };

    /// Shannon entropy of a probability vector, in nats.
    fn entropy(p: &[f32]) -> f64 {
        p.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| -(x as f64) * (x as f64).ln())
            .sum()
    }

    /// `target_c_scale` / `target_rescale_q` sharpen the *recorded* improved
    /// policy (lower entropy, more mass on the completed-Q argmax) while
    /// `candidate_score` -- the Sequential-Halving survivor key and the
    /// returned-action score -- is byte-identical, because it never reads the
    /// `target_*` fields.
    #[test]
    fn target_overrides_sharpen_the_recorded_target_only() {
        // A near-balanced root: small completed-Q spread, so the Mctx-verbatim
        // target barely moves off the prior.
        let logits = [0.20, 0.10, -0.05, 0.15];
        let visits = [10u32, 8, 6, 8];
        let q = [0.62, 0.55, 0.40, 0.58];
        let completed = completed_q(0.55, &logits, &visits, &q);

        let base = GumbelConfig::default();
        let sharp = GumbelConfig {
            target_c_scale: Some(1.0),
            target_rescale_q: Some(false),
            ..GumbelConfig::default()
        };

        // Played-move machinery is untouched: the target fields never reach
        // `candidate_score`, and `recorded_target_config` leaves every other
        // field alone.
        let max_visits = *visits.iter().max().unwrap();
        let gumbel = [0.0; 4];
        for idx in 0..4 {
            let a = candidate_score(idx, &gumbel, &logits, &completed, &base, &visits, max_visits);
            let b = candidate_score(idx, &gumbel, &logits, &completed, &sharp, &visits, max_visits);
            assert_eq!(a, b, "candidate_score changed at {idx}");
        }
        assert_eq!(base.recorded_target_config().c_scale, 0.1);
        assert_eq!(sharp.recorded_target_config().c_scale, 1.0);
        assert!(!sharp.recorded_target_config().rescale_q);

        // The recorded target itself: sharper under the overrides.
        let baseline_target = improved_policy(&logits, &visits, &completed, &base);
        let baseline_recorded =
            improved_policy(&logits, &visits, &completed, &base.recorded_target_config());
        assert_eq!(
            baseline_target, baseline_recorded,
            "default config: recorded target unchanged"
        );
        let sharp_recorded =
            improved_policy(&logits, &visits, &completed, &sharp.recorded_target_config());
        assert!(
            entropy(&sharp_recorded) < entropy(&baseline_recorded) - 0.05,
            "sharpened target entropy {} vs baseline {}",
            entropy(&sharp_recorded),
            entropy(&baseline_recorded),
        );
        let argmax = |p: &[f32]| {
            (0..p.len())
                .max_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap())
                .unwrap()
        };
        assert_eq!(argmax(&sharp_recorded), 0, "sharpened toward completed-Q argmax");
        assert!(sharp_recorded[0] > baseline_recorded[0] + 0.1);
    }

    #[test]
    fn most_visited_action_breaks_ties_toward_the_lowest_index() {
        assert_eq!(most_visited_action(&[1, 2, 3]), 2);
        assert_eq!(most_visited_action(&[3, 5, 5, 2]), 1);
        assert_eq!(most_visited_action(&[0, 0, 0]), 0);
        assert_eq!(most_visited_action(&[7]), 0);
    }

    #[test]
    fn visit_count_play_ignores_the_completed_q_ranking_key() {
        // Action 2 has the most visits; action 0 would win the completed-Q
        // `candidate_score` ranking on its extreme Q. Visit-count play returns
        // the most-visited action regardless of that key.
        let gumbel = [0.0, 0.0, 0.0];
        let logits = [0.0, 0.0, 0.0];
        let visits = [1u32, 2, 5];
        let completed = [1.0, 0.0, 0.0];
        let max_visits = 5;
        let cfg = GumbelConfig::default();
        let by_score = (0..3)
            .max_by(|&a, &b| {
                let sa = candidate_score(a, &gumbel, &logits, &completed, &cfg, &visits, max_visits);
                let sb = candidate_score(b, &gumbel, &logits, &completed, &cfg, &visits, max_visits);
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap();
        assert_eq!(by_score, 0);
        assert_eq!(most_visited_action(&visits), 2);
    }

    #[test]
    fn improved_policy_target_is_identical_under_both_root_move_modes() {
        let logits = [2.0, -1.0, 0.5];
        let visits = [4, 1, 0];
        let q = [0.5, -0.5, 0.1];
        let completed_q_mode = GumbelConfig {
            root_move_selection: RootMoveSelection::CompletedQ,
            ..GumbelConfig::default()
        };
        let visit_count_mode = GumbelConfig {
            root_move_selection: RootMoveSelection::VisitCount,
            ..GumbelConfig::default()
        };
        let completed = completed_q(0.25, &logits, &visits, &q);
        assert_eq!(
            improved_policy(&logits, &visits, &completed, &completed_q_mode),
            improved_policy(&logits, &visits, &completed, &visit_count_mode),
        );
    }

    /// Independent transcription of DeepMind Mctx
    /// `get_sequence_of_considered_visits` (`mctx/_src/seq_halving.py`),
    /// returning the per-simulation sequence of considered-visit counts.
    fn mctx_considered_visits(m: usize, n: usize) -> Vec<usize> {
        if m <= 1 {
            return (0..n).collect();
        }
        let log2max = (u32::BITS - (m as u32 - 1).leading_zeros()) as usize;
        let mut sequence: Vec<usize> = Vec::new();
        let mut visits = vec![0usize; m];
        let mut num_considered = m;
        while sequence.len() < n {
            let num_extra_visits = std::cmp::max(1, n / (log2max * num_considered));
            for _ in 0..num_extra_visits {
                sequence.extend_from_slice(&visits[..num_considered]);
                for v in visits[..num_considered].iter_mut() {
                    *v += 1;
                }
            }
            num_considered = std::cmp::max(2, num_considered / 2);
        }
        sequence.truncate(n);
        sequence
    }

    /// Per-candidate total visit allocation (rank-sorted) and the survivor-count
    /// sequence that the Mctx schedule produces, derived independently from
    /// [`mctx_considered_visits`] by replaying its round structure.
    fn mctx_summary(m: usize, n: u32) -> (Vec<u32>, Vec<usize>) {
        let n = n as usize;
        // Confirm our phase-derived helper agrees with the flat sequence port.
        let flat = mctx_considered_visits(m, n);
        assert_eq!(flat.len(), n);
        if m <= 1 {
            return (vec![n as u32], if n == 0 { vec![] } else { vec![1] });
        }
        let log2max = (u32::BITS - (m as u32 - 1).leading_zeros()) as usize;
        let mut alloc = vec![0u32; m];
        let mut survivors = Vec::new();
        let mut num_considered = m;
        let mut appended = 0usize;
        let mut assigned = 0usize;
        while appended < n {
            let num_extra_visits = std::cmp::max(1, n / (log2max * num_considered));
            survivors.push(num_considered);
            for _ in 0..num_extra_visits {
                for slot in alloc.iter_mut().take(num_considered) {
                    appended += 1;
                    if assigned < n {
                        *slot += 1;
                        assigned += 1;
                    }
                }
            }
            num_considered = std::cmp::max(2, num_considered / 2);
        }
        (alloc, survivors)
    }

    fn our_summary(m: usize, n: u32) -> (Vec<u32>, Vec<usize>) {
        let schedule = mctx_sh_schedule(m, n);
        let mut alloc = vec![0u32; m];
        for phase in &schedule {
            for (rank, &v) in phase.visits.iter().enumerate() {
                alloc[rank] += v;
            }
        }
        let survivors = schedule.iter().map(|p| p.num_considered).collect();
        (alloc, survivors)
    }

    /// Pins our Sequential-Halving schedule to DeepMind Mctx's
    /// `get_sequence_of_considered_visits`: the rank-sorted per-candidate visit
    /// allocation and the survivor-count sequence must match exactly, and the
    /// schedule must spend the whole budget.
    #[test]
    fn sh_schedule_matches_mctx_considered_visits() {
        for &(n, m) in &[(32u32, 7usize), (32, 8), (16, 16), (8, 4)] {
            let (ours_alloc, ours_surv) = our_summary(m, n);
            let (mctx_alloc, mctx_surv) = mctx_summary(m, n);
            assert_eq!(ours_alloc, mctx_alloc, "allocation mismatch n={n} m={m}");
            assert_eq!(ours_surv, mctx_surv, "survivor sequence mismatch n={n} m={m}");
            assert_eq!(
                ours_alloc.iter().sum::<u32>(),
                n,
                "schedule must spend the whole budget n={n} m={m}"
            );
        }
        // Explicit reference vectors for the schedules exercised by Connect Four self-play.
        assert_eq!(our_summary(7, 32).0, vec![12, 12, 4, 1, 1, 1, 1]);
        assert_eq!(our_summary(7, 32).1, vec![7, 3, 2, 2]);
        assert_eq!(our_summary(8, 32).0, vec![11, 11, 3, 3, 1, 1, 1, 1]);
        assert_eq!(our_summary(4, 8).0, vec![3, 3, 1, 1]);
        assert_eq!(our_summary(16, 16).0, vec![1; 16]);
        assert_eq!(our_summary(16, 16).1, vec![16]);
    }

    #[test]
    fn completed_q_uses_the_explicit_root_value_for_unvisited_actions() {
        let completed = completed_q(0.4, &[0.0, 0.0, 0.0], &[10, 0, 2], &[0.8, 99.0, -0.2]);
        assert_eq!(completed[0], 0.8);
        assert!((completed[1] - 4.0 / 13.0).abs() < 1e-12);
        assert_eq!(completed[2], -0.2);
    }

    #[test]
    fn improved_policy_is_finite_and_preserves_equal_q_priors() {
        let cfg = GumbelConfig::default();
        let p = improved_policy(&[2.0, -1.0, 0.5], &[0, 0, 0], &[0.25; 3], &cfg);
        let prior = improved_policy(&[2.0, -1.0, 0.5], &[0, 0, 0], &[0.0; 3], &cfg);
        assert_eq!(p, prior);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0));
    }

    #[test]
    fn visit_offset_controls_how_hard_completed_q_overrides_the_prior() {
        // `sigma = (c_visit + max_visit) * c_scale * q`. At a small simulation
        // budget `max_visit` is tiny, so `c_visit` alone decides whether one
        // shallow completed-Q sample can swamp the prior logits. A large
        // `c_visit` collapses the improved policy onto the top-Q action after a
        // single visit; a near-zero `c_visit` leaves the prior largely intact.
        let logits = [0.0, 0.0, 0.0];
        let visits = [1, 1, 1];
        let q = [0.9, 0.1, 0.1];
        let heavy = improved_policy(
            &logits,
            &visits,
            &q,
            &GumbelConfig {
                c_visit: 50.0,
                c_scale: 0.1,
                rescale_q: false,
                ..GumbelConfig::default()
            },
        );
        let light = improved_policy(
            &logits,
            &visits,
            &q,
            &GumbelConfig {
                c_visit: 0.0,
                c_scale: 0.1,
                rescale_q: false,
                ..GumbelConfig::default()
            },
        );
        assert!(heavy[0] > 0.95, "heavy c_visit collapses onto top Q: {heavy:?}");
        assert!(light[0] < 0.45, "light c_visit keeps the policy broad: {light:?}");
    }

    #[test]
    fn improved_policy_has_additive_logit_invariance() {
        let cfg = GumbelConfig::default();
        let a = improved_policy(&[0.0, 1.0], &[4, 1], &[0.5, -0.5], &cfg);
        let b = improved_policy(&[17.0, 18.0], &[4, 1], &[0.5, -0.5], &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn min_max_rescaling_is_optional_and_equal_q_is_neutral() {
        assert_eq!(
            transform_completed_q(&[-2.0, 0.0, 2.0], true),
            vec![0.0, 0.5, 1.0]
        );
        assert_eq!(transform_completed_q(&[0.25; 3], true), vec![0.25; 3]);
        assert_eq!(
            transform_completed_q(&[-2.0, 0.0, 2.0], false),
            vec![-2.0, 0.0, 2.0]
        );
    }

    /// Cross-checks the Sequential Halving survivor key against DeepMind's
    /// Mctx `seq_halving.score_considered`, which ranks by
    /// `gumbel + (logits - max(logits)) + qtransform(completed_q)` where
    /// `qtransform` is `(maxvisit_init + max_visit) * value_scale * rescaled`
    /// with the reference constants `maxvisit_init = 50`, `value_scale = 0.1`.
    /// Our `candidate_score` omits the additive `-max(logits)` term, which is
    /// a constant shift across actions and therefore cannot change the ranking.
    #[test]
    fn root_survivor_ranking_matches_mctx_score_considered() {
        let cfg = GumbelConfig::default();
        let visits = [5, 2, 0];
        let q = [0.6, -0.2, 99.0];
        let logits = [0.2, -0.1, 0.3];
        let gumbel = [0.0, 2.0, 1.0];
        let completed = completed_q(0.25, &logits, &visits, &q);
        let max_visits = *visits.iter().max().unwrap();

        // Reference scores hand-computed from the Mctx equations:
        //   transformed completed Q = [1.0, 0.0, 0.5729497]
        //   qtransform = (50 + 5) * 0.1 * transformed = [5.5, 0.0, 3.1512234]
        //   score = gumbel + (logits - 0.3) + qtransform
        let reference = [
            0.0 + (0.2 - 0.3) + 5.5,
            2.0 + (-0.1 - 0.3) + 0.0,
            1.0 + (0.3 - 0.3) + 3.1512233621661086,
        ];
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (idx, expected) in reference.iter().enumerate() {
            let ours = candidate_score(idx, &gumbel, &logits, &completed, &cfg, &visits, max_visits);
            assert!((ours - max_logit - expected).abs() < 1e-9);
        }
        // Mctx survivor order: action 0, then action 2, then action 1.
        assert!(reference[0] > reference[2] && reference[2] > reference[1]);
    }

    /// `NodeFloor` is the Mctx-verbatim default: `sigma_visit_scale` returns
    /// the `c_visit + max_visit` per-node floor for every action regardless of
    /// its own realized visits, so `candidate_score` is byte-unchanged.
    #[test]
    fn node_floor_sigma_scale_is_the_mctx_per_node_floor() {
        for realized in [0u32, 1, 3, 7] {
            assert_eq!(
                sigma_visit_scale(SigmaMode::NodeFloor, realized, 7, 50.0),
                57.0
            );
        }
    }

    /// Under the visit-aware modes a 0- or 1-visit action's `sigma` multiplier
    /// is ~0, so its (possibly wild) completed-Q cannot outrank a well-visited
    /// action on the `sigma` term alone. Same logits and Gumbel draw; only the
    /// completed-Q and visits differ.
    #[test]
    fn barely_visited_noisy_q_cannot_win_on_sigma_under_visit_aware_modes() {
        let gumbel = [0.0, 0.0];
        let logits = [0.0, 0.0];
        // action 0: 6 visits, modest Q; action 1: 1 visit, extreme Q.
        let visits = [6u32, 1];
        let completed = [0.30, 1.0];
        let max_visits = 6;
        for mode in [
            SigmaMode::Smooth,
            SigmaMode::HardGate(2),
            SigmaMode::HardGate(3),
            SigmaMode::HardGate(4),
            SigmaMode::RealizedOnly,
        ] {
            let cfg = GumbelConfig {
                c_visit: 0.0,
                c_scale: 0.1,
                rescale_q: false,
                sigma_mode: mode,
                ..GumbelConfig::default()
            };
            let s0 = candidate_score(0, &gumbel, &logits, &completed, &cfg, &visits, max_visits);
            let s1 = candidate_score(1, &gumbel, &logits, &completed, &cfg, &visits, max_visits);
            assert!(s0 > s1, "{mode:?}: barely-visited noisy Q won ({s0} vs {s1})");
        }
    }

    /// A heavily-visited action (`realized ~ max_visit`) reproduces the old
    /// `NodeFloor` sigma within tolerance once `c_visit = 0`.
    #[test]
    fn heavily_visited_action_reproduces_node_floor_sigma() {
        let realized = 8u32;
        let max_visits = 8u32;
        let floor = sigma_visit_scale(SigmaMode::NodeFloor, realized, max_visits, 0.0);
        for mode in [SigmaMode::Smooth, SigmaMode::HardGate(4), SigmaMode::RealizedOnly] {
            let scale = sigma_visit_scale(mode, realized, max_visits, 0.0);
            assert!((scale - floor).abs() < 1e-9, "{mode:?}: {scale} vs {floor}");
        }
        // The hard gate is exactly the floor above its threshold, zero below.
        assert_eq!(sigma_visit_scale(SigmaMode::HardGate(3), 2, 8, 50.0), 0.0);
        assert_eq!(sigma_visit_scale(SigmaMode::HardGate(3), 3, 8, 50.0), 58.0);
    }

    /// Additive-invariance of the ranking key under a constant logit shift
    /// holds under every sigma mode (the sigma term does not touch logits).
    #[test]
    fn candidate_score_ranking_is_logit_shift_invariant_under_every_mode() {
        let gumbel = [0.3, -0.7, 1.1];
        let logits = [0.2, -0.1, 0.4];
        let shifted = [17.2, 16.9, 17.4];
        let visits = [4u32, 1, 0];
        let completed = completed_q(0.1, &logits, &visits, &[0.5, -0.3, 0.0]);
        let max_visits = 4;
        for mode in [
            SigmaMode::NodeFloor,
            SigmaMode::Smooth,
            SigmaMode::HardGate(2),
            SigmaMode::RealizedOnly,
        ] {
            let cfg = GumbelConfig {
                sigma_mode: mode,
                ..GumbelConfig::default()
            };
            let base: Vec<f64> = (0..3)
                .map(|i| candidate_score(i, &gumbel, &logits, &completed, &cfg, &visits, max_visits))
                .collect();
            let bumped: Vec<f64> = (0..3)
                .map(|i| candidate_score(i, &gumbel, &shifted, &completed, &cfg, &visits, max_visits))
                .collect();
            for i in 0..3 {
                assert!((base[i] + 17.0 - bumped[i]).abs() < 1e-9, "{mode:?}");
            }
        }
    }

    #[test]
    fn improved_policy_stays_finite_nonneg_and_normalized() {
        let cfg = GumbelConfig::default();
        let p = improved_policy(&[2.0, -1.0, 0.5], &[4, 1, 0], &[0.5, -0.5, 0.1], &cfg);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0));
    }

    #[test]
    fn completed_q_reference_vector_freezes_root_inputs_and_output_policy() {
        let cfg = GumbelConfig::default();
        let visits = [5, 2, 0];
        let q = [0.6, -0.2, 99.0];
        let logits = [0.2, -0.1, 0.3];
        let completed = completed_q(0.25, &logits, &visits, &q);
        assert_eq!(completed, vec![0.6, -0.2, 0.2583597617681613]);
        assert!((transform_completed_q(&completed, true)[2] - 0.5729497022102017).abs() < 1e-12);
        let policy = improved_policy(&logits, &visits, &completed, &cfg);
        let expected = [0.9020746, 0.0027310802, 0.09519435];
        for (actual, expected) in policy.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
