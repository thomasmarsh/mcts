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
//! The improved-policy training target this writes back is the Sequential
//! Halving visit distribution over the root's children.
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
use crate::game::{Game, PlayerIndex};

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
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            sims: 32,
            max_considered: 8,
            c_visit: 50.0,
            c_scale: 0.1,
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
}

/// Number of Sequential Halving phases for `m` candidates: `ceil(log2 m)`,
/// and `0` when there is nothing to halve.
pub(crate) fn num_phases(m: usize) -> usize {
    if m <= 1 {
        0
    } else {
        (m - 1).ilog2() as usize + 1
    }
}

/// A standard Gumbel(0, 1) draw, `-ln(-ln u)` for `u` uniform on `(0, 1]`.
fn sample_gumbel(rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.gen_range(f64::MIN_POSITIVE..=1.0);
    -(-(u.ln())).ln()
}

fn root_child_max_visits<G, S>(search: &TreeSearch<G, S>, root_id: Id) -> u32
where
    G: Game,
    S: PolicyProfile<G>,
    G::S: std::fmt::Display,
{
    let children = search.index.get(root_id).children();
    (0..children.len())
        .map(|i| children.num_visits(i))
        .max()
        .unwrap_or(0)
}

/// Complete unvisited action values with the visit-weighted mix of the root
/// evaluation and the observed child values.  All values are from the root
/// player's perspective.
fn completed_q(root_value: f64, visits: &[u32], q_values: &[f64]) -> Vec<f64> {
    assert_eq!(visits.len(), q_values.len());
    let (weighted_q, total_visits) = visits
        .iter()
        .zip(q_values)
        .fold((0.0, 0u32), |(sum, count), (&visits, &q)| {
            (sum + visits as f64 * q, count + visits)
        });
    let mixed_value = (root_value + weighted_q) / (1 + total_visits) as f64;
    visits
        .iter()
        .zip(q_values)
        .map(|(&visits, &q)| if visits == 0 { mixed_value } else { q })
        .collect()
}

fn root_completed_q<G, S>(
    search: &TreeSearch<G, S>,
    root_id: Id,
    player: usize,
    root_value: f64,
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
    completed_q(root_value, &visits, &q_values)
}

/// `g_a + logit_a + sigma(q_a)` -- the Sequential Halving ranking key and the
/// final-selection score, using a completed root Q value.
fn candidate_score(
    idx: usize,
    gumbel: &[f64],
    logits: &[f64],
    completed_q: &[f64],
    cfg: &GumbelConfig,
    max_visits: u32,
) -> f64 {
    let sigma = (cfg.c_visit + max_visits as f64) * cfg.c_scale * completed_q[idx];
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

    let logits: Vec<f64> = {
        let raw = search
            .config
            .prior
            .as_deref_mut()
            .map(|p| p.evaluate_children(state, &actions))
            .unwrap_or_default();
        if raw.len() == k {
            raw
        } else {
            vec![0.0; k]
        }
    };
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

    let phases = num_phases(m);
    let mut budget_left = cfg.sims;
    for phase in 0..phases {
        let n_actions = considered.len() as u32;
        let last_phase = phase == phases - 1;
        let per_action = if last_phase {
            (budget_left / n_actions).max(1)
        } else {
            (cfg.sims / (phases as u32 * n_actions)).max(1)
        };
        for &a in &considered {
            for _ in 0..per_action {
                if budget_left == 0 {
                    break;
                }
                run_forced_iteration(search, root_id, state, &actions[a]);
                budget_left -= 1;
            }
        }
        if last_phase {
            break;
        }
        let max_visits = root_child_max_visits(search, root_id);
        let completed_q = root_completed_q(search, root_id, player, root_value);
        considered.sort_by(|&a, &b| {
            let sb = candidate_score(b, &gumbel, &logits, &completed_q, cfg, max_visits);
            let sa = candidate_score(a, &gumbel, &logits, &completed_q, cfg, max_visits);
            sb.partial_cmp(&sa).unwrap()
        });
        considered.truncate(considered.len().div_ceil(2).max(1));
    }

    let max_visits = root_child_max_visits(search, root_id);
    let completed_q = root_completed_q(search, root_id, player, root_value);
    let best = *considered
        .iter()
        .max_by(|&&a, &&b| {
            let sa = candidate_score(a, &gumbel, &logits, &completed_q, cfg, max_visits);
            let sb = candidate_score(b, &gumbel, &logits, &completed_q, cfg, max_visits);
            sa.partial_cmp(&sb).unwrap()
        })
        .expect("Gumbel keeps at least one candidate");

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

    GumbelOutcome {
        action: actions[best].clone(),
        visit_distribution,
    }
}

#[cfg(test)]
mod tests {
    use super::{completed_q, num_phases};

    #[test]
    fn phase_count_is_ceil_log2() {
        assert_eq!(num_phases(0), 0);
        assert_eq!(num_phases(1), 0);
        assert_eq!(num_phases(2), 1);
        assert_eq!(num_phases(3), 2);
        assert_eq!(num_phases(4), 2);
        assert_eq!(num_phases(5), 3);
        assert_eq!(num_phases(8), 3);
        assert_eq!(num_phases(16), 4);
    }

    #[test]
    fn completed_q_uses_the_explicit_root_value_for_unvisited_actions() {
        let completed = completed_q(0.4, &[10, 0, 2], &[0.8, 99.0, -0.2]);
        assert_eq!(completed[0], 0.8);
        assert!((completed[1] - 8.0 / 13.0).abs() < 1e-12);
        assert_eq!(completed[2], -0.2);
    }
}
