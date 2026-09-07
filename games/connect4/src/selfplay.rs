//! The Gumbel self-play player for Connect Four -- the search configuration
//! shared by `dump --label gumbel` and the generation gate. The tic-tac-toe
//! counterpart is `games/ttt/src/selfplay.rs`.

use mcts::algorithms::mcts::gumbel::{
    gumbel_search, gumbel_search_with_root_value, GumbelConfig, GumbelOutcome,
};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;

use crate::policynet::NTuplePolicyNet;
use crate::valuenet::NTupleValueNet;
use crate::{Move, Standard, State};

/// Completed-Q interior selection (a PUCT stub today) over the n-tuple value
/// head consulted at every leaf (`max_playout_depth == 0`).
pub type GumbelProfile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Standard, NTupleValueNet>>;

/// One generation's player: a persistent `TreeSearch` re-rooted per move by
/// the Gumbel schedule.
pub struct GumbelPlayer {
    search: TreeSearch<Standard, GumbelProfile>,
    value_net: NTupleValueNet,
    cfg: GumbelConfig,
    name: String,
}

impl GumbelPlayer {
    pub fn new(net: NTupleValueNet, cfg: GumbelConfig, seed: u64) -> Self {
        Self::with_policy_and_playout_depth(net, NTuplePolicyNet::default(), cfg, seed, 0)
    }

    pub fn with_policy(
        net: NTupleValueNet,
        policy: NTuplePolicyNet,
        cfg: GumbelConfig,
        seed: u64,
    ) -> Self {
        Self::with_policy_and_playout_depth(net, policy, cfg, seed, 0)
    }

    /// As [`GumbelPlayer::new`], but with an explicit playout depth. `0` is
    /// the AlphaZero-style leaf evaluation the self-play loop uses (the value
    /// head is the leaf value); a large depth instead rolls out to a natural
    /// terminal and backs up the true result, making the value head moot --
    /// the baseline for "does the net help at all".
    pub fn with_playout_depth(
        net: NTupleValueNet,
        cfg: GumbelConfig,
        seed: u64,
        max_playout_depth: usize,
    ) -> Self {
        Self::with_policy_and_playout_depth(
            net,
            NTuplePolicyNet::default(),
            cfg,
            seed,
            max_playout_depth,
        )
    }

    pub fn with_policy_and_playout_depth(
        net: NTupleValueNet,
        policy: NTuplePolicyNet,
        cfg: GumbelConfig,
        seed: u64,
        max_playout_depth: usize,
    ) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(max_playout_depth)
                .q_init(QInit::Loss)
                .select(GumbelCompletedQ::with_config(cfg))
                .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                .with_policy_logits(policy)
                .seed(seed),
        );
        Self {
            search,
            value_net: net,
            cfg,
            name: "gumbel".to_string(),
        }
    }

    /// The full Gumbel outcome, including its completed-Q policy target.
    pub fn choose(&mut self, state: &State<6, 7>) -> GumbelOutcome<Move> {
        gumbel_search_with_root_value(
            &mut self.search,
            state,
            &self.cfg,
            self.value_net.value(state) as f64,
        )
    }
}

impl Search for GumbelPlayer {
    type G = Standard;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State<6, 7>) -> Move {
        self.choose(state).action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::mcts::node::Node;
    use mcts::algorithms::mcts::policy::PolicyLogits;
    use mcts::algorithms::mcts::search::shared::expand;
    use mcts::game::{Game, PlayerIndex};

    use crate::valuenet::NT_WEIGHTS;
    use crate::Player;

    fn wide_cfg() -> GumbelConfig {
        // Consider every legal column so a sign check never depends on which
        // candidates Gumbel-top-k happened to keep.
        GumbelConfig {
            sims: 48,
            max_considered: 7,
            c_scale: 1.0,
            ..GumbelConfig::default()
        }
    }

    fn player_with(net: NTupleValueNet, seed: u64) -> GumbelPlayer {
        GumbelPlayer::new(net, wide_cfg(), seed)
    }

    fn column_bias(col: usize, amount: f32) -> NTuplePolicyNet {
        let mut weights = vec![0.0; crate::policynet::POLICY_WEIGHTS];
        weights[col] = amount;
        NTuplePolicyNet::from_weights(weights)
    }

    #[test]
    fn policy_sidecar_shifts_fixed_state_gumbel_candidates() {
        let cfg = GumbelConfig {
            sims: 1,
            max_considered: 1,
            ..GumbelConfig::default()
        };
        let state = State::default();
        let mut uniform_col3 = 0;
        let mut biased_col3 = 0;
        for seed in 0..200 {
            if GumbelPlayer::with_policy(
                NTupleValueNet::default(),
                NTuplePolicyNet::default(),
                cfg,
                seed,
            )
            .choose(&state)
            .action
                == Move(3)
            {
                uniform_col3 += 1;
            }
            if GumbelPlayer::with_policy(NTupleValueNet::default(), column_bias(3, 20.0), cfg, seed)
                .choose(&state)
                .action
                == Move(3)
            {
                biased_col3 += 1;
            }
        }
        assert!(
            biased_col3 > uniform_col3 + 100,
            "uniform={uniform_col3}, biased={biased_col3}"
        );
    }

    #[test]
    fn policy_logits_do_not_seed_child_scores_or_visits() {
        let mut search: TreeSearch<Standard, GumbelProfile> = TreeSearch::default().config(
            SearchConfig::default()
                .with_policy_logits(column_bias(3, 20.0))
                .seed(3),
        );
        let state = State::default();
        let root = search.reset(0, Standard::zobrist_hash(&state));
        expand::<Standard>(
            &search.index,
            root,
            &state,
            false,
            Default::default(),
            false,
            false,
            None,
            None,
            |_| None,
        );
        let children = search.index.get(root).children();
        for i in 0..children.len() {
            assert_eq!(children.num_visits(i), 0);
            assert_eq!(children.expected_score(i, 0), 0.0);
        }
    }

    #[test]
    fn canonical_expansion_aligns_cached_value_logits_and_actions() {
        let state = Standard::apply(State::default(), &Move(6));
        let (canonical, _) = Standard::canonical_representation(mcts::game::Real(state));
        let canonical = canonical.into_inner();
        assert_ne!(state, canonical, "fixture must require reflection");

        let net = corner_poison_net(-1.0);
        let mut policy = column_bias(0, 3.0);
        let index = mcts::algorithms::mcts::search::TreeIndex::new();
        let node = index.insert(Node::new(
            Standard::player_to_move(&state).to_index(),
            Standard::zobrist_hash(&state),
        ));
        expand::<Standard>(
            &index,
            node,
            &state,
            false,
            Default::default(),
            true,
            false,
            None,
            Some(&mut policy),
            |expanded_state| Some(net.value(expanded_state) as f64),
        );

        let mut expected_actions = Vec::new();
        Standard::generate_actions(&canonical, &mut expected_actions);
        let expected_logits = policy.logits(&canonical, &expected_actions);
        let children = index.get(node).children();
        assert_eq!(
            (0..children.len())
                .map(|i| children.action(i))
                .collect::<Vec<_>>(),
            expected_actions
        );
        assert_eq!(children.policy_logits(), expected_logits);
        assert_eq!(
            children.raw_evaluator_value(),
            Some(net.value(&canonical) as f64)
        );
    }

    /// Black holds the bottom row's cols 0..3 and it is Black to move -- a
    /// drop into column 3 completes the four. Reached by a real move
    /// sequence (Black on 0,1,2; White stacking column 6).
    fn one_move_from_win() -> State<6, 7> {
        let mut state = State::<6, 7>::default();
        // Black on cols 0,1,2 of the bottom row; White stacks column 6 (only
        // three high, no win) so the turn returns to Black.
        for col in [0u8, 6, 1, 6, 2, 6] {
            state = Standard::apply(state, &Move(col));
        }
        assert!(matches!(state.turn(), Player::Black));
        state
    }

    /// An n-tuple net whose only signal is: the mover seeing the *opponent*
    /// on the bottom-left cell (cell 0) is worth `sign * 10` (pre-`tanh`).
    /// Cell 0 is the least-significant digit of horizontal window 0
    /// (`[0, 1, 2, 3]`), so "opponent at 0, rest empty" is feature index 2.
    fn corner_poison_net(sign: f32) -> NTupleValueNet {
        let mut w = vec![0.0f32; NT_WEIGHTS];
        // Weight tables: bias at 0, then window 0's 81-wide table at offset 1.
        w[1 + 2] = sign * 10.0;
        NTupleValueNet::from_weights(w)
    }

    /// A terminal win reached through the forced root edge must credit the
    /// root player with `+1`, so the Gumbel schedule plays it -- with the
    /// zero net, only the terminal result can move the score.
    #[test]
    fn gumbel_takes_an_immediate_win() {
        let state = one_move_from_win();
        for seed in [1u64, 2, 3, 4, 5] {
            let action = player_with(NTupleValueNet::default(), seed).choose_action(&state);
            assert_eq!(action, Move(3), "seed {seed} missed the winning drop");
        }
    }

    /// The value head is consulted at the child, from the child mover's (the
    /// opponent's) perspective. A weight that makes "the opponent has cell 0"
    /// score badly *for the opponent* must translate to the root player
    /// wanting to play into column 0 -- i.e. the nega conversion and the
    /// per-player edge stats keep the leaf value pointed the right way.
    #[test]
    fn gumbel_value_sign_favours_a_move_that_is_bad_for_the_opponent() {
        let state = State::<6, 7>::default(); // empty board, Black to move

        for seed in [1u64, 2, 3] {
            assert_eq!(
                player_with(corner_poison_net(-1.0), seed).choose_action(&state),
                Move(0),
                "seed {seed}: Black should drop into the column the net rates as opponent-poison"
            );
            assert_ne!(
                player_with(corner_poison_net(1.0), seed).choose_action(&state),
                Move(0),
                "seed {seed}: Black should avoid handing the opponent a net-favoured cell"
            );
        }
    }

    /// The lower-level seam on its own: a single forced descent into a won
    /// position leaves the root child's expected score positive for the root
    /// player.
    #[test]
    fn forced_descent_credits_the_root_player() {
        let net = NTupleValueNet::default();
        let mut search: TreeSearch<Standard, GumbelProfile> = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .simulate(EvaluatedCutoff::new().evaluator(net))
                .seed(1),
        );
        let state = one_move_from_win();
        let outcome = gumbel_search(&mut search, &state, &wide_cfg());

        let forced = Move(3);
        assert!(
            outcome.visit_distribution.iter().any(|(m, _)| *m == forced),
            "winning drop received visits"
        );
        let root = search.index.get(search.root_id);
        let idx = (0..root.children().len())
            .find(|&i| root.children().action(i) == forced)
            .unwrap();
        assert!(
            root.children().expected_score(idx, 0) > 0.5,
            "winning edge should be ~+1 for Black"
        );
    }
}
