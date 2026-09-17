//! The Gumbel self-play player for Othello -- the search configuration
//! shared by `dump --label gumbel` and the generation gate. Ported from
//! `games/connect4/src/selfplay.rs::GumbelPlayer`; the completed-Q root
//! schedule and interior selection (`GumbelCompletedQ`) are game-generic and
//! reused verbatim across games, so the only real porting work is plugging
//! in Othello's n-tuple value head and policy sidecar. Unlike
//! Connect Four's `NTupleValueNet` (a `Default`-able fixed-shape struct),
//! Othello's tuple geometry is data (`model.toml`), so [`NTupleModelEval`]
//! wraps an explicitly-constructed [`NTupleModel`] instance rather than a
//! process-wide env-resolved singleton (`crate::ntuple::NTupleEval`) --
//! a self-play loop loads a fresh `weights.bin` every generation and needs
//! an owned instance per load, not process-global state.
//!
//! [`CnnGumbelPlayer`] is the CNN-backed alternative, mirroring Connect
//! Four's `CnnGumbelPlayer` (`games/connect4/src/selfplay.rs`): a joint
//! value+policy container (`crate::convnet::CnnValueNet`, which implements
//! both `Evaluator<Othello>` and `PolicyLogits<Othello>`) plugged into the
//! same `GumbelCompletedQ`/`EvaluatedCutoff` machinery as a single type
//! parameter, rather than `GumbelProfile` itself becoming generic over the
//! value/policy model class -- `GumbelProfile`'s two type parameters
//! (`NTupleModelEval` as both the `Evaluator` and, via `with_policy_logits`,
//! the value-prior seam) are concrete, not bounded by a shared trait
//! `GumbelPlayer` and `CnnGumbelPlayer` could both implement, so a separate
//! struct is the smaller change and matches the precedent already landed for
//! Connect Four.

use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig, GumbelOutcome};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::evaluator::{Evaluator, EVAL_MAGNITUDE_LIMIT};

use crate::ntuple::NTupleModelEval;
use crate::policy::NTuplePolicyNet;
use crate::{Move, Othello, State};

/// Completed-Q interior selection over the n-tuple value head consulted at
/// every leaf (`max_playout_depth == 0`).
pub type GumbelProfile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Othello, NTupleModelEval>>;

/// One generation's player: a persistent `TreeSearch` re-rooted per move by
/// the Gumbel schedule.
pub struct GumbelPlayer {
    search: TreeSearch<Othello, GumbelProfile>,
    value_net: NTupleModelEval,
    cfg: GumbelConfig,
    name: String,
}

impl GumbelPlayer {
    pub fn with_policy(
        net: NTupleModelEval,
        policy: NTuplePolicyNet,
        cfg: GumbelConfig,
        seed: u64,
    ) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
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
    pub fn choose(&mut self, state: &State) -> GumbelOutcome<Move> {
        gumbel_search_with_root_value(
            &mut self.search,
            state,
            &self.cfg,
            self.value_net.value(state) as f64,
        )
    }
}

impl Search for GumbelPlayer {
    type G = Othello;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State) -> Move {
        self.choose(state).action
    }
}

/// Gumbel player backed by a joint value-and-policy container implementing
/// both `Evaluator<Othello>` and `PolicyLogits<Othello>` -- the Othello
/// analogue of Connect Four's `CnnGumbelPlayer`. Generic over the evaluator
/// so the same search wiring works with `crate::convnet::CnnValueNet` (CPU)
/// or `crate::convnet::mlx::MlxCnnValueNet` (GPU, behind the `mlx` feature)
/// without duplicating this struct per backend. The shared container
/// supplies both contracts without seeding policy logits as child values,
/// exactly as `GumbelPlayer` above does for the n-tuple heads.
pub struct CnnGumbelPlayer<E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static> {
    search: TreeSearch<Othello, Mcts<GumbelCompletedQ, EvaluatedCutoff<Othello, E>>>,
    net: E,
    cfg: GumbelConfig,
    name: String,
}

impl<E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static> CnnGumbelPlayer<E> {
    pub fn new(net: E, cfg: GumbelConfig, seed: u64) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .select(GumbelCompletedQ::with_config(cfg))
                .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
                .with_policy_logits(net.clone())
                .seed(seed),
        );
        Self {
            search,
            net,
            cfg,
            name: "gumbel-cnn".to_string(),
        }
    }

    /// The full Gumbel outcome, including its completed-Q policy target.
    pub fn choose(&mut self, state: &State) -> GumbelOutcome<Move> {
        let root_value = self.net.evaluate(state) as f64 / EVAL_MAGNITUDE_LIMIT as f64;
        gumbel_search_with_root_value(&mut self.search, state, &self.cfg, root_value)
    }
}

impl<E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Default + 'static> Search for CnnGumbelPlayer<E> {
    type G = Othello;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State) -> Move {
        self.choose(state).action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convnet::CnnValueNet;
    use crate::ntuple::ModelGeometry;
    use crate::{Player, BB};
    use mcts::game::Game;

    fn tiny_geom() -> ModelGeometry {
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/tests/tiny.toml"),
        )
        .unwrap();
        ModelGeometry::parse(&bytes)
    }

    fn zero_net() -> NTupleModelEval {
        NTupleModelEval::default()
    }

    fn zero_policy() -> NTuplePolicyNet {
        NTuplePolicyNet::zeros(tiny_geom())
    }

    fn wide_cfg() -> GumbelConfig {
        // Consider every legal move so a sign check never depends on which
        // candidates Gumbel-top-k happened to keep.
        GumbelConfig {
            sims: 48,
            max_considered: 64,
            c_scale: 1.0,
            ..GumbelConfig::default()
        }
    }

    fn player_with(cfg: GumbelConfig, seed: u64) -> GumbelPlayer {
        GumbelPlayer::with_policy(zero_net(), zero_policy(), cfg, seed)
    }

    fn state(black: u64, white: u64, turn: Player) -> State {
        State {
            black: BB::from_bits(black),
            white: BB::from_bits(white),
            turn,
            last_pass: false,
            hashes: [0u64; 8],
        }
    }

    /// A near-full board where the only legal move is a single flanking
    /// capture that ends the game with Black far ahead: Black holds every
    /// square except one White disc at 61 (f8) and one empty square at 60
    /// (e8); playing 60 sandwiches 61 against Black's own disc at 62 (g8) in
    /// the same row, flipping it and filling the board. Square 63 (h8) is
    /// also empty but has no opponent-adjacent capture (its only neighbor,
    /// 62, is already Black), so it never becomes a legal move.
    fn near_full_black_win() -> State {
        let mut black = 0u64;
        let mut white = 0u64;
        for i in 0..64u32 {
            if i == 61 {
                white |= 1 << i;
            } else if i != 60 && i != 63 {
                black |= 1 << i;
            }
        }
        state(black, white, Player::Black)
    }

    /// A forced descent into the only legal, terminal, board-filling move
    /// must be played -- with the zero net, only the terminal result can
    /// move the score, so this pins the sign convention end to end.
    #[test]
    fn gumbel_takes_the_only_real_move_and_it_is_terminal_and_a_win() {
        let s = near_full_black_win();
        let mut actions = Vec::new();
        Othello::generate_actions(&s, &mut actions);
        assert_eq!(
            actions,
            vec![Move(60)],
            "fixture must have exactly one legal move"
        );

        for seed in [1u64, 2, 3] {
            let action = player_with(wide_cfg(), seed).choose_action(&s);
            assert_eq!(action, Move(60));
        }
    }

    /// The Sequential-Halving schedule must not panic or mis-divide on a
    /// forced-pass root -- a board-square branching factor of 1, distinct
    /// from Connect Four's always-at-least-2 legal columns except at the
    /// very end.
    #[test]
    fn gumbel_search_handles_a_forced_pass_root_without_panicking() {
        // Black at a1 only, White at h8 only -- Black to move has no legal
        // action and gets a single PASS.
        let s = state(1 << 0, 1 << 63, Player::Black);
        let mut actions = Vec::new();
        Othello::generate_actions(&s, &mut actions);
        assert_eq!(actions, vec![Move::PASS]);

        for seed in [1u64, 2, 3] {
            let action = player_with(wide_cfg(), seed).choose_action(&s);
            assert_eq!(action, Move::PASS);
        }
    }

    /// Policy sidecar logits shift which Gumbel-top-k candidates are drawn
    /// at a fixed branching root, over many seeds -- the Othello analogue of
    /// Connect Four's `policy_sidecar_shifts_fixed_state_gumbel_candidates`.
    ///
    /// The opening position's 4 legal moves are themselves exactly one D4
    /// orbit, so biasing a single canonical-frame column there boosts all 4
    /// equally (the sidecar's D4 averaging spreads one column's weight over
    /// its whole geometric orbit -- see `policy.rs`'s own equivariance
    /// test). Using White's reply after Black plays d3 instead breaks that
    /// symmetry: the position is no longer D4-invariant, so its legal moves
    /// are not a single orbit, and biasing one of them measurably shifts the
    /// distribution among the actually-legal actions.
    #[test]
    fn policy_sidecar_shifts_fixed_state_gumbel_candidates() {
        let cfg = GumbelConfig {
            sims: 1,
            max_considered: 1,
            ..GumbelConfig::default()
        };
        let s = Othello::apply(State::default(), &Move(19));
        let mut actions = Vec::new();
        Othello::generate_actions(&s, &mut actions);
        assert!(actions.len() > 1, "fixture needs a real branching choice");
        let target = actions[0];
        assert_ne!(target, Move::PASS);

        let geom = tiny_geom();
        let n = geom.n_weights() * 64;
        let mut biased = vec![0.0f32; n];
        for row in biased.chunks_mut(64) {
            row[target.0 as usize] = 20.0;
        }

        let mut uniform_count = 0;
        let mut biased_count = 0;
        for seed in 0..200u64 {
            if GumbelPlayer::with_policy(zero_net(), zero_policy(), cfg, seed)
                .choose(&s)
                .action
                == target
            {
                uniform_count += 1;
            }
            if GumbelPlayer::with_policy(
                zero_net(),
                NTuplePolicyNet::from_weights(tiny_geom(), biased.clone()),
                cfg,
                seed,
            )
            .choose(&s)
            .action
                == target
            {
                biased_count += 1;
            }
        }
        assert!(
            biased_count > uniform_count + 50,
            "uniform={uniform_count}, biased={biased_count}"
        );
    }

    /// `CnnGumbelPlayer` wiring smoke tests: the same two sign-audit fixtures
    /// `GumbelPlayer` uses above, re-run through the CNN seam so a wiring bug
    /// (wrong `Evaluator`/`PolicyLogits` plumbing, a panic on the CNN's own
    /// forward pass under real search) would show up here rather than only
    /// in a slow graded run. `CnnValueNet`'s own value/policy correctness
    /// (D4 equivariance, cross-language fixtures) is already covered in
    /// `convnet.rs`; these tests are about the search seam, not the network.
    fn cnn_player_with(cfg: GumbelConfig, seed: u64) -> CnnGumbelPlayer<CnnValueNet> {
        CnnGumbelPlayer::new(CnnValueNet::default(), cfg, seed)
    }

    #[test]
    fn cnn_gumbel_takes_the_only_real_move_and_it_is_terminal_and_a_win() {
        let s = near_full_black_win();
        for seed in [1u64, 2, 3] {
            let action = cnn_player_with(wide_cfg(), seed).choose_action(&s);
            assert_eq!(action, Move(60));
        }
    }

    #[test]
    fn cnn_gumbel_search_handles_a_forced_pass_root_without_panicking() {
        let s = state(1 << 0, 1 << 63, Player::Black);
        for seed in [1u64, 2, 3] {
            let action = cnn_player_with(wide_cfg(), seed).choose_action(&s);
            assert_eq!(action, Move::PASS);
        }
    }
}
