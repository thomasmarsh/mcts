//! The Gumbel self-play player for tic-tac-toe -- the search configuration
//! shared by `dump --label gumbel` and the generation gate.

use mcts::algorithms::mcts::gumbel::{gumbel_search, GumbelConfig, GumbelOutcome};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;

use crate::valuenet::NTupleValueNet;
use crate::{HashedPosition, Move, TicTacToe};

/// Completed-Q interior selection (a PUCT stub today) over a linear value
/// head consulted at every leaf (`max_playout_depth == 0`).
pub type GumbelProfile = Mcts<GumbelCompletedQ, EvaluatedCutoff<TicTacToe, NTupleValueNet>>;

/// One generation's player: a persistent `TreeSearch` re-rooted per move by
/// the Gumbel schedule.
pub struct GumbelPlayer {
    search: TreeSearch<TicTacToe, GumbelProfile>,
    cfg: GumbelConfig,
    name: String,
}

impl GumbelPlayer {
    pub fn new(net: NTupleValueNet, cfg: GumbelConfig, seed: u64) -> Self {
        Self::with_playout_depth(net, cfg, seed, 0)
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
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(max_playout_depth)
                .q_init(QInit::Loss)
                .simulate(EvaluatedCutoff::new().evaluator(net))
                .seed(seed),
        );
        Self {
            search,
            cfg,
            name: "gumbel".to_string(),
        }
    }

    /// The full Gumbel outcome -- action plus the visit-distribution policy
    /// target -- for the self-play dump path.
    pub fn choose(&mut self, state: &HashedPosition) -> GumbelOutcome<Move> {
        gumbel_search(&mut self.search, state, &self.cfg)
    }
}

impl Search for GumbelPlayer {
    type G = TicTacToe;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &HashedPosition) -> Move {
        gumbel_search(&mut self.search, state, &self.cfg).action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::mcts::node::QInit;
    use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
    use mcts::algorithms::mcts::{SearchConfig, TreeSearch};

    use crate::valuenet::NT_WEIGHTS;
    use crate::{Piece, Position};

    fn wide_cfg() -> GumbelConfig {
        // Consider every legal move so a sign check never depends on which
        // candidates Gumbel-top-k happened to keep.
        GumbelConfig {
            sims: 48,
            max_considered: 16,
            ..GumbelConfig::default()
        }
    }

    fn player_with(net: NTupleValueNet, seed: u64) -> GumbelPlayer {
        GumbelPlayer::new(net, wide_cfg(), seed)
    }

    /// An n-tuple net whose only signal is: along either main diagonal, the
    /// side to move seeing the *opponent* on the centre cell is worth
    /// `sign * 10` (pre-`tanh`). Centre is index 1 (place 3) in both diagonal
    /// tuples `[0,4,8]` and `[2,4,6]`; digit 2 there means `(feat / 3) % 3 == 2`
    /// for any occupancy of the diagonal's end cells.
    fn centre_poison_net(sign: f32) -> NTupleValueNet {
        let mut w = vec![0.0f32; NT_WEIGHTS];
        // Weight tables: bias at 0, then eight 27-wide line tables. The two
        // main diagonals are lines 6 and 7 (offsets 1 + 6*27 and 1 + 7*27).
        for line in [6usize, 7] {
            let base = 1 + line * 27;
            for feat in 0..27 {
                if (feat / 3) % 3 == 2 {
                    w[base + feat] = sign * 10.0;
                }
            }
        }
        NTupleValueNet::from_weights(w)
    }

    /// A terminal win reached through the forced root edge must credit the
    /// root player with `+1`, so the Gumbel schedule plays it. This exercises
    /// the whole chain -- `descend_from` forced edge, terminal utilities,
    /// completed-Q sigma, final `argmax` -- for sign, with the zero net so
    /// only the terminal result can move the score.
    #[test]
    fn gumbel_takes_an_immediate_win() {
        let mut pos = Position::new();
        pos.set(0, Piece::X);
        pos.set(1, Piece::X);
        pos.set(3, Piece::O);
        pos.set(4, Piece::O);
        pos.turn = Piece::X; // X to move, cell 2 completes the top row
        let state = HashedPosition::from_position(pos);

        for seed in [1u64, 2, 3, 4, 5] {
            let action = player_with(NTupleValueNet::default(), seed).choose_action(&state);
            assert_eq!(action, Move(2), "seed {seed} missed the winning move");
        }
    }

    /// The value head is consulted at the child, from the child mover's
    /// (the opponent's) perspective. A weight that makes "the opponent has
    /// cell 4" score badly *for the opponent* must translate to the root
    /// player wanting cell 4 -- i.e. the nega conversion and the per-player
    /// edge stats keep the leaf value pointed the right way.
    #[test]
    fn gumbel_value_sign_favours_a_move_that_is_bad_for_the_opponent() {
        let state = HashedPosition::new(); // empty board, X to move

        for seed in [1u64, 2, 3] {
            assert_eq!(
                player_with(centre_poison_net(-1.0), seed).choose_action(&state),
                Move(4),
                "seed {seed}: X should grab the cell the net rates as opponent-poison"
            );
            assert_ne!(
                player_with(centre_poison_net(1.0), seed).choose_action(&state),
                Move(4),
                "seed {seed}: X should avoid handing the opponent a net-favoured cell"
            );
        }
    }

    /// The lower-level seam on its own: a single forced descent into a won
    /// position leaves the root child's expected score positive for the root
    /// player.
    #[test]
    fn forced_descent_credits_the_root_player() {
        fn root_child_score(pos: Position, forced: Move) -> f64 {
            let net = NTupleValueNet::default();
            let mut search: TreeSearch<TicTacToe, GumbelProfile> =
                TreeSearch::default().config(
                    SearchConfig::default()
                        .expand_threshold(1)
                        .max_playout_depth(0)
                        .q_init(QInit::Loss)
                        .simulate(EvaluatedCutoff::new().evaluator(net))
                        .seed(1),
                );
            let state = HashedPosition::from_position(pos);
            let outcome = gumbel_search(&mut search, &state, &wide_cfg());
            outcome
                .visit_distribution
                .iter()
                .find(|(m, _)| *m == forced)
                .map(|(m, _)| {
                    let root = search.index.get(search.root_id);
                    let idx = (0..root.children().len())
                        .find(|&i| root.children().action(i) == *m)
                        .unwrap();
                    root.children().expected_score(idx, 0)
                })
                .expect("forced move received visits")
        }

        // X at 0,1; O at 3,4; X to move -> X plays 2 and wins.
        let mut win = Position::new();
        win.set(0, Piece::X);
        win.set(1, Piece::X);
        win.set(3, Piece::O);
        win.set(4, Piece::O);
        win.turn = Piece::X;
        assert!(root_child_score(win, Move(2)) > 0.5, "winning edge should be ~+1 for X");
    }
}
