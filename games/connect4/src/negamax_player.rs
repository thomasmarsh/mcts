//! A fixed-depth negamax player usable as a [`mcts::util::battle_royale`]
//! opponent, for placing trained Connect Four nets on a negamax-depth ladder.
//!
//! Connect Four is solved, so "gen K at S sims holds parity with negamax
//! depth D" is a meaningful strength unit. This player delegates every move to
//! a bounded-depth [`Negamax`] search over the strict-turn [`Connect4Negamax`]
//! newtype with the signal-free [`MaterialBlind`] evaluator, so a shallow rung
//! is a purely *tactical* opponent -- exactly what the ladder calibrates.

use crate::reference_diagnostic::Connect4Negamax;
use crate::{Move, Standard, State};
use mcts::algorithms::negamax::{MaterialBlind, Negamax, NegamaxOptions};
use mcts::algorithms::Search;

/// Transposition-table size for the ladder solver, as a power of two. Matches
/// `reference_diagnostic`'s reference solver.
const TABLE_BITS: u32 = 20;

/// Plays the fixed-depth `MaterialBlind` bounded-negamax best move each turn.
///
/// [`Negamax::bounded_negamax`] returns `(best_action, score)` with the score
/// -- and the action's ranking -- taken from the perspective of the player to
/// move at `state`. `battle_royale::<Standard, _, _>` always hands
/// `choose_action` the live position with the mover to act, so the returned
/// action is already the best move *for us*; no sign flip is applied here. Ties
/// between equal-scoring moves resolve to the lowest column, because
/// `bounded_negamax` ranks with a stable sort over `generate_actions` order and
/// this game generates columns in ascending order.
pub struct Connect4NegamaxPlayer {
    solver: Negamax<Connect4Negamax, MaterialBlind>,
    max_depth: u32,
    name: String,
}

impl Connect4NegamaxPlayer {
    /// `max_depth` is the fixed negamax search depth in plies (clamped to at
    /// least 1). Tie-breaking is fully deterministic without a seed.
    pub fn new(max_depth: u32) -> Self {
        let max_depth = max_depth.max(1);
        let solver = Negamax::<Connect4Negamax, MaterialBlind>::new_with_options(
            MaterialBlind,
            NegamaxOptions::default()
                .with_max_depth(max_depth)
                .with_table_bits(TABLE_BITS),
        );
        Self {
            solver,
            max_depth,
            name: format!("negamax-d{max_depth}"),
        }
    }

    /// The fixed-depth best move for `state`.
    pub fn best_move(&mut self, state: &State<6, 7>) -> Move {
        self.solver.bounded_negamax(state, self.max_depth).0
    }
}

impl Search for Connect4NegamaxPlayer {
    type G = Standard;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State<6, 7>) -> Move {
        self.best_move(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_diagnostic::reference_negamax_score;
    use mcts::game::Game;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn play(cols: &[u8]) -> State<6, 7> {
        let mut state = State::<6, 7>::default();
        for &c in cols {
            assert!(!Standard::is_terminal(&state), "scripted line ended early");
            state = Standard::apply(state, &Move(c));
        }
        state
    }

    fn ply(state: &State<6, 7>) -> u32 {
        state.black().count_ones() + state.white().count_ones()
    }

    /// Proven value of playing `mv` in `state`, from `state`'s mover's
    /// perspective, as a sign in `{-1, 0, 1}`. The child is searched to the
    /// end (`MaterialBlind` blind cutoffs never apply because `depth` covers
    /// every remaining ply), so this is an exact optimal-play label.
    fn proven_sign(state: &State<6, 7>, mv: Move) -> i32 {
        let child = Standard::apply(*state, &mv);
        if child.has_winner() {
            return 1; // we just connected four
        }
        let remaining = 42 - ply(&child);
        (-reference_negamax_score(&child, remaining.max(1))).signum()
    }

    fn legal(state: &State<6, 7>) -> Vec<Move> {
        let mut v = Vec::new();
        Standard::generate_actions(state, &mut v);
        v
    }

    /// A depth-1 player plays an immediate winning move when one exists.
    #[test]
    fn takes_immediate_win() {
        // Black holds row 0 columns 0,1,2; White scattered on 4,5,6.
        let state = play(&[0, 4, 1, 5, 2, 6]);
        assert_eq!(Standard::player_to_move(&state), crate::Player::Black);
        assert_eq!(Connect4NegamaxPlayer::new(1).choose_action(&state), Move(3));
    }

    /// A depth-2 player blocks the opponent's immediate winning threat when it
    /// has no faster win of its own. (Depth 1 cannot see an opponent reply, so
    /// blocking is inherently a depth-2 property.)
    #[test]
    fn blocks_immediate_threat() {
        // Black row 0 columns 0,1,2 threatens column 3; White (discs on 4,5)
        // has no immediate win and must block.
        let state = play(&[0, 4, 1, 5, 2]);
        assert_eq!(Standard::player_to_move(&state), crate::Player::White);
        assert_eq!(Connect4NegamaxPlayer::new(2).choose_action(&state), Move(3));
    }

    /// On a fixed set of near-terminal mid-game positions, increasing the
    /// search depth never regresses the proven quality of the chosen move: the
    /// sequence of proven-value signs for the depth-2, depth-4 and
    /// search-to-the-end players is non-decreasing, and the exact player always
    /// picks a proven-optimal move.
    #[test]
    fn deeper_search_never_regresses_proven_quality() {
        let mut fixtures: Vec<State<6, 7>> = Vec::new();
        for seed in 0..400u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = State::<6, 7>::default();
            let mut ok = true;
            for _ in 0..35 {
                if Standard::is_terminal(&state) {
                    ok = false;
                    break;
                }
                let moves = legal(&state);
                state = Standard::apply(state, &moves[rng.gen_range(0..moves.len())]);
            }
            let remaining = 42 - ply(&state);
            if ok && !Standard::is_terminal(&state) && (6..=8).contains(&remaining) {
                fixtures.push(state);
            }
            if fixtures.len() == 4 {
                break;
            }
        }
        assert_eq!(fixtures.len(), 4, "expected 4 near-terminal fixtures");

        for state in &fixtures {
            let remaining = 42 - ply(state);
            let best = legal(state)
                .into_iter()
                .map(|m| proven_sign(state, m))
                .max()
                .unwrap();

            let mut prev = i32::MIN;
            for depth in [2u32, 4, remaining] {
                let choice = Connect4NegamaxPlayer::new(depth).choose_action(state);
                let sign = proven_sign(state, choice);
                assert!(
                    sign >= prev,
                    "depth {depth} regressed proven quality ({sign} < {prev})"
                );
                prev = sign;
                if depth == remaining {
                    assert_eq!(sign, best, "exact player did not pick a proven-optimal move");
                }
            }
        }
    }
}
