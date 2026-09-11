//! A classical (non-learned) MCTS player for Connect Four, usable as a
//! [`mcts::util::battle_royale`] opponent alongside [`crate::negamax_player::
//! Connect4NegamaxPlayer`] and [`crate::selfplay::CnnGumbelPlayer`].
//!
//! The search is plain UCT (`select::Ucb1`) over a depth-limited random
//! playout that falls back to [`Heuristic`] once the cutoff depth is
//! reached (`simulate::EvaluatedCutoff<Standard, Heuristic>`), rather than
//! always rolling every playout out to a natural terminal -- Connect Four
//! games run up to 42 plies, so a short random-then-evaluate playout is far
//! cheaper per iteration than a full rollout while still being informed by
//! the same tactical `Heuristic` the negamax ladder uses at its own
//! cutoffs. Connect Four has never been SMAC-tuned (unlike breakthrough/
//! othello/tak, none of which have a `presets.json` for it either), so this
//! is a reasonable-effort classical opponent built from defaults, not a
//! tuned configuration -- see the constants below for what was chosen and
//! why.

use std::time::Duration;

use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::Ucb1;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;

use crate::{Heuristic, Move, Standard, State};

/// Root-parallel worker count. Root parallelism spawns one independent
/// search tree per thread, so memory scales with this number; this machine
/// (8 cores, ~8.6 GB RAM) runs `cargo test --lib` with its own per-binary
/// test concurrency on top, so a modest fixed count -- well short of the
/// full core count -- keeps a battle_royale match using this player from
/// compounding into the kind of memory pressure documented in AGENTS.md's
/// "Rust tests" section, while still getting real wall-clock speedup over a
/// single tree.
const NUM_THREADS: usize = 2;

/// Playout depth cutoff in plies before [`Heuristic`] is consulted in place
/// of continuing the random rollout to a natural terminal. Chosen as a
/// round, moderate number -- deep enough that a playout usually resolves a
/// local tactical skirmish before being cut off, shallow enough to keep
/// each iteration cheap against Connect Four's up-to-42-ply game length.
/// Not tuned; see this module's doc comment.
const PLAYOUT_CUTOFF_DEPTH: usize = 8;

/// Plain UCT selection over a depth-cutoff playout evaluated by
/// [`Heuristic`] once cut off.
pub type ClassicalMctsProfile = Mcts<Ucb1, EvaluatedCutoff<Standard, Heuristic>>;

/// Plays a fixed-time classical UCT search each turn, using [`Heuristic`]
/// as the depth-cutoff evaluator instead of a learned value head.
pub struct ClassicalMctsPlayer {
    search: TreeSearch<Standard, ClassicalMctsProfile>,
    name: String,
}

impl ClassicalMctsPlayer {
    /// `max_time` is the search budget spent on each `choose_action` call.
    /// Thread count and playout cutoff depth are fixed (see this module's
    /// doc comment and constants); only the per-move time budget varies.
    pub fn new(max_time: Duration) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .max_playout_depth(PLAYOUT_CUTOFF_DEPTH)
                .num_threads(NUM_THREADS)
                .max_time(max_time),
        );
        Self {
            search,
            name: "classical-mcts".to_string(),
        }
    }
}

impl Search for ClassicalMctsPlayer {
    type G = Standard;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State<6, 7>) -> Move {
        self.search.choose_action(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::parallel_test_guard;
    use mcts::game::Game;

    fn play(cols: &[u8]) -> State<6, 7> {
        let mut state = State::<6, 7>::default();
        for &c in cols {
            assert!(!Standard::is_terminal(&state), "scripted line ended early");
            state = Standard::apply(state, &Move(c));
        }
        state
    }

    /// A one-ply-deep immediate winning move is trivial for even a small
    /// budget to find every time: every child of the winning move is
    /// terminal, so every playout through it backs up a certain win.
    #[test]
    fn takes_immediate_win() {
        let _guard = parallel_test_guard();
        // Black holds row 0 columns 0,1,2; White scattered on 4,5,6.
        let state = play(&[0, 4, 1, 5, 2, 6]);
        assert_eq!(Standard::player_to_move(&state), crate::Player::Black);
        let mut player = ClassicalMctsPlayer::new(Duration::from_millis(150));
        assert_eq!(player.choose_action(&state), Move(3));
    }

    /// A position where the opponent threatens an immediate win and we have
    /// no faster win of our own: the only move that avoids a certain loss
    /// next ply is the block, so any reasonable budget finds it every time.
    #[test]
    fn blocks_immediate_threat() {
        let _guard = parallel_test_guard();
        // Black row 0 columns 0,1,2 threatens column 3; White (discs on
        // 4,5) has no immediate win and must block.
        let state = play(&[0, 4, 1, 5, 2]);
        assert_eq!(Standard::player_to_move(&state), crate::Player::White);
        let mut player = ClassicalMctsPlayer::new(Duration::from_millis(150));
        assert_eq!(player.choose_action(&state), Move(3));
    }
}
