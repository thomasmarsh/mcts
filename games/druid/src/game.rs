//! The state wrapper that carries the incremental caches, the `Druid<M>`
//! game type generic over a move encoding, and the `Game` impl. Everything
//! terminal/evaluation here (winner, `terminal_status`, `compute_utilities`)
//! is shared between the flat and move-split encodings -- only action
//! generation/application/notation are dispatch to `moves::MoveEncoding`.

use std::marker::PhantomData;

use mcts::game::{Game, PlayerIndex, TerminalStatus};

use crate::connectivity::Connectivity;
use crate::moves::{MoveEncoding, Move as SplitMove, Split};
use crate::movecache::MoveCache;
use crate::state::State;
use crate::types::{Piece, PieceKind, Player, PlacedPiece, Square};
use crate::zobrist::{
    cell_zobrist, full_hash, hand_zobrist, player_zobrist, zobrist_height_bits,
};

/// A board position plus the three incremental caches derived from it: the
/// Zobrist hash, `Connectivity`, and `MoveCache`. Only `State` (`.0`)
/// round-trips over the wire; the other three are pure caches bumped by
/// `apply`.
#[derive(Debug, Default, Clone)]
pub struct HashedState(
    pub(crate) State,
    pub(crate) u64,
    pub(crate) Connectivity,
    pub(crate) MoveCache,
);

// Deliberately excludes `Connectivity` (field 2) and `MoveCache` (field 3)
// from equality -- both are pure caches derived from `State`, but comparing
// them would either be unsound (`Connectivity`) or merely redundant
// (`MoveCache`); see each type's own doc comment for which applies and why.
// Comparing either would make this `PartialEq`/`Eq` impl unsound for the
// transposition-table dedupe check (`table.rs`'s `entry.state == state`)
// that relies on it, in `Connectivity`'s case.
impl PartialEq for HashedState {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for HashedState {}

impl HashedState {
    /// Panics if `size` isn't `Size::is_supported` -- callers that accept a
    /// size from outside this module (e.g. an API request) should check that
    /// first and reject unsupported sizes there instead of hitting this.
    pub fn new(size: crate::types::Size) -> Self {
        assert!(size.is_supported(), "unsupported board size: {size:?}");
        let state = State::new(size);
        let cache = MoveCache::new(&state);
        let bits = zobrist_height_bits(size);
        let hash = full_hash(&state, bits);
        HashedState(state, hash, Connectivity::new(size), cache)
    }

    pub fn state(&self) -> &State {
        &self.0
    }

    /// Build a `HashedState` from an arbitrary `State`, recomputing the
    /// Zobrist hash, `Connectivity`, and `MoveCache` from scratch rather than
    /// deriving them incrementally via `Game::apply`. For a `State` that
    /// wasn't built by replaying moves through `apply` -- the case for one
    /// deserialized from a client-supplied JSON state, since only `State`
    /// (not the derived caches) round-trips over the wire -- this is the only
    /// way to get a `HashedState` at all. Panics if `state.size` isn't
    /// `Size::is_supported`, matching `HashedState::new`.
    pub fn from_state(state: State) -> Self {
        assert!(
            state.size.is_supported(),
            "unsupported board size: {:?}",
            state.size
        );
        let bits = zobrist_height_bits(state.size);
        let hash = full_hash(&state, bits);
        let mut connectivity = Connectivity::new(state.size);
        for color in [Player::Black, Player::White] {
            connectivity.rebuild(state.size, &state.board, color);
        }
        let cache = MoveCache::new(&state);
        HashedState(state, hash, connectivity, cache)
    }

    /// Rebuild `Connectivity` and `MoveCache` from the current board.
    /// `Game::apply` is what normally keeps both in sync incrementally;
    /// this is only needed after mutating `.0.board` directly (bypassing
    /// `apply`), which only test code that hand-constructs a position
    /// should ever do.
    #[cfg(test)]
    pub(crate) fn resync_caches(&mut self) {
        self.2 = Connectivity::new(self.0.size);
        for color in [Player::Black, Player::White] {
            self.2.rebuild(self.0.size, &self.0.board, color);
        }
        self.3.rebuild(&self.0);
    }
}

impl std::fmt::Display for HashedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The Druid `Game`, generic over a move encoding `M` (see `moves`): `Split`
/// (the shipped Piece/Orientation/Cell sub-action sequence, the default) or
/// `Flat` (whole-turn `PlacedPiece`s, the pre-move-splitting representation
/// used by the `strength_move_splitting` comparison).
#[derive(Clone, Debug)]
pub struct DruidGame<M: MoveEncoding = Split>(pub(crate) PhantomData<M>);

/// Apply a whole-turn placement by sequencing the move-split sub-actions.
/// Used by the server adapter and by tests that do not exercise the MCTS
/// tree; the flat encoding's `Game::apply` is the direct one-action
/// equivalent.
pub fn apply_placed(mut state: HashedState, placed: PlacedPiece) -> HashedState {
    let kind = placed.0.kind();
    state = DruidGame::<Split>::apply(state, &SplitMove::Piece(kind));
    if let Piece::Lintel(o) = placed.0 {
        state = DruidGame::<Split>::apply(state, &SplitMove::Orientation(o));
    }
    state = DruidGame::<Split>::apply(state, &SplitMove::Cell(placed.1));
    state
}

/// The shared board-mutation half of applying a whole-turn placement: deplete
/// the mover's hand, write the piece, flip the turn, and update the hash for
/// the board delta, the player toggle, and the hand-count delta -- plus the
/// `Connectivity`/`MoveCache` caches. Does *not* touch `pending`; the
/// caller-owned policy (move-split's `Cell` arm resets and re-hashes the
/// pending phase transition; flat leaves it at `None`) does that. Keeping
/// this one function shared is how the two encodings stay byte-identical on
/// the board.
pub(crate) fn apply_turn(mut state: HashedState, placed: PlacedPiece) -> HashedState {
    let bits = zobrist_height_bits(state.0.size);
    debug_assert!(
        state.0.size.is_supported(),
        "HASHES table is too small for this board size; HashedState::new should have rejected it"
    );
    let (cells, n) = state.0.move_cells(placed);
    let old: [Square; 3] = std::array::from_fn(|i| state.0.board[cells[i]]);
    let mover = state.0.player;
    let kind = placed.0.kind();
    let old_hand_count = match kind {
        PieceKind::Sarsen => state.0.hand(mover).sarsens,
        PieceKind::Lintel => state.0.hand(mover).lintels,
    };
    state.0.apply(placed);
    // `State::apply` flips player and depletes `mover`'s hand; read the
    // post-state hand count now that the hand is the mover's (the mover's
    // hand is indexed by color, not "current player", so it's still the same
    // hand after the flip).
    let new_hand_count = match kind {
        PieceKind::Sarsen => state.0.hand(mover).sarsens,
        PieceKind::Lintel => state.0.hand(mover).lintels,
    };
    debug_assert!(
        state.0.board.iter().all(|square| (square.height as usize) < (1usize << bits)),
        "cell height exceeded the {bits}-bit Zobrist encoding for {:?}; max_cell_height's bound was wrong",
        state.0.size
    );
    let mut hash = state.1;
    hash ^= player_zobrist(mover);
    hash ^= player_zobrist(state.0.player);
    hash ^= hand_zobrist(mover, kind, old_hand_count, bits);
    hash ^= hand_zobrist(mover, kind, new_hand_count, bits);
    for k in 0..n {
        let i = cells[k];
        hash ^= cell_zobrist(i, old[k].height, old[k].piece, bits);
        let sq = state.0.board[i];
        hash ^= cell_zobrist(i, sq.height, sq.piece, bits);
    }
    state.1 = hash;
    state
        .2
        .update(state.0.size, &state.0.board, &cells[..n], &old[..n], mover);
    state.3.update(&state.0, &cells[..n]);
    state
}

impl<M: MoveEncoding> Game for DruidGame<M> {
    type S = HashedState;
    type A = M::Action;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        M::generate_actions(state, actions)
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.1
    }

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        M::apply(state, m)
    }

    fn is_terminal(state: &Self::S) -> bool {
        !matches!(Self::terminal_status(state), TerminalStatus::NotTerminal)
    }

    /// Single source of truth for both `is_terminal` and `winner`: both are
    /// answered by `Connectivity` (see above), so computing them separately
    /// (as the default `Game::terminal_status` does) means every caller that
    /// needs both -- e.g. the end of an MCTS rollout, which checks
    /// `is_terminal` to stop and then `winner`/`compute_utilities` to score
    /// it -- would otherwise redo the same connectivity read twice.
    /// Overriding this lets callers that go through `terminal_status` get
    /// both from one read; `is_terminal` and `winner` (below) still each do
    /// their own read when called alone, same as before.
    fn terminal_status(state: &Self::S) -> TerminalStatus<Player> {
        // Per the ruleset (http://cambolbro.com/games/druid/), the game is
        // won by completing a cross-board connection. That's the only real
        // win condition -- a depleted hand alone does *not* end the game,
        // since the other piece type may still have legal moves (that was
        // the bug: this used to trigger on either hand alone).
        if let Some(winner) = state.2.winner(state.0.size) {
            return TerminalStatus::Winner(winner);
        }

        // But the physical game's fallback for running out of pieces --
        // picking up and relocating a placed piece, or doubling the piece
        // count -- isn't implemented here, so this engine *can* reach a
        // true no-legal-moves state that the real game never would. Left
        // unterminated, that state feeds MCTS an empty action list (a
        // rollout crash) or lets a random playout burn its whole budget
        // re-stacking sarsens with no path to a connection. So: treat "no
        // legal moves" as a terminal draw, but only pay for the
        // `moves()` check once a hand is actually at zero for the mover --
        // that's the only situation where running dry is possible, so it's
        // a cheap, rare trigger rather than a call on every ply.
        let hand = state.0.current_hand();
        if hand.sarsens == 0 || hand.lintels == 0 {
            let mut actions = Vec::new();
            state.0.moves(&mut actions);
            if actions.is_empty() {
                return TerminalStatus::Draw;
            }
        }
        TerminalStatus::NotTerminal
    }

    fn notation(state: &Self::S, m: &Self::A) -> String {
        M::notation(state, m)
    }

    fn winner(state: &Self::S) -> Option<Player> {
        state.2.winner(state.0.size)
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.0.player
    }

    /// The default (`game.rs`) scores a non-terminal state as a flat 0. for
    /// both players. That default is only ever reached here via a playout
    /// hitting `max_playout_depth` before either side connects (a real
    /// winner is already handled by `terminal_status`/`trial.terminal` --
    /// see the backprop comment at
    /// `strategies/mcts/backprop.rs:95-103`, which only falls back to this
    /// function when there is genuinely nothing cached). Scoring every such
    /// cutoff as a draw throws away whatever progress either side has made
    /// -- this is the "max_depth ... reduces the quality of playouts" issue
    /// noted at the top of this file. Replace it with a cheap proxy for
    /// Cameron Browne's suggested fitness = your_best_path_prob /
    /// opponent's_best_path_prob: the difference in each color's shortest
    /// remaining border-to-border path (`State::connect_distance`),
    /// normalized to stay strictly inside (-1, 1) so it can never be
    /// confused with a real win/loss.
    fn compute_utilities(state: &Self::S) -> Vec<f64> {
        if let Some(winner) = Self::winner(state) {
            let wi = winner.to_index();
            return (0..Self::num_players())
                .map(|i| if i == wi { 1. } else { -1. })
                .collect();
        }

        // Neither color has connected (checked above), so both distances
        // are strictly positive: a distance of 0 would mean that color's
        // border-to-border path is already all their own cells, i.e. a
        // connection, which `Self::winner` would have already caught.
        let black_dist = state.0.connect_distance(Player::Black) as f64;
        let white_dist = state.0.connect_distance(Player::White) as f64;
        let black_score = (white_dist - black_dist) / (black_dist + white_dist);

        (0..Self::num_players())
            .map(|i| {
                if i == Player::Black.to_index() {
                    black_score
                } else {
                    -black_score
                }
            })
            .collect()
    }
}