use rand::rngs::SmallRng;
use serde::Serialize;

// Refers to a player index. Expectation is that these values
// are small and monotonically increasing. Stored as a usize for ease
// of use as an array index.
pub trait PlayerIndex {
    fn to_index(&self) -> usize;
}

// A proxy trait to simplify some implementation
pub trait Action: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize {}

// Blanket implementation
impl<T: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize> Action for T {}

pub trait Game: Sized {
    /// The type representing the state of your game. Ideally, this
    /// should be as small as possible and have a cheap Clone or Copy
    /// implementation.
    type S: Clone + std::fmt::Debug;

    /// The type representing actions, or moves, in your game. These
    /// also should be very cheap to clone.
    type A: Action;

    /// The player type. This value only needs to conform to PlayerIndex.
    type P: PlayerIndex;

    /// Given a state, apply an action to it producing a new state.
    fn apply(state: Self::S, action: &Self::A) -> Self::S;

    /// All possible actions from a given state. This is expected to
    /// be deterministic. (Subsequent invocations on the same state
    /// should produce the same set of actions.) This will not be
    /// invoked if `is_terminal` returns `true`.
    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>);

    /// Returns `true` if the game has ended and there are no more
    /// possible actions. The default implementation calls
    /// `generate_actions` which may be expensive. Ideally this can
    /// be computed more cheaply.
    fn is_terminal(state: &Self::S) -> bool {
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        actions.is_empty()
    }

    /// For games with hidden information, state may be determinized
    /// for the sake of sampling via a playout. Essentially, this
    /// amounts to shuffling the hidden state around. Please note,
    /// however, that determinization can be difficult to perform
    /// uniformly and may introduce bias in the the playouts.
    #[allow(unused_variables)]
    fn determinize(state: Self::S, rng: &mut SmallRng) -> Self::S {
        state
    }

    /// Assuming a zero-sum game, the player who has won.
    fn winner(state: &Self::S) -> Option<Self::P>;

    /// Returns the rank of the player in a given game state. The
    /// current implementation assumes a two-player game. Rank is
    /// a value between 1.0 and num_players, with 1.0 being best
    /// and higher numbers being worse.
    //
    // NOTE: this is too expensive. Maybe `rank(S) -> Vec<f64>`
    fn rank(state: &Self::S, player_index: usize) -> f64 {
        match Self::winner(state) {
            Some(w) if w.to_index() == player_index => 1.,
            Some(_) => 2.,
            None => 1.5,
        }
    }

    /// Returns the play whose turn it is to move for the given
    /// state.
    fn player_to_move(state: &Self::S) -> Self::P;

    /// A constant value that indicates the number of players
    /// in the game.
    fn num_players() -> usize {
        2
    }

    /// Move notation for a given move relative to a given state.
    #[allow(unused)]
    fn notation(state: &Self::S, action: &Self::A) -> String {
        "??".into()
    }

    #[deprecated]
    fn gen_moves(state: &Self::S) -> Vec<Self::A> {
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        actions
    }

    #[deprecated]
    fn get_reward(init: &Self::S, term: &Self::S) -> f64 {
        match Self::winner(term) {
            None => 0.,
            Some(w) if w.to_index() == Self::player_to_move(init).to_index() => 1.,
            _ => -1.,
        }
    }
}
