use rand::rngs::SmallRng;
use serde::Serialize;

// Refers to a player index. Expectation is that these values
// are small and monotonically increasing. Stored as a usize for ease
// of use as an array index. TODO: maybe this should be a newtype.
#[derive(Copy, Clone, Default, Serialize, Debug, PartialEq, Eq)]
pub struct PlayerIndex(pub usize);

impl<T> std::ops::Index<PlayerIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: PlayerIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl<T> std::ops::Index<PlayerIndex> for [T] {
    type Output = T;

    fn index(&self, index: PlayerIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl From<usize> for PlayerIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

// An index into the symmetries for a state.
#[derive(Copy, Clone, Default, Serialize, Debug)]
pub struct Symmetry(pub usize);

impl From<usize> for Symmetry {
    fn from(value: usize) -> Self {
        Symmetry(value)
    }
}

/// Container which holds both the hash for a state as well as
/// a symmetry that was used for this hash.
#[derive(Copy, Clone, Default, Serialize, Debug)]
pub struct ZobristHash<K: HashKey> {
    pub hash: K,
    pub symmetry: Symmetry,
}

impl<K: HashKey> From<K> for ZobristHash<K> {
    fn from(hash: K) -> Self {
        Self {
            hash,
            symmetry: Symmetry::default(),
        }
    }
}

// A proxy trait to simplify some implementation.
//
// NOTE: the `Hash` requirement is less strong than the Zobrist requirement for
// transposition tables. However, it would be nice to use the zobrist hash if it
// is available since it may be cheaper.
pub trait Action: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send {}

// Blanket implementation
impl<T: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send> Action for T {}

pub trait HashKey:
    Clone + Copy + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send + Default
{
}

// Blanket implementation
impl<
        T: Clone + Copy + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send + Default,
    > HashKey for T
{
}

pub trait Game: Sized + Clone + Sync + Send {
    /// The type representing the state of your game. Ideally, this
    /// should be as small as possible and have a cheap Clone or Copy
    /// implementation.
    type S: Clone + Default + std::fmt::Debug + Sized + Sync + Send + Eq + std::fmt::Display;

    /// The type representing actions, or moves, in your game. These
    /// also should be very cheap to clone.
    type A: Action;

    /// The type to use for the hash key for a state.
    type K: HashKey;

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
    fn winner(state: &Self::S) -> Option<PlayerIndex>;

    /// Returns the rank of the player in a given game state. The
    /// current implementation assumes a two-player game. Rank is
    /// a value between 1.0 and num_players, with 1.0 being best
    /// and higher numbers being worse.
    //
    // NOTE: this is too expensive. Maybe `rank(S) -> Vec<f64>`
    fn rank(state: &Self::S, player_index: PlayerIndex) -> f64 {
        match Self::winner(state) {
            Some(w) if w == player_index => 1.,
            Some(_) => 2.,
            None => 1.5,
        }
    }

    /// Returns the play whose turn it is to move for the given
    /// state.
    fn player_to_move(state: &Self::S) -> PlayerIndex;

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

    #[inline]
    fn get_reward(init: &Self::S, term: &Self::S) -> f64 {
        Self::compute_utilities(term)[Self::player_to_move(init)]
    }

    #[allow(unused_variables)]
    fn parse_action(state: &Self::S, input: &str) -> Option<Self::A> {
        unimplemented!();
    }

    // #[inline]
    // fn rank_to_util(rank: f64, num_players: usize) -> f64 {
    //     let n = num_players as f64;

    //     if n == 1. {
    //         2. * rank - 1.
    //     } else {
    //         1. - ((rank - 1.) * (2. / (n - 1.)))
    //     }
    // }

    #[inline]
    fn compute_utilities(state: &Self::S) -> Vec<f64> {
        let winner = Self::winner(state);
        (0..Self::num_players())
            .map(|i| match winner {
                None => 0.,
                Some(w) if w == PlayerIndex(i) => 1.,
                _ => -1.,
            })
            .collect()

        // TODO: think about the best way to handle ranking
        //
        // (0..Self::num_players())
        //     .map(|i| {
        //         let n = Self::num_players();
        //         let rank = Self::rank(state, i);
        //         rank_to_util(rank, n)
        //     })
        //     .collect()
    }

    /// In the presence of symmetries, we need to translate actions from one
    /// frame of reference to another.
    #[allow(unused_variables)]
    fn canonicalize_action(state: &Self::S, action: Self::A) -> Self::A {
        action
    }

    /// In the presence of symmetries, we need to translate actions from one
    /// frame of reference to another.
    #[allow(unused_variables)]
    fn relativize_action(state: &Self::S, action: Self::A) -> Self::A {
        action
    }

    /// A zobrist hash is expected to be cheap and precomputed upon move
    /// application. The hash also includes the symmetry of the state provided
    /// relative to a canonical symmetry.
    #[allow(unused_variables)]
    fn zobrist_hash(state: &Self::S) -> ZobristHash<Self::K> {
        Self::K::default().into()
    }
}
