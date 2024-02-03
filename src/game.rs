use log::trace;

pub trait Game: Sized {
    type S: Clone + std::fmt::Debug; // TODO: remove debugs here
    type M: Clone + std::fmt::Debug;
    type P: PartialEq + std::fmt::Debug;

    /// Apply the move/action, producing a new state.
    /// TODO: minimax-rs supports an optional mutative interface to consider.
    fn apply(state: &Self::S, m: Self::M) -> Self::S;

    /// The available moves/actions from this state. It is an error to call
    /// this on a terminal node.
    /// TODO: should this be a NonEmpty<Self::M> result?
    /// TODO: should take a mutable Vec?
    fn gen_moves(state: &Self::S) -> Vec<Self::M>;

    /// If this state is not terminal, we expect that gen_moves will return
    /// at least one move.
    fn is_terminal(state: &Self::S) -> bool;

    /// For MCTS with UCT, use +1 for a win, -1 for a loss, and 0 for a draw.
    fn get_reward(init_state: &Self::S, term_state: &Self::S) -> i32 {
        if !Self::is_terminal(term_state) {
            // Maybe return 0?
            panic!();
        }

        let winner = Self::winner(term_state);

        if winner.is_some() {
            if Some(Self::player_to_move(init_state)) == winner {
                1
            } else {
                -1
            }
        } else {
            0
        }
    }

    /// A user visible display representation for the move
    fn notation(state: &Self::S, m: &Self::M) -> String;

    /// Which player is the winner. It is an error to call this on a
    /// non-terminal state.
    fn winner(state: &Self::S) -> Option<Self::P>;

    /// The current player
    fn player_to_move(state: &Self::S) -> Self::P;

    // TODO: branching_factor() -- would help reserve nodes in arena
}
