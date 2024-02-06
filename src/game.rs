// NOTE: Zobrist hashing is not a necessary approach to hashing in MCTS
// like it is in minimax since we don't rely on undo operaitons. We could
// just stipulate that `S` conform to `Hash + Eq`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ZobristHash(pub u64);

pub trait Game: Sized {
    type S: Clone + std::fmt::Debug; // TODO: remove debugs here
    type M: Clone + std::hash::Hash + Eq + std::fmt::Debug;
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

    /// Used in hidden information games
    fn determinize(state: &Self::S, _rng: &mut rand_xorshift::XorShiftRng) -> Self::S {
        state.clone()
    }

    /// A user visible display representation for the move
    fn notation(state: &Self::S, m: &Self::M) -> String;

    /// Which player is the winner. It is an error to call this on a
    /// non-terminal state.
    fn winner(state: &Self::S) -> Option<Self::P>;

    /// The current player
    fn player_to_move(state: &Self::S) -> Self::P;

    /// The current player
    fn hash(state: &Self::S) -> ZobristHash {
        unimplemented!()
    }
}

// TODO: this is just a sketch
pub mod safe {
    use nonempty::NonEmpty;

    use super::Game;

    pub struct ActiveState<S>(S);
    pub struct TerminalState<S>(S);

    pub enum State<G: Game> {
        Unknown(G::S),
        Active(ActiveState<G::S>),
        Terminal(TerminalState<G::S>),
    }

    impl<G: Game> State<G> {
        pub fn resolve(self) -> Self {
            match self {
                State::Unknown(state) => {
                    if G::is_terminal(&state) {
                        State::Terminal(TerminalState(state))
                    } else {
                        State::Active(ActiveState(state))
                    }
                }
                _ => self,
            }
        }

        pub fn get(&self) -> &G::S {
            match self {
                State::Unknown(state) => state,
                State::Active(state) => &state.0,
                State::Terminal(state) => &state.0,
            }
        }

        pub fn get_mut(&mut self) -> &mut G::S {
            match self {
                State::Unknown(state) => state,
                State::Active(state) => &mut state.0,
                State::Terminal(state) => &mut state.0,
            }
        }
    }

    pub struct ParsedState<G: Game> {
        state: State<G>,
    }

    impl<G: Game> ParsedState<G> {
        pub fn get_raw(&self) -> &G::S {
            self.state.get()
        }

        pub fn get_raw_mut(&mut self) -> &mut G::S {
            self.state.get_mut()
        }

        pub fn apply(state: &ActiveState<G::S>, m: G::M) -> State<G> {
            State::Unknown(G::apply(&state.0, m)).resolve()
        }

        pub fn gen_moves(state: &ActiveState<G::S>) -> NonEmpty<G::M> {
            NonEmpty::from_vec(G::gen_moves(&state.0)).unwrap_or_else(||
            panic!("Game::is_terminal() reported the state is not terminal, but there are no moves provided by Game::gen_moves()"))
        }

        pub fn get_reward(init_state: &ActiveState<G::S>, term_state: &TerminalState<G::S>) -> i32 {
            G::get_reward(&init_state.0, &term_state.0)
        }
    }
}
