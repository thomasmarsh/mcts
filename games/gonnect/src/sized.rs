//! A fixed-size view of Gonnect for code that builds games from `S::default()`.
//!
//! `State::default()` is the traditional 13x13 board. The game-agnostic n-tuple
//! trainer starts every episode from `<G::S>::default()`, so training on 5x5
//! needs a game whose default state is 5x5: `SizedGonnect<5>`. Every rule is
//! delegated to [`Gonnect`], so a `SizedState<N>` plays exactly like a
//! `State::new(N)`.

use std::fmt;

use mcts::game::{Canonical, Game, Real, Transform};

use crate::{Gonnect, Move, Player, State};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizedState<const N: usize>(pub State);

impl<const N: usize> Default for SizedState<N> {
    fn default() -> Self {
        SizedState(State::new(N))
    }
}

impl<const N: usize> fmt::Display for SizedState<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone)]
pub struct SizedGonnect<const N: usize>;

impl<const N: usize> Game for SizedGonnect<N> {
    type S = SizedState<N>;
    type A = Move;
    type P = Player;

    fn apply(state: Self::S, action: &Move) -> Self::S {
        SizedState(Gonnect::apply(state.0, action))
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Move>) {
        Gonnect::generate_actions(&state.0, actions)
    }

    fn random_action(state: &Self::S, rng: &mut rand::rngs::SmallRng) -> Option<Move> {
        Gonnect::random_action(&state.0, rng)
    }

    fn is_terminal(state: &Self::S) -> bool {
        Gonnect::is_terminal(&state.0)
    }

    fn player_to_move(state: &Self::S) -> Player {
        Gonnect::player_to_move(&state.0)
    }

    fn winner(state: &Self::S) -> Option<Player> {
        Gonnect::winner(&state.0)
    }

    fn parse_action(state: &Self::S, input: &str) -> Option<Move> {
        Gonnect::parse_action(&state.0, input)
    }

    fn notation(state: &Self::S, action: &Move) -> String {
        Gonnect::notation(&state.0, action)
    }

    fn num_players() -> usize {
        Gonnect::num_players()
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        Gonnect::zobrist_hash(&state.0)
    }

    fn symmetry_ply_limit(state: &Self::S) -> usize {
        Gonnect::symmetry_ply_limit(&state.0)
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let (canon, t) = Gonnect::canonical_representation(Real(state.0 .0));
        (Canonical(SizedState(canon.0)), t)
    }

    fn apply_to_action(action: Real<Move>, sym: Transform) -> Canonical<Move> {
        Gonnect::apply_to_action(action, sym)
    }

    fn invert_action(action: Canonical<Move>, sym: Transform) -> Real<Move> {
        Gonnect::invert_action(action, sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn sized_gonnect_plays_the_same_game_as_gonnect() {
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..20 {
            let mut a = State::new(5);
            let mut b = SizedState::<5>::default();
            while !Gonnect::is_terminal(&a) {
                let (mut xs, mut ys) = (Vec::new(), Vec::new());
                Gonnect::generate_actions(&a, &mut xs);
                SizedGonnect::<5>::generate_actions(&b, &mut ys);
                assert_eq!(xs, ys);
                let m = xs[rng.gen_range(0..xs.len())];
                a = Gonnect::apply(a, &m);
                b = SizedGonnect::<5>::apply(b, &m);
                assert_eq!(a, b.0);
                assert_eq!(
                    Gonnect::zobrist_hash(&a),
                    SizedGonnect::<5>::zobrist_hash(&b)
                );
            }
            assert!(SizedGonnect::<5>::is_terminal(&b));
            assert_eq!(Gonnect::winner(&a), SizedGonnect::<5>::winner(&b));
        }
    }

    #[test]
    fn default_state_is_the_requested_size() {
        assert_eq!(SizedState::<5>::default().0.black().rows(), 5);
        assert_eq!(SizedState::<7>::default().0.black().rows(), 7);
    }
}
