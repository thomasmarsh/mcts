use crate::game::{Game, PlayerIndex};

use nimlib::{moves, NimAction, NimGame, NimRule, Split, Stack, TakeSize};

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum Player {
    Black,
    White,
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl Player {
    pub fn next(self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NimState {
    pub game: NimGame,
    pub rules: Vec<NimRule>, // NimGame has `rules`, but it's private...
    pub turn: Player,
}

impl NimState {
    pub fn new() -> Self {
        let stacks = vec![Stack(1), Stack(3), Stack(5), Stack(7)];
        let rules = vec![NimRule {
            take: TakeSize::Any,
            split: Split::Optional,
        }];

        Self {
            game: NimGame::new(rules.clone(), stacks),
            turn: Player::Black,
            rules,
        }
    }
}

impl Default for NimState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NimState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

#[derive(Clone)]
pub struct Nim;

impl Game for Nim {
    type S = NimState;
    type A = NimAction;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        actions.extend(moves::calculate_legal_moves(
            state.game.get_stacks(),
            &state.rules,
            (0, 0),
        ))
    }

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {
        moves::apply_move(&mut state.game, m).expect("error in nimlib");
        state.turn = state.turn.next();
        state
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        format!("{:?}", m)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.game.get_stacks().iter().all(|x| x.0 == 0)
    }

    fn winner(state: &Self::S) -> Option<Player> {
        if !Self::is_terminal(state) {
            panic!();
        }
        Some(Self::player_to_move(state).next())
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.turn
    }
}
