use crate::game::Game;

use nimlib::{moves, NimAction, NimGame, NimRule, Split, Stack, TakeSize};

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Player {
    Black,
    White,
}

impl Player {
    pub fn next(self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

#[derive(Clone, Debug)]
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

pub struct Nim;

impl Game for Nim {
    type S = NimState;
    type M = NimAction;
    type P = Player;

    fn gen_moves(state: &Self::S) -> Vec<Self::M> {
        moves::calculate_legal_moves(state.game.get_stacks(), &state.rules, (0, 0))
    }

    fn apply(state: &Self::S, m: Self::M) -> Self::S {
        let mut tmp = state.clone();
        moves::apply_move(&mut tmp.game, &m).expect("error in nimlib");
        tmp.turn = tmp.turn.next();
        tmp
    }

    fn notation(_state: &Self::S, m: &Self::M) -> String {
        format!("{:?}", m)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.game.get_stacks().iter().all(|x| x.0 == 0)
    }

    fn get_reward(init_state: &Self::S, term_state: &Self::S) -> i32 {
        let current_player = init_state.turn;
        if term_state.turn.next() == current_player {
            1
        } else {
            -1
        }
    }

    fn empty_move(_state: &Self::S) -> Self::M {
        NimAction::Place(nimlib::PlaceAction {
            stack_index: 0,
            amount: 0,
        })
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        if !Self::is_terminal(state) {
            panic!();
        }
        Some(Self::player_just_moved(state))
    }

    fn player_just_moved(state: &Self::S) -> Self::P {
        state.turn.next()
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        state.turn
    }
}
