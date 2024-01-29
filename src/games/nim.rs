use crate::game::{Game, Winner};

use nimlib::{moves, NimAction, NimGame, NimRule, Split, Stack, TakeSize};

#[derive(Clone)]
pub struct NimState {
    pub game: NimGame,
    pub rules: Vec<NimRule>, // NimGame has `rules`, but it's private...
}

impl NimState {
    fn new() -> Self {
        let stacks = vec![Stack(1), Stack(3), Stack(5), Stack(7)];
        let rules = vec![NimRule {
            take: TakeSize::Any,
            split: Split::Optional,
        }];

        Self {
            game: NimGame::new(rules.clone(), stacks),
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

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) {
        moves.extend(moves::calculate_legal_moves(
            state.game.get_stacks(),
            &state.rules,
            (0, 0),
        ));
    }

    fn apply(state: &mut Self::S, m: Self::M) -> Option<Self::S> {
        let mut tmp = state.clone();
        moves::apply_move(&mut tmp.game, &m).expect("error in nimlib");
        Some(tmp)
    }

    fn notation(_state: &Self::S, m: Self::M) -> Option<String> {
        Some(format!("{:?}", m))
    }

    fn get_winner(state: &Self::S) -> Option<Winner> {
        if state.game.get_stacks().iter().all(|x| x.0 == 0) {
            Some(Winner::PlayerJustMoved)
        } else {
            None
        }
    }
}
