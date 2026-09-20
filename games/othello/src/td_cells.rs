//! Othello's adapter for the game-agnostic `ntuple` trainer: 64 cells, king-move
//! adjacency for random-walk tuples, the 8 D4 board orientations, and the
//! per-cell code from the side to move's point of view.
//!
//! Code 0 is an empty cell, 1 a disc of the side to move, 2 a disc of the
//! opponent. With four states per cell, an empty cell the side to move can
//! legally play on is 3 instead of 0. The 3-state coding is exactly the digit
//! layout of `crate::ntuple`, so a 3-state model trained here loads there.

use ntuple::CellFeatures;

use crate::ntuple::D4;
use crate::{generate_moves, Othello, Player, State};

#[derive(Clone, Copy, Debug, Default)]
pub struct OthelloCells;

impl CellFeatures for OthelloCells {
    type G = Othello;

    fn num_cells(&self) -> usize {
        64
    }

    fn neighbors(&self, cell: usize) -> Vec<usize> {
        let (r, c) = ((cell / 8) as i32, (cell % 8) as i32);
        let mut out = Vec::with_capacity(8);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (rr, cc) = (r + dr, c + dc);
                if (dr, dc) != (0, 0) && (0..8).contains(&rr) && (0..8).contains(&cc) {
                    out.push((rr * 8 + cc) as usize);
                }
            }
        }
        out
    }

    fn orientations(&self) -> Vec<Vec<u8>> {
        D4.iter().map(|perm| perm.to_vec()).collect()
    }

    fn max_states_per_cell(&self) -> usize {
        4
    }

    fn cell_codes(&self, state: &State, states_per_cell: usize, out: &mut [u8]) {
        let (me, opp) = match state.turn {
            Player::Black => (state.black, state.white),
            Player::White => (state.white, state.black),
        };
        let (me_bits, opp_bits) = (me.bits(), opp.bits());
        let playable = if states_per_cell >= 4 {
            generate_moves(me, opp).bits()
        } else {
            0
        };
        for (i, o) in out.iter_mut().enumerate().take(64) {
            let bit = 1u64 << i;
            *o = if me_bits & bit != 0 {
                1
            } else if opp_bits & bit != 0 {
                2
            } else if playable & bit != 0 {
                3
            } else {
                0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntuple::{ModelGeometry, NTupleModel};
    use mcts::game::Game;
    use ntuple::{random_walk_tuples, terminal_value, Geometry, Model, PuctConfig, PuctPlayer};
    use mcts::algorithms::Search;
    use mcts::game::PlayerIndex;
    use std::sync::Arc;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn opening_codes_mark_own_opponent_and_playable_cells() {
        let mut codes = [0u8; 64];
        OthelloCells.cell_codes(&State::default(), 4, &mut codes);
        // Black to move: black discs on e4 (28) and d5 (35), white on d4 (27) and e5 (36).
        assert_eq!((codes[28], codes[35]), (1, 1));
        assert_eq!((codes[27], codes[36]), (2, 2));
        // Black's four legal replies: d3 (19), c4 (26), f5 (37), e6 (44).
        let playable: Vec<usize> = (0..64).filter(|&i| codes[i] == 3).collect();
        assert_eq!(playable, vec![19, 26, 37, 44]);
        // Three-state coding leaves them empty.
        OthelloCells.cell_codes(&State::default(), 3, &mut codes);
        assert!(codes.iter().all(|&c| c <= 2));
        // Colours swap with the side to move.
        let white_to_move = State {
            turn: Player::White,
            ..State::default()
        };
        OthelloCells.cell_codes(&white_to_move, 3, &mut codes);
        assert_eq!((codes[28], codes[27]), (2, 1));
    }

    #[test]
    fn three_state_logit_matches_the_existing_evaluator_on_the_same_weights() {
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/model.toml"),
        )
        .unwrap();
        let old_geom = ModelGeometry::parse(&bytes);
        let n = old_geom.n_weights();
        let w: Vec<f32> = (0..n).map(|i| ((i as f64) * 0.1).sin() as f32).collect();
        let old = NTupleModel::from_weights(old_geom, w.clone());
        let new = Model::from_weights(
            Geometry::from_toml(&bytes, &OthelloCells.orientations()),
            w,
        );
        assert_eq!(new.geometry().sha256_hex(), old.geometry().sha256_hex());
        assert_eq!(new.geometry().n_weights(), n);

        let mut s = State::default();
        let mut actions = Vec::new();
        let mut codes = [0u8; 64];
        // Walk the first moves of a game, comparing on every position.
        for _ in 0..12 {
            OthelloCells.cell_codes(&s, 3, &mut codes);
            let (a, b) = (old.logit(&s), new.logit(&codes));
            assert!((a - b).abs() < 1e-3, "old {a} new {b}");
            actions.clear();
            Othello::generate_actions(&s, &mut actions);
            s = Othello::apply(s, &actions[actions.len() / 2]);
        }
    }

    fn transform(bits: u64, k: usize) -> u64 {
        let mut out = 0u64;
        for i in 0..64 {
            if bits & (1u64 << i) != 0 {
                out |= 1u64 << D4[k][i];
            }
        }
        out
    }

    #[test]
    fn value_is_invariant_under_all_eight_orientations_for_three_and_four_states() {
        let mut rng = SmallRng::seed_from_u64(9);
        for spc in [3usize, 4] {
            let neighbors: Vec<Vec<usize>> = (0..64).map(|c| OthelloCells.neighbors(c)).collect();
            let tuples = random_walk_tuples(&neighbors, 20, 6, &mut rng);
            let geom = Geometry::from_tuples(spc, tuples, &OthelloCells.orientations());
            let w: Vec<f32> = (0..geom.n_weights()).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let model = Model::from_weights(geom, w);

            let mut s = State::default();
            let mut actions = Vec::new();
            let (mut c0, mut c1) = ([0u8; 64], [0u8; 64]);
            for ply in 0..20 {
                actions.clear();
                Othello::generate_actions(&s, &mut actions);
                s = Othello::apply(s, &actions[rng.gen_range(0..actions.len())]);
                OthelloCells.cell_codes(&s, spc, &mut c0);
                let v0 = model.logit(&c0);
                for k in 1..8 {
                    let mut g = s;
                    g.black = crate::BB::from_bits(transform(s.black.bits(), k));
                    g.white = crate::BB::from_bits(transform(s.white.bits(), k));
                    OthelloCells.cell_codes(&g, spc, &mut c1);
                    let vk = model.logit(&c1);
                    assert!((v0 - vk).abs() < 1e-3, "spc {spc} ply {ply} orient {k}: {v0} vs {vk}");
                }
            }
        }
    }

    /// Exact result of `s` for its player to move (+1, 0, -1), by exhaustive
    /// negamax over the game's own actions (so passes count as moves), and
    /// whether any line of play passes.
    fn solve(s: &State) -> (f32, bool) {
        let mover = Othello::player_to_move(s).to_index();
        if Othello::is_terminal(s) {
            return (terminal_value::<Othello>(s, mover), false);
        }
        let mut actions = Vec::new();
        Othello::generate_actions(s, &mut actions);
        let (mut best, mut passes) = (f32::NEG_INFINITY, false);
        for a in &actions {
            let (v, p) = solve_child(s, a);
            best = best.max(v);
            passes |= p || *a == crate::Move::PASS;
        }
        (best, passes)
    }

    /// Value of playing `a` in `s`, for the player to move in `s`.
    fn solve_child(s: &State, a: &crate::Move) -> (f32, bool) {
        let c = Othello::apply(*s, a);
        let (v, p) = solve(&c);
        let same = Othello::player_to_move(&c).to_index() == Othello::player_to_move(s).to_index();
        (if same { v } else { -v }, p)
    }

    #[test]
    fn endgame_search_picks_an_exactly_best_move_through_passes_with_an_untrained_model() {
        // All-zero weights: every non-terminal value is 0, so only the exact
        // terminal scores and the negamax signs can steer the search.
        let mut rng = SmallRng::seed_from_u64(5);
        let neighbors: Vec<Vec<usize>> = (0..64).map(|c| OthelloCells.neighbors(c)).collect();
        let tuples = random_walk_tuples(&neighbors, 4, 4, &mut rng);
        let geom = Geometry::from_tuples(4, tuples, &OthelloCells.orientations());
        let model = Arc::new(Model::zeros(geom));
        let cfg = PuctConfig { iterations: 3000, c_puct: 1.0, prior_temperature: 1.0 };

        let (mut checked, mut with_passes) = (0, 0);
        while checked < 24 {
            // Random play to 6 empties, keeping games that have not ended.
            let mut s = State::default();
            let mut actions = Vec::new();
            while !Othello::is_terminal(&s) && (s.black.count_ones() + s.white.count_ones()) < 58 {
                actions.clear();
                Othello::generate_actions(&s, &mut actions);
                s = Othello::apply(s, &actions[rng.gen_range(0..actions.len())]);
            }
            actions.clear();
            if Othello::is_terminal(&s) {
                continue;
            }
            Othello::generate_actions(&s, &mut actions);
            if actions.len() < 2 {
                continue;
            }
            let (best, passes) = solve(&s);
            let mut player = PuctPlayer::new(OthelloCells, model.clone(), cfg.clone());
            let chosen = player.choose_action(&s);
            let (got, _) = solve_child(&s, &chosen);
            assert_eq!(got, best, "chose {chosen:?} worth {got}, best is {best}, in\n{s}");
            checked += 1;
            with_passes += passes as u32;
        }
        assert!(with_passes > 0, "no sampled endgame contained a pass line");
    }
}
