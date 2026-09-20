//! Gonnect's adapter for the game-agnostic `ntuple` trainer: an `N x N` grid,
//! king-move adjacency for the random-walk tuples, the 8 D4 board orientations,
//! and one code per cell from the side to move's point of view.
//!
//! Code 0 is an empty cell, 1 a stone of the side to move, 2 a stone of the
//! opponent (the stones-only coding, `states_per_cell = 3`). Ko and the swap
//! option are not cell properties and are not encoded: the ko snapshot only
//! removes at most one placement from the legal set, and the swap window exists
//! only for White's first reply, so a value read from cells alone cannot see
//! either. The trainer and the search see them only through the actions they
//! enumerate.

use ntuple::CellFeatures;

use crate::sized::{SizedGonnect, SizedState};
use crate::Player;

/// `(n, row, col) -> (row, col)` for one D4 element on an `n x n` grid.
type GridMap = fn(usize, usize, usize) -> (usize, usize);

#[derive(Clone, Copy, Debug, Default)]
pub struct GonnectCells<const N: usize>;

impl<const N: usize> CellFeatures for GonnectCells<N> {
    type G = SizedGonnect<N>;

    fn num_cells(&self) -> usize {
        N * N
    }

    fn neighbors(&self, cell: usize) -> Vec<usize> {
        let (r, c) = ((cell / N) as i32, (cell % N) as i32);
        let mut out = Vec::with_capacity(8);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (rr, cc) = (r + dr, c + dc);
                if (dr, dc) != (0, 0) && (0..N as i32).contains(&rr) && (0..N as i32).contains(&cc)
                {
                    out.push((rr * N as i32 + cc) as usize);
                }
            }
        }
        out
    }

    fn orientations(&self) -> Vec<Vec<u8>> {
        // The 8 elements of D4 as (row, col) maps on an N x N grid; element 0 is the identity.
        let n = N;
        let maps: [GridMap; 8] = [
            |_, r, c| (r, c),
            |n, r, c| (c, n - 1 - r),
            |n, r, c| (n - 1 - r, n - 1 - c),
            |n, r, c| (n - 1 - c, r),
            |n, r, c| (r, n - 1 - c),
            |n, r, c| (n - 1 - r, c),
            |_, r, c| (c, r),
            |n, r, c| (n - 1 - c, n - 1 - r),
        ];
        maps.iter()
            .map(|f| {
                (0..n * n)
                    .map(|i| {
                        let (r, c) = f(n, i / n, i % n);
                        (r * n + c) as u8
                    })
                    .collect()
            })
            .collect()
    }

    fn max_states_per_cell(&self) -> usize {
        3
    }

    fn cell_codes(&self, state: &SizedState<N>, states_per_cell: usize, out: &mut [u8]) {
        debug_assert_eq!(states_per_cell, 3);
        let s = &state.0;
        let (me, opp) = match s.turn() {
            Player::Black => (s.black(), s.white()),
            Player::White => (s.white(), s.black()),
        };
        for (i, o) in out.iter_mut().enumerate().take(N * N) {
            *o = if me.get_index(i) {
                1
            } else if opp.get_index(i) {
                2
            } else {
                0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::Search;
    use mcts::game::Game;
    use ntuple::{random_walk_tuples, Geometry, Model, PuctConfig, PuctPlayer};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn orientations_are_the_eight_distinct_grid_symmetries() {
        let perms = GonnectCells::<5>.orientations();
        assert_eq!(perms.len(), 8);
        assert!(perms[0].iter().enumerate().all(|(i, &c)| c as usize == i));
        for p in &perms {
            let mut sorted = p.clone();
            sorted.sort_unstable();
            assert!(sorted.iter().enumerate().all(|(i, &c)| c as usize == i), "a permutation");
        }
        let mut uniq = perms.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 8);
        // A corner maps only to corners, the centre to itself.
        for p in &perms {
            assert!([0u8, 4, 20, 24].contains(&p[0]));
            assert_eq!(p[12], 12);
        }
    }

    #[test]
    fn codes_are_relative_to_the_side_to_move() {
        let feats = GonnectCells::<5>;
        let mut codes = [0u8; 25];
        let mut s = SizedState::<5>::default();
        feats.cell_codes(&s, 3, &mut codes);
        assert!(codes.iter().all(|&c| c == 0));
        // Black plays the centre; White (to move) sees it as an opponent stone.
        let m = parse(&s, "C3");
        s = SizedGonnect::<5>::apply(s, &m);
        feats.cell_codes(&s, 3, &mut codes);
        assert_eq!(codes[12], 2);
        assert_eq!(codes.iter().filter(|&&c| c != 0).count(), 1);
        // White plays A1; Black (to move) sees the centre as its own stone.
        let m = parse(&s, "A1");
        s = SizedGonnect::<5>::apply(s, &m);
        feats.cell_codes(&s, 3, &mut codes);
        assert_eq!((codes[12], codes[0]), (1, 2));
    }

    fn parse(s: &SizedState<5>, text: &str) -> crate::Move {
        SizedGonnect::<5>::parse_action(s, text).expect("legal move")
    }

    fn play(moves: &[&str]) -> SizedState<5> {
        let mut s = SizedState::<5>::default();
        for m in moves {
            let a = parse(&s, m);
            s = SizedGonnect::<5>::apply(s, &a);
        }
        s
    }

    /// A searcher with an all-zero model: only exact terminal results steer it, so it can
    /// only play a forced win or a forced block by getting the sign of a win right.
    fn zero_model_player() -> PuctPlayer<GonnectCells<5>> {
        let feats = GonnectCells::<5>;
        let nb: Vec<Vec<usize>> = (0..25).map(|c| feats.neighbors(c)).collect();
        let tuples = random_walk_tuples(&nb, 4, 4, &mut SmallRng::seed_from_u64(3));
        let model = Arc::new(Model::zeros(Geometry::from_tuples(3, tuples, &feats.orientations())));
        let cfg = PuctConfig { iterations: 600, c_puct: 1.0, prior_temperature: 1.0, empties_exact: 0 };
        PuctPlayer::new(feats, model, cfg)
    }

    #[test]
    fn the_search_takes_a_win_in_one_and_blocks_a_loss_in_one() {
        // Black holds A3 B3 C3 D3 (E3 completes the left-right connection); White holds A1 B1 C1.
        let win_in_one = play(&["A3", "A1", "B3", "B1", "C3", "C1", "D3", "D1"]);
        let mut p = zero_model_player();
        let a = p.choose_action(&win_in_one);
        assert_eq!(SizedGonnect::<5>::notation(&win_in_one, &a), "E3", "Black to move wins at E3");

        let block = play(&["A3", "A1", "B3", "B1", "C3", "C1", "D3"]);
        let mut p = zero_model_player();
        let a = p.choose_action(&block);
        assert_eq!(SizedGonnect::<5>::notation(&block, &a), "E3", "White to move must block E3");
    }

    #[test]
    fn a_win_by_connection_is_a_win_for_the_mover_and_no_move_is_a_loss_for_the_mover() {
        let win = SizedGonnect::<5>::apply(
            play(&["A3", "A1", "B3", "B1", "C3", "C1", "D3", "D1"]),
            &parse(&play(&["A3", "A1", "B3", "B1", "C3", "C1", "D3", "D1"]), "E3"),
        );
        assert!(SizedGonnect::<5>::is_terminal(&win));
        assert_eq!(SizedGonnect::<5>::winner(&win), Some(crate::Player::Black));

        let s = play(&["A3", "A1", "B3", "B1", "C3", "C1", "D3", "D1"]);
        let stuck = SizedGonnect::<5>::apply(s, &crate::Move::NO_MOVE);
        assert!(SizedGonnect::<5>::is_terminal(&stuck));
        assert_eq!(SizedGonnect::<5>::winner(&stuck), Some(crate::Player::White), "Black had no move");
    }
}
