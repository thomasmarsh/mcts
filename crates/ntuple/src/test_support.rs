//! Shared fixtures for this crate's tests.

use mcts::game::Game;

use crate::geometry::CellFeatures;
use game_ttt::TicTacToe;

/// Tic-tac-toe as a `CellFeatures` game: 9 cells, king-move adjacency, the
/// 8 symmetries of the square.
#[derive(Clone)]
pub struct TttCells;

impl CellFeatures for TttCells {
    type G = TicTacToe;

    fn num_cells(&self) -> usize {
        9
    }

    fn neighbors(&self, cell: usize) -> Vec<usize> {
        let (r, c) = ((cell / 3) as i32, (cell % 3) as i32);
        let mut out = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (rr, cc) = (r + dr, c + dc);
                if (dr, dc) != (0, 0) && (0..3).contains(&rr) && (0..3).contains(&cc) {
                    out.push((rr * 3 + cc) as usize);
                }
            }
        }
        out
    }

    fn orientations(&self) -> Vec<Vec<u8>> {
        type Map = fn(usize, usize) -> (usize, usize);
        let maps: [Map; 8] = [
            |r, c| (r, c),
            |r, c| (r, 2 - c),
            |r, c| (2 - r, c),
            |r, c| (c, r),
            |r, c| (2 - r, 2 - c),
            |r, c| (c, 2 - r),
            |r, c| (2 - c, r),
            |r, c| (2 - c, 2 - r),
        ];
        maps.iter()
            .map(|f| {
                (0..9)
                    .map(|i| {
                        let (r, c) = f(i / 3, i % 3);
                        (r * 3 + c) as u8
                    })
                    .collect()
            })
            .collect()
    }

    fn max_states_per_cell(&self) -> usize {
        3
    }

    fn cell_codes(&self, s: &<TicTacToe as Game>::S, _: usize, out: &mut [u8]) {
        let turn = s.position.turn;
        for (i, o) in out.iter_mut().enumerate() {
            *o = match s.position.get(i) {
                None => 0,
                Some(p) if p == turn => 1,
                Some(_) => 2,
            };
        }
    }
}
