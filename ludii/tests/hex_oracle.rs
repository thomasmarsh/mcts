//! Oracle test proving the `style_c` sexpr -> Core IR -> interpreter chain agrees, on Hex, with an
//! independent, from-scratch hand-rolled connectivity reference -- there's no existing
//! `games/hex` crate to reuse the way `games/ttt` was reused for `tests/oracle.rs`, per
//! `DESIGN.md`'s corpus notes, so this test *is* the second implementation rather than a wrapper
//! around one.
//!
//! [`HexOracle`] deliberately doesn't call into `ludii::core::hex` or `game_core::bitboard`'s
//! `flood6` at all -- it recomputes hex adjacency and connected components from first principles
//! (a plain BFS over an explicit six-neighbor delta list), so a bug shared between the oracle and
//! the interpreter is very unlikely to be a coincidence.

use ludii::core::interp::State;
use ludii::core::Program;
use ludii::style_c::parse_game;

const SIDE: usize = 3;
const SITES: usize = SIDE * SIDE;

fn hex_program() -> Program {
    parse_game(include_str!("../style-c/sexpr/hex.sc")).unwrap()
}

/// A from-scratch reference implementation of Hex on a `SIDE x SIDE` board: player 0 ("P1")
/// connects row 0 to row `SIDE - 1`, player 1 ("P2") connects column 0 to column `SIDE - 1` --
/// matching `Hex::edge_for_compass`'s NE/SW -> North/South, NW/SE -> West/East convention (see
/// `ludii::core::hex`'s doc comment), but re-derived independently rather than calling into it.
struct HexOracle {
    occupied: [[bool; SITES]; 2],
    to_move: usize,
}

impl HexOracle {
    fn new() -> Self {
        HexOracle {
            occupied: [[false; SITES]; 2],
            to_move: 0,
        }
    }

    fn legal_moves(&self) -> Vec<usize> {
        (0..SITES)
            .filter(|&s| !self.occupied[0][s] && !self.occupied[1][s])
            .collect()
    }

    fn apply(&mut self, site: usize) {
        assert!(!self.occupied[0][site] && !self.occupied[1][site]);
        self.occupied[self.to_move][site] = true;
        self.to_move = 1 - self.to_move;
    }

    fn last_mover(&self) -> usize {
        1 - self.to_move
    }

    /// The six hex-adjacent neighbors of `site`: north/south/east/west, plus the
    /// northeast/southwest diagonal (not northwest/southeast) -- see `game_core::bitboard`'s
    /// `flood6` doc comment for why that's the diagonal pair this corpus picked.
    fn neighbors(site: usize) -> Vec<usize> {
        let row = (site / SIDE) as isize;
        let col = (site % SIDE) as isize;
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
            .into_iter()
            .filter_map(|(dr, dc)| {
                let r = row + dr;
                let c = col + dc;
                (r >= 0 && (r as usize) < SIDE && c >= 0 && (c as usize) < SIDE)
                    .then(|| r as usize * SIDE + c as usize)
            })
            .collect()
    }

    /// The winner, if the player who just moved has a hex-connected group of their own stones
    /// touching both of their edges. Computed via plain BFS over every connected component of
    /// that player's stones, independent of any seed-from-one-edge strategy.
    fn winner(&self) -> Option<usize> {
        let player = self.last_mover();
        let stones = &self.occupied[player];
        let mut visited = [false; SITES];
        for start in 0..SITES {
            if !stones[start] || visited[start] {
                continue;
            }
            let mut stack = vec![start];
            visited[start] = true;
            let mut component = Vec::new();
            while let Some(site) = stack.pop() {
                component.push(site);
                for n in Self::neighbors(site) {
                    if stones[n] && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
            let (touches_a, touches_b) = if player == 0 {
                (
                    component.iter().any(|&s| s / SIDE == 0),
                    component.iter().any(|&s| s / SIDE == SIDE - 1),
                )
            } else {
                (
                    component.iter().any(|&s| s % SIDE == 0),
                    component.iter().any(|&s| s % SIDE == SIDE - 1),
                )
            };
            if touches_a && touches_b {
                return Some(player);
            }
        }
        None
    }
}

fn legal_sites_ludii(state: &State<SIDE, SIDE>, program: &Program) -> Vec<usize> {
    let mut sites: Vec<usize> = state.legal_moves(program).collect();
    sites.sort_unstable();
    sites
}

/// Replays `sites` through both the oracle and the interpreter move by move, asserting legal
/// moves and terminal result agree after every move.
fn assert_agrees(sites: &[usize]) {
    let program = hex_program();
    let mut oracle = HexOracle::new();
    let mut interp = State::<SIDE, SIDE>::new(&program);

    assert_eq!(
        oracle.legal_moves(),
        legal_sites_ludii(&interp, &program),
        "initial legal moves disagree"
    );
    assert_eq!(oracle.winner(), interp.winner(&program));

    for (step, &site) in sites.iter().enumerate() {
        oracle.apply(site);
        interp.apply(site);

        assert_eq!(
            oracle.legal_moves(),
            legal_sites_ludii(&interp, &program),
            "legal moves disagree after move {step} (site {site})"
        );
        assert_eq!(
            oracle.winner(),
            interp.winner(&program),
            "terminal result disagrees after move {step} (site {site})"
        );
    }
}

#[test]
fn in_progress_two_moves() {
    // P1 takes the center, P2 takes a corner. No winner yet.
    assert_agrees(&[4, 0]);
}

#[test]
fn in_progress_four_moves() {
    // A longer in-progress game with no connection for either player yet.
    assert_agrees(&[0, 1, 3, 5]);
}

#[test]
fn p1_wins_straight_column() {
    // P1 (row 0 <-> row 2): the middle column (1, 4, 7) is a straight north/south chain.
    // P2: 0, 3 (off to the side, no connection).
    assert_agrees(&[1, 0, 4, 3, 7]);
}

#[test]
fn p2_wins_northeast_southwest_diagonal() {
    // P2 (col 0 <-> col 2): the northeast/southwest diagonal (0, 4, 8) is hex-adjacent.
    // P1: 1, 2, 3 (no connection).
    assert_agrees(&[1, 0, 2, 4, 3, 8]);
}

#[test]
fn northwest_southeast_diagonal_does_not_connect() {
    // The *other* diagonal (2, 4, 6) touches both of P2's edges (site 6 on column 0, site 2 on
    // column 2) but is deliberately not hex-adjacent -- must not count as connected.
    // P1: 0, 1, 3 (no connection).
    assert_agrees(&[0, 2, 1, 4, 3, 6]);
}

#[test]
fn full_board_checkerboard_p1_wins_on_the_last_move() {
    // Filling the board in index order gives P1 the "even" checkerboard cells (0, 2, 4, 6, 8) and
    // P2 the "odd" ones (1, 3, 5, 7). The northeast/southwest diagonal is exactly the direction
    // that *preserves* checkerboard parity (both row and column shift by the same amount), so
    // corners 0 and 8 both land in P1's own diagonal component through the center (4) -- 2 and 6
    // stay isolated (their only same-parity neighbor would need a nonexistent row/column). P1
    // only completes row 0 <-> row 2 on their final move (site 8, move 9 of 9).
    assert_agrees(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);

    let mut interp = State::<SIDE, SIDE>::new(&hex_program());
    for site in [0, 1, 2, 3, 4, 5, 6, 7, 8] {
        interp.apply(site);
    }
    assert_eq!(interp.winner(&hex_program()), Some(0));
}
