//! Oracle test for Y, mirroring `tests/hex_oracle.rs`'s methodology: there's no existing
//! `games/y` crate to reuse, and (per `style-c/sexpr/y.gdls`'s header) Y isn't pushed through the
//! `.lud`/`ast`/`elaborate` pipeline this round -- so `Y` here is the `style_c`-lowered
//! `style-c/sexpr/y.gdls` `Program`, checked against an independent, from-scratch hand-rolled
//! reference rather than against a second lowering of the same `.lud` source the way
//! `tests/oracle.rs`/`tests/hex_oracle.rs` check `ttt`/Hex.
//!
//! [`YOracle`] deliberately doesn't call into `gdl::core::hex` or `gdl::core::interp` at all
//! -- it recomputes the triangular board's valid sites, six-way hex adjacency, and three-edge
//! connectivity from first principles (a plain BFS over an explicit six-neighbor delta list plus
//! its own row/col edge membership tests), so a bug shared between the oracle and the interpreter
//! is very unlikely to be a coincidence.

use gdl::core::interp::State;
use gdl::core::Program;
use gdl::style_c::parse_game;

const SIDE: usize = 4;
const SITES: usize = SIDE * SIDE;

fn y_program() -> Program {
    parse_game(include_str!("../style-c/sexpr/y.gdls")).unwrap()
}

/// A from-scratch reference implementation of Y on a side-`SIDE` triangular board: a site
/// `(row, col)` is on the board iff `row + col < SIDE` (see `core::hex::Hex`'s module doc for why
/// this is the same underlying `SIDE x SIDE` grid a rhombus board also uses, just masked
/// smaller). The three edges are `Bottom` (row 0), `Left` (col 0), and `Hypotenuse`
/// (`row + col == SIDE - 1`) -- re-derived independently rather than calling into
/// `core::hex::Hex::triangle_edge`.
struct YOracle {
    occupied: [[bool; SITES]; 2],
    to_move: usize,
}

impl YOracle {
    fn new() -> Self {
        YOracle {
            occupied: [[false; SITES]; 2],
            to_move: 0,
        }
    }

    fn on_board(site: usize) -> bool {
        let row = site / SIDE;
        let col = site % SIDE;
        row + col < SIDE
    }

    fn legal_moves(&self) -> Vec<usize> {
        (0..SITES)
            .filter(|&s| Self::on_board(s) && !self.occupied[0][s] && !self.occupied[1][s])
            .collect()
    }

    fn apply(&mut self, site: usize) {
        assert!(Self::on_board(site));
        assert!(!self.occupied[0][site] && !self.occupied[1][site]);
        self.occupied[self.to_move][site] = true;
        self.to_move = 1 - self.to_move;
    }

    fn last_mover(&self) -> usize {
        1 - self.to_move
    }

    /// The six hex-adjacent neighbors of `site` that are still on the (triangular) board --
    /// north/south/east/west, plus the northeast/southwest diagonal, matching
    /// `bitboard::Board::flood6`'s convention (see `hex_oracle.rs`'s identical
    /// neighbor list -- the adjacency itself is unaffected by which sites the board masks out).
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
            .filter(|&s| Self::on_board(s))
            .collect()
    }

    /// The winner, if the player who just moved has a hex-connected group of their own stones
    /// touching all three board edges. Computed via plain BFS over every connected component of
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
            let touches_bottom = component.iter().any(|&s| s / SIDE == 0);
            let touches_left = component.iter().any(|&s| s % SIDE == 0);
            let touches_hyp = component.iter().any(|&s| s / SIDE + s % SIDE == SIDE - 1);
            if touches_bottom && touches_left && touches_hyp {
                return Some(player);
            }
        }
        None
    }
}

fn legal_sites_gdl(state: &State<SIDE, SIDE>, program: &Program) -> Vec<usize> {
    let mut sites: Vec<usize> = state.legal_moves(program).collect();
    sites.sort_unstable();
    sites
}

/// Replays `sites` through both the oracle and the interpreter move by move, asserting legal
/// moves and terminal result agree after every move.
fn assert_agrees(sites: &[usize]) {
    let program = y_program();
    let mut oracle = YOracle::new();
    let mut interp = State::<SIDE, SIDE>::new(&program);

    assert_eq!(
        oracle.legal_moves(),
        legal_sites_gdl(&interp, &program),
        "initial legal moves disagree"
    );
    assert_eq!(oracle.winner(), interp.winner(&program));

    for (step, &site) in sites.iter().enumerate() {
        oracle.apply(site);
        interp.apply(site);

        assert_eq!(
            oracle.legal_moves(),
            legal_sites_gdl(&interp, &program),
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
fn initial_legal_moves_are_exactly_the_triangle() {
    // A side-4 triangle has 4+3+2+1 = 10 valid sites out of the 16-cell grid it's carved from.
    let program = y_program();
    let state = State::<SIDE, SIDE>::new(&program);
    assert_eq!(state.legal_moves(&program).count_ones(), 10);
}

#[test]
fn in_progress_two_moves() {
    assert_agrees(&[0, 1]);
}

#[test]
fn touching_only_two_of_three_edges_is_not_a_win() {
    // P1 plays 0 (row0,col0 -- Bottom and Left) and 4 (row1,col0 -- Left only), a connected pair
    // (adjacent via the (1,0) delta) that never reaches the Hypotenuse (row+col==3): 0 -> 0+0=0,
    // 4 -> 1+0=1. P2 plays off to the side (8, 9), no connection either.
    assert_agrees(&[0, 8, 4, 9]);
}

#[test]
fn p1_wins_by_connecting_all_three_sides() {
    // P1 fills the entire Bottom row (0, 1, 2, 3), one straight connected run (adjacent via the
    // (0,1) delta): touches Left through 0 (col 0), Hypotenuse through 3 (row+col == 3), and
    // Bottom trivially (it *is* row 0) -- all from one connected component.
    // P2 plays off to the side (8, 9, 12), no connection.
    assert_agrees(&[0, 8, 1, 9, 2, 12, 3]);
}

#[test]
fn full_triangle_p1_wins_by_filling_the_bottom_row() {
    let mut interp = State::<SIDE, SIDE>::new(&y_program());
    let sites = [0, 8, 1, 9, 2, 12, 3];
    for site in sites {
        interp.apply(site);
    }
    assert_eq!(interp.winner(&y_program()), Some(0));
}
