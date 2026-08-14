//! A tree-walking evaluator that binds a Core IR [`super::Program`] directly to a concrete
//! `Rect`-shaped board (`game_core::bitboard::BitBoard<N, M>`), rather than compiling it to Rust
//! source. Per `DESIGN.md`'s bootstrap order ("Interpret Core, don't codegen yet"), this is
//! deliberately the slow, obviously-correct path -- an oracle to check codegen against later,
//! not a performance target itself.
//!
//! The caller picks `N`/`M` to match `program.topology` (there is currently no dynamic-topology
//! `BitBoard`, so this only works for a topology known at the call site -- see
//! `core::lower::lower_game`'s tests and the oracle test in `tests/` for the Tic-Tac-Toe case).

use game_core::bitboard::BitBoard;

use super::{EndRule, Player, Program, Region, Topology};

fn eval_region<const N: usize, const M: usize>(
    region: &Region,
    occupied: &[BitBoard<N, M>],
) -> BitBoard<N, M> {
    match region {
        Region::Occupied(Player(i)) => occupied[*i],
        Region::Union(a, b) => eval_region(a, occupied) | eval_region(b, occupied),
        Region::Complement(a) => !eval_region(a, occupied),
        Region::Sites(sites) => sites
            .iter()
            .fold(BitBoard::EMPTY, |acc, &s| acc | BitBoard::from_index(s)),
    }
}

/// The state of an in-progress game: each player's occupied region, plus whose turn it is.
#[derive(Debug, Clone, PartialEq)]
pub struct State<const N: usize, const M: usize> {
    pub occupied: Vec<BitBoard<N, M>>,
    pub to_move: usize,
}

impl<const N: usize, const M: usize> State<N, M> {
    /// A fresh, empty board for `program`. Panics (via `debug_assert`) if `N`/`M` don't match
    /// `program.topology` -- the caller is expected to already know the topology it's
    /// interpreting for.
    pub fn new(program: &Program) -> Self {
        match &program.topology {
            Topology::Rect(rect) => {
                debug_assert_eq!(rect.rows, N);
                debug_assert_eq!(rect.cols, M);
            }
            Topology::Hex(hex) => {
                debug_assert_eq!(hex.side, N);
                debug_assert_eq!(hex.side, M);
            }
        }
        State {
            occupied: vec![BitBoard::EMPTY; program.num_players],
            to_move: 0,
        }
    }

    /// The sites `program.move_gen` currently permits placing a piece on.
    pub fn legal_moves(&self, program: &Program) -> BitBoard<N, M> {
        eval_region(&program.move_gen.to, &self.occupied)
    }

    /// Places a piece for the current player at `site` and advances to the next player.
    /// Does not check legality -- callers should check `legal_moves` first.
    pub fn apply(&mut self, site: usize) {
        self.occupied[self.to_move].set(site);
        self.to_move = (self.to_move + 1) % self.occupied.len();
    }

    /// The player who moved most recently, if `apply` has been called at least once.
    fn last_mover(&self) -> usize {
        (self.to_move + self.occupied.len() - 1) % self.occupied.len()
    }

    /// The winner, if the player who just moved satisfies one of `program.end`'s end conditions.
    /// `None` if the game isn't over from any of those conditions' perspective -- this doesn't
    /// check for a full board/draw, since neither Tic-Tac-Toe's nor Hex's `.lud` declares one.
    pub fn winner(&self, program: &Program) -> Option<usize> {
        let last_mover = self.last_mover();
        let board = self.occupied[last_mover];
        program.end.iter().find_map(|rule| {
            match rule {
                EndRule::Line { length } => match &program.topology {
                    Topology::Rect(rect) => rect.lines(*length).into_iter().any(|line| {
                        let mask = line
                            .into_iter()
                            .fold(BitBoard::EMPTY, |acc, s| acc | BitBoard::from_index(s));
                        mask.is_subset(board)
                    }),
                    Topology::Hex(_) => false,
                },
                EndRule::Connected => {
                    let (side_a, side_b) = &program.player_regions[last_mover];
                    let seed = eval_region(side_a, &self.occupied);
                    let target = eval_region(side_b, &self.occupied);
                    board.flood6(seed).intersects(target)
                }
            }
            .then_some(last_mover)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::game::Description;
    use crate::core::{lower_game, EndRule, Hex, MoveGen, Rect};
    use crate::elaborate::game::elaborate_description;
    use crate::parse::parse;

    fn tic_tac_toe_program() -> Program {
        let forms = parse(include_str!("../../lud/Tic-Tac-Toe.lud")).unwrap();
        let Description::Game(game) = elaborate_description(&forms[0]).unwrap() else {
            panic!("expected Description::Game");
        };
        lower_game(&game).unwrap()
    }

    fn hex_program() -> Program {
        let forms = parse(include_str!("../../lud/Hex.lud")).unwrap();
        let Description::Game(game) = elaborate_description(&forms[0]).unwrap() else {
            panic!("expected Description::Game");
        };
        lower_game(&game).unwrap()
    }

    #[test]
    fn empty_board_has_nine_legal_moves() {
        let program = tic_tac_toe_program();
        let state = State::<3, 3>::new(&program);
        assert_eq!(state.legal_moves(&program).count_ones(), 9);
    }

    #[test]
    fn placing_reduces_legal_moves_and_alternates_turns() {
        let program = tic_tac_toe_program();
        let mut state = State::<3, 3>::new(&program);
        assert_eq!(state.to_move, 0);
        state.apply(4);
        assert_eq!(state.to_move, 1);
        assert_eq!(state.legal_moves(&program).count_ones(), 8);
        assert!(!state.legal_moves(&program).get(4));
    }

    #[test]
    fn top_row_win_is_detected() {
        let program = tic_tac_toe_program();
        let mut state = State::<3, 3>::new(&program);
        // X: 0, O: 3, X: 1, O: 4, X: 2 -- X takes the top row.
        for site in [0, 3, 1, 4, 2] {
            state.apply(site);
        }
        assert_eq!(state.winner(&program), Some(0));
    }

    #[test]
    fn no_winner_mid_game() {
        let program = tic_tac_toe_program();
        let mut state = State::<3, 3>::new(&program);
        state.apply(0);
        assert_eq!(state.winner(&program), None);
    }

    #[test]
    fn manual_program_matches_lowered_one() {
        // Core IR should be constructible and checkable by hand, independent of elaboration --
        // this pins down that a hand-built Program behaves identically to the lowered one.
        let manual = Program {
            topology: Topology::Rect(Rect { rows: 3, cols: 3 }),
            num_players: 2,
            move_gen: MoveGen {
                to: Region::Complement(Box::new(Region::Union(
                    Box::new(Region::Occupied(Player(0))),
                    Box::new(Region::Occupied(Player(1))),
                ))),
            },
            end: vec![EndRule::Line { length: 3 }],
            player_regions: Vec::new(),
        };
        assert_eq!(manual, tic_tac_toe_program());
    }

    #[test]
    fn manual_hex_program_matches_lowered_one() {
        let manual = Program {
            topology: Topology::Hex(Hex { side: 3 }),
            num_players: 2,
            move_gen: MoveGen {
                to: Region::Complement(Box::new(Region::Union(
                    Box::new(Region::Occupied(Player(0))),
                    Box::new(Region::Occupied(Player(1))),
                ))),
            },
            end: vec![EndRule::Connected],
            player_regions: vec![
                (Region::Sites(vec![6, 7, 8]), Region::Sites(vec![0, 1, 2])),
                (Region::Sites(vec![0, 3, 6]), Region::Sites(vec![2, 5, 8])),
            ],
        };
        assert_eq!(manual, hex_program());
    }

    #[test]
    fn hex_empty_board_has_nine_legal_moves() {
        let program = hex_program();
        let state = State::<3, 3>::new(&program);
        assert_eq!(state.legal_moves(&program).count_ones(), 9);
    }

    #[test]
    fn hex_no_winner_mid_game() {
        let program = hex_program();
        let mut state = State::<3, 3>::new(&program);
        state.apply(4); // center
        assert_eq!(state.winner(&program), None);
    }

    #[test]
    fn hex_p1_wins_by_connecting_north_and_south_edges() {
        // P1's edges are NE (row 2: sites 6, 7, 8) and SW (row 0: sites 0, 1, 2). A straight
        // vertical chain up the middle column (1, 4, 7) connects them.
        let program = hex_program();
        let mut state = State::<3, 3>::new(&program);
        // P1: 1, 4, 7 (middle column). P2: 0, 3 (off to the side, no connection).
        for site in [1, 0, 4, 3, 7] {
            state.apply(site);
        }
        assert_eq!(state.winner(&program), Some(0));
    }

    #[test]
    fn hex_p2_wins_by_connecting_west_and_east_edges_via_diagonal() {
        // P2's edges are NW (col 0: sites 0, 3, 6) and SE (col 2: sites 2, 5, 8). The
        // northeast/southwest diagonal (0, 4, 8) is hex-adjacent (see game_core::bitboard's
        // flood6 doc comment) and connects west to east.
        let program = hex_program();
        let mut state = State::<3, 3>::new(&program);
        // P1: 1, 2, 3 (no connection). P2: 0, 4, 8 (the hex-adjacent diagonal), moving last.
        for site in [1, 0, 2, 4, 3, 8] {
            state.apply(site);
        }
        assert_eq!(state.winner(&program), Some(1));
    }

    #[test]
    fn hex_northwest_diagonal_does_not_connect() {
        // The *other* diagonal (2, 4, 6 -- northwest/southeast) is deliberately not hex-adjacent,
        // so it must not satisfy P2's west/east connection despite touching both edges (site 6
        // is on the west edge, site 2 on the east edge).
        let program = hex_program();
        let mut state = State::<3, 3>::new(&program);
        // P1: 0, 1, 3 (no connection). P2: 2, 4, 6 (the non-hex-adjacent diagonal), moving last.
        for site in [0, 2, 1, 4, 3, 6] {
            state.apply(site);
        }
        assert_eq!(state.winner(&program), None);
    }
}
