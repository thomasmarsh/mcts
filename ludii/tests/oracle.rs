//! Oracle test proving the whole `.lud` -> lex -> s-expr -> ast -> elaborate -> Core IR ->
//! interpreter chain agrees with `games/ttt`'s hand-written implementation, for a handful of
//! fixed positions (a couple of in-progress boards, a P1 win, a P2 win, a draw). See the "prove
//! the whole chain on Tic-Tac-Toe" session charter in `README.md`.
//!
//! `games/ttt::Position` and the Core interpreter's `State<3, 3>` happen to agree on cell
//! indexing (row-major, 0..9) and player numbering (player 0 == `Piece::X` moves first), so
//! walking the same site sequence through both and comparing legal moves + terminal result at
//! every step is a direct oracle check, not a translation exercise.

use game_ttt::{Piece, Position};
use ludii::ast::game::Description;
use ludii::core::interp::State;
use ludii::core::lower_game;
use ludii::core::Program;
use ludii::elaborate::game::elaborate_description;
use ludii::parse::parse;

fn tic_tac_toe_program() -> Program {
    let forms = parse(include_str!("../lud/Tic-Tac-Toe.lud")).unwrap();
    let Description::Game(game) = elaborate_description(&forms[0]).unwrap() else {
        panic!("expected Description::Game");
    };
    lower_game(&game).unwrap()
}

fn oracle_winner(position: &Position) -> Option<usize> {
    position.winner().map(|piece| match piece {
        Piece::X => 0,
        Piece::O => 1,
    })
}

fn legal_sites(position: &Position) -> Vec<usize> {
    let mut moves = Vec::new();
    position.gen_moves(&mut moves);
    let mut sites: Vec<usize> = moves.into_iter().map(|m| m.0 as usize).collect();
    sites.sort_unstable();
    sites
}

fn legal_sites_ludii(state: &State<3, 3>, program: &Program) -> Vec<usize> {
    let mut sites: Vec<usize> = state.legal_moves(program).collect();
    sites.sort_unstable();
    sites
}

/// Replays `sites` through both oracles move by move, asserting legal moves and terminal result
/// agree after every move.
fn assert_agrees(sites: &[usize]) {
    let program = tic_tac_toe_program();
    let mut oracle = Position::new();
    let mut interp = State::<3, 3>::new(&program);

    assert_eq!(
        legal_sites(&oracle),
        legal_sites_ludii(&interp, &program),
        "initial legal moves disagree"
    );
    assert_eq!(oracle_winner(&oracle), interp.winner(&program));

    for (step, &site) in sites.iter().enumerate() {
        oracle.apply(game_ttt::Move(site as u8));
        interp.apply(site);

        assert_eq!(
            legal_sites(&oracle),
            legal_sites_ludii(&interp, &program),
            "legal moves disagree after move {step} (site {site})"
        );
        assert_eq!(
            oracle_winner(&oracle),
            interp.winner(&program),
            "terminal result disagrees after move {step} (site {site})"
        );
    }
}

#[test]
fn in_progress_two_moves() {
    // X takes the center, O takes a corner. No winner yet.
    assert_agrees(&[4, 0]);
}

#[test]
fn in_progress_four_moves() {
    // A longer in-progress game with no line for either player yet.
    assert_agrees(&[0, 1, 3, 5]);
}

#[test]
fn p1_wins_top_row() {
    // X: 0, 1, 2 (top row). O: 3, 4.
    assert_agrees(&[0, 3, 1, 4, 2]);
}

#[test]
fn p2_wins_middle_row() {
    // O: 3, 4, 5 (middle row). X: 0, 1, 8 (no line).
    assert_agrees(&[0, 3, 1, 4, 8, 5]);
}

#[test]
fn draw() {
    // Final board (row-major):
    //   X O X
    //   X O O
    //   O X X
    // No line for either player at any point along the way.
    assert_agrees(&[0, 4, 2, 1, 3, 6, 7, 5, 8]);
}
