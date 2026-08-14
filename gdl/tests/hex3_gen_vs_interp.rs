//! Confirms `gdl::codegen::hex`'s Rust-source backend and `core::interp`'s tree-walking backend
//! agree directly with each other, not just transitively via both separately agreeing with
//! `tests/hex_oracle.rs`'s from-scratch reference (`tests/hex_gen_oracle.rs`). Mirrors
//! `tests/ttt_gen_vs_interp.rs` -- same structure, `games/hex3-gen` (`codegen::hex`'s
//! `BitBoard`-branch output for Hex) in place of `games/ttt-gen`.
//!
//! Deliberately still a 3x3 board (`games/hex3-gen`, not the 11x11
//! `games/hex-gen` the UI actually plays): `core::interp::State<N, M>` is
//! itself `game_core::bitboard::BitBoard<N, M>`-backed, capped at 64 cells,
//! so it cannot represent an 11x11 (121-cell) position at all -- this
//! comparison can only ever run against codegen::hex's `BitBoard` branch.
//! `tests/hex_gen_oracle.rs` covers the `BigBitBoard` branch instead, via a
//! from-scratch oracle that has no such cap.

use game_hex3_gen::{Move as GenMove, Player, Position as GenPosition};
use gdl::core::interp::State;
use gdl::core::Program;
use gdl::style_c::parse_game;

const SIDE: usize = 3;

fn hex_program() -> Program {
    parse_game(include_str!("../style-c/sexpr/hex.gdls")).unwrap()
}

fn legal_sites_interp(state: &State<SIDE, SIDE>, program: &Program) -> Vec<usize> {
    let mut sites: Vec<usize> = state.legal_moves(program).collect();
    sites.sort_unstable();
    sites
}

fn legal_sites_generated(position: &GenPosition) -> Vec<usize> {
    let mut actions = Vec::new();
    position.gen_moves(&mut actions);
    let mut sites: Vec<usize> = actions.into_iter().map(|m| m.0 as usize).collect();
    sites.sort_unstable();
    sites
}

fn assert_agrees(sites: &[usize]) {
    let program = hex_program();
    let mut interp = State::<SIDE, SIDE>::new(&program);
    let mut generated = GenPosition::new();

    assert_eq!(
        legal_sites_interp(&interp, &program),
        legal_sites_generated(&generated),
        "initial legal moves disagree"
    );
    assert_eq!(
        interp.winner(&program),
        generated.winner().map(|p: Player| p as usize)
    );

    for (step, &site) in sites.iter().enumerate() {
        interp.apply(site);
        generated.apply(GenMove(site as u8));

        assert_eq!(
            legal_sites_interp(&interp, &program),
            legal_sites_generated(&generated),
            "legal moves disagree after move {step} (site {site})"
        );
        assert_eq!(
            interp.winner(&program),
            generated.winner().map(|p: Player| p as usize),
            "winner disagrees after move {step} (site {site})"
        );
    }
}

#[test]
fn in_progress_two_moves() {
    assert_agrees(&[4, 0]);
}

#[test]
fn in_progress_four_moves() {
    assert_agrees(&[0, 1, 3, 5]);
}

#[test]
fn p1_wins_straight_column() {
    assert_agrees(&[1, 0, 4, 3, 7]);
}

#[test]
fn p2_wins_northeast_southwest_diagonal() {
    assert_agrees(&[1, 0, 2, 4, 3, 8]);
}

#[test]
fn northwest_southeast_diagonal_does_not_connect() {
    assert_agrees(&[0, 2, 1, 4, 3, 6]);
}

#[test]
fn full_board_checkerboard_p1_wins_on_the_last_move() {
    assert_agrees(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
}
