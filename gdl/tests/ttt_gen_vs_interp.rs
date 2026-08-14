//! Confirms `gdl::codegen::rect`'s Rust-source backend and `core::interp`'s tree-walking backend
//! agree directly with each other, not just transitively via both separately agreeing with
//! `games/ttt` (`tests/oracle.rs`, `tests/ttt_gen_oracle.rs`). Walks the same move sequences
//! `tests/oracle.rs` already uses through both `core::interp::State` (the backend every
//! `Region`/`BoolExpr` combinator is developed and checked against first) and `games/ttt-gen`
//! (`codegen::rect`'s output for the same `Program`).

use game_ttt_gen::{Move as GenMove, Player, Position as GenPosition};
use gdl::core::interp::State;
use gdl::core::Program;
use gdl::style_c::parse_game;

fn tic_tac_toe_program() -> Program {
    parse_game(include_str!("../style-c/sexpr/tic-tac-toe.gdls")).unwrap()
}

fn legal_sites_interp(state: &State<3, 3>, program: &Program) -> Vec<usize> {
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
    let program = tic_tac_toe_program();
    let mut interp = State::<3, 3>::new(&program);
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
fn p1_wins_top_row() {
    assert_agrees(&[0, 3, 1, 4, 2]);
}

#[test]
fn p2_wins_middle_row() {
    assert_agrees(&[0, 3, 1, 4, 8, 5]);
}

#[test]
fn draw() {
    assert_agrees(&[0, 4, 2, 1, 3, 6, 7, 5, 8]);
}
