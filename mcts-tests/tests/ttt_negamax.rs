use mcts::game::Game;
use mcts::negamax::{MaterialBlind, Negamax, NegamaxOptions};
use mcts::strategies::Search;

// Tic-tac-toe never runs past 9 plies, so a `MaterialBlind` evaluator (an
// always-draw score) is fine here -- with `max_depth` at least 9, negamax
// always reaches a real terminal state before ever consulting it.
type TS = Negamax<game_ttt::TicTacToe, MaterialBlind>;

fn solver() -> TS {
    Negamax::new_with_options(MaterialBlind, NegamaxOptions::default().with_max_depth(9))
}

#[test]
fn finds_the_same_forced_win_mcts_solver_proves() {
    // Same position as `ttt_strategies.rs`'s
    // `test_root_report_flags_the_proven_winning_move`: O to move, a
    // unique immediate win at Move(2).
    // O O .
    // X . .
    // X . .
    use game_ttt::*;

    let init_state = HashedPosition {
        position: Position {
            turn: Piece::O,
            board: [(0, Piece::O), (1, Piece::O), (3, Piece::X), (6, Piece::X)]
                .iter()
                .fold(0, |board, (i, piece)| {
                    let value = match piece {
                        Piece::X => 0b01,
                        Piece::O => 0b10,
                    };
                    board | (value << (i << 1))
                }),
        },
        hashes: [0; 8],
    };

    let mut search = solver();
    let chosen = search.choose_action(&init_state);
    assert_eq!(chosen, Move(2), "should find the immediate winning move");

    let report = search.root_report(&init_state);
    let winning = report
        .actions
        .iter()
        .find(|a| a.action == Move(2))
        .expect("Move(2) should be a reported root action");
    assert!(
        winning.is_proven,
        "winning move should be reported as proven"
    );
    assert_eq!(
        report.principal_variation.first(),
        Some(&Move(2)),
        "PV should start with the winning move"
    );
}

#[test]
fn picks_a_legal_move_from_the_opening_position() {
    use game_ttt::*;
    type G = TicTacToe;
    let init_state = HashedPosition::new();

    let mut legal = Vec::new();
    G::generate_actions(&init_state, &mut legal);

    let mut search = solver();
    let action = search.choose_action(&init_state);
    assert!(legal.contains(&action));

    // The empty board is a known draw with best play from both sides.
    assert_eq!(
        search.root_score(),
        0,
        "tic-tac-toe from the opening position is a proven draw"
    );
}
