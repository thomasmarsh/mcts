use mcts::game::Game;
use mcts::search::TreeStats;
use mcts::simulate::{DecisiveMove, DecisiveMoveMode, SimulateStrategy};

use game_ttt::{HashedPosition, Move, Piece, Position, TicTacToe};

use rand::SeedableRng;

/// Anti-decisive move (Teytaud & Teytaud 2010, Algorithm 4): O threatens an
/// immediate win at cell 2 (top row, cells 0/1/2). X's own pieces (cells 3
/// and 7) don't share a line, so X has no winning move of its own. Every X
/// move other than blocking cell 2 hands O that win on the very next reply,
/// so cell 2 is the unique anti-decisive choice even though it isn't a win
/// for X.
#[test]
fn anti_decisive_mode_blocks_the_opponents_only_winning_reply() {
    let mut position = Position::new();
    position.set(0, Piece::O);
    position.set(1, Piece::O);
    position.set(3, Piece::X);
    position.set(7, Piece::X);
    position.turn = Piece::X;
    let state = HashedPosition::from_position(position);

    let mut available = Vec::new();
    TicTacToe::generate_actions(&state, &mut available);
    assert_eq!(available, vec![Move(2), Move(4), Move(5), Move(6), Move(8)]);

    let mut strategy = DecisiveMove::<TicTacToe>::new().mode(DecisiveMoveMode::AntiDecisive);
    let stats = TreeStats::<TicTacToe>::default();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

    let chosen = strategy.select_move(&state, &available, &stats, 0, None, None, &mut rng);
    assert_eq!(*chosen, Move(2));
}

/// A winning move always takes priority over an anti-decisive block, even
/// when one exists elsewhere in the action list.
#[test]
fn anti_decisive_mode_still_prefers_an_immediate_win() {
    let mut position = Position::new();
    // X threatens a win at cell 2 (top row) and O separately threatens a
    // win at cell 8 (bottom row, cells 6/7/8) -- X must take its own win
    // rather than block O's.
    position.set(0, Piece::X);
    position.set(1, Piece::X);
    position.set(6, Piece::O);
    position.set(7, Piece::O);
    position.turn = Piece::X;
    let state = HashedPosition::from_position(position);

    let mut available = Vec::new();
    TicTacToe::generate_actions(&state, &mut available);

    let mut strategy = DecisiveMove::<TicTacToe>::new().mode(DecisiveMoveMode::AntiDecisive);
    let stats = TreeStats::<TicTacToe>::default();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

    let chosen = strategy.select_move(&state, &available, &stats, 0, None, None, &mut rng);
    assert_eq!(*chosen, Move(2));
}
