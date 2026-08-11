//! Stress tests: correct but slow (multi-second, real-time-budgeted, or
//! many-games) checks that don't belong in `cargo test --lib`'s fast path.
//! Living in `tests/` (a separate integration-test binary) keeps them out of
//! that command automatically -- `cargo test --lib` never compiles or runs
//! this file. Run explicitly with `cargo test --test stress`.
//!
//! Each test here should still run alone comfortably; the guard below only
//! protects against *this binary's own* tests overlapping under cargo's
//! default per-binary test concurrency, the same problem
//! `crate::strategies::parallel_test_guard` solves for the unit-test binary.

use std::sync::{Mutex, OnceLock};

fn stress_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[test]
fn test_tree_parallel_transpositions_survive_many_real_time_games() {
    let _guard = stress_test_guard();
    // Regression guard for a race between `Node::is_terminal()` and
    // `Node::is_leaf()` in `select_step` (search.rs): those used to be
    // two separate `OnceLock::get()` reads with a decision gap between
    // them. Under transpositions, a *different* thread can resolve the
    // very same node (reached via a different move order) from Leaf to
    // Terminal in that gap: `is_terminal()` (checked first) sees the
    // still-unresolved leaf and returns `false`, then `is_leaf()`
    // (checked moments later) sees the now-resolved node and *also*
    // returns `false` -- falling through both branches into
    // `best_child()`/`Node::edges()` on a node that's actually
    // Terminal, tripping `edges()`'s `unreachable!()`. Fixed by
    // `Node::status()`, a single snapshot both decisions are now
    // derived from.
    //
    // This didn't show up in the fast `cargo test --lib` tree-parallel
    // test because that one budgets by *iteration count*: a few thousand
    // iterations split across a handful of threads on trivially-cheap
    // TicTacToe finishes in microseconds of real wall-clock time,
    // sampling very few actual thread interleavings. Budgeting by *time*
    // instead forces every thread to keep racing for the same real
    // duration regardless of how fast an iteration is, sampling far more
    // interleavings per test-second -- which is what actually caught
    // this originally (on Druid, under a real multi-hundred-ms budget).
    // Playing many full games (not just one `choose_action` call) adds
    // further exposure across many distinct board positions. That
    // combination is exactly why this test takes several real seconds
    // and belongs here rather than in the unit-test suite.
    use game_ttt::*;
    use mcts::game::Game;
    use mcts::strategies::Search;
    type G = TicTacToe;

    type TS = mcts::strategies::mcts::TreeSearch<G, mcts::strategies::mcts::strategy::Ucb1>;
    let mut ts = TS::default().config(
        mcts::strategies::mcts::SearchConfig::default()
            .max_time(std::time::Duration::from_millis(30))
            .use_transpositions(true)
            .num_tree_threads(4),
    );

    for _ in 0..20 {
        let mut state = HashedPosition::new();
        while !G::is_terminal(&state) {
            let action = ts.choose_action(&state);
            state = G::apply(state, &action);
        }
    }
}

#[test]
fn test_druid_hash_no_collision_across_many_random_games() {
    let _guard = stress_test_guard();
    // `Druid::zobrist_hash` used to cover only board cells + the pending
    // sub-move, on the assumption that player-to-move and hand counts are
    // always recoverable from the board. That's false once lintels are in
    // play -- a lintel placement can raise a cell's height by more than 1,
    // decoupling turn-count from board appearance -- so two different,
    // both-legally-reachable states could hash identically and silently
    // alias in the MCTS transposition table. The fix extended the hash to
    // cover player-to-move and both hands' remaining counts; this plays many
    // random games and asserts no two distinct states ever share a hash,
    // the same technique `examples/test_hash_collision.rs` used to find the
    // original collision (in 2,485 games -- this runs comfortably past that
    // margin). That example is kept separately for bigger ad-hoc runs (its
    // default is 200,000 games), since a run that size is too slow for even
    // this suite.
    use game_druid::{Druid, HashedState, Size, State};
    use mcts::game::Game;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::HashMap;

    let size = Size { w: 5, h: 5 };
    let mut rng = SmallRng::seed_from_u64(1);
    let mut seen: HashMap<u64, State> = HashMap::new();

    for _game in 0..5_000u64 {
        let mut state = HashedState::new(size);
        for _ply in 0..400 {
            if Druid::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Druid::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }

            let hash = Druid::zobrist_hash(&state);
            if let Some(prev) = seen.insert(hash, state.state().clone()) {
                assert_eq!(
                    prev,
                    *state.state(),
                    "hash collision: two distinct states shared one Zobrist hash"
                );
            }

            let action = *actions.choose(&mut rng).unwrap();
            state = Druid::apply(state, &action);
        }
    }
}

#[test]
fn test_othello_many_random_games_complete() {
    let _guard = stress_test_guard();
    // Verifies that many random Othello games always terminate without
    // panicking, and collects basic statistics (winner distribution, move
    // counts) as a sanity check on the game implementation.
    use game_othello::*;
    use mcts::game::Game;
    use rand::seq::SliceRandom;

    let mut rng = rand::thread_rng();
    const NUM_GAMES: usize = 2_000;

    let mut move_counts: Vec<usize> = Vec::with_capacity(NUM_GAMES);
    let mut non_pass_counts: Vec<usize> = Vec::with_capacity(NUM_GAMES);
    let mut black_wins: u64 = 0;
    let mut white_wins: u64 = 0;
    let mut draws: u64 = 0;

    for _ in 0..NUM_GAMES {
        let mut state = State::default();
        let mut total_actions = 0usize;
        let mut non_pass_actions = 0usize;

        while !Othello::is_terminal(&state) {
            let mut actions = Vec::new();
            Othello::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "non-terminal state produced zero actions"
            );

            let action = *actions.choose(&mut rng).unwrap();
            if action != Move::PASS {
                non_pass_actions += 1;
            }
            state = Othello::apply(state, &action);
            total_actions += 1;
            assert!(
                total_actions <= 200,
                "game exceeded 200 actions ({total_actions})"
            );
        }

        move_counts.push(total_actions);
        non_pass_counts.push(non_pass_actions);
        match Othello::winner(&state) {
            Some(Player::Black) => black_wins += 1,
            Some(Player::White) => white_wins += 1,
            None => draws += 1,
        }
    }

    let total = black_wins + white_wins + draws;
    assert_eq!(total, NUM_GAMES as u64, "all games must produce a result");

    let avg_moves = move_counts.iter().sum::<usize>() as f64 / NUM_GAMES as f64;
    let avg_non_pass = non_pass_counts.iter().sum::<usize>() as f64 / NUM_GAMES as f64;

    eprintln!("Othello random game stats ({NUM_GAMES} games):");
    eprintln!("  Avg total actions: {avg_moves:.1}");
    eprintln!("  Avg disc placements: {avg_non_pass:.1}");
    eprintln!(
        "  Black wins: {black_wins} ({:.1}%)",
        100.0 * black_wins as f64 / NUM_GAMES as f64
    );
    eprintln!(
        "  White wins: {white_wins} ({:.1}%)",
        100.0 * white_wins as f64 / NUM_GAMES as f64
    );
    eprintln!(
        "  Draws: {draws} ({:.1}%)",
        100.0 * draws as f64 / NUM_GAMES as f64
    );

    // All games had at least some disc placements (can't all pass immediately
    // from the initial position, which has 4 legal moves).
    assert!(
        avg_non_pass >= 4.0,
        "implausibly few non-pass moves: {avg_non_pass:.1}"
    );
}

/// Stress-test the Othello engine against the naive loop-based oracle for
/// every position reachable during random play, checking all 8 symmetries.
///
/// Runs 10 000 random games (~400 000 positions), each checked against 8
/// symmetries = ~3.2 million oracle comparisons.  Every mismatch is reported
/// with the game seed and ply.
#[test]
fn test_othello_oracle_symmetry_stress() {
    let _guard = stress_test_guard();
    type BB = game_core::bitboard::BitBoard<8, 8>;
    use game_core::symmetry::D4Symmetry;
    use game_othello::{
        self, naive_apply, naive_generate_moves, naive_get_flips, Move, Othello, Player, State,
    };
    use mcts::game::Game;

    // Seeded RNG (xoroshiro-like for speed; just use a simple LCG for Rust).
    // Seed chosen arbitrarily: 0xdead_beef_cafe_babe
    let mut rng: u64 = 0xdead_beef_cafe_babe;
    let next_rand = |rng: &mut u64| -> u64 {
        // xorshift64*
        let mut x = *rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *rng = x;
        x.wrapping_mul(0x9e3779b97f4a7c15)
    };

    const NUM_GAMES: u64 = 100_000;
    const MAX_PLIES: usize = 128;

    for game in 0..NUM_GAMES {
        let game_seed = rng; // snapshot for diagnostics
        let mut state = State::default();

        for ply in 0..MAX_PLIES {
            let (player, opponent) = match state.turn {
                Player::Black => (state.black, state.white),
                Player::White => (state.white, state.black),
            };

            // ---- Check invariants ----
            assert_eq!(
                state.black.bits() & state.white.bits(),
                0,
                "game={game} ply={ply}: overlapping bits"
            );

            // ---- Compare generate_moves vs oracle, all symmetries ----
            for sym_idx in 0..8 {
                // Build symmetric bitboards
                let mut black_sym = BB::EMPTY;
                let mut white_sym = BB::EMPTY;
                for i in 0..64 {
                    let si = D4Symmetry::<8>::index_symmetries(i)[sym_idx];
                    if state.black.get_at(i / 8, i % 8) {
                        black_sym |= BB::from_index(si);
                    }
                    if state.white.get_at(i / 8, i % 8) {
                        white_sym |= BB::from_index(si);
                    }
                }

                // Determine player/opponent in symmetric frame
                let (p_sym, o_sym) = match state.turn {
                    Player::Black => (black_sym, white_sym),
                    Player::White => (white_sym, black_sym),
                };

                let prod = game_othello::generate_moves(p_sym, o_sym);
                let naive = naive_generate_moves(p_sym, o_sym);
                if prod != naive {
                    panic!(
                        "game={game} (seed={game_seed:#x}) ply={ply} sym={sym_idx}: \
                         generate_moves mismatch\n  player={:#018x}\n  opponent={:#018x}\n  \
                         prod={:#018x}\n  naive={:#018x}",
                        p_sym.bits(),
                        o_sym.bits(),
                        prod.bits(),
                        naive.bits(),
                    );
                }
            }

            // ---- Pick a random legal move ----
            let legal = game_othello::generate_moves(player, opponent);
            let naive_legal = naive_generate_moves(player, opponent);
            if legal != naive_legal {
                panic!(
                    "game={game} (seed={game_seed:#x}) ply={ply}: generate_moves mismatch at identity\n  \
                     prod={:#018x}\n  naive={:#018x}",
                    legal.bits(), naive_legal.bits(),
                );
            }

            if legal.is_empty() {
                // No legal moves: try pass
                // Check that naive also has no legal moves
                assert!(
                    naive_legal.is_empty(),
                    "game={game} ply={ply}: naive has moves but prod doesn't"
                );

                let after_pass = game_othello::generate_moves(opponent, player);
                let after_naive = naive_generate_moves(opponent, player);
                if after_pass != after_naive {
                    panic!(
                        "game={game} (seed={game_seed:#x}) ply={ply}: after-pass generate_moves mismatch\n  \
                         prod={:#018x}\n  naive={:#018x}",
                        after_pass.bits(), after_naive.bits(),
                    );
                }
                if after_pass.is_empty() {
                    // Double pass: game over
                    break;
                }
                // Apply pass
                let prod_state = Othello::apply(state, &Move::PASS);
                let naive_state = naive_apply(state, &Move::PASS);
                assert_eq!(
                    prod_state.black, naive_state.black,
                    "game={game} ply={ply}: pass: black mismatch"
                );
                assert_eq!(
                    prod_state.white, naive_state.white,
                    "game={game} ply={ply}: pass: white mismatch"
                );
                assert_eq!(
                    prod_state.turn, naive_state.turn,
                    "game={game} ply={ply}: pass: turn mismatch"
                );
                state = prod_state;
                continue;
            }

            // Pick a random legal move
            let legal_bits = legal.bits();
            let num_legal = legal_bits.count_ones();
            // Choose uniformly among legal moves
            let choice = (next_rand(&mut rng) as usize) % (num_legal as usize);
            let mut mv_idx = 0;
            let mut found = 0u8;
            for i in 0..64 {
                if (legal_bits >> i) & 1 != 0 {
                    if found == choice as u8 {
                        mv_idx = i as u8;
                        break;
                    }
                    found += 1;
                }
            }

            // ---- Compare get_flips ----
            let mv_bb = BB::from_index(mv_idx as usize);
            let prod_flips = game_othello::get_flips(player, opponent, mv_bb);
            let naive_flips = naive_get_flips(player, opponent, mv_bb);
            if prod_flips != naive_flips {
                panic!(
                    "game={game} (seed={game_seed:#x}) ply={ply} move={}: \
                     get_flips mismatch\n  prod={:#018x}\n  naive={:#018x}",
                    mv_idx,
                    prod_flips.bits(),
                    naive_flips.bits(),
                );
            }

            // ---- Apply move and compare states ----
            let prod_state = Othello::apply(state, &Move(mv_idx));
            let naive_state = naive_apply(state, &Move(mv_idx));
            if prod_state.black != naive_state.black
                || prod_state.white != naive_state.white
                || prod_state.turn != naive_state.turn
                || prod_state.last_pass != naive_state.last_pass
            {
                // Debug: compare moves for both turns on the same bitboard
                let black_moves_prod =
                    game_othello::generate_moves(prod_state.black, prod_state.white);
                let white_moves_prod =
                    game_othello::generate_moves(prod_state.white, prod_state.black);
                let black_moves_naive = naive_generate_moves(naive_state.black, naive_state.white);
                let white_moves_naive = naive_generate_moves(naive_state.white, naive_state.black);
                panic!(
                    "game={game} (seed={game_seed:#x}) ply={ply} move={}: state mismatch\n  \
                     prod: B={:#018x} W={:#018x} turn={:?} last_pass={}\n  \
                     naive: B={:#018x} W={:#018x} turn={:?} last_pass={}\n  \
                     prod: Black_moves={:#018x} White_moves={:#018x}\n  \
                     naive: Black_moves={:#018x} White_moves={:#018x}",
                    mv_idx,
                    prod_state.black.bits(),
                    prod_state.white.bits(),
                    prod_state.turn,
                    prod_state.last_pass,
                    naive_state.black.bits(),
                    naive_state.white.bits(),
                    naive_state.turn,
                    naive_state.last_pass,
                    black_moves_prod.bits(),
                    white_moves_prod.bits(),
                    black_moves_naive.bits(),
                    white_moves_naive.bits(),
                );
            }

            state = prod_state;
        }

        if game % 1000 == 0 {
            eprintln!("  Othello oracle stress: {game}/{NUM_GAMES} games complete");
        }
    }
}

// ============================================================================
// Breakthrough & Knightthrough oracle stress tests
// ============================================================================

use game_breakthrough as breakthrough;
use game_core::bitboard::BitBoard;
use game_knightthrough as knightthrough;
use mcts::game::Game;

// ---- Custom Player enum for the naive representation (matches both game
// Player types structurally) ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    Black,
    White,
}

impl Player {
    fn next(self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

// ---- Naive 8×8 board: flat array, row-major (row 0 = south wall) -----------

type NaiveBoard = [Option<Player>; 64];

fn naive_from_bitboards(black: &BitBoard<8, 8>, white: &BitBoard<8, 8>) -> NaiveBoard {
    let mut board = [None; 64];
    for (i, cell) in board.iter_mut().enumerate() {
        if black.get(i) {
            *cell = Some(Player::Black);
        } else if white.get(i) {
            *cell = Some(Player::White);
        }
    }
    board
}

// ---- Naive breakthrough move generation (loops and branches) ----------------

fn naive_breakthrough_moves(board: &NaiveBoard, turn: Player) -> Vec<breakthrough::Move> {
    let mut moves = Vec::new();
    for src in 0..64 {
        if board[src] != Some(turn) {
            continue;
        }
        let row = src / 8;
        let col = src % 8;

        let (straight, sw, se) = match turn {
            Player::Black => {
                if row == 0 {
                    continue;
                }
                (
                    src - 8,
                    if col > 0 { Some(src - 9) } else { None },
                    if col < 7 { Some(src - 7) } else { None },
                )
            }
            Player::White => {
                if row == 7 {
                    continue;
                }
                (
                    src + 8,
                    if col > 0 { Some(src + 7) } else { None },
                    if col < 7 { Some(src + 9) } else { None },
                )
            }
        };

        // Straight: must be empty
        if board[straight].is_none() {
            moves.push(breakthrough::Move(src as u8, straight as u8));
        }

        // South-west / north-west: must not be own piece
        if let Some(dst) = sw {
            if board[dst] != Some(turn) {
                moves.push(breakthrough::Move(src as u8, dst as u8));
            }
        }
        // South-east / north-east: must not be own piece
        if let Some(dst) = se {
            if board[dst] != Some(turn) {
                moves.push(breakthrough::Move(src as u8, dst as u8));
            }
        }
    }
    moves
}

// ---- Naive knightthrough move generation (loops and branches) ---------------

const KNIGHT_OFFSETS: [(isize, isize); 8] = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
];

fn naive_knightthrough_moves(board: &NaiveBoard, turn: Player) -> Vec<knightthrough::Move> {
    let mut moves = Vec::new();
    for src in 0..64 {
        if board[src] != Some(turn) {
            continue;
        }
        let row = src / 8;
        let col = src % 8;

        for (dr, dc) in &KNIGHT_OFFSETS {
            let r = row as isize + dr;
            let c = col as isize + dc;
            if (0..8).contains(&r) && (0..8).contains(&c) {
                let dst = (r as usize) * 8 + (c as usize);
                if board[dst] != Some(turn) {
                    moves.push(knightthrough::Move(src as u8, dst as u8));
                }
            }
        }
    }
    moves
}

// ---- Naive apply (shared between breakthrough and knightthrough) ------------

/// Apply a move to the naive board. Returns `true` if the game ended (goal
/// reached or all opponent pieces captured).
fn naive_apply(board: &mut NaiveBoard, turn: &mut Player, src: u8, dst: u8) -> bool {
    let src = src as usize;
    let dst = dst as usize;
    let piece = board[src].take().expect("src must contain a piece");
    board[dst] = Some(piece);

    // Check win: reached opponent's back rank
    let reached_goal = match piece {
        Player::Black => dst < 8,   // south wall = row 0
        Player::White => dst >= 56, // north wall = row 7
    };
    if reached_goal {
        return true;
    }

    // Check win: all opponent pieces captured
    let opponent = match piece {
        Player::Black => Player::White,
        Player::White => Player::Black,
    };
    if !board.contains(&Some(opponent)) {
        return true;
    }

    *turn = turn.next();
    false
}

// ---- Coordinate helpers for diagnostics ------------

fn coord_str(index: usize) -> String {
    let col = index % 8;
    let row = index / 8;
    format!("{}{}", (b'a' + col as u8) as char, row + 1)
}

// ---- Stress test generated per game kind via macro -----------------------

macro_rules! stress_oracle_test {
    ($test_name:ident, $game_mod:ident, $Game:ident, $naive_moves_fn:ident, $label:expr, $num_games:expr, $games_label:expr) => {
        #[test]
        fn $test_name() {
            let _guard = stress_test_guard();
            let label = $label;
            let games_label = $games_label;
            let num_games = $num_games;

            let mut rng: u64 = 0xcafe_babe_dead_beef;
            let next_rand = |rng: &mut u64| -> u64 {
                let mut x = *rng;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *rng = x;
                x.wrapping_mul(0x9e3779b97f4a7c15)
            };

            let mut goal_wins = 0u64;
            let mut capture_wins = 0u64;
            let mut stuck_games = 0u64;

            use mcts::game::PlayerIndex;
            use $game_mod::$Game as GameT;
            use $game_mod::State as GameState;

            for game in 0..num_games {
                let game_seed = rng;

                let mut bb_state = GameState::<8, 8>::default();
                let mut naive_board = naive_from_bitboards(&bb_state.black(), &bb_state.white());
                let mut naive_turn = Player::Black;

                for ply in 0..512 {
                    // -- Verify state matches --
                    let bb_board = naive_from_bitboards(&bb_state.black(), &bb_state.white());
                    assert_eq!(
                        bb_board, naive_board,
                        "{label} game={game} (seed={game_seed:#x}) ply={ply}: \
                         board state mismatch",
                    );

                    // -- Verify piece counts --
                    let black_count = naive_board
                        .iter()
                        .filter(|&&p| p == Some(Player::Black))
                        .count();
                    let white_count = naive_board
                        .iter()
                        .filter(|&&p| p == Some(Player::White))
                        .count();
                    assert_eq!(
                        bb_state.black().count_ones() as usize,
                        black_count,
                        "{label} game={game} ply={ply}: black piece count mismatch",
                    );
                    assert_eq!(
                        bb_state.white().count_ones() as usize,
                        white_count,
                        "{label} game={game} ply={ply}: white piece count mismatch",
                    );

                    // -- Generate moves from both --
                    let mut bb_actions = Vec::new();
                    GameT::<8, 8>::generate_actions(&bb_state, &mut bb_actions);

                    let naive_actions = $naive_moves_fn(&naive_board, naive_turn);

                    // -- Sort both for comparison (ordering may differ) --
                    let mut bb_sorted: Vec<_> = bb_actions.iter().map(|m| (m.0, m.1)).collect();
                    let mut naive_sorted: Vec<_> =
                        naive_actions.iter().map(|m| (m.0, m.1)).collect();
                    bb_sorted.sort();
                    naive_sorted.sort();

                    assert_eq!(
                        bb_sorted, naive_sorted,
                        "{label} game={game} (seed={game_seed:#x}) ply={ply} \
                         turn={naive_turn:?}: move list mismatch\n  \
                         bb: {:?}\n  naive: {:?}",
                        bb_sorted, naive_sorted,
                    );

                    // -- Check terminal / stuck --
                    let bb_terminal = GameT::<8, 8>::is_terminal(&bb_state);

                    if bb_actions.is_empty() && !bb_terminal {
                        stuck_games += 1;
                        // Engine doesn't detect capture-as-loss — end game.
                        // Determine winner from naive board.
                        if black_count == 0 {
                            capture_wins += 1;
                        } else if white_count == 0 {
                            capture_wins += 1;
                        }
                        break;
                    }

                    if bb_actions.is_empty() {
                        break;
                    }

                    // -- Pick a random legal move --
                    let pick = (next_rand(&mut rng) as usize) % naive_actions.len();
                    let mv = naive_actions[pick];

                    // -- Apply to both and verify --
                    let bb_next = GameT::<8, 8>::apply(bb_state, &mv);
                    let game_ended = naive_apply(&mut naive_board, &mut naive_turn, mv.0, mv.1);

                    {
                        let bb_next_board =
                            naive_from_bitboards(&bb_next.black(), &bb_next.white());
                        assert_eq!(
                            bb_next_board,
                            naive_board,
                            "{label} game={game} (seed={game_seed:#x}) ply={ply} \
                             move={}: state mismatch after apply",
                            coord_str(mv.0 as usize),
                        );
                    }

                    if game_ended {
                        assert!(
                            bb_next.has_winner(),
                            "{label} game={game} (seed={game_seed:#x}) ply={ply}: \
                             naive says game ended but bitboard doesn't have winner flag",
                        );
                        assert_eq!(
                            bb_next.turn().to_index(),
                            naive_turn as usize,
                            "{label} game={game} (seed={game_seed:#x}) ply={ply}: \
                             turn mismatch after winning move",
                        );
                        goal_wins += 1;
                        break;
                    } else {
                        assert_eq!(
                            bb_next.turn().to_index(),
                            naive_turn as usize,
                            "{label} game={game} (seed={game_seed:#x}) ply={ply}: \
                             turn mismatch after non-winning move",
                        );
                    }

                    bb_state = bb_next;
                }

                if game % 100 == 99 {
                    eprintln!(
                        "  {label} oracle stress ({games_label}): {}/{} games complete",
                        game,
                        num_games - 1,
                    );
                }
            }

            eprintln!(
                "{label} oracle stress ({games_label}): {num_games} games, \
                 goal_wins={goal_wins}, capture_wins={capture_wins}, stuck_games={stuck_games}",
            );
        }
    };
}

stress_oracle_test!(
    test_breakthrough_oracle_stress_5000,
    breakthrough,
    Breakthrough,
    naive_breakthrough_moves,
    "Breakthrough",
    5_000,
    "5k games"
);

stress_oracle_test!(
    test_knightthrough_oracle_stress_5000,
    knightthrough,
    Knightthrough,
    naive_knightthrough_moves,
    "Knightthrough",
    5_000,
    "5k games"
);
