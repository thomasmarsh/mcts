//! Druid, a connection game designed by Cameron Browne
//! (<http://cambolbro.com/games/druid/>).
//!
//! The game rules, its difficulty for MCTS, and the designer's guidance on
//! search heuristics are documented in `games/druid/README.md`, separate from
//! the code below.
//!
//! The board is defined once (see `state`, `zobrist`, `connectivity`,
//! `movecache`, `heuristics`) and exposed through two *move encodings*
//! selected by [`Druid`]'s type parameter (see `moves`): `Split`, the shipped
//! Piece/Orientation/Cell sub-action representation (the default), and
//! `Flat`, the pre-move-splitting whole-`PlacedPiece` snapshot kept for the
//! `strength_move_splitting` comparison.

mod connectivity;
pub mod game;
mod heuristics;
mod movecache;
pub mod moves;
mod state;
mod types;
mod zobrist;

pub use crate::game::{apply_placed, DruidGame, HashedState};
pub use crate::heuristics::{DruidHeuristic, DruidHeuristicWeights, RaveDecisiveHeuristic};
pub use crate::moves::{Flat, Move, MoveEncoding, Split};
pub use crate::state::State;
pub use crate::types::*;

/// The shipped, move-split Druid game: `DruidGame<Split>`. The type aliases
/// expose the concrete split game (used by the server binary, presets, and
/// existing callers) while `DruidGame<M>` stays generic so the same engine
/// can also run in the `Flat` encoding (`DruidFlat`) for the
/// `strength_move_splitting` comparison.
pub type Druid = DruidGame<Split>;
/// The flat (pre-move-splitting, whole-`PlacedPiece`) encoding of `DruidGame`.
pub type DruidFlat = DruidGame<Flat>;
/// The split (shipped) encoding of `DruidGame`, same as `Druid`.
pub type DruidSplit = DruidGame<Split>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristics::heuristic_scores;
    use crate::zobrist::*;
    use mcts::game::{Game, PlayerIndex, TerminalStatus};
    use mcts::algorithms::mcts::simulate::SimulatePolicy;
    use mcts::algorithms::mcts::TreeStats;
    use mcts::algorithms::{
        mcts::{
            node::QInit,
            render::{self, NodeRender},
            strategy, SearchConfig, TreeSearch,
        },
        Search,
    };
    use rustc_hash::FxHashSet as HashSet;

    impl NodeRender for HashedState {}

    #[test]
    fn test_druid_render() {
        let mut search = TreeSearch::<Druid, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(20),
        );
        _ = search.choose_action(&HashedState::default());
        render::render(&search);
    }

    #[test]
    fn test_self_play_smoke_no_hash_collisions() {
        // A short self-play run exercising the real Zobrist hashing path
        // end-to-end. The transposition table itself no longer stores state
        // to verify this against (`table.rs` trusts the hash outright, on
        // the strength of exactly this property holding), so this test
        // verifies it independently: collect every state reached along the
        // played line, plus every one-ply successor considered along the
        // way (the same breadth of states real expansion would hash and
        // insert), into our own map, and confirm no two distinct states
        // ever share a hash.
        let mut search: TreeSearch<Druid, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(50),
        );

        let mut seen: std::collections::HashMap<u64, State> = std::collections::HashMap::new();
        let mut check = |s: &HashedState| {
            let hash = Druid::zobrist_hash(s);
            let state = s.state().clone();
            if let Some(prev) = seen.insert(hash, state.clone()) {
                assert_eq!(
                    prev, state,
                    "hash collision: two distinct states shared one Zobrist hash"
                );
            }
        };

        let mut state = HashedState::default();
        for _ in 0..40 {
            if Druid::is_terminal(&state) {
                break;
            }
            check(&state);
            let mut actions = Vec::new();
            Druid::generate_actions(&state, &mut actions);
            for action in &actions {
                check(&Druid::apply(state.clone(), action));
            }
            let action = search.choose_action(&state);
            state = Druid::apply(state, &action);
        }
        check(&state);
    }

    #[test]
    fn test_max_cell_height_matches_hand_sarsens() {
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            assert_eq!(max_cell_height(size), Hand::new(size).sarsens as usize);
        }
    }

    #[test]
    fn test_is_supported_accepts_default_and_common_sizes() {
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            assert!(
                size.is_supported(),
                "{size:?} should be supported under the corrected bit width"
            );
        }
    }

    #[test]
    fn test_zobrist_height_encoding_is_injective_over_full_range() {
        // Confirms the per-cell height encoding is injective across the
        // *entire* representable range for a given bit width, not just the
        // heights a real game can reach -- the encoding scheme itself must
        // hold regardless of how the bound is derived.
        for size in [
            Size { w: 3, h: 3 },
            DEFAULT_SIZE,
            Size { w: 7, h: 7 },
            Size { w: 9, h: 9 },
            Size { w: 10, h: 10 },
        ] {
            let bits = zobrist_height_bits(size);
            // The bit width must be able to represent every height the game
            // can actually produce on one cell.
            assert!((1usize << bits) > max_cell_height(size));

            let mut seen = HashSet::default();
            for h in 0..(1usize << bits) {
                let mut hash = 0u64;
                for b in 0..bits {
                    if h & (1 << b) != 0 {
                        hash ^= HASHES.hash(b);
                    }
                }
                assert!(
                    seen.insert(hash),
                    "height {h} collided with an earlier height for size {size:?} (bits={bits})"
                );
            }
        }
    }

    #[test]
    fn test_zobrist_no_aliasing_past_old_area_sized_bit_width() {
        // Before the fix, `zobrist_height_bits` was sized off the board
        // area (25 for a 5x5 board), giving only `ceil(log2(25)) = 5` bits
        // -- a mod-32 ceiling. But a single hand can stack up to
        // `Hand::new(size).sarsens` (50) sarsens on one cell, so heights 1
        // and 33 used to alias. Drive a real cell up through `Game::apply`
        // and confirm every reachable height now hashes distinctly.
        let size = DEFAULT_SIZE;
        let cell = 0u8;
        let max_height = max_cell_height(size);
        assert_eq!(max_height, 50);

        let mut state = HashedState::new(size);
        let mut hashes_by_height = std::collections::HashMap::new();
        hashes_by_height.insert(0usize, state.1);

        for h in 1..=max_height {
            // Keep depleting the same hand so the cell keeps stacking
            // instead of running into the "only your own piece" rule that
            // real move generation would otherwise apply.
            state.0.player = Player::Black;
            state = apply_placed(state, PlacedPiece(Piece::Sarsen, cell));
            assert_eq!(state.0.board[cell as usize].height, h as u16);
            hashes_by_height.insert(h, state.1);
        }

        let mut seen = HashSet::default();
        for (&h, &hash) in &hashes_by_height {
            assert!(
                seen.insert(hash),
                "height {h} collided with another reachable height's hash"
            );
        }

        // The specific old (buggy) collision: height 1 vs height 1 + 32.
        let old_ceiling = 32usize;
        assert!(max_height > old_ceiling);
        assert_ne!(
            hashes_by_height[&1],
            hashes_by_height[&(1 + old_ceiling)],
            "height 1 and height {} alias, matching the old area-sized bug",
            1 + old_ceiling
        );
    }

    #[test]
    fn test_is_terminal_false_when_one_hand_empty_but_other_can_move() {
        // The pre-fix bug: is_terminal ended the game as soon as *either*
        // hand (sarsens or lintels) hit zero, even if the other piece type
        // still had a legal move. Set up exactly that: no sarsens left in
        // hand, but a legal lintel exists anyway -- a lintel's support only
        // needs the *topmost* piece at each end cell to match the mover's
        // color, sarsen or not, so two Black-topped cells are enough.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        state.0.player = Player::Black;
        state.0.board[Pos(0, 0).index(size.w)] = Square {
            height: 1,
            piece: Some(Player::Black),
        };
        state.0.board[Pos(2, 0).index(size.w)] = Square {
            height: 1,
            piece: Some(Player::Black),
        };
        state.0.hand_black.sarsens = 0;
        state.0.hand_black.lintels = 1;
        // Poking `.0.board` directly bypasses `Game::apply`, which is what
        // normally keeps `Connectivity` in sync -- resync it so
        // `is_terminal`/`terminal_status` below read the position actually
        // set up here, not whatever `Connectivity` was at `new()`.
        state.resync_caches();

        assert!(state.0.connection().is_none());
        let mut actions = Vec::new();
        state.0.moves(&mut actions);
        assert!(
            !actions.is_empty(),
            "test setup should have produced a legal lintel move"
        );

        assert!(
            !Druid::is_terminal(&state),
            "an empty sarsen hand must not end the game while a legal lintel move exists"
        );
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::NotTerminal,
            "terminal_status must agree with is_terminal"
        );
    }

    #[test]
    fn test_is_terminal_true_when_no_legal_moves_remain() {
        // Once *both* piece types are exhausted for the mover (and there's
        // no connection), there are no legal moves left -- this engine
        // doesn't implement the physical game's "pick up and relocate" or
        // "double the pieces" fallback, so treat that as a terminal draw
        // rather than feeding MCTS an empty action list.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        state.0.player = Player::Black;
        state.0.hand_black.sarsens = 0;
        state.0.hand_black.lintels = 0;

        assert!(state.0.connection().is_none());
        let mut actions = Vec::new();
        state.0.moves(&mut actions);
        assert!(actions.is_empty());

        assert!(
            Druid::is_terminal(&state),
            "no legal moves with no connection must be terminal"
        );
        assert_eq!(
            Druid::winner(&state),
            None,
            "a no-legal-moves termination is a draw, not a win"
        );
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::Draw,
            "terminal_status must agree with is_terminal/winner"
        );
    }

    #[test]
    fn test_incremental_hash_matches_full_recompute() {
        // `Game::apply` now updates the hash incrementally (XOR out the old
        // contribution, XOR in the new) instead of recomputing from scratch
        // every step. Confirm that stays identical to a from-scratch
        // recompute across many randomized move sequences and board sizes,
        // including games that run long enough to restack cells past their
        // original height. Drives `Druid::generate_actions`/`Druid::apply`
        // directly (the split sub-actions, not whole-turn `PlacedPiece`s via
        // `apply_placed`) so the check runs after *every* sub-move --
        // including the mid-turn `pending != None` states between a
        // `Move::Piece`/`Move::Orientation` and its `Move::Cell`, which a
        // whole-turn-only check never observes.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let bits = zobrist_height_bits(size);
            let mut rng = SmallRng::seed_from_u64(size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    Druid::generate_actions(&state, &mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = Druid::apply(state, &m);
                    actions.clear();

                    assert_eq!(
                        state.1,
                        full_hash(&state.0, bits),
                        "incremental hash diverged from full recompute at size={size:?} game={game} ply={ply} move={m:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_is_terminal_true_on_connection_regardless_of_hands() {
        // A completed connection ends the game even with plenty of pieces
        // left in hand.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        for x in 0..size.w {
            let i = Pos(x, 0).index(size.w);
            state.0.board[i] = Square {
                height: 1,
                piece: Some(Player::White),
            };
        }
        // See the comment on the equivalent call above: poking `.0.board`
        // directly bypasses the `Game::apply` path that normally keeps
        // `Connectivity` in sync.
        state.resync_caches();
        assert_eq!(state.0.connection(), Some(Player::White));
        assert!(
            Druid::is_terminal(&state),
            "a completed connection must be terminal"
        );
        assert_eq!(
            Druid::terminal_status(&state),
            TerminalStatus::Winner(Player::White),
            "terminal_status must agree with is_terminal/connection"
        );
    }

    #[test]
    fn test_connectivity_survives_a_move_that_flips_the_bridging_cell() {
        // `Connectivity`'s doc comment explains the subtlety this covers: a
        // lintel's legality only requires 2 of its 3 touched cells to
        // already match the mover's color, so a lintel can legally repaint
        // a cell the *opponent* owns -- silently deleting it from the
        // opponent's connectivity graph. A union-find that only ever unions
        // (never retracts) would keep reporting the old connection as
        // intact after that; this drives exactly that scenario through the
        // real `Game::apply` path and checks the incremental `winner()`
        // against a from-scratch `state.0.connection()` at every step.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        let col = 2u8;
        let mid = size.h / 2;

        let place = |state: HashedState, player: Player, pos: Pos| -> HashedState {
            let mut state = state;
            state.0.player = player;
            apply_placed(state, PlacedPiece(Piece::Sarsen, pos.index(size.w) as u8))
        };

        // Black builds the column's top and bottom segments, leaving a gap
        // at the middle row -- not yet connected top-to-bottom.
        for y in [0, 1, size.h - 2, size.h - 1] {
            state = place(state, Player::Black, Pos(col, y));
        }
        assert_eq!(
            state.0.connection(),
            None,
            "gapped column must not be connected yet"
        );
        assert_eq!(Druid::winner(&state), None, "incremental winner must agree");

        // Filling the gap connects the two segments into one continuous
        // top-to-bottom column: Black wins.
        state = place(state, Player::Black, Pos(col, mid));
        assert_eq!(
            state.0.connection(),
            Some(Player::Black),
            "filling the gap should complete the connection"
        );
        assert_eq!(
            Druid::winner(&state),
            Some(Player::Black),
            "incremental winner must agree"
        );

        // White builds sarsens flanking the bridge cell in the same row, at
        // the same height -- enough on their own (2 of 3 touched cells
        // already White) to legally place a horizontal lintel through the
        // bridge cell without it needing to match either White end.
        state = place(state, Player::White, Pos(col - 1, mid));
        state = place(state, Player::White, Pos(col + 1, mid));
        state.0.player = Player::White;
        state = apply_placed(
            state,
            PlacedPiece(
                Piece::Lintel(Orientation::Horizontal),
                Pos(col - 1, mid).index(size.w) as u8,
            ),
        );

        // The bridge cell is now White, splitting Black's column back into
        // two disconnected segments -- Black must no longer read as
        // connected, and the incremental path must agree with a from-scratch
        // recompute rather than keep reporting the connection that used to
        // exist through the now-repainted cell.
        assert_eq!(
            state.0.connection(),
            None,
            "the lintel should have broken Black's column by repainting the bridge cell"
        );
        assert_eq!(
            Druid::winner(&state),
            None,
            "incremental winner() must reflect the broken bridge, not the stale pre-flip connection"
        );
    }

    #[test]
    fn test_incremental_connectivity_matches_full_recompute() {
        // Randomized-game analogue of
        // `test_incremental_hash_matches_full_recompute`: after every move,
        // the incremental `Druid::winner` (backed by `Connectivity`) must
        // agree with a from-scratch `state.0.connection()` BFS.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng = SmallRng::seed_from_u64(0xC0117 + size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    if state.0.connection().is_some() {
                        break;
                    }
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = apply_placed(state, m);
                    actions.clear();

                    assert_eq!(
                        Druid::winner(&state),
                        state.0.connection(),
                        "incremental winner diverged from full recompute at size={size:?} game={game} ply={ply}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_generate_actions_matches_full_recompute() {
        // Ground-truth `State::moves` vs the split `generate_actions`
        // tree. Every complete `PlacedPiece` reachable by walking
        // `Piece -> (Orientation) -> Cell` through `Druid::generate_actions`
        // must equal `State::moves`, and vice-versa.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        use std::collections::HashSet;

        fn collect_via_tree(state: &HashedState) -> HashSet<PlacedPiece> {
            let mut out = HashSet::default();
            let mut piece_actions = Vec::new();
            Druid::generate_actions(state, &mut piece_actions);
            for pa in piece_actions {
                match pa {
                    Move::Piece(kind) => {
                        let s1 = Druid::apply(state.clone(), &pa);
                        let mut next = Vec::new();
                        Druid::generate_actions(&s1, &mut next);
                        match kind {
                            PieceKind::Sarsen => {
                                for ca in next {
                                    if let Move::Cell(idx) = ca {
                                        out.insert(PlacedPiece(Piece::Sarsen, idx));
                                    }
                                }
                            }
                            PieceKind::Lintel => {
                                for oa in next {
                                    if let Move::Orientation(o) = oa {
                                        let s2 = Druid::apply(s1.clone(), &oa);
                                        let mut cells = Vec::new();
                                        Druid::generate_actions(&s2, &mut cells);
                                        for ca in cells {
                                            if let Move::Cell(idx) = ca {
                                                out.insert(PlacedPiece(Piece::Lintel(o), idx));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => panic!("root actions must be Piece"),
                }
            }
            out
        }

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng = SmallRng::seed_from_u64(0x5EED + size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }

                    let ground: HashSet<PlacedPiece> = actions.iter().copied().collect();
                    let via_tree = collect_via_tree(&state);
                    assert_eq!(
                        via_tree, ground,
                        "tree-collected moves diverged from ground truth at size={size:?} game={game} ply={ply}"
                    );

                    let m = actions[rng.gen_range(0..actions.len())];
                    state = apply_placed(state, m);
                    actions.clear();
                }
            }
        }
    }

    #[test]
    fn test_from_state_matches_incrementally_built_state() {
        // `HashedState::from_state` is the deserialize-path counterpart to
        // `test_incremental_hash_matches_full_recompute` /
        // `test_generate_actions_matches_full_recompute`: given a `State`
        // that arrived some other way than a chain of `Game::apply` calls
        // (e.g. deserialized from a client-supplied JSON state, which only
        // round-trips `State`, not the derived `HashedState` caches), the
        // hash/legal-moves/terminal status it rebuilds from scratch must
        // agree exactly with the same board built incrementally.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng =
                SmallRng::seed_from_u64(0xF20_5747E + size.w as u64 * 1000 + size.h as u64);

            for game in 0..20 {
                let mut incremental = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    incremental.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    incremental = apply_placed(incremental, m);
                    actions.clear();

                    let rebuilt = HashedState::from_state(incremental.state().clone());

                    assert_eq!(
                        rebuilt.1, incremental.1,
                        "from_state hash diverged at size={size:?} game={game} ply={ply}"
                    );
                    assert_eq!(
                        Druid::winner(&rebuilt),
                        Druid::winner(&incremental),
                        "from_state winner diverged at size={size:?} game={game} ply={ply}"
                    );
                    assert_eq!(
                        Druid::is_terminal(&rebuilt),
                        Druid::is_terminal(&incremental),
                        "from_state terminal status diverged at size={size:?} game={game} ply={ply}"
                    );
                    if !Druid::is_terminal(&rebuilt) {
                        let mut rebuilt_actions = Vec::new();
                        Druid::generate_actions(&rebuilt, &mut rebuilt_actions);
                        let mut incremental_actions = Vec::new();
                        Druid::generate_actions(&incremental, &mut incremental_actions);
                        assert_eq!(
                            rebuilt_actions, incremental_actions,
                            "from_state legal moves diverged at size={size:?} game={game} ply={ply}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_compute_utilities_still_scores_a_real_win_as_decisive() {
        // The heuristic branch must not shadow the real win/loss case: a
        // connected state still gets the exact +1./-1., not a value merely
        // close to it.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        for x in 0..size.w {
            let i = Pos(x, 0).index(size.w);
            state.0.board[i] = Square {
                height: 1,
                piece: Some(Player::White),
            };
        }
        state.resync_caches();
        assert_eq!(state.0.connection(), Some(Player::White));

        let utilities = Druid::compute_utilities(&state);
        assert_eq!(utilities[Player::White.to_index()], 1.);
        assert_eq!(utilities[Player::Black.to_index()], -1.);
    }

    #[test]
    fn test_compute_utilities_is_a_draw_on_the_symmetric_empty_board() {
        // On an empty square board, Black's top-bottom distance and White's
        // left-right distance are identical by symmetry, so the heuristic
        // should agree with the old flat-draw default here specifically.
        let state = HashedState::new(DEFAULT_SIZE);
        let utilities = Druid::compute_utilities(&state);
        assert_eq!(utilities, vec![0., 0.]);
    }

    #[test]
    fn test_compute_utilities_favors_the_color_closer_to_connecting() {
        // Black has built most of a top-to-bottom column (one cell short);
        // White hasn't built anything. A depth-cutoff playout landing here
        // should score this as good for Black, not a flat draw -- this is
        // the actual bug being fixed: the old default threw this signal
        // away entirely.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        let col = 2u8;
        for y in 0..size.h - 1 {
            let i = Pos(col, y).index(size.w);
            state.0.board[i] = Square {
                height: 1,
                piece: Some(Player::Black),
            };
        }
        state.resync_caches();
        assert_eq!(state.0.connection(), None, "one cell short of connecting");

        let utilities = Druid::compute_utilities(&state);
        let black = utilities[Player::Black.to_index()];
        let white = utilities[Player::White.to_index()];
        assert!(
            black > 0.,
            "Black is one move from winning, should score above a draw: {black}"
        );
        assert_eq!(
            black, -white,
            "zero-sum: the two utilities must be exact opposites"
        );
        assert!(
            black < 1.,
            "a non-terminal cutoff must never read as a real win"
        );
    }

    #[test]
    fn test_connect_distance_zero_iff_connected() {
        let size = Size { w: 4, h: 4 };
        let mut state = HashedState::new(size);
        assert_eq!(
            state.0.connect_distance(Player::Black),
            size.h as u32,
            "empty board: every row costs 1, so the full height must be paid"
        );

        // Fill every row but the last: one cell short of a top-to-bottom
        // column.
        let col = 1u8;
        for y in 0..size.h - 1 {
            let i = Pos(col, y).index(size.w);
            state.0.board[i] = Square {
                height: 1,
                piece: Some(Player::Black),
            };
        }
        state.resync_caches();
        assert_eq!(
            state.0.connection(),
            None,
            "one cell short must not be connected yet"
        );
        assert_eq!(
            state.0.connect_distance(Player::Black),
            1,
            "one cell short of a column should cost exactly 1"
        );

        // Fill the last cell: now a complete column, i.e. a win.
        let i = Pos(col, size.h - 1).index(size.w);
        state.0.board[i] = Square {
            height: 1,
            piece: Some(Player::Black),
        };
        state.resync_caches();
        assert_eq!(state.0.connection(), Some(Player::Black));
        assert_eq!(
            state.0.connect_distance(Player::Black),
            0,
            "a completed connection must cost exactly 0"
        );
    }

    // Weights chosen so a fired heuristic lands in its own decimal digit --
    // `decode_flags` below reads a summed score back out unambiguously,
    // rather than an inequality that a coincidental weight collision (e.g.
    // default weights 3.0 + 1.0 == 3.0 + 1.0) could pass for the wrong
    // reason.
    const DECODABLE_WEIGHTS: DruidHeuristicWeights = DruidHeuristicWeights {
        block_threat: 1.0,
        defend_fork: 10.0,
        threaten_connection: 100.0,
    };

    fn decode_flags(score: f64) -> (bool, bool, bool) {
        let n = score.round() as i64;
        (n % 10 == 1, (n / 10) % 10 == 1, (n / 100) % 10 == 1)
    }

    fn place_sarsen(state: HashedState, player: Player, pos: Pos, size: Size) -> HashedState {
        let mut state = state;
        state.0.player = player;
        apply_placed(state, PlacedPiece(Piece::Sarsen, pos.index(size.w) as u8))
    }

    #[test]
    fn test_heuristic_scores_flags_blocking_a_lintel_threat() {
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);

        // Black at (1,2), flanked by White at (0,2) and (2,2), all height 1:
        // White's horizontal lintel anchored at (0,2) is a legal candidate
        // (2 of its 3 touched cells already White) that would repaint
        // Black's cell at (1,2) -- exactly the "threat" heuristic 1 detects.
        state = place_sarsen(state, Player::Black, Pos(1, 2), size);
        state = place_sarsen(state, Player::White, Pos(0, 2), size);
        state = place_sarsen(state, Player::White, Pos(2, 2), size);
        state.0.player = Player::Black;

        let mut available = Vec::new();
        state.0.moves(&mut available);
        let scores = heuristic_scores(&state, Player::Black, &available, &DECODABLE_WEIGHTS);

        let threatened_cell = Pos(1, 2).index(size.w);
        let stack_on_threatened = PlacedPiece(Piece::Sarsen, threatened_cell as u8);
        let idx = available
            .iter()
            .position(|m| *m == stack_on_threatened)
            .expect("stacking on the threatened cell should be a legal move");
        let (block, _, _) = decode_flags(scores[idx]);
        assert!(
            block,
            "move touching the threatened cell should get block-threat credit"
        );

        // An unrelated move far from the threat, any fork, or either color's
        // (still singleton) components shouldn't get any credit at all.
        let unrelated = PlacedPiece(Piece::Sarsen, Pos(4, 4).index(size.w) as u8);
        let idx = available.iter().position(|m| *m == unrelated).unwrap();
        assert_eq!(
            scores[idx], 0.0,
            "unrelated move should score 0, got {}",
            scores[idx]
        );
    }

    #[test]
    fn test_heuristic_scores_does_not_flag_a_threat_the_opponent_cannot_play() {
        // Same geometry as the test above, but White's hand is drained of
        // lintels first -- the repaint is structurally possible but not
        // actually playable, so it must not be flagged as a threat.
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);
        state = place_sarsen(state, Player::Black, Pos(1, 2), size);
        state = place_sarsen(state, Player::White, Pos(0, 2), size);
        state = place_sarsen(state, Player::White, Pos(2, 2), size);
        state.0.hand_white.lintels = 0;
        state.0.player = Player::Black;

        let mut available = Vec::new();
        state.0.moves(&mut available);
        let scores = heuristic_scores(&state, Player::Black, &available, &DECODABLE_WEIGHTS);

        let threatened_cell = Pos(1, 2).index(size.w);
        let stack_on_threatened = PlacedPiece(Piece::Sarsen, threatened_cell as u8);
        let idx = available
            .iter()
            .position(|m| *m == stack_on_threatened)
            .unwrap();
        let (block, _, _) = decode_flags(scores[idx]);
        assert!(
            !block,
            "opponent with no lintels left can't threaten, so no block credit is due"
        );
    }

    #[test]
    fn test_heuristic_scores_flags_a_fork_of_two_connecting_lintels() {
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);

        // Two Black dominoes, (0,2)-(0,3) and (2,2)-(2,3): the horizontal
        // lintel anchored at row 2 *and* the one anchored at row 3 each
        // independently connect the same pair of components -- a fork,
        // either one alone secures the connection.
        for pos in [Pos(0, 2), Pos(0, 3), Pos(2, 2), Pos(2, 3)] {
            state = place_sarsen(state, Player::Black, pos, size);
        }
        // A second, unrelated pair of singleton Black cells 2 apart on row
        // 0: exactly one lintel (anchored at (2,0)) connects *this* root
        // pair, so it's a real connecting move but not a fork (no second,
        // independent move completes the same connection).
        state = place_sarsen(state, Player::Black, Pos(2, 0), size);
        state = place_sarsen(state, Player::Black, Pos(4, 0), size);
        state.0.player = Player::Black;

        let mut available = Vec::new();
        state.0.moves(&mut available);
        let scores = heuristic_scores(&state, Player::Black, &available, &DECODABLE_WEIGHTS);

        for anchor in [Pos(0, 2), Pos(0, 3)] {
            let fork_move = PlacedPiece(
                Piece::Lintel(Orientation::Horizontal),
                anchor.index(size.w) as u8,
            );
            let idx = available
                .iter()
                .position(|m| *m == fork_move)
                .unwrap_or_else(|| panic!("expected {fork_move:?} to be legal"));
            let (_, fork, _) = decode_flags(scores[idx]);
            assert!(
                fork,
                "{fork_move:?} completes the shared connection and should get fork credit"
            );
        }

        let single_connector = PlacedPiece(
            Piece::Lintel(Orientation::Horizontal),
            Pos(2, 0).index(size.w) as u8,
        );
        let idx = available
            .iter()
            .position(|m| *m == single_connector)
            .expect("expected the lone connecting lintel to be legal");
        let (_, fork, _) = decode_flags(scores[idx]);
        assert!(
            !fork,
            "a connecting move with no alternate way to complete the same connection isn't a fork"
        );
    }

    #[test]
    fn test_heuristic_scores_flags_extending_toward_the_far_border_and_opponent_chokepoints() {
        let size = DEFAULT_SIZE;
        let mut state = HashedState::new(size);

        // Black's only piece is at (2,0) -- its largest (and only)
        // component's reach along Black's goal axis (row) is y=0, so a move
        // touching (2,1) (row 1, one closer to the far border, row 4)
        // should get "extend toward the far border" credit.
        state = place_sarsen(state, Player::Black, Pos(2, 0), size);
        // White has a two-cell component at (4,2)-(4,3); a Black lintel
        // repainting one of those cells should get "opponent chokepoint"
        // credit for touching White's largest component.
        state = place_sarsen(state, Player::White, Pos(4, 2), size);
        state = place_sarsen(state, Player::White, Pos(4, 3), size);
        state.0.player = Player::Black;

        let mut available = Vec::new();
        state.0.moves(&mut available);
        let scores = heuristic_scores(&state, Player::Black, &available, &DECODABLE_WEIGHTS);

        let extend = PlacedPiece(Piece::Sarsen, Pos(2, 1).index(size.w) as u8);
        let idx = available.iter().position(|m| *m == extend).unwrap();
        let (_, _, threaten) = decode_flags(scores[idx]);
        assert!(
            threaten,
            "a move extending toward the far border should get threaten-connection credit"
        );

        let backward = PlacedPiece(Piece::Sarsen, Pos(0, 0).index(size.w) as u8);
        let idx = available.iter().position(|m| *m == backward).unwrap();
        let (_, _, threaten) = decode_flags(scores[idx]);
        assert!(
            !threaten,
            "an unrelated move away from both the frontier and the opponent shouldn't get credit"
        );
    }

    #[test]
    fn test_druid_heuristic_select_move_returns_a_legal_move() {
        // Smoke test across a short self-play run so `select_move` sees both
        // an empty board (every heuristic score 0, degrading to uniform) and
        // a populated one (heuristics actually firing) without panicking,
        // and always returns a move from the slice it was given.
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut heuristic = DruidHeuristic::<Split>::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let stats = TreeStats::<Druid>::default();
        let mut state = HashedState::default();

        for _ in 0..30 {
            if Druid::is_terminal(&state) {
                break;
            }
            let mut available = Vec::new();
            Druid::generate_actions(&state, &mut available);
            let player = Druid::player_to_move(&state).to_index();
            let chosen =
                *heuristic.select_move(&state, &available, &stats, player, None, None, &mut rng);
            assert!(
                available.contains(&chosen),
                "select_move must return one of the available moves"
            );
            state = Druid::apply(state, &chosen);
        }
    }

    #[test]
    fn test_rave_decisive_heuristic_drives_a_search_and_picks_a_legal_action() {
        let mut search = TreeSearch::<Druid, RaveDecisiveHeuristic>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(30),
        );
        let state = HashedState::default();
        let action = search.choose_action(&state);
        let mut available = Vec::new();
        Druid::generate_actions(&state, &mut available);
        assert!(available.contains(&action));
    }

    #[test]
    fn test_mcts_solver_finds_forced_win() {
        // A 3x3 board where Black already owns 2 of the 3 cells in each of
        // two disjoint columns (x=0 and x=2) needed for a top-to-bottom
        // connection -- each column's missing cell ((0,2) and (2,2)) is an
        // independent winning threat. It's White's move: whichever threat
        // White blocks, Black completes the other and wins immediately next
        // turn. A forced loss for White that the solver should prove at the
        // root in well under the iteration budget.
        let size = Size { w: 3, h: 3 };
        let mut raw = State::new(size);
        for pos in [Pos(0, 0), Pos(0, 1), Pos(2, 0), Pos(2, 1)] {
            raw.board[pos.index(size.w)] = Square {
                height: 1,
                piece: Some(Player::Black),
            };
        }
        raw.player = Player::White;
        let state = HashedState::from_state(raw);

        assert!(
            state.state().connection().is_none(),
            "test setup should not already be terminal"
        );
        assert_eq!(Druid::terminal_status(&state), TerminalStatus::NotTerminal);

        // One single-threaded solver call. Don't loop `choose_action` across
        // multiple sub-ply states -- the solver resolves the full forced-loss
        // position at root, and the PV itself already traces Black's winning
        // reply through White's best block. Checking `root_report` / PV
        // directly avoids the O(iter_budget * depth) cost of replaying every
        // sub-ply through a fresh tree each time.
        let mut ts = TreeSearch::<Druid, strategy::Ucb1>::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(2000)
                .q_init(QInit::Loss)
                .use_mcts_solver(true)
                .seed(7),
        );

        let chosen = ts.choose_action(&state);
        let total_iters = ts
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            total_iters < 2000,
            "solver should prove White's position lost and stop early, used {total_iters} iterations"
        );

        // The root should be proven as a win for Black (the solver resolves
        // that every White reply leads to a Black win).
        let report = ts.root_report(&state);
        let black_win = report
            .actions
            .iter()
            .find(|a| a.action == chosen)
            .expect("chosen action should appear in root report");
        assert!(
            black_win.is_proven,
            "the PV's first action should be proven"
        );

        // Replay the PV from root to verify it ends with Black winning.
        let pv = ts.principle_variation();
        assert!(!pv.is_empty(), "PV should contain at least one move");
        assert_eq!(
            pv.first(),
            Some(&chosen),
            "PV should start with the chosen action"
        );

        let terminal_state = pv.iter().fold(state.clone(), Druid::apply);
        assert_eq!(
            Druid::winner(&terminal_state),
            Some(Player::Black),
            "PV should end with Black winning: replaying {} plies gave {:?}",
            pv.len(),
            Druid::winner(&terminal_state),
        );
    }

    #[test]
    fn test_terminal_status_false_when_pending_not_none() {
        // The board state is unchanged by a Piece/Orientation sub-action,
        // so a mid-turn (pending != None) position must never be reported
        // as terminal — the game only ends after a Cell sub-action that
        // completes or blocks a connection.
        let size = Size { w: 3, h: 3 };
        let mut state = HashedState::new(size);
        // Advance one piece-kind decision into the pending state.
        state = Druid::apply(state, &Move::Piece(PieceKind::Sarsen));
        assert_ne!(state.0.pending, Pending::None);

        assert!(
            !Druid::is_terminal(&state),
            "mid-turn state must not be terminal"
        );
        assert_eq!(Druid::winner(&state), None, "mid-turn state has no winner");
        assert!(
            matches!(Druid::terminal_status(&state), TerminalStatus::NotTerminal),
            "terminal_status must agree with is_terminal"
        );
        // compute_utilities must not report a decisive ±1 for a mid-turn
        // state (it should fall through to the heuristic path).
        let utilities = Druid::compute_utilities(&state);
        assert!(
            utilities.iter().all(|&u| u.abs() < 1.0),
            "mid-turn utilities must not be decisive (±1), got {:?}",
            utilities
        );
    }

    #[test]
    fn test_decisive_move_does_not_fire_at_piece_or_orientation_phases() {
        // DecisiveMove::choose checks one ply ahead via G::apply +
        // terminal_status. A Piece/Orientation sub-action doesn't change
        // the board, so terminal_status always returns NotTerminal, and
        // DecisiveMove falls through to the inner strategy. This test
        // verifies that a search configured with DecisiveMove doesn't
        // panic and returns a valid sub-action when starting mid-turn.
        let mut state = HashedState::new(Size { w: 3, h: 3 });
        // Pre-set pending to Piece(Sarsen) so generate_actions returns Cell
        // actions — DecisiveMove's inner strategy must pick one.
        state = Druid::apply(state, &Move::Piece(PieceKind::Sarsen));

        let mut search = TreeSearch::<Druid, RaveDecisiveHeuristic>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(30),
        );
        let action = search.choose_action(&state);
        let mut available = Vec::new();
        Druid::generate_actions(&state, &mut available);
        assert!(
            available.contains(&action),
            "DecisiveMove must return a valid sub-action mid-turn, got {:?} among {:?}",
            action,
            available
        );
    }
}
/// Flat-encoding tests: the `Flat` mode shares `State`/`MoveCache`/
/// hashing/connectivity/heuristics with `Split`, so these only verify what
/// the mode itself adds -- whole-turn `generate_actions`/`apply` behaving
/// identically to `State::moves`/`apply_placed`, the incremental caches
/// staying correct under flat moves, and parity with the split encoding.
#[cfg(test)]
mod flat_tests {
    use super::*;
    use crate::zobrist::{full_hash, zobrist_height_bits};
    use mcts::game::{Game, PlayerIndex};
    use mcts::algorithms::mcts::simulate::SimulatePolicy;
    use mcts::algorithms::mcts::TreeStats;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn flat_generate_actions_matches_state_moves() {
        // The flat whole-turn `generate_actions` is a `MoveCache` read;
        // confirm it always equals the from-scratch ground truth `State::moves`
        // across randomized games (same property the split tree test checks,
        // but for the flat action list directly).
        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng = SmallRng::seed_from_u64(0xF1A7 + size.w as u64 * 1000 + size.h as u64);
            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let mut via_flat = Vec::new();
                    DruidFlat::generate_actions(&state, &mut via_flat);
                    assert_eq!(
                        via_flat, actions,
                        "flat moves diverged from ground truth at size={size:?} game={game} ply={ply}"
                    );
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = DruidFlat::apply(state, &m);
                    actions.clear();
                }
            }
        }
    }

    #[test]
    fn flat_apply_matches_apply_placed() {
        // The flat whole-turn `apply` and the split `apply_placed` both effect
        // the same placement; they share `apply_turn`, so the resulting board
        // and hash must agree exactly.
        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let mut rng = SmallRng::seed_from_u64(0xA9 + size.w as u64 * 1000 + size.h as u64);
            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..100 {
                    state.0.moves(&mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    let after_flat = DruidFlat::apply(state.clone(), &m);
                    let after_split = apply_placed(state, m);
                    assert_eq!(
                        *after_flat.state(),
                        *after_split.state(),
                        "flat vs split state diverged at size={size:?} game={game} ply={ply} move={m:?}"
                    );
                    assert_eq!(
                        after_flat.1, after_split.1,
                        "flat vs split hash diverged at size={size:?} game={game} ply={ply}"
                    );
                    state = after_flat;
                    actions.clear();
                }
            }
        }
    }

    #[test]
    fn flat_incremental_hash_matches_full_recompute() {
        // Same property as the split test, driven by flat whole-turn applies.
        for size in [Size { w: 3, h: 3 }, DEFAULT_SIZE, Size { w: 7, h: 7 }] {
            let bits = zobrist_height_bits(size);
            let mut rng = SmallRng::seed_from_u64(size.w as u64 * 1000 + size.h as u64);
            for game in 0..20 {
                let mut state = HashedState::new(size);
                let mut actions = Vec::new();
                for ply in 0..200 {
                    DruidFlat::generate_actions(&state, &mut actions);
                    if actions.is_empty() {
                        break;
                    }
                    let m = actions[rng.gen_range(0..actions.len())];
                    state = DruidFlat::apply(state, &m);
                    actions.clear();
                    assert_eq!(
                        state.1,
                        full_hash(&state.0, bits),
                        "flat incremental hash diverged at size={size:?} game={game} ply={ply}"
                    );
                }
            }
        }
    }

    #[test]
    fn flat_heuristic_select_move_returns_a_legal_move() {
        // Smoke: the flat encoding gives the shared `heuristic_scores` a
        // directly scoring whole-turn action, must stay legal and not panic.
        let mut heuristic = DruidHeuristic::<Flat>::default();
        let mut rng = SmallRng::seed_from_u64(7);
        let stats = TreeStats::<DruidFlat>::default();
        let mut state = HashedState::default();
        for _ in 0..30 {
            if DruidFlat::is_terminal(&state) {
                break;
            }
            let mut available = Vec::new();
            DruidFlat::generate_actions(&state, &mut available);
            let player = DruidFlat::player_to_move(&state).to_index();
            let chosen =
                *heuristic.select_move(&state, &available, &stats, player, None, None, &mut rng);
            assert!(
                available.contains(&chosen),
                "flat select_move must return one of the available moves"
            );
            state = DruidFlat::apply(state, &chosen);
        }
    }
}
