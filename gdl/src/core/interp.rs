//! A tree-walking evaluator that binds a Core IR [`super::Program`] directly to a concrete
//! `Rect`-shaped board (`game_core::bitboard::BitBoard<N, M>`), rather than compiling it to Rust
//! source. Per `DESIGN.md`'s bootstrap order ("Interpret Core, don't codegen yet"), this is
//! deliberately the slow, obviously-correct path -- an oracle to check codegen against later,
//! not a performance target itself.
//!
//! The caller picks `N`/`M` to match `program.topology` (there is currently no dynamic-topology
//! `BitBoard`, so this only works for a topology known at the call site -- see this module's own
//! tests and the oracle tests in `tests/` for concrete examples).

use game_core::bitboard::BitBoard;

use super::{BoolExpr, Connectivity, Direction, Player, Program, Region, Topology};

/// `region` shifted one step in `dir` -- the backend realization of [`Region::Shift`]/DESIGN.md's
/// `shift(dir): Region -> Region`. Each arm is a direct, unmodified call into an existing, proven
/// `BitBoard::shift_*` method (`game_core::bitboard::BitBoard`'s "Board displacement" section) --
/// this function adds no new bit-twiddling of its own, it only gives the Core IR a name for what
/// already exists.
fn shift<const N: usize, const M: usize>(region: BitBoard<N, M>, dir: Direction) -> BitBoard<N, M> {
    match dir {
        Direction::North => region.shift_north(),
        Direction::East => region.shift_east(),
        Direction::South => region.shift_south(),
        Direction::West => region.shift_west(),
        Direction::Northeast => region.shift_northeast(),
        Direction::Northwest => region.shift_northwest(),
        Direction::Southeast => region.shift_southeast(),
        Direction::Southwest => region.shift_southwest(),
    }
}

/// The cells adjacent to (but not inside) `region`, under `conn`-adjacency -- the backend
/// realization of [`Region::Adjacent`]/DESIGN.md's `adjacent(conn): Region -> Region`, and the
/// per-iteration step [`bounded_fixpoint`] uses to realize [`Region::Flood`].
///
/// Every direction's shift is computed from the same input `region` and OR'd together in one
/// expression, deliberately -- splitting this across multiple statements (each shift computed
/// from the *previous* statement's already-shifted result) is exactly the latent bug
/// `game_core::bitboard::BitBoard::flood6`'s doc comment documents: a compound shift can bridge
/// through a cell that isn't actually `conn`-adjacent to the original region. Folding over a
/// direction list computed from the one unmodified `region` value structurally can't reintroduce
/// that bug, unlike a hand-written sequence of `|=` statements.
fn adjacent<const N: usize, const M: usize>(
    region: BitBoard<N, M>,
    conn: Connectivity,
) -> BitBoard<N, M> {
    let dirs: &[Direction] = match conn {
        Connectivity::Four => &[
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ],
        Connectivity::Six => &[
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
            Direction::Northeast,
            Direction::Southwest,
        ],
        Connectivity::Eight => &[
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
            Direction::Northeast,
            Direction::Northwest,
            Direction::Southeast,
            Direction::Southwest,
        ],
    };
    dirs.iter()
        .fold(BitBoard::EMPTY, |acc, &dir| acc | shift(region, dir))
        & !region
}

/// A bounded fixpoint/trace: iterates `step` from `(seed, aux_seed)`, unioning the `Region`-valued
/// half into an accumulator (masked to stay within `bound`) until it stops growing or `max_iters`
/// is reached -- DESIGN.md's "Categorical structure" section (`bounded_fixpoint` as a bounded
/// trace, `Tr: Hom(A x X, B x X) -> Hom(A, B)`). `max_iters` is always safe to set to `bound`'s
/// cell count: `COMPLETENESS.md` classifies this as "LFP, directly" precisely because a monotone
/// operator over subsets of an `n`-element domain is guaranteed to stabilize within `n`
/// iterations, the same fact `DESIGN.md` already cites to justify `max_iters` being statically
/// board-size-derivable rather than an independently-chosen safety margin.
///
/// Generic over an auxiliary threaded state `Aux` so this one node shape can also hold `has_cycle`
/// -- README.md's design-spike case 5, and `COMPLETENESS.md`'s "LFP, simultaneous" classification
/// -- whose threaded state is `(visited: Region, parent: Raster<Direction>, cycle: Bool)`, not
/// bare `Region`. [`Region::Flood`] is this shape's `Aux = ()` instantiation, the only one wired
/// into `Program`/`eval_region` this session; `has_cycle_shape_holds_a_parent_and_cycle_flag`
/// below confirms (by actually compiling and running a non-`()` `Aux` that threads a parent map
/// and a cycle flag through the same function) that the shape genuinely generalizes, rather than
/// asserting it in prose. Landing `has_cycle` itself as a `Region`/`BoolExpr`/`Program` primitive
/// is future work -- see `README.md`'s session note and `DESIGN.md`'s "Worked pass" table.
fn bounded_fixpoint<Aux, const N: usize, const M: usize>(
    seed: BitBoard<N, M>,
    bound: BitBoard<N, M>,
    aux_seed: Aux,
    max_iters: usize,
    mut step: impl FnMut(BitBoard<N, M>, Aux) -> (BitBoard<N, M>, Aux),
) -> (BitBoard<N, M>, Aux) {
    let mut state = seed & bound;
    let mut aux = aux_seed;
    for _ in 0..max_iters {
        let (next, next_aux) = step(state, aux);
        let next = next & bound;
        aux = next_aux;
        if next == state {
            return (state, aux);
        }
        state = next;
    }
    (state, aux)
}

/// The `conn`-connected component(s) of `bound` reachable from `seed` -- the backend realization
/// of [`Region::Flood`]/DESIGN.md's `flood(seed, conn): Region -> Region`, and
/// [`bounded_fixpoint`]'s `Aux = ()` instantiation (`step` just unions in `adjacent`). This
/// replaces what was previously a direct, non-composable call to
/// `game_core::bitboard::BitBoard::flood6` in `State::winner` -- same underlying bit operations
/// (`adjacent`'s `Connectivity::Six` arm ORs the identical six shifts `flood6` did), now expressed
/// as a real Region-algebra combinator instead of a bespoke per-arity `BitBoard` method call.
fn flood<const N: usize, const M: usize>(
    bound: BitBoard<N, M>,
    seed: BitBoard<N, M>,
    conn: Connectivity,
) -> BitBoard<N, M> {
    bounded_fixpoint(seed, bound, (), bound.count_ones() as usize, |state, ()| {
        (state | adjacent(state, conn), ())
    })
    .0
}

fn eval_region<const N: usize, const M: usize>(
    region: &Region,
    occupied: &[BitBoard<N, M>],
) -> BitBoard<N, M> {
    match region {
        Region::Occupied(Player(i)) => occupied[*i],
        Region::Union(a, b) => eval_region(a, occupied) | eval_region(b, occupied),
        Region::Complement(a) => !eval_region(a, occupied),
        Region::Intersect(a, b) => eval_region(a, occupied) & eval_region(b, occupied),
        Region::Sites(sites) => sites
            .iter()
            .fold(BitBoard::EMPTY, |acc, &s| acc | BitBoard::from_index(s)),
        Region::Shift { region, dir } => shift(eval_region(region, occupied), *dir),
        Region::Adjacent { region, conn } => adjacent(eval_region(region, occupied), *conn),
        Region::Flood { region, seed, conn } => flood(
            eval_region(region, occupied),
            eval_region(seed, occupied),
            *conn,
        ),
    }
}

/// Evaluates a [`BoolExpr`] against `board` (the region under test -- `State::winner` always
/// passes the mover's own occupied region, matching what `EndRule`'s doc comment promises).
/// `edges` is `Some(list)` from `Program.player_regions[mover]` when the program declares any --
/// required by `BoolExpr::Connects` (which floods from `edges[0]` and checks the result
/// intersects every remaining entry), unused otherwise; see that variant's doc comment in
/// `core::mod` for why the list is threaded in from the caller rather than embedded in the
/// expression.
fn eval_bool<const N: usize, const M: usize>(
    expr: &BoolExpr,
    board: BitBoard<N, M>,
    edges: Option<&[BitBoard<N, M>]>,
    occupied: &[BitBoard<N, M>],
) -> bool {
    match expr {
        BoolExpr::Contains(sites) => eval_region(sites, occupied).is_subset(board),
        BoolExpr::Connects { conn } => {
            let edges = edges.expect("BoolExpr::Connects requires Program.player_regions");
            let [first, rest @ ..] = edges else {
                panic!("BoolExpr::Connects requires at least one player region");
            };
            let flooded = flood(board, *first, *conn);
            rest.iter().all(|&edge| flooded.intersects(edge))
        }
        BoolExpr::Any(exprs) => exprs.iter().any(|e| eval_bool(e, board, edges, occupied)),
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
        let edges: Option<Vec<_>> = program.player_regions.get(last_mover).map(|regions| {
            regions
                .iter()
                .map(|r| eval_region(r, &self.occupied))
                .collect()
        });
        program
            .end
            .iter()
            .any(|rule| eval_bool(&rule.condition, board, edges.as_deref(), &self.occupied))
            .then_some(last_mover)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::hex::HexShape;
    use crate::core::{EndRule, Hex, MoveGen, Rect};
    use crate::style_c::parse_game;
    use std::collections::HashMap;

    fn tic_tac_toe_program() -> Program {
        parse_game(include_str!("../../style-c/sexpr/tic-tac-toe.gdls")).unwrap()
    }

    fn hex_program() -> Program {
        parse_game(include_str!("../../style-c/sexpr/hex.gdls")).unwrap()
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
    fn manual_program_matches_parsed_one() {
        // Core IR should be constructible and checkable by hand, independent of any parser --
        // this pins down that a hand-built Program behaves identically to the parsed one.
        let rect = Rect { rows: 3, cols: 3 };
        let manual = Program {
            topology: Topology::Rect(rect),
            num_players: 2,
            move_gen: MoveGen {
                to: Region::Complement(Box::new(Region::Union(
                    Box::new(Region::Occupied(Player(0))),
                    Box::new(Region::Occupied(Player(1))),
                ))),
            },
            end: vec![EndRule {
                condition: BoolExpr::Any(
                    rect.lines(3)
                        .into_iter()
                        .map(|line| BoolExpr::Contains(Region::Sites(line)))
                        .collect(),
                ),
            }],
            player_regions: Vec::new(),
        };
        assert_eq!(manual, tic_tac_toe_program());
    }

    #[test]
    fn manual_hex_program_matches_parsed_one() {
        let manual = Program {
            topology: Topology::Hex(Hex {
                side: 3,
                shape: HexShape::Rhombus,
            }),
            num_players: 2,
            move_gen: MoveGen {
                to: Region::Complement(Box::new(Region::Union(
                    Box::new(Region::Occupied(Player(0))),
                    Box::new(Region::Occupied(Player(1))),
                ))),
            },
            end: vec![EndRule {
                condition: BoolExpr::Connects {
                    conn: Connectivity::Six,
                },
            }],
            player_regions: vec![
                vec![Region::Sites(vec![6, 7, 8]), Region::Sites(vec![0, 1, 2])],
                vec![Region::Sites(vec![0, 3, 6]), Region::Sites(vec![2, 5, 8])],
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

    // -- Direct, hand-built tests of the new Region-algebra combinators themselves (not routed
    // through any particular game's Program), per "Core IR should be constructible and checkable
    // by hand" -- see `core::mod`'s module doc.

    #[test]
    fn region_shift_moves_a_single_site() {
        // A 3x3 board, one stone at the center (site 4, row 1 col 1). Shifting it north should
        // land on site 7 (row 2 col 1) -- matches BitBoard's row-major/bottom-left-origin
        // indexing, the same convention `Rect`/`Hex` already use.
        let occupied = vec![BitBoard::<3, 3>::from_index(4)];
        let shifted = Region::Shift {
            region: Box::new(Region::Occupied(Player(0))),
            dir: Direction::North,
        };
        assert_eq!(
            eval_region(&shifted, &occupied),
            BitBoard::<3, 3>::from_index(7)
        );
    }

    #[test]
    fn region_shift_northwest_matches_direct_bitboard_call() {
        // A diagonal direction, to exercise the four `Eight`-only Direction variants (not just
        // the four cardinal ones every other test already covers via Rect's lines/Hex's edges).
        let bits = BitBoard::<3, 3>::from_index(4);
        let occupied = vec![bits];
        let shifted = Region::Shift {
            region: Box::new(Region::Occupied(Player(0))),
            dir: Direction::Northwest,
        };
        assert_eq!(eval_region(&shifted, &occupied), bits.shift_northwest());
    }

    #[test]
    fn region_adjacent_four_excludes_the_region_itself() {
        // Center of a 3x3 board, four-way adjacency: the four orthogonal neighbors (1, 3, 5, 7),
        // never the seed site (4) itself even though a shift-and-mask could otherwise re-include
        // it on a board with wraparound.
        let occupied = vec![BitBoard::<3, 3>::from_index(4)];
        let adj = Region::Adjacent {
            region: Box::new(Region::Occupied(Player(0))),
            conn: Connectivity::Four,
        };
        let expected = [1usize, 3, 5, 7]
            .into_iter()
            .fold(BitBoard::<3, 3>::EMPTY, |acc, s| {
                acc | BitBoard::from_index(s)
            });
        assert_eq!(eval_region(&adj, &occupied), expected);
    }

    #[test]
    fn region_adjacent_eight_includes_diagonals() {
        let occupied = vec![BitBoard::<3, 3>::from_index(4)];
        let adj = Region::Adjacent {
            region: Box::new(Region::Occupied(Player(0))),
            conn: Connectivity::Eight,
        };
        // Every other site on a 3x3 board is a queen-move neighbor of the center.
        let expected = (0..9)
            .filter(|&s| s != 4)
            .fold(BitBoard::<3, 3>::EMPTY, |acc, s| {
                acc | BitBoard::from_index(s)
            });
        assert_eq!(eval_region(&adj, &occupied), expected);
    }

    #[test]
    fn region_flood_matches_direct_flood6_call() {
        // The exact scenario `hex_p2_wins_by_connecting_west_and_east_edges_via_diagonal` above
        // already proves end-to-end through a real Program -- this pins the same fact down
        // directly against `Region::Flood`/`eval_region`, independent of any parser, and checks
        // it against a direct `BitBoard::flood6` call (the code path `Region::Flood`
        // replaced in `State::winner`).
        let occupied = vec![
            BitBoard::<3, 3>::EMPTY,
            BitBoard::<3, 3>::from_index(0) | BitBoard::from_index(4) | BitBoard::from_index(8),
        ];
        let board = occupied[1];
        let seed = BitBoard::<3, 3>::from_index(0);
        let flooded = Region::Flood {
            region: Box::new(Region::Occupied(Player(1))),
            seed: Box::new(Region::Sites(vec![0])),
            conn: Connectivity::Six,
        };
        assert_eq!(eval_region(&flooded, &occupied), board.flood6(seed));
    }

    #[test]
    fn bool_expr_any_short_circuits_on_the_first_true_contains() {
        let occupied = vec![BitBoard::<3, 3>::from_index(0) | BitBoard::from_index(1)];
        let expr = BoolExpr::Any(vec![
            BoolExpr::Contains(Region::Sites(vec![0, 1])),
            BoolExpr::Contains(Region::Sites(vec![5])), // not present -- must not affect the result
        ]);
        assert!(eval_bool(&expr, occupied[0], None, &occupied));
    }

    #[test]
    fn region_intersect_keeps_only_sites_in_both_operands() {
        let occupied: Vec<BitBoard<3, 3>> = Vec::new();
        let expr = Region::Intersect(
            Box::new(Region::Sites(vec![0, 1, 2, 3])),
            Box::new(Region::Sites(vec![2, 3, 4, 5])),
        );
        let expected = BitBoard::<3, 3>::from_index(2) | BitBoard::from_index(3);
        assert_eq!(eval_region(&expr, &occupied), expected);
    }

    #[test]
    fn bool_expr_connects_requires_touching_every_declared_edge() {
        // A three-edge Connects (Y's shape, not Hex's): a group spanning only two of the three
        // declared edges must not satisfy it, even though it would satisfy a two-edge Connects
        // over the same pair. Board: edges are three sides of a 3x3 grid used as stand-ins (top
        // row, left column, bottom row) -- not an actual triangle, just enough to exercise arity.
        let top =
            BitBoard::<3, 3>::from_index(6) | BitBoard::from_index(7) | BitBoard::from_index(8);
        let left =
            BitBoard::<3, 3>::from_index(0) | BitBoard::from_index(3) | BitBoard::from_index(6);
        let bottom =
            BitBoard::<3, 3>::from_index(0) | BitBoard::from_index(1) | BitBoard::from_index(2);
        let expr = BoolExpr::Connects {
            conn: Connectivity::Six,
        };

        // {8, 4}: a northeast/southwest-diagonal chain touching only `top` (site 8) -- never
        // reaches `left` or `bottom`. Must fail the three-edge Connects even though it trivially
        // satisfies a (degenerate) one-edge Connects over `top` alone.
        let two_edges_only = vec![BitBoard::<3, 3>::from_index(8) | BitBoard::from_index(4)];
        assert!(!eval_bool(
            &expr,
            two_edges_only[0],
            Some(&[top, left, bottom]),
            &two_edges_only
        ));
        assert!(eval_bool(
            &expr,
            two_edges_only[0],
            Some(&[top]),
            &two_edges_only
        ));

        // Extending the same chain down to site 0 (shared by both `left` and `bottom`) now
        // touches all three declared edges from one connected component.
        let all_three = vec![
            BitBoard::<3, 3>::from_index(8) | BitBoard::from_index(4) | BitBoard::from_index(0),
        ];
        assert!(eval_bool(
            &expr,
            all_three[0],
            Some(&[top, left, bottom]),
            &all_three
        ));
    }

    /// Confirms `bounded_fixpoint`'s shape -- see that function's doc comment -- actually holds
    /// `has_cycle`'s simultaneous multi-relation induction (README.md's design-spike case 5): a
    /// `visited: Region` half (the fixpoint's own `state`) plus an `Aux = (parent map, cycle
    /// flag)` half, threaded together across iterations rather than three separate loops. This is
    /// a real, hand-verifiable instance of that algorithm (not a stub): a 2x2 board under
    /// four-way adjacency is itself a 4-cycle (0-1-3-2-0, since every cell has exactly two
    /// neighbors), so flooding from any seed and flagging a site reached from two *distinct*
    /// already-visited neighbors in the same round (the batched form of README.md's `not
    /// visited(S2, _)`-guarded Datalog rule) must detect it; a 3-cell sub-board with the same
    /// seed (a path, not a cycle) is the negative control. `has_cycle` itself is not landed as a
    /// `Region`/`BoolExpr`/`Program` primitive this session -- see `README.md`'s session note.
    fn detect_cycle<const N: usize, const M: usize>(
        bound: BitBoard<N, M>,
        seed: usize,
    ) -> (bool, HashMap<usize, Direction>) {
        let dirs = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ];
        let (_, (parent, cycle)) = bounded_fixpoint::<(HashMap<usize, Direction>, bool), N, M>(
            BitBoard::from_index(seed),
            bound,
            (HashMap::new(), false),
            bound.count_ones() as usize,
            |visited, (mut parent, mut cycle)| {
                let candidates = adjacent(visited, Connectivity::Four) & bound & !visited;
                let mut next = visited;
                for s2 in candidates {
                    let mut parent_dirs = Vec::new();
                    for &dir in &dirs {
                        if (shift(BitBoard::<N, M>::from_index(s2), dir) & visited)
                            != BitBoard::EMPTY
                        {
                            parent_dirs.push(dir);
                        }
                    }
                    if parent_dirs.len() >= 2 {
                        cycle = true;
                    }
                    if let Some(&dir) = parent_dirs.first() {
                        parent.insert(s2, dir);
                    }
                    next.set(s2);
                }
                (next, (parent, cycle))
            },
        );
        (cycle, parent)
    }

    #[test]
    fn has_cycle_shape_holds_a_parent_and_cycle_flag() {
        let full_board = BitBoard::<2, 2>::ONES;
        let (cycle, parent) = detect_cycle(full_board, 0);
        assert!(cycle, "a 2x2 board under 4-way adjacency is a 4-cycle");
        // The `Aux`-threaded parent map survives the fixpoint and is retrievable -- not just the
        // cycle flag -- confirming both halves of the simultaneous state are real, not vestigial.
        assert_eq!(parent.get(&1), Some(&Direction::West));
        assert_eq!(parent.get(&2), Some(&Direction::South));

        let path_only =
            BitBoard::<2, 2>::from_index(0) | BitBoard::from_index(1) | BitBoard::from_index(2);
        let (cycle, _) = detect_cycle(path_only, 0);
        assert!(!cycle, "a 3-cell sub-board (a path) has no cycle");
    }
}
