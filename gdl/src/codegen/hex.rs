//! `Topology::Hex` codegen: lowers a `Program` into the text of a standalone `games/*`-shaped
//! Rust crate, on the same `side x side` grid `core::interp::State`/`core::hex::Hex` already use
//! for a Rhombus board -- see this module's own doc comment in `mod.rs` for the scope this
//! backend covers.
//!
//! Unlike `rect::generate` (which bakes one concrete board size into the generated types), this
//! backend emits a *const-generic* `Position<const N: usize, const WORDS: usize>` (plus
//! `HashedPosition`/the `Game`-implementing marker struct) parameterized over board side `N` and
//! `bitboard::Board<[u64; WORDS], Const<N>, Const<N>>`'s own `WORDS` parameter -- the same
//! `<const N, const WORDS>` shape `games/gonnect` already hand-writes for its own multiple board
//! sizes (9/13/19).
//! A single generated crate can then be instantiated at several concrete sizes from one hand-
//! written `main.rs` dispatch (`games/hex-gen/src/main.rs`'s `dispatch_size!` macro, mirroring
//! `games/gonnect/src/main.rs`'s), rather than needing one generated crate per board size.
//!
//! The board type is always `Board<[u64; WORDS], Const<N>, Const<N>>` (the multi-word storage
//! backend), never `rect::generate`'s single-word `Board<u64, Const<N>, Const<M>>`: a single
//! generic function body has to work for every instantiated `N`, and `[u64; WORDS]` is the one
//! storage backend that's already generic over its own word count. `WORDS` is a genuine second
//! const-generic parameter (not derivable from `N` on stable Rust -- see `bitboard::Storage`'s
//! own doc comment), supplied by whichever concrete instantiation a caller picks.
//!
//! The real consequence of going const-generic: a `Region::Sites` region (`Program`'s `(regions
//! ...)` edges) can no longer lower to a literal bitmask, since one generic function body has to
//! produce the right mask for *every* instantiated `N`, not just the one `N` the source `.gdls`
//! file happened to name. Instead, each `Region::Sites` list is matched (via `identify_edge`)
//! against what `core::hex::Hex::edge` computes for each of the four compass edges at that
//! `Program`'s own concrete `side`; a match lowers to a call to a generated `side_north`/
//! `side_south`/`side_east`/`side_west` function -- a small generic formula (a loop setting `N`
//! bits), not a lookup table -- instead of a literal. A `Region::Sites` that isn't one of those
//! four edges has no generic lowering and is rejected, same scoping discipline as `rect`/`hex`
//! already apply to `Intersect`/`Shift`/`Adjacent`/`Flood`.
//!
//! Only lowers what a Rhombus-shaped Hex-topology `Program` actually needs: the same
//! `Region::Occupied`/`Union`/`Complement`/`Sites` shapes `rect` already lowers (Hex's own
//! `(sites Empty)` move generator and `(regions ...)` edges are built only from those), plus
//! `BoolExpr::Connects` itself, specialized to `Connectivity::Six` via a generated `hex_connects`
//! helper that calls `Board::flood6` directly. `HexShape::Triangle` (Y) is out of scope --
//! see `mod.rs`.
//!
//! Zobrist hashing also had to change shape: a `static LazyZobristTable<NUM_HASHES>` needs
//! `NUM_HASHES` fixed at the item level, but `NUM_HASHES` would need to depend on the generic `N`
//! -- an array length depending on another const generic isn't expressible on stable Rust (the
//! same limitation that forces `WORDS` to be its own parameter rather than computed from `N`).
//! `HashedPosition::compute_hash` recomputes an FNV-1a-style hash from `Position`'s raw board
//! words on every `apply` instead -- non-incremental, but `WORDS` is always tiny (1-2 for every
//! board size this backend targets), so the recomputation is nowhere near a hot-path cost.

use crate::core::hex::{Edge, HexShape};
use crate::core::{BoolExpr, Connectivity, EndRule, Hex, Player, Program, Region};

use super::{fnv1a, Error};

/// If `region` is exactly the site list `core::hex::Hex::edge` computes for one of the four
/// compass edges (at `hex`'s own concrete `side`), the generated function name that computes the
/// same edge generically over any `N`. `None` if `region` isn't a `Region::Sites` or doesn't
/// match any edge -- see this module's doc comment for why there's no literal-mask fallback.
fn identify_edge(region: &Region, hex: &Hex) -> Option<&'static str> {
    let Region::Sites(sites) = region else {
        return None;
    };
    let mut sites = sites.clone();
    sites.sort_unstable();
    let candidates = [
        ("side_north", Edge::North),
        ("side_south", Edge::South),
        ("side_east", Edge::East),
        ("side_west", Edge::West),
    ];
    candidates.into_iter().find_map(|(name, edge)| {
        let mut expected = hex.edge(edge);
        expected.sort_unstable();
        (expected == sites).then_some(name)
    })
}

/// The generic Rust source of the named edge function (`identify_edge`'s return value), mirroring
/// `core::hex::Hex::edge`'s own formula for that `Edge` variant exactly, just expressed as a loop
/// over the generic `N` instead of a `Vec<usize>` built from a concrete `side`.
fn edge_fn_source(name: &str) -> String {
    let body = match name {
        "side_north" => "(N - 1) * N + c",
        "side_south" => "c",
        "side_west" => "c * N",
        "side_east" => "c * N + (N - 1)",
        _ => unreachable!("identify_edge only returns these four names"),
    };
    format!(
        r#"fn {name}<const N: usize, const WORDS: usize>() -> Board<[u64; WORDS], Const<N>, Const<N>> {{
    let mut b = Board::new_const();
    for c in 0..N {{
        b.set_index({body});
    }}
    b
}}
"#
    )
}

/// Lowers `region` into a Rust expression of type `Board<[u64; WORDS], Const<N>, Const<N>>`,
/// evaluated relative to `self.occupied: [Board<[u64; WORDS], Const<N>, Const<N>>; NUM_PLAYERS]` -- see `rect::region_expr`'s
/// identical doc comment for why this stays a separate copy rather than shared with `rect`'s.
fn region_expr(region: &Region, hex: &Hex) -> Result<String, Error> {
    Ok(match region {
        Region::Occupied(Player(i)) => format!("self.occupied[{i}]"),
        Region::Union(a, b) => format!("({} | {})", region_expr(a, hex)?, region_expr(b, hex)?),
        Region::Complement(a) => format!("!{}", region_expr(a, hex)?),
        Region::Sites(_) => match identify_edge(region, hex) {
            Some(name) => format!("{name}::<N, WORDS>()"),
            None => {
                return Err(Error(format!(
                    "codegen::hex: {region:?} is a Region::Sites that isn't one of Hex's four \
                     named compass edges -- the const-generic codegen backend can only lower a \
                     Sites region it can express as a formula over N (see this module's doc \
                     comment), and no corpus Hex game needs anything else yet"
                )));
            }
        },
        Region::Intersect(..)
        | Region::Shift { .. }
        | Region::Adjacent { .. }
        | Region::Flood { .. } => {
            return Err(Error(format!(
                "codegen::hex: {region:?} has no lowering yet -- no Rhombus-shaped Hex-topology \
                 corpus game routed through codegen needs it yet (a Triangle board's \
                 (sites Empty) masking is the next real forcing case for `Intersect`, per \
                 DESIGN.md's \"grow from real lowerings\")"
            )));
        }
    })
}

/// True if `expr` (or anything nested inside an `Any`) is a `BoolExpr::Connects` -- used to decide
/// whether the generated source needs the `hex_connects` helper and its `edges` lookup table at
/// all, so a hypothetical Hex-topology game whose end rule is pure `Contains` doesn't get dead
/// code that would trip this workspace's `-D warnings`.
fn contains_connects(expr: &BoolExpr) -> bool {
    match expr {
        BoolExpr::Connects { .. } => true,
        BoolExpr::Contains(_) => false,
        BoolExpr::Any(exprs) => exprs.iter().any(contains_connects),
    }
}

/// Lowers `expr` into a `bool`-typed Rust expression. `board` is the (already-lowered) name of
/// the region-under-test variable in scope; `edges` is `Some(name)` of an in-scope
/// `&[Board<[u64; WORDS], Const<N>, Const<N>>]` binding (`Program.player_regions[last_mover]`, lowered by
/// `generate` below) when the program declares any, required by `BoolExpr::Connects`.
fn bool_expr(
    expr: &BoolExpr,
    board: &str,
    edges: Option<&str>,
    hex: &Hex,
) -> Result<String, Error> {
    Ok(match expr {
        BoolExpr::Contains(region) => {
            format!("({}).is_subset({board})", region_expr(region, hex)?)
        }
        BoolExpr::Any(exprs) => {
            if exprs.is_empty() {
                "false".to_string()
            } else {
                let parts = exprs
                    .iter()
                    .map(|e| bool_expr(e, board, edges, hex))
                    .collect::<Result<Vec<_>, _>>()?;
                parts.join(" || ")
            }
        }
        BoolExpr::Connects { conn } => {
            if *conn != Connectivity::Six {
                return Err(Error(format!(
                    "codegen::hex: Connects{{conn: {conn:?}}} has no lowering yet -- every \
                     corpus Hex-topology game's end rule uses Six-connectivity (the topology's \
                     own adjacency), the only one the generated `hex_connects` helper implements"
                )));
            }
            let edges = edges.ok_or_else(|| {
                Error(
                    "codegen::hex: BoolExpr::Connects requires Program.player_regions, which \
                     this program doesn't declare"
                        .into(),
                )
            })?;
            format!("hex_connects({board}, {edges})")
        }
    })
}

fn end_expr(end: &[EndRule], board: &str, edges: Option<&str>, hex: &Hex) -> Result<String, Error> {
    if end.is_empty() {
        return Ok("false".to_string());
    }
    let parts = end
        .iter()
        .map(|rule| bool_expr(&rule.condition, board, edges, hex))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(parts.join(" || "))
}

pub fn generate(
    game_name: &str,
    struct_name: &str,
    source_path: &str,
    hex: Hex,
    program: &Program,
) -> Result<String, Error> {
    if hex.shape != HexShape::Rhombus {
        return Err(Error(
            "codegen::hex: only HexShape::Rhombus is supported so far -- a Triangle board (Y) \
             additionally needs Region::Intersect to mask (sites Empty) down to the triangular \
             half of the grid, not attempted yet"
                .into(),
        ));
    }
    if program.num_players == 0 {
        return Err(Error(
            "codegen::hex: a game needs at least one player".into(),
        ));
    }
    if program.num_players > 26 {
        // `display_char_at` below spends one letter A..Z per player -- no corpus game is
        // anywhere close to this, so a real per-player display scheme isn't worth building yet.
        return Err(Error(format!(
            "codegen::hex: {} players exceeds the 26-player display-char scheme",
            program.num_players
        )));
    }

    let side = hex.side;
    let words = (side * side).div_ceil(64);
    let num_players = program.num_players;
    let seed = fnv1a(struct_name);

    let needs_connects = program
        .end
        .iter()
        .any(|rule| contains_connects(&rule.condition));
    if needs_connects && program.player_regions.is_empty() {
        return Err(Error(
            "codegen::hex: an end rule uses Connects but Program.player_regions is empty".into(),
        ));
    }
    if !program.player_regions.is_empty() && program.player_regions.len() != num_players {
        return Err(Error(format!(
            "codegen::hex: Program.player_regions has {} entries for {num_players} players",
            program.player_regions.len()
        )));
    }

    let move_expr = region_expr(&program.move_gen.to, &hex)?;
    let edges_var = needs_connects.then_some("edges");
    let win_expr = end_expr(&program.end, "board", edges_var, &hex)?;

    let player_variants: String = (0..num_players).map(|i| format!("    P{i},\n")).collect();
    let to_index_arms: String = (0..num_players)
        .map(|i| format!("            Player::P{i} => {i},\n"))
        .collect();
    let from_index_arms: String = (0..num_players)
        .map(|i| format!("            {i} => Player::P{i},\n"))
        .collect();

    let mut used_edges: Vec<&'static str> = Vec::new();
    let edges_arms: String = if needs_connects {
        program
            .player_regions
            .iter()
            .enumerate()
            .map(|(player, regions)| {
                let items = regions
                    .iter()
                    .map(|r| {
                        let expr = region_expr(r, &hex)?;
                        if let Some(name) = identify_edge(r, &hex) {
                            if !used_edges.contains(&name) {
                                used_edges.push(name);
                            }
                        }
                        Ok(expr)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(format!(
                    "            {player} => &[{}],\n",
                    items.join(", ")
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .join("")
    } else {
        String::new()
    };

    let edges_binding = if needs_connects {
        format!(
            r#"        let edges: &[Board<[u64; WORDS], Const<N>, Const<N>>] = match last_mover {{
{edges_arms}            _ => unreachable!(),
        }};
"#
        )
    } else {
        String::new()
    };

    used_edges.sort_unstable();
    let edge_fns: String = used_edges.iter().map(|name| edge_fn_source(name)).collect();

    let hex_connects_fn = if needs_connects {
        r#"
/// The mover's stones (`board`, floodfilled from `edges[0]`) reach every remaining entry of
/// `edges` -- the Rust realization of `BoolExpr::Connects{ conn: Connectivity::Six }`, matching
/// `gdl::core::interp::eval_bool`'s own `Connects` arm but specialized once, at generation time,
/// to a direct `Board::flood6` call instead of `core::interp::bounded_fixpoint`'s general
/// iterate-to-a-fixpoint loop.
fn hex_connects<const N: usize, const WORDS: usize>(
    board: Board<[u64; WORDS], Const<N>, Const<N>>,
    edges: &[Board<[u64; WORDS], Const<N>, Const<N>>],
) -> bool {
    let [first, rest @ ..] = edges else {
        return false;
    };
    let flooded = board.flood6(*first);
    rest.iter().all(|&e| flooded.intersects(e))
}
"#
        .to_string()
    } else {
        String::new()
    };

    Ok(format!(
        r#"// @generated by `cargo run -p gdl --bin codegen -- {source_path} {struct_name}` --
// do not edit by hand. Regenerate instead of patching -- see gdl/src/codegen/hex.rs.
//
// {game_name}, lowered from {source_path} via gdl::core::Program. Const-generic over board side
// `N` and `bitboard::Board<[u64; WORDS], Const<N>, Const<N>>`'s own `WORDS` parameter -- see a caller crate's
// `main.rs` (e.g. `games/hex-gen/src/main.rs`'s `dispatch_size!` macro) for how a concrete board
// size is picked at request time, the same pattern `games/gonnect` hand-writes for itself.

use bitboard::{{Board, Const}};
use game_core::display::{{RectangularBoard, RectangularBoardDisplay}};
use mcts::game::{{Game, PlayerIndex}};
use serde::{{Deserialize, Serialize}};
use std::fmt;

const NUM_PLAYERS: usize = {num_players};
const SEED: u64 = 0x{seed:x};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Player {{
{player_variants}}}

impl PlayerIndex for Player {{
    fn to_index(&self) -> usize {{
        match self {{
{to_index_arms}        }}
    }}
}}

impl Player {{
    fn from_index(index: usize) -> Self {{
        match index {{
{from_index_arms}            _ => unreachable!(),
        }}
    }}

    fn next(self) -> Self {{
        Player::from_index((self.to_index() + 1) % NUM_PLAYERS)
    }}
}}
{hex_connects_fn}
{edge_fns}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Position<const N: usize, const WORDS: usize> {{
    pub turn: Player,
    pub occupied: [Board<[u64; WORDS], Const<N>, Const<N>>; NUM_PLAYERS],
}}

impl<const N: usize, const WORDS: usize> Default for Position<N, WORDS> {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl<const N: usize, const WORDS: usize> Position<N, WORDS> {{
    pub fn new() -> Self {{
        Self {{
            turn: Player::P0,
            occupied: [Board::new_const(); NUM_PLAYERS],
        }}
    }}

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {{
        let legal = {move_expr};
        for s in legal {{
            actions.push(Move(s as u8));
        }}
    }}

    pub fn apply(&mut self, m: Move) {{
        self.occupied[self.turn.to_index()].set_index(m.0 as usize);
        self.turn = self.turn.next();
    }}

    /// The player who moved most recently, and whether their move satisfied one of `program`'s
    /// end rules -- mirrors `core::interp::State::winner`, checking only the mover who just
    /// moved (per `core::EndRule`'s own doc comment), not every player unconditionally.
    pub fn winner(&self) -> Option<Player> {{
        let last_mover = (self.turn.to_index() + NUM_PLAYERS - 1) % NUM_PLAYERS;
        let board = self.occupied[last_mover];
{edges_binding}        let won = {{
            {win_expr}
        }};
        if won {{
            Some(Player::from_index(last_mover))
        }} else {{
            None
        }}
    }}
}}

/// Wraps `Position` with a hash recomputed from the raw board words on every `apply` -- see this
/// module's doc comment for why it's not an incrementally-maintained `LazyZobristTable` the way
/// `games/ttt-gen`/`games/hex3-gen` use (an array sized by `N` isn't expressible as a static item
/// on stable Rust). `WORDS` is always small (1-2) for every board size this backend targets, so
/// recomputing on every move is nowhere near a hot-path cost.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct HashedPosition<const N: usize, const WORDS: usize> {{
    pub position: Position<N, WORDS>,
    pub hash: u64,
}}

impl<const N: usize, const WORDS: usize> Default for HashedPosition<N, WORDS> {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl<const N: usize, const WORDS: usize> HashedPosition<N, WORDS> {{
    pub fn new() -> Self {{
        let position = Position::new();
        let hash = Self::compute_hash(&position);
        Self {{ position, hash }}
    }}

    /// FNV-1a over `position`'s occupied-board words plus whose turn it is -- deterministic and
    /// well-distributed, not cryptographic, matching `fnv1a`'s own role picking `SEED`.
    fn compute_hash(position: &Position<N, WORDS>) -> u64 {{
        let mut h: u64 = SEED;
        for board in position.occupied {{
            for w in board.words() {{
                h ^= w;
                h = h.wrapping_mul(0x100000001b3);
            }}
        }}
        h ^= position.turn.to_index() as u64;
        h
    }}

    fn apply(&mut self, m: Move) {{
        self.position.apply(m);
        self.hash = Self::compute_hash(&self.position);
    }}
}}

#[derive(Debug, Clone)]
pub struct {struct_name}<const N: usize, const WORDS: usize>;

impl<const N: usize, const WORDS: usize> Game for {struct_name}<N, WORDS> {{
    type S = HashedPosition<N, WORDS>;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {{
        state.position.gen_moves(actions);
    }}

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {{
        state.apply(*m);
        state
    }}

    fn is_terminal(state: &Self::S) -> bool {{
        if state.position.winner().is_some() {{
            return true;
        }}
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        actions.is_empty()
    }}

    fn winner(state: &Self::S) -> Option<Self::P> {{
        state.position.winner()
    }}

    fn player_to_move(state: &Self::S) -> Self::P {{
        state.position.turn
    }}

    fn num_players() -> usize {{
        NUM_PLAYERS
    }}

    fn zobrist_hash(state: &Self::S) -> u64 {{
        state.hash
    }}

    fn notation(_state: &Self::S, m: &Self::A) -> String {{
        format!("({{}}, {{}})", m.0 as usize % N, m.0 as usize / N)
    }}
}}

impl<const N: usize, const WORDS: usize> RectangularBoard for HashedPosition<N, WORDS> {{
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {{
        let site = row * N + col;
        for p in 0..NUM_PLAYERS {{
            if self.position.occupied[p].get_index(site) {{
                return (b'A' + p as u8) as char;
            }}
        }}
        '.'
    }}
}}

impl<const N: usize, const WORDS: usize> fmt::Display for HashedPosition<N, WORDS> {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        RectangularBoardDisplay(self).fmt(f)
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    use mcts::util::random_play;

    #[test]
    fn random_playouts_terminate() {{
        random_play::<{struct_name}<{side}, {words}>>();
    }}
}}
"#
    ))
}
