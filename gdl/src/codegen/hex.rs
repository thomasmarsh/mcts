//! `Topology::Hex` codegen: lowers a `Program` into the text of a standalone `games/*`-shaped
//! Rust crate, on the same `side x side` grid `core::interp::State`/`core::hex::Hex` already use
//! for a Rhombus board -- see this module's own doc comment in `mod.rs` for the scope this
//! backend covers. The generated `Board` type is `game_core::bitboard::BitBoard<side, side>`
//! when `side * side <= 64` (fits one `u64`), else
//! `game_core::bigbitboard::BigBitBoard<side, side, WORDS>` -- the same threshold and swap
//! `games/gonnect` makes by hand for its own larger board sizes, done here at generation time
//! instead (see `board_type`/`region_expr`'s `words` parameter).
//!
//! Only lowers what a Rhombus-shaped Hex-topology `Program` actually needs: the same
//! `Region::Occupied`/`Union`/`Complement`/`Sites` shapes `rect` already lowers (Hex's own
//! `(sites Empty)` move generator and `(regions ...)` edges are built only from those), plus
//! `BoolExpr::Connects` itself, specialized to `Connectivity::Six` via a generated `hex_connects`
//! helper that calls `game_core::bitboard::BitBoard::flood6` directly. `HexShape::Triangle` (Y)
//! is out of scope -- see `mod.rs`.

use crate::core::hex::HexShape;
use crate::core::{BoolExpr, Connectivity, EndRule, Hex, Player, Program, Region};

use super::{fnv1a, Error};

/// Lowers `region` into a Rust expression of type `Board`, evaluated relative to `self.occupied:
/// [Board; NUM_PLAYERS]` -- see `rect::region_expr`'s identical doc comment; the two functions
/// are kept as separate small copies (one per backend module, matching `mod.rs`'s "one pass per
/// Topology variant" architecture) rather than shared, since `rect`'s copy will grow
/// `Intersect`/`Shift`/`Adjacent`/`Flood` lowerings independently once a Rect-topology game forces
/// them.
///
/// `words` mirrors `generate`'s own board-type choice: `None` emits a plain `u64` literal for
/// `Region::Sites` (`Board::new(0b...)`, `BitBoard::new`'s signature), `Some(n)` emits an
/// `n`-element word array (`Board::new([0x..., ...])`, `BigBitBoard::new`'s signature) instead.
fn region_expr(region: &Region, words: Option<usize>) -> Result<String, Error> {
    Ok(match region {
        Region::Occupied(Player(i)) => format!("self.occupied[{i}]"),
        Region::Union(a, b) => format!("({} | {})", region_expr(a, words)?, region_expr(b, words)?),
        Region::Complement(a) => format!("!{}", region_expr(a, words)?),
        Region::Sites(sites) => match words {
            None => {
                let mask = sites.iter().fold(0u64, |acc, &s| acc | (1 << s));
                format!("Board::new(0b{mask:b})")
            }
            Some(words) => {
                let mut word_masks = vec![0u64; words];
                for &s in sites {
                    word_masks[s / 64] |= 1 << (s % 64);
                }
                let items: Vec<String> =
                    word_masks.iter().map(|w| format!("0x{w:016x}")).collect();
                format!("Board::new([{}])", items.join(", "))
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
/// the region-under-test variable in scope; `edges` is `Some(name)` of an in-scope `&[Board]`
/// binding (`Program.player_regions[last_mover]`, lowered by `generate` below) when the program
/// declares any, required by `BoolExpr::Connects`. Unlike `rect::bool_expr`, `Connects` always
/// lowers to a plain function call (`hex_connects(...)`) rather than an inline block, so it composes
/// safely as an `Any` operand without needing disambiguating parens (a leading `{ ... }` block
/// followed by `||` is itself ambiguous with an empty-parameter closure at statement start).
fn bool_expr(
    expr: &BoolExpr,
    board: &str,
    edges: Option<&str>,
    words: Option<usize>,
) -> Result<String, Error> {
    Ok(match expr {
        BoolExpr::Contains(region) => {
            format!("({}).is_subset({board})", region_expr(region, words)?)
        }
        BoolExpr::Any(exprs) => {
            if exprs.is_empty() {
                "false".to_string()
            } else {
                let parts = exprs
                    .iter()
                    .map(|e| bool_expr(e, board, edges, words))
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

fn end_expr(
    end: &[EndRule],
    board: &str,
    edges: Option<&str>,
    words: Option<usize>,
) -> Result<String, Error> {
    if end.is_empty() {
        return Ok("false".to_string());
    }
    let parts = end
        .iter()
        .map(|rule| bool_expr(&rule.condition, board, edges, words))
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
    let num_players = program.num_players;
    let seed = fnv1a(struct_name);

    let cells = side * side;
    let words_count = cells.div_ceil(64);
    let words_param = (cells > 64).then_some(words_count);
    let (board_import, board_type) = match words_param {
        None => (
            "use game_core::bitboard::BitBoard;".to_string(),
            format!("BitBoard<{side}, {side}>"),
        ),
        Some(words_count) => (
            "use game_core::bigbitboard::BigBitBoard;".to_string(),
            format!("BigBitBoard<{side}, {side}, {words_count}>"),
        ),
    };

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

    let move_expr = region_expr(&program.move_gen.to, words_param)?;
    let edges_var = needs_connects.then_some("edges");
    let win_expr = end_expr(&program.end, "board", edges_var, words_param)?;

    let player_variants: String = (0..num_players).map(|i| format!("    P{i},\n")).collect();
    let to_index_arms: String = (0..num_players)
        .map(|i| format!("            Player::P{i} => {i},\n"))
        .collect();
    let from_index_arms: String = (0..num_players)
        .map(|i| format!("            {i} => Player::P{i},\n"))
        .collect();

    let edges_arms: String = if needs_connects {
        program
            .player_regions
            .iter()
            .enumerate()
            .map(|(player, regions)| {
                let items = regions
                    .iter()
                    .map(|r| region_expr(r, words_param))
                    .collect::<Result<Vec<_>, _>>()?;
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
            r#"        let edges: &[Board] = match last_mover {{
{edges_arms}            _ => unreachable!(),
        }};
"#
        )
    } else {
        String::new()
    };

    let hex_connects_fn = if needs_connects {
        r#"
/// The mover's stones (`board`, floodfilled from `edges[0]`) reach every remaining entry of
/// `edges` -- the Rust realization of `BoolExpr::Connects{ conn: Connectivity::Six }`, matching
/// `gdl::core::interp::eval_bool`'s own `Connects` arm but specialized once, at generation time,
/// to a direct `BitBoard::flood6` call instead of `core::interp::bounded_fixpoint`'s general
/// iterate-to-a-fixpoint loop.
fn hex_connects(board: Board, edges: &[Board]) -> bool {
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
// {game_name}, lowered from {source_path} via gdl::core::Program.

{board_import}
use game_core::display::{{RectangularBoard, RectangularBoardDisplay}};
use mcts::game::{{Game, PlayerIndex}};
use mcts::zobrist::LazyZobristTable;
use serde::{{Deserialize, Serialize}};
use std::fmt;

const SIDE: usize = {side};
const NUM_PLAYERS: usize = {num_players};

type Board = {board_type};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Position {{
    pub turn: Player,
    pub occupied: [Board; NUM_PLAYERS],
}}

impl Default for Position {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl Position {{
    pub fn new() -> Self {{
        Self {{
            turn: Player::P0,
            occupied: [Board::EMPTY; NUM_PLAYERS],
        }}
    }}

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {{
        let legal = {move_expr};
        for s in legal {{
            actions.push(Move(s as u8));
        }}
    }}

    pub fn apply(&mut self, m: Move) {{
        self.occupied[self.turn.to_index()].set(m.0 as usize);
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

const NUM_SITES: usize = SIDE * SIDE;
const NUM_HASHES: usize = NUM_SITES * NUM_PLAYERS;
static HASHES: LazyZobristTable<NUM_HASHES> = LazyZobristTable::new(0x{seed:x});

/// Wraps `Position` with an incrementally-maintained zobrist hash -- unlike `games/ttt`'s own
/// `HashedPosition`, this doesn't track a symmetry-aware hash per symmetry (that's a hand
/// optimization `Program` has no way to request yet, not part of Core IR's own semantics).
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct HashedPosition {{
    pub position: Position,
    pub hash: u64,
}}

impl Default for HashedPosition {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl HashedPosition {{
    pub fn new() -> Self {{
        Self {{
            position: Position::new(),
            hash: 0,
        }}
    }}

    fn apply(&mut self, m: Move) {{
        self.hash ^= HASHES.hash((m.0 as usize) * NUM_PLAYERS + self.position.turn.to_index());
        self.position.apply(m);
    }}
}}

#[derive(Debug, Clone)]
pub struct {struct_name};

impl Game for {struct_name} {{
    type S = HashedPosition;
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
        format!("({{}}, {{}})", m.0 as usize % SIDE, m.0 as usize / SIDE)
    }}
}}

impl RectangularBoard for HashedPosition {{
    const NUM_DISPLAY_ROWS: usize = SIDE;
    const NUM_DISPLAY_COLS: usize = SIDE;

    fn display_char_at(&self, row: usize, col: usize) -> char {{
        let site = row * SIDE + col;
        for p in 0..NUM_PLAYERS {{
            if self.position.occupied[p].get(site) {{
                return (b'A' + p as u8) as char;
            }}
        }}
        '.'
    }}
}}

impl fmt::Display for HashedPosition {{
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
        random_play::<{struct_name}>();
    }}
}}
"#
    ))
}
