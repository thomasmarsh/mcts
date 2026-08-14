//! `Topology::Rect` codegen: lowers a `Program` into the text of a standalone `games/*`-shaped
//! Rust crate, built on `game_core::bitboard::BitBoard<N, M>` -- the same backend
//! `core::interp::State` already binds `Program` to at runtime, but here specialized once, at
//! generation time, into ordinary monomorphic Rust rather than re-walked per call. Compare
//! `games/ttt/src/lib.rs` (hand-written) against this module's output for Tic-Tac-Toe
//! (`games/ttt-gen/src/lib.rs`, checked in via `cargo run -p ludii --bin codegen`):
//! representation choices differ (one `BitBoard` per player here vs. `games/ttt`'s packed 2-bit
//! cells, no D4-symmetry-aware zobrist here), but both implement `mcts::game::Game` for the same
//! rules and both pass the same oracle-comparison test (`tests/ttt_gen_oracle.rs`).
//!
//! Only lowers the `Region`/`BoolExpr` shapes Tic-Tac-Toe's `Program` uses -- see this module's
//! own doc comment in `mod.rs`.

use crate::core::{BoolExpr, EndRule, Player, Program, Rect, Region};

use super::Error;

/// Lowers `region` into a Rust expression of type `Board` (a `game_core::bitboard::BitBoard<N,
/// M>`), evaluated relative to `self.occupied: [Board; NUM_PLAYERS]` -- so the returned text is
/// valid in any method with a `&self` receiver of that shape (`Position::gen_moves` and
/// `Position::winner`, the only two call sites so far).
fn region_expr(region: &Region) -> Result<String, Error> {
    Ok(match region {
        Region::Occupied(Player(i)) => format!("self.occupied[{i}]"),
        Region::Union(a, b) => format!("({} | {})", region_expr(a)?, region_expr(b)?),
        // No extra enclosing parens: every other arm here already parenthesizes itself where an
        // operator's precedence would otherwise need it (`Union` below), or is a single atom
        // (`Occupied`/`Sites`) -- so `!<inner>` is always already unambiguous, and adding a
        // second, redundant pair would trip clippy's `double_parens` once nested under `Union`.
        Region::Complement(a) => format!("!{}", region_expr(a)?),
        Region::Sites(sites) => {
            let mask = sites.iter().fold(0u64, |acc, &s| acc | (1 << s));
            format!("Board::new(0b{mask:b})")
        }
        Region::Intersect(..)
        | Region::Shift { .. }
        | Region::Adjacent { .. }
        | Region::Flood { .. } => {
            return Err(Error(format!(
                "codegen::rect: {region:?} has no lowering yet -- no Rect-topology corpus game \
                 routed through codegen needs it yet (DESIGN.md's \"grow from real lowerings\")"
            )));
        }
    })
}

/// Lowers `expr` into a `bool`-typed Rust expression, `board` being the (already-lowered) name of
/// the region-under-test variable in scope (`Position::winner`'s `board: Board` local, per
/// `EndRule`'s own doc comment: the mover's own occupied region).
fn bool_expr(expr: &BoolExpr, board: &str) -> Result<String, Error> {
    Ok(match expr {
        BoolExpr::Contains(region) => format!("({}).is_subset({board})", region_expr(region)?),
        BoolExpr::Any(exprs) => {
            if exprs.is_empty() {
                "false".to_string()
            } else {
                // No enclosing parens: every operand here is itself only ever built from `||`
                // (there's no `&&`/mixed-precedence combinator in `BoolExpr` yet), so grouping
                // is never semantically required -- and this expression also lands directly in
                // syntactic positions (a `let` value, a block's tail) where `-D unused-parens`
                // rejects a redundant enclosing pair outright.
                let parts = exprs
                    .iter()
                    .map(|e| bool_expr(e, board))
                    .collect::<Result<Vec<_>, _>>()?;
                parts.join(" || ")
            }
        }
        BoolExpr::Connects { .. } => {
            return Err(Error(
                "codegen::rect: BoolExpr::Connects has no lowering yet -- Hex/Y's end rules stay \
                 on core::interp; a Rect-topology game with a Connects end rule is the forcing \
                 case, not encountered yet"
                    .into(),
            ));
        }
    })
}

fn end_expr(end: &[EndRule], board: &str) -> Result<String, Error> {
    if end.is_empty() {
        return Ok("false".to_string());
    }
    let parts = end
        .iter()
        .map(|rule| bool_expr(&rule.condition, board))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(parts.join(" || "))
}

/// FNV-1a over `s`'s bytes -- just needs to be a stable, well-distributed seed for this game's
/// `LazyZobristTable`, not cryptographic; deterministic on `struct_name` so regenerating the same
/// game twice produces byte-identical output.
fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn generate(
    game_name: &str,
    struct_name: &str,
    source_path: &str,
    rect: Rect,
    program: &Program,
) -> Result<String, Error> {
    if program.num_players == 0 {
        return Err(Error(
            "codegen::rect: a game needs at least one player".into(),
        ));
    }
    if program.num_players > 26 {
        // `display_char_at` below spends one letter A..Z per player -- no corpus game is
        // anywhere close to this, so a real per-player display scheme isn't worth building yet.
        return Err(Error(format!(
            "codegen::rect: {} players exceeds the 26-player display-char scheme",
            program.num_players
        )));
    }
    if !program.player_regions.is_empty() {
        return Err(Error(
            "codegen::rect: Program.player_regions is only consulted by BoolExpr::Connects, \
             which this backend doesn't lower yet -- see bool_expr"
                .into(),
        ));
    }

    let rows = rect.rows;
    let cols = rect.cols;
    let num_players = program.num_players;
    let seed = fnv1a(struct_name);

    let move_expr = region_expr(&program.move_gen.to.clone())?;
    let win_expr = end_expr(&program.end, "board")?;

    let player_variants: String = (0..num_players).map(|i| format!("    P{i},\n")).collect();
    let to_index_arms: String = (0..num_players)
        .map(|i| format!("            Player::P{i} => {i},\n"))
        .collect();
    let from_index_arms: String = (0..num_players)
        .map(|i| format!("            {i} => Player::P{i},\n"))
        .collect();

    Ok(format!(
        r#"// @generated by `cargo run -p ludii --bin codegen -- {source_path} {struct_name}` --
// do not edit by hand. Regenerate instead of patching -- see ludii/src/codegen/rect.rs.
//
// {game_name}, lowered from {source_path} via ludii::core::Program.

use game_core::bitboard::BitBoard;
use game_core::display::{{RectangularBoard, RectangularBoardDisplay}};
use mcts::game::{{Game, PlayerIndex}};
use mcts::zobrist::LazyZobristTable;
use serde::{{Deserialize, Serialize}};
use std::fmt;

const ROWS: usize = {rows};
const COLS: usize = {cols};
const NUM_PLAYERS: usize = {num_players};

type Board = BitBoard<{rows}, {cols}>;

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
        let won = {{
            {win_expr}
        }};
        if won {{
            Some(Player::from_index(last_mover))
        }} else {{
            None
        }}
    }}
}}

const NUM_SITES: usize = ROWS * COLS;
const NUM_HASHES: usize = NUM_SITES * NUM_PLAYERS;
static HASHES: LazyZobristTable<NUM_HASHES> = LazyZobristTable::new(0x{seed:x});

/// Wraps `Position` with an incrementally-maintained zobrist hash -- unlike `games/ttt`'s own
/// `HashedPosition`, this doesn't track a D4-symmetry-aware hash per symmetry (that's a
/// `Rect`-square-specific hand optimization `Program` has no way to request yet, not part of
/// Core IR's own semantics).
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
        format!("({{}}, {{}})", m.0 as usize % COLS, m.0 as usize / COLS)
    }}
}}

impl RectangularBoard for HashedPosition {{
    const NUM_DISPLAY_ROWS: usize = ROWS;
    const NUM_DISPLAY_COLS: usize = COLS;

    fn display_char_at(&self, row: usize, col: usize) -> char {{
        let site = row * COLS + col;
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
