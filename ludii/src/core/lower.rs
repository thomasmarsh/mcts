//! Lowers an elaborated [`crate::ast::game::Game`] into a Core IR [`super::Program`].
//!
//! Scoped exactly to what `lud/Tic-Tac-Toe.lud` elaborates to (see `crate::elaborate`'s own
//! per-ludeme scoping notes) -- a `(square <n>)` board, one `(piece ...)` per player, a
//! `(move Add (to (sites Empty)))` play rule, and a single `(if (is Line <n>) (result Mover
//! Win))` end rule. Any other shape is a lowering error, not a panic: this is meant to grow one
//! real `.lud` file at a time, not to silently do the wrong thing on a shape it doesn't
//! recognize yet.

use crate::ast::boolean::{BooleanFunction, Is};
use crate::ast::common::SiteOrRegion;
use crate::ast::equipment::Item;
use crate::ast::game::{Game, Players};
use crate::ast::graph::generator::Extent;
use crate::ast::graph::GraphFunction;
use crate::ast::moves::decision::{Decision, MoveSiteType};
use crate::ast::moves::Moves;
use crate::ast::numeric::dim::DimFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::{RegionFunction, Sites, SitesIndexType};
use crate::ast::rules::end::EndRule as AstEndRule;
use crate::ast::types::{ResultType, RoleType};
use crate::core::{EndRule, MoveGen, Player, Program, Rect, Region};

/// A lowering failure: `.lud` shapes the Core lowering doesn't (yet) recognize.
#[derive(Debug, Clone, PartialEq)]
pub struct LowerError(pub String);

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LowerError {}

fn err(message: impl Into<String>) -> LowerError {
    LowerError(message.into())
}

fn lower_topology(game: &Game) -> Result<Rect, LowerError> {
    let mut boards = game.equipment.items.iter().filter_map(|item| match item {
        Item::Board(board) => Some(board),
        _ => None,
    });
    let board = boards
        .next()
        .ok_or_else(|| err("expected exactly one (board ...) in equipment"))?;
    if boards.next().is_some() {
        return Err(err("expected exactly one (board ...) in equipment"));
    }
    let GraphFunction::Square(square) = &board.graph.node else {
        return Err(err("only (square ...) boards are lowered so far"));
    };
    if square.shape.is_some() || square.modifier.is_some() {
        return Err(err("(square ...) shape/modifier aren't lowered yet"));
    }
    let Extent::Dims(rows, cols) = &square.extent else {
        return Err(err("only a plain (square <dim>) extent is lowered so far"));
    };
    if cols.is_some() {
        return Err(err(
            "(square rows columns) isn't lowered yet -- only square boards",
        ));
    }
    let DimFunction::Int(n) = rows.node else {
        return Err(err("only an integer board dimension is lowered so far"));
    };
    if n <= 0 {
        return Err(err("board dimension must be positive"));
    }
    let n = n as usize;
    Ok(Rect { rows: n, cols: n })
}

fn num_players(game: &Game) -> Result<usize, LowerError> {
    let Players::Count(count) = &game.players else {
        return Err(err("only a plain (players <int>) count is lowered so far"));
    };
    let IntFunction::Int(n) = count.node else {
        return Err(err("only an integer player count is lowered so far"));
    };
    if n <= 0 {
        return Err(err("player count must be positive"));
    }
    Ok(n as usize)
}

/// The union of every player's occupied region -- "any site with a piece on it."
fn all_occupied(num_players: usize) -> Region {
    (1..num_players).fold(Region::Occupied(Player(0)), |acc, i| {
        Region::Union(Box::new(acc), Box::new(Region::Occupied(Player(i))))
    })
}

fn lower_move_gen(game: &Game, num_players: usize) -> Result<MoveGen, LowerError> {
    let play = game
        .rules
        .play
        .as_ref()
        .ok_or_else(|| err("expected a (play ...) rule"))?;
    let Moves::Decision(decision) = &play.0.node else {
        return Err(err(
            "only a (move ...) decision play rule is lowered so far",
        ));
    };
    let Decision::Site {
        kind: MoveSiteType::Add,
        piece: None,
        to,
        count: None,
        stack: None,
        then: None,
    } = decision.as_ref()
    else {
        return Err(err(
            "only (move Add (to ...)) with no piece/count/stack/then is lowered so far",
        ));
    };
    let Some(SiteOrRegion::Region(region)) = &to.location else {
        return Err(err(
            "(move Add (to ...)) must target a region, not a single site",
        ));
    };
    let RegionFunction::Sites(sites) = &region.node else {
        return Err(err("only (to (sites ...)) is lowered so far"));
    };
    let Sites::Index {
        kind: SitesIndexType::Empty,
        site_type: None,
        index: None,
    } = sites.as_ref()
    else {
        return Err(err("only (to (sites Empty)) is lowered so far"));
    };
    Ok(MoveGen {
        to: Region::Complement(Box::new(all_occupied(num_players))),
    })
}

fn lower_end(game: &Game) -> Result<Vec<EndRule>, LowerError> {
    let end = game
        .rules
        .end
        .as_ref()
        .ok_or_else(|| err("expected an (end ...) rule"))?;
    let [AstEndRule::If(if_rule)] = end.rules.as_slice() else {
        return Err(err("only a single (end (if ...)) rule is lowered so far"));
    };
    if !if_rule.subconditions.is_empty() {
        return Err(err("(if ...) subconditions aren't lowered yet"));
    }
    let condition = if_rule
        .condition
        .as_ref()
        .ok_or_else(|| err("(if ...) requires a condition"))?;
    let BooleanFunction::Is(is) = &condition.node else {
        return Err(err("only (is Line ...) is lowered so far"));
    };
    let Is::Line(line) = is.as_ref() else {
        return Err(err("only (is Line ...) is lowered so far"));
    };
    if line.site_type.is_some()
        || line.direction.is_some()
        || line.through.is_some()
        || line.owner.is_some()
        || line.what.is_some()
        || line.exact.is_some()
        || line.contiguous.is_some()
        || line.condition.is_some()
        || line.by_level.is_some()
    {
        return Err(err(
            "only a bare (is Line <int>) -- no site type/direction/through/owner/what/exact/\
             contiguous/condition/byLevel -- is lowered so far",
        ));
    }
    let IntFunction::Int(length) = line.min_length.node else {
        return Err(err("only an integer minLength is lowered so far"));
    };
    if length <= 0 {
        return Err(err("minLength must be positive"));
    }
    let result = if_rule
        .result
        .ok_or_else(|| err("(if ...) requires a result -- default-only ifs aren't lowered yet"))?;
    if result.role != RoleType::Mover || result.result != ResultType::Win {
        return Err(err("only (result Mover Win) is lowered so far"));
    }
    Ok(vec![EndRule {
        line_length: length as usize,
    }])
}

/// Lowers a self-contained `ast::game::Game` into a Core IR [`Program`]. See the module doc for
/// exactly which `.lud` shapes are currently supported.
pub fn lower_game(game: &Game) -> Result<Program, LowerError> {
    let topology = lower_topology(game)?;
    let players = num_players(game)?;
    let move_gen = lower_move_gen(game, players)?;
    let end = lower_end(game)?;
    Ok(Program {
        topology,
        num_players: players,
        move_gen,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::game::Description;
    use crate::elaborate::game::elaborate_description;
    use crate::parse::parse;

    fn lower_fixture(src: &str) -> Program {
        let forms = parse(src).unwrap();
        let Description::Game(game) = elaborate_description(&forms[0]).unwrap() else {
            panic!("expected Description::Game");
        };
        lower_game(&game).unwrap()
    }

    #[test]
    fn tic_tac_toe_fixture_lowers() {
        let program = lower_fixture(include_str!("../../lud/Tic-Tac-Toe.lud"));
        assert_eq!(program.topology, Rect { rows: 3, cols: 3 });
        assert_eq!(
            program.move_gen.to,
            Region::Complement(Box::new(Region::Union(
                Box::new(Region::Occupied(Player(0))),
                Box::new(Region::Occupied(Player(1))),
            )))
        );
        assert_eq!(program.end, vec![EndRule { line_length: 3 }]);
    }
}
