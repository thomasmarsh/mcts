//! Lowers an elaborated [`crate::ast::game::Game`] into a Core IR [`super::Program`].
//!
//! Scoped exactly to what `lud/Tic-Tac-Toe.lud` and `lud/Hex.lud` elaborate to (see
//! `crate::elaborate`'s own per-ludeme scoping notes) -- a `(square <n>)` or `(hex Diamond <n>)`
//! board, a `(move Add (to (sites Empty)))` play rule, and a single end rule that's either
//! `(if (is Line <n>) (result Mover Win))` or `(if (is Connected Mover) (result Mover Win))`. Any
//! other shape is a lowering error, not a panic: this is meant to grow one real `.lud` file at a
//! time, not to silently do the wrong thing on a shape it doesn't recognize yet.

use crate::ast::boolean::{BooleanFunction, ConnectRegions, Is, IsConnectType};
use crate::ast::common::SiteOrRegion;
use crate::ast::equipment::other::RegionsSpec;
use crate::ast::equipment::Item;
use crate::ast::game::{Game, Players};
use crate::ast::graph::generator::{Extent, HexShapeType};
use crate::ast::graph::GraphFunction;
use crate::ast::located::LBox;
use crate::ast::moves::decision::{Decision, MoveSiteType};
use crate::ast::moves::Moves;
use crate::ast::numeric::dim::DimFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::{RegionFunction, SideTarget, Sites, SitesIndexType};
use crate::ast::rules::end::EndRule as AstEndRule;
use crate::ast::types::{ResultType, RoleType};
use crate::core::hex::{Edge, Hex, HexShape};
use crate::core::{
    BoolExpr, Connectivity, EndRule, MoveGen, Player, Program, Rect, Region, Topology,
};

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

/// A single-dimension `Extent::Dims(<dim>, None)`, the only extent shape either `(square <n>)` or
/// `(hex Diamond <n>)` is lowered for so far.
fn lower_square_dim(extent: &Extent, context: &str) -> Result<usize, LowerError> {
    let Extent::Dims(dim, cols) = extent else {
        return Err(err(format!(
            "only a plain {context} <dim> extent is lowered so far"
        )));
    };
    if cols.is_some() {
        return Err(err(format!(
            "{context} rows columns isn't lowered yet -- only square boards"
        )));
    }
    let DimFunction::Int(n) = dim.node else {
        return Err(err("only an integer board dimension is lowered so far"));
    };
    if n <= 0 {
        return Err(err("board dimension must be positive"));
    }
    Ok(n as usize)
}

fn lower_topology(game: &Game) -> Result<Topology, LowerError> {
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
    match &board.graph.node {
        GraphFunction::Square(square) => {
            if square.shape.is_some() || square.modifier.is_some() {
                return Err(err("(square ...) shape/modifier aren't lowered yet"));
            }
            let n = lower_square_dim(&square.extent, "(square ...)")?;
            Ok(Topology::Rect(Rect { rows: n, cols: n }))
        }
        GraphFunction::Hex(hex) => {
            if hex.shape != Some(HexShapeType::Diamond) {
                return Err(err("only (hex Diamond <dim>) boards are lowered so far"));
            }
            let side = lower_square_dim(&hex.extent, "(hex Diamond ...)")?;
            Ok(Topology::Hex(Hex {
                side,
                shape: HexShape::Rhombus,
            }))
        }
        _ => Err(err(
            "only (square ...) or (hex Diamond ...) boards are lowered so far",
        )),
    }
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

/// The union of every player's occupied region -- "any site with a piece on it." `pub(crate)`
/// since [`crate::style_c`]'s independent sexpr frontend reuses it for its own `(sites Empty)`
/// sugar, rather than duplicating the fold.
pub(crate) fn all_occupied(num_players: usize) -> Region {
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

fn lower_end(game: &Game, topology: &Topology) -> Result<Vec<EndRule>, LowerError> {
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
        return Err(err(
            "only (is Line ...) or (is Connected ...) is lowered so far",
        ));
    };
    let result = if_rule
        .result
        .ok_or_else(|| err("(if ...) requires a result -- default-only ifs aren't lowered yet"))?;
    if result.role != RoleType::Mover || result.result != ResultType::Win {
        return Err(err("only (result Mover Win) is lowered so far"));
    }
    match is.as_ref() {
        Is::Line(line) => {
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
                    "only a bare (is Line <int>) -- no site type/direction/through/owner/what/\
                     exact/contiguous/condition/byLevel -- is lowered so far",
                ));
            }
            let IntFunction::Int(length) = line.min_length.node else {
                return Err(err("only an integer minLength is lowered so far"));
            };
            if length <= 0 {
                return Err(err("minLength must be positive"));
            }
            let Topology::Rect(rect) = topology else {
                return Err(err(
                    "(is Line ...) is only lowered for a Rect topology so far",
                ));
            };
            Ok(vec![EndRule {
                condition: BoolExpr::Any(
                    rect.lines(length as usize)
                        .into_iter()
                        .map(|line| BoolExpr::Contains(Region::Sites(line)))
                        .collect(),
                ),
            }])
        }
        Is::Connect {
            kind,
            min_regions,
            site_type,
            at,
            direction,
            regions,
        } => {
            if *kind != IsConnectType::Connected {
                return Err(err(
                    "only (is Connected ...) is lowered so far -- not Blocked",
                ));
            }
            if min_regions.is_some() || site_type.is_some() || at.is_some() || direction.is_some() {
                return Err(err(
                    "(is Connected ...) minRegions/siteType/at/direction aren't lowered yet",
                ));
            }
            if !matches!(regions, ConnectRegions::Role(RoleType::Mover)) {
                return Err(err("only (is Connected Mover) is lowered so far"));
            }
            let Topology::Hex(_) = topology else {
                return Err(err(
                    "(is Connected ...) is only lowered for a Hex topology so far",
                ));
            };
            Ok(vec![EndRule {
                condition: BoolExpr::Connects {
                    conn: Connectivity::Six,
                },
            }])
        }
        _ => Err(err(
            "only (is Line ...) or (is Connected ...) is lowered so far",
        )),
    }
}

/// A single `(regions <roleType> {(sites Side <compassDirection>) (sites Side
/// <compassDirection>)})` equipment entry, lowered to a `(Region, Region)` pair of static edge
/// site lists -- so far only meaningful for a `Hex` topology.
fn lower_side_region(
    func: &LBox<RegionFunction>,
    topology: &Topology,
) -> Result<Region, LowerError> {
    let RegionFunction::Sites(sites) = &func.node else {
        return Err(err(
            "only (sites Side <compassDirection>) is lowered for (regions ...) entries",
        ));
    };
    let Sites::Side {
        site_type: None,
        target: Some(SideTarget::Compass(compass)),
    } = sites.as_ref()
    else {
        return Err(err(
            "only (sites Side <compassDirection>) is lowered so far",
        ));
    };
    let Topology::Hex(hex) = topology else {
        return Err(err(
            "(sites Side ...) is only lowered for a Hex topology so far",
        ));
    };
    let edge: Edge = Hex::edge_for_compass(*compass).ok_or_else(|| {
        err(format!(
            "unsupported compass direction for a Hex side: {compass:?}"
        ))
    })?;
    Ok(Region::Sites(hex.edge(edge)))
}

/// Every `(regions <roleType> {...})` equipment item, indexed by player. Empty when the game
/// declares none (e.g. Tic-Tac-Toe) -- only a game with a `BoolExpr::Connects` end rule looks
/// this table up. Each player's entry is a list of named regions, not a fixed pair -- Hex
/// declares two (`(sites Side NE)`/`(sites Side SW)`), but `core::Program.player_regions`'s
/// shape is arbitrary-length per DESIGN.md's "Y's three-edge win is about to be a third [data
/// point]" note, so this lowering accepts any nonempty list rather than pattern-matching a fixed
/// arity.
fn lower_player_regions(
    game: &Game,
    topology: &Topology,
    num_players: usize,
) -> Result<Vec<Vec<Region>>, LowerError> {
    let mut slots: Vec<Option<Vec<Region>>> = vec![None; num_players];
    let mut found_any = false;
    for item in &game.equipment.items {
        let Item::Regions(regions) = item else {
            continue;
        };
        found_any = true;
        let owner = regions
            .owner
            .ok_or_else(|| err("(regions ...) requires an owner role"))?;
        let player = match owner {
            RoleType::P1 => 0,
            RoleType::P2 => 1,
            other => {
                return Err(err(format!(
                    "(regions ...) owner {other:?} isn't lowered yet -- only P1/P2"
                )))
            }
        };
        let slot = slots.get_mut(player).ok_or_else(|| {
            err(format!(
                "(regions ...) owner player {player} is out of range for {num_players} players"
            ))
        })?;
        let RegionsSpec::Regions(funcs) = &regions.spec else {
            return Err(err(
                "only (regions <roleType> {(sites Side ...) ...}) is lowered so far",
            ));
        };
        if funcs.is_empty() {
            return Err(err(
                "(regions ...) must have at least one (sites Side ...) entry",
            ));
        }
        *slot = Some(
            funcs
                .iter()
                .map(|f| lower_side_region(f, topology))
                .collect::<Result<Vec<_>, LowerError>>()?,
        );
    }
    if !found_any {
        return Ok(Vec::new());
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, slot)| slot.ok_or_else(|| err(format!("missing (regions ...) for player {i}"))))
        .collect()
}

/// Lowers a self-contained `ast::game::Game` into a Core IR [`Program`]. See the module doc for
/// exactly which `.lud` shapes are currently supported.
pub fn lower_game(game: &Game) -> Result<Program, LowerError> {
    let topology = lower_topology(game)?;
    let players = num_players(game)?;
    let move_gen = lower_move_gen(game, players)?;
    let end = lower_end(game, &topology)?;
    let player_regions = lower_player_regions(game, &topology, players)?;
    Ok(Program {
        topology,
        num_players: players,
        move_gen,
        end,
        player_regions,
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
        assert_eq!(program.topology, Topology::Rect(Rect { rows: 3, cols: 3 }));
        assert_eq!(
            program.move_gen.to,
            Region::Complement(Box::new(Region::Union(
                Box::new(Region::Occupied(Player(0))),
                Box::new(Region::Occupied(Player(1))),
            )))
        );
        let [EndRule {
            condition: BoolExpr::Any(arms),
        }] = program.end.as_slice()
        else {
            panic!("expected a single Any-of-Contains end rule");
        };
        assert_eq!(arms.len(), 8); // 3 rows, 3 columns, 2 diagonals -- see Rect::lines's own tests.
        assert!(arms.iter().all(
            |arm| matches!(arm, BoolExpr::Contains(Region::Sites(sites)) if sites.len() == 3)
        ));
        assert!(program.player_regions.is_empty());
    }

    #[test]
    fn hex_fixture_lowers() {
        let program = lower_fixture(include_str!("../../lud/Hex.lud"));
        assert_eq!(
            program.topology,
            Topology::Hex(Hex {
                side: 3,
                shape: HexShape::Rhombus,
            })
        );
        assert_eq!(
            program.move_gen.to,
            Region::Complement(Box::new(Region::Union(
                Box::new(Region::Occupied(Player(0))),
                Box::new(Region::Occupied(Player(1))),
            )))
        );
        assert_eq!(
            program.end,
            vec![EndRule {
                condition: BoolExpr::Connects {
                    conn: Connectivity::Six,
                },
            }]
        );
        assert_eq!(
            program.player_regions,
            vec![
                vec![
                    Region::Sites(vec![6, 7, 8]), // NE -> North edge
                    Region::Sites(vec![0, 1, 2]), // SW -> South edge
                ],
                vec![
                    Region::Sites(vec![0, 3, 6]), // NW -> West edge
                    Region::Sites(vec![2, 5, 8]), // SE -> East edge
                ],
            ]
        );
    }
}
