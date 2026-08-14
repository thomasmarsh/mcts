//! The real, working frontend onto [`crate::core::Program`] (see `README.md`'s "Current status"):
//! a small, direct s-expression encoding of Core IR, parsed with [`crate::parse::sexpr`]'s
//! generic reader (parens for calls, `{}` for lists, ordinary literals). This is *not* an attempt
//! at Style C's planned human-friendly surface notation (`def`, `guard`, primed fields,
//! `fixpoint`, ... -- see `style-c/games/tak.md`) -- the grammar below is a near-literal
//! parenthesized rendering of [`crate::core::Program`]/[`Region`]/[`BoolExpr`]'s own Rust shape,
//! "Core IR as data." Per `HISTORY.md`'s session note: several rounds of live syntax review left
//! Style C's own grammar unstable, so this frontend exists to stop that from blocking real
//! progress -- nothing about writing a game's declarative subset and running it through Core IR
//! needs a human-friendly lexer to exist first. A pretty-printer from Style C's eventual surface
//! syntax down to this s-expression form remains a plausible future addition; it's not required to
//! make this frontend useful on its own, and no longer planned per `ROADMAP.md`'s phase 2 (which
//! may instead promote this sexpr form to the canonical surface syntax outright).
//!
//! This module used to also advertise independence from a `.lud`-parsing `ast`/`elaborate`
//! pipeline that lowered Ludii's own ludeme AST into a `Program`; that pipeline has since been
//! deleted outright (`ROADMAP.md`'s decision to stop loading `.lud` source in code at all), so
//! there's nothing left to be independent *from* -- this is simply the frontend now.
//!
//! # Grammar
//!
//! ```text
//! Game     := "(" "game" Str Clause* ")"
//! Clause   := "(" "topology" TopologyExpr ")"
//!           | "(" "players" Int ")"
//!           | "(" "moves" Region ")"
//!           | "(" "end" Bool ")"                        -- zero or more; each is one EndRule
//!           | "(" "regions" Int Side+ ")"                -- zero or more; player_regions[Int]
//! TopologyExpr := "(" "rect" Int Int ")" | "(" "hex" Int ")" | "(" "hex_triangle" Int ")"
//! Region   := "(" "occupied" Int ")"
//!           | "(" "union" Region Region ")"
//!           | "(" "intersect" Region Region ")"
//!           | "(" "complement" Region ")"
//!           | "(" "sites" Int* ")" | "(" "sites" "Empty" ")"
//!           | "(" "shift" Region Direction ")"
//!           | "(" "adjacent" Region Connectivity ")"
//!           | "(" "flood" Region Region Connectivity ")"
//! Bool     := "(" "contains" Region ")"
//!           | "(" "connects" Connectivity ")"
//!           | "(" "any" Bool* ")"
//!           | "(" "has_line" Int ")"                     -- sugar, Rect only, see lower_bool
//! Side     := "(" "side" Compass ")"                     -- Hex { Rhombus } only, see lower_side
//!           | "(" "tri_side" TriEdge ")"                 -- Hex { Triangle } only, see lower_side
//! Direction    := North | East | South | West | Northeast | Northwest | Southeast | Southwest
//! Connectivity := Four | Six | Eight
//! Compass       := NE | SE | SW | NW
//! TriEdge       := Bottom | Left | Hypotenuse
//! ```
//!
//! `(sites Empty)` on a `Hex { Triangle }` topology additionally intersects the usual
//! "complement of every player's occupied region" with the board's valid triangular sites (see
//! [`crate::core::hex::Hex::valid_sites`]) -- a rhombus or rect board's valid sites are the whole
//! grid, so this is a no-op there and left unapplied, matching every other game's existing
//! `Program` value exactly rather than adding a redundant `Intersect` wrapper.

use crate::core::hex::{Edge, HexShape, TriangleEdge};
use crate::core::{
    all_occupied, BoolExpr, Connectivity, Direction, EndRule, Hex, MoveGen, Player, Program, Rect,
    Region, Topology,
};
use crate::parse::sexpr::{self, Call, Head, SExpr};
use crate::parse::ParseError;

/// A parse or lowering failure: not a panic, since this is meant to grow one accepted shape at a
/// time (matching `core::lower::LowerError`'s own discipline) rather than silently do the wrong
/// thing on a shape it doesn't recognize yet.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error(e.to_string())
    }
}

fn err(message: impl Into<String>) -> Error {
    Error(message.into())
}

/// Parses and lowers a single `(game "Name" clause...)` source text straight to a [`Program`].
pub fn parse_game(src: &str) -> Result<Program, Error> {
    let forms = sexpr::parse(src)?;
    let [form] = forms.as_slice() else {
        return Err(err(format!(
            "expected exactly one top-level (game ...) form, found {}",
            forms.len()
        )));
    };
    lower_game(&form.node)
}

#[derive(Debug, Clone, Copy)]
struct Ctx {
    topology: Topology,
    num_players: usize,
}

fn as_call<'a>(node: &'a SExpr, context: &str) -> Result<&'a Call, Error> {
    match node {
        SExpr::Call(c) => Ok(c),
        other => Err(err(format!(
            "expected a call in {context}, found {other:?}"
        ))),
    }
}

fn head_name(call: &Call) -> Result<&str, Error> {
    match &call.head {
        Head::Ident(s) => Ok(s.as_str()),
        other => Err(err(format!(
            "expected a plain identifier call head, found {other:?}"
        ))),
    }
}

fn as_int(node: &SExpr) -> Result<i64, Error> {
    match node {
        SExpr::Int(v) => Ok(*v),
        other => Err(err(format!("expected an integer, found {other:?}"))),
    }
}

fn as_str(node: &SExpr) -> Result<&str, Error> {
    match node {
        SExpr::Str(s) => Ok(s.as_str()),
        other => Err(err(format!("expected a string, found {other:?}"))),
    }
}

fn as_ident(node: &SExpr) -> Result<&str, Error> {
    match node {
        SExpr::Ident(s) => Ok(s.as_str()),
        other => Err(err(format!("expected a bare identifier, found {other:?}"))),
    }
}

/// A call's positional args, requiring exactly `n` of them (this grammar has no named args).
fn exact_args<'a, const N: usize>(call: &'a Call, context: &str) -> Result<[&'a SExpr; N], Error> {
    let nodes: Vec<&SExpr> = call.args.iter().map(|a| &a.value.node).collect();
    nodes.try_into().map_err(|nodes: Vec<&SExpr>| {
        err(format!(
            "{context} takes exactly {N} argument(s), found {}",
            nodes.len()
        ))
    })
}

fn lower_topology_expr(node: &SExpr) -> Result<Topology, Error> {
    let call = as_call(node, "(topology ...)")?;
    match head_name(call)? {
        "rect" => {
            let [rows, cols] = exact_args::<2>(call, "(rect rows cols)")?;
            Ok(Topology::Rect(Rect {
                rows: as_int(rows)? as usize,
                cols: as_int(cols)? as usize,
            }))
        }
        "hex" => {
            let [side] = exact_args::<1>(call, "(hex side)")?;
            Ok(Topology::Hex(Hex {
                side: as_int(side)? as usize,
                shape: HexShape::Rhombus,
            }))
        }
        "hex_triangle" => {
            let [side] = exact_args::<1>(call, "(hex_triangle side)")?;
            Ok(Topology::Hex(Hex {
                side: as_int(side)? as usize,
                shape: HexShape::Triangle,
            }))
        }
        other => Err(err(format!("unknown topology shape ({other} ...)"))),
    }
}

fn lower_direction(s: &str) -> Result<Direction, Error> {
    match s {
        "North" => Ok(Direction::North),
        "East" => Ok(Direction::East),
        "South" => Ok(Direction::South),
        "West" => Ok(Direction::West),
        "Northeast" => Ok(Direction::Northeast),
        "Northwest" => Ok(Direction::Northwest),
        "Southeast" => Ok(Direction::Southeast),
        "Southwest" => Ok(Direction::Southwest),
        other => Err(err(format!("unknown direction {other}"))),
    }
}

fn lower_connectivity(s: &str) -> Result<Connectivity, Error> {
    match s {
        "Four" => Ok(Connectivity::Four),
        "Six" => Ok(Connectivity::Six),
        "Eight" => Ok(Connectivity::Eight),
        other => Err(err(format!("unknown connectivity {other}"))),
    }
}

fn lower_region(node: &SExpr, ctx: &Ctx) -> Result<Region, Error> {
    let call = as_call(node, "a region expression")?;
    match head_name(call)? {
        "occupied" => {
            let [player] = exact_args::<1>(call, "(occupied player)")?;
            Ok(Region::Occupied(Player(as_int(player)? as usize)))
        }
        "union" => {
            let [a, b] = exact_args::<2>(call, "(union a b)")?;
            Ok(Region::Union(
                Box::new(lower_region(a, ctx)?),
                Box::new(lower_region(b, ctx)?),
            ))
        }
        "intersect" => {
            let [a, b] = exact_args::<2>(call, "(intersect a b)")?;
            Ok(Region::Intersect(
                Box::new(lower_region(a, ctx)?),
                Box::new(lower_region(b, ctx)?),
            ))
        }
        "complement" => {
            let [a] = exact_args::<1>(call, "(complement a)")?;
            Ok(Region::Complement(Box::new(lower_region(a, ctx)?)))
        }
        "sites" => {
            if let [only] = call.args.as_slice() {
                if let SExpr::Ident(s) = &only.value.node {
                    if s == "Empty" {
                        let empty = Region::Complement(Box::new(all_occupied(ctx.num_players)));
                        return Ok(match ctx.topology {
                            Topology::Hex(
                                hex @ Hex {
                                    shape: HexShape::Triangle,
                                    ..
                                },
                            ) => Region::Intersect(
                                Box::new(empty),
                                Box::new(Region::Sites(hex.valid_sites())),
                            ),
                            _ => empty,
                        });
                    }
                }
            }
            let sites = call
                .args
                .iter()
                .map(|a| Ok(as_int(&a.value.node)? as usize))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Region::Sites(sites))
        }
        "shift" => {
            let [region, dir] = exact_args::<2>(call, "(shift region direction)")?;
            Ok(Region::Shift {
                region: Box::new(lower_region(region, ctx)?),
                dir: lower_direction(as_ident(dir)?)?,
            })
        }
        "adjacent" => {
            let [region, conn] = exact_args::<2>(call, "(adjacent region connectivity)")?;
            Ok(Region::Adjacent {
                region: Box::new(lower_region(region, ctx)?),
                conn: lower_connectivity(as_ident(conn)?)?,
            })
        }
        "flood" => {
            let [region, seed, conn] = exact_args::<3>(call, "(flood region seed connectivity)")?;
            Ok(Region::Flood {
                region: Box::new(lower_region(region, ctx)?),
                seed: Box::new(lower_region(seed, ctx)?),
                conn: lower_connectivity(as_ident(conn)?)?,
            })
        }
        other => Err(err(format!("unknown region shape ({other} ...)"))),
    }
}

fn lower_bool(node: &SExpr, ctx: &Ctx) -> Result<BoolExpr, Error> {
    let call = as_call(node, "a boolean expression")?;
    match head_name(call)? {
        "contains" => {
            let [region] = exact_args::<1>(call, "(contains region)")?;
            Ok(BoolExpr::Contains(lower_region(region, ctx)?))
        }
        "connects" => {
            let [conn] = exact_args::<1>(call, "(connects connectivity)")?;
            Ok(BoolExpr::Connects {
                conn: lower_connectivity(as_ident(conn)?)?,
            })
        }
        "any" => Ok(BoolExpr::Any(
            call.args
                .iter()
                .map(|a| lower_bool(&a.value.node, ctx))
                .collect::<Result<Vec<_>, Error>>()?,
        )),
        // Sugar: expands to the same Any-of-Contains shape core::lower::lower_end produces for
        // `.lud`'s (is Line <n>), via the identical Rect::lines helper -- not a new IR shape.
        "has_line" => {
            let [length] = exact_args::<1>(call, "(has_line length)")?;
            let Topology::Rect(rect) = ctx.topology else {
                return Err(err("(has_line ...) is only meaningful for a Rect topology"));
            };
            let length = as_int(length)?;
            if length <= 0 {
                return Err(err("(has_line ...) length must be positive"));
            }
            Ok(BoolExpr::Any(
                rect.lines(length as usize)
                    .into_iter()
                    .map(|line| BoolExpr::Contains(Region::Sites(line)))
                    .collect(),
            ))
        }
        other => Err(err(format!("unknown boolean shape ({other} ...)"))),
    }
}

/// `(side <compass>)` or `(tri_side <edge>)`, valid only inside a `(regions ...)` clause. `side`'s
/// NE/SE/SW/NW-to-edge mapping matches `core::hex::Hex`'s own diamond-rotation convention (see
/// that module's doc comment) -- spelled out here directly since this frontend takes compass
/// names as plain identifiers, not a typed `CompassDirection`. `tri_side` is the equivalent for
/// `Hex { Triangle }`'s three edges, spelled out by name (`Bottom`/`Left`/`Hypotenuse`) rather
/// than a compass point, since a triangle doesn't have four sides to name that way.
fn lower_side(node: &SExpr, ctx: &Ctx) -> Result<Region, Error> {
    let call = as_call(node, "a (regions ...) endpoint")?;
    match head_name(call)? {
        "side" => {
            let [compass] = exact_args::<1>(call, "(side compass)")?;
            let Topology::Hex(hex) = ctx.topology else {
                return Err(err("(side ...) is only meaningful for a Hex topology"));
            };
            let edge = match as_ident(compass)? {
                "NE" => Edge::North,
                "SE" => Edge::East,
                "SW" => Edge::South,
                "NW" => Edge::West,
                other => return Err(err(format!("unsupported compass direction {other}"))),
            };
            Ok(Region::Sites(hex.edge(edge)))
        }
        "tri_side" => {
            let [name] = exact_args::<1>(call, "(tri_side edge)")?;
            let Topology::Hex(
                hex @ Hex {
                    shape: HexShape::Triangle,
                    ..
                },
            ) = ctx.topology
            else {
                return Err(err(
                    "(tri_side ...) is only meaningful for a Hex { Triangle } topology",
                ));
            };
            let edge = match as_ident(name)? {
                "Bottom" => TriangleEdge::Bottom,
                "Left" => TriangleEdge::Left,
                "Hypotenuse" => TriangleEdge::Hypotenuse,
                other => return Err(err(format!("unsupported triangle edge {other}"))),
            };
            Ok(Region::Sites(hex.triangle_edge(edge)))
        }
        other => Err(err(format!(
            "(regions ...) endpoints must each be (side ...) or (tri_side ...), found ({other} ...)"
        ))),
    }
}

fn lower_game(node: &SExpr) -> Result<Program, Error> {
    let call = as_call(node, "the top-level game form")?;
    let name = head_name(call)?;
    if name != "game" {
        return Err(err(format!("expected (game ...), found ({name} ...)")));
    }
    let mut args = call.args.iter();
    let first = args
        .next()
        .ok_or_else(|| err("(game ...) requires a name string as its first argument"))?;
    as_str(&first.value.node)?;

    let clauses: Vec<&Call> = args
        .map(|a| as_call(&a.value.node, "a (game ...) clause"))
        .collect::<Result<_, Error>>()?;

    // Pass 1: topology and player count, needed by name-only region expressions like
    // `(sites Empty)` and by Hex-only sugar (`has_line`/`side`) before the rest can be lowered.
    let mut topology = None;
    let mut num_players = None;
    for clause in &clauses {
        match head_name(clause)? {
            "topology" => {
                let [expr] = exact_args::<1>(clause, "(topology ...)")?;
                topology = Some(lower_topology_expr(expr)?);
            }
            "players" => {
                let [n] = exact_args::<1>(clause, "(players ...)")?;
                num_players = Some(as_int(n)? as usize);
            }
            _ => {}
        }
    }
    let topology = topology.ok_or_else(|| err("missing (topology ...) clause"))?;
    let num_players = num_players.ok_or_else(|| err("missing (players ...) clause"))?;
    let ctx = Ctx {
        topology,
        num_players,
    };

    // Pass 2: everything else, now that topology/num_players are in scope.
    let mut move_gen = None;
    let mut end = Vec::new();
    let mut region_slots: Vec<Option<Vec<Region>>> = vec![None; num_players];
    let mut any_regions = false;

    for clause in &clauses {
        match head_name(clause)? {
            "topology" | "players" => {}
            "moves" => {
                let [to] = exact_args::<1>(clause, "(moves ...)")?;
                move_gen = Some(MoveGen {
                    to: lower_region(to, &ctx)?,
                });
            }
            "end" => {
                let [condition] = exact_args::<1>(clause, "(end ...)")?;
                end.push(EndRule {
                    condition: lower_bool(condition, &ctx)?,
                });
            }
            "regions" => {
                any_regions = true;
                let mut args = clause.args.iter();
                let player_node = args.next().ok_or_else(|| {
                    err("(regions ...) requires a player index as its first argument")
                })?;
                let player = as_int(&player_node.value.node)? as usize;
                let edges: Vec<Region> = args
                    .map(|a| lower_side(&a.value.node, &ctx))
                    .collect::<Result<_, Error>>()?;
                if edges.is_empty() {
                    return Err(err("(regions player ...) requires at least one endpoint"));
                }
                let slot = region_slots.get_mut(player).ok_or_else(|| {
                    err(format!(
                        "(regions {player} ...) is out of range for {num_players} players"
                    ))
                })?;
                *slot = Some(edges);
            }
            other => return Err(err(format!("unknown game clause ({other} ...)"))),
        }
    }

    let move_gen = move_gen.ok_or_else(|| err("missing (moves ...) clause"))?;
    let player_regions = if any_regions {
        region_slots
            .into_iter()
            .enumerate()
            .map(|(i, slot)| {
                slot.ok_or_else(|| err(format!("missing (regions {i} ...) for player {i}")))
            })
            .collect::<Result<Vec<_>, Error>>()?
    } else {
        Vec::new()
    };

    Ok(Program {
        topology,
        num_players,
        move_gen,
        end,
        player_regions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tic_tac_toe_matches_a_hand_built_program() {
        // Same hand-built value `core::interp`'s own `manual_program_matches_parsed_one` test
        // checks the interpreter against -- duplicated here rather than shared, since the two
        // tests are checking different things (this parser's output vs. the interpreter's
        // fixture-loading), per "Core IR should be constructible and checkable by hand."
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
        let program = parse_game(include_str!("../../style-c/sexpr/tic-tac-toe.sc")).unwrap();
        assert_eq!(program, manual);
    }

    #[test]
    fn hex_matches_a_hand_built_program() {
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
        let program = parse_game(include_str!("../../style-c/sexpr/hex.sc")).unwrap();
        assert_eq!(program, manual);
    }

    #[test]
    fn y_matches_a_hand_built_program() {
        // Checked against a hand-built `Program`, the same "Core IR should be constructible and
        // checkable by hand" discipline `core::interp`'s own manual-Program tests use.
        let hex = Hex {
            side: 4,
            shape: HexShape::Triangle,
        };
        let empty = Region::Complement(Box::new(Region::Union(
            Box::new(Region::Occupied(Player(0))),
            Box::new(Region::Occupied(Player(1))),
        )));
        let three_sides = vec![
            Region::Sites(hex.triangle_edge(TriangleEdge::Bottom)),
            Region::Sites(hex.triangle_edge(TriangleEdge::Left)),
            Region::Sites(hex.triangle_edge(TriangleEdge::Hypotenuse)),
        ];
        let manual = Program {
            topology: Topology::Hex(hex),
            num_players: 2,
            move_gen: MoveGen {
                to: Region::Intersect(Box::new(empty), Box::new(Region::Sites(hex.valid_sites()))),
            },
            end: vec![EndRule {
                condition: BoolExpr::Connects {
                    conn: Connectivity::Six,
                },
            }],
            player_regions: vec![three_sides.clone(), three_sides],
        };
        let program = parse_game(include_str!("../../style-c/sexpr/y.sc")).unwrap();
        assert_eq!(program, manual);
    }

    #[test]
    fn missing_topology_is_a_lowering_error_not_a_panic() {
        let result = parse_game(r#"(game "X" (players 2) (moves (sites Empty)))"#);
        assert!(result.is_err());
    }

    #[test]
    fn unknown_clause_is_a_lowering_error() {
        let result = parse_game(
            r#"(game "X" (topology (rect 3 3)) (players 2) (moves (sites Empty)) (bogus 1))"#,
        );
        assert!(result.is_err());
    }
}
