//! Graph function ludemes (Language Reference chapter 4): build and transform the graph
//! (vertices, edges, faces) that underlies a board.

pub mod generator;
pub mod operator;
pub mod shape;

use crate::ast::common::GraphLiteral;

/// Any ludeme that produces a board graph: a generator, a shape, an operator applied to a
/// sub-graph, or an explicit `(graph vertices:{...} edges:{...})` literal.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphFunction {
    Literal(GraphLiteral),

    Brick(generator::Brick),
    Celtic(generator::Celtic),
    Hex(generator::Hex),
    Quadhex(generator::Quadhex),
    Square(generator::Square),
    Tiling(generator::Tiling),
    Tri(generator::Tri),

    Rectangle(shape::Rectangle),
    Regular(shape::Regular),
    Repeat(shape::Repeat),
    Spiral(shape::Spiral),
    Wedge(shape::Wedge),
    Concentric(shape::Concentric),

    Add(operator::Add),
    Clip(operator::Clip),
    Complete(operator::Complete),
    Dual(operator::Dual),
    Hole(operator::Hole),
    Intersect(operator::Intersect),
    Keep(operator::Keep),
    Layers(operator::Layers),
    MakeFaces(operator::MakeFaces),
    Merge(operator::Merge),
    Recoordinate(operator::Recoordinate),
    Remove(operator::Remove),
    Renumber(operator::Renumber),
    Rotate(operator::Rotate),
    Scale(operator::Scale),
    Shift(operator::Shift),
    Skew(operator::Skew),
    SplitCrossings(operator::SplitCrossings),
    Subdivide(operator::Subdivide),
    Trim(operator::Trim),
    Union(operator::Union),
}
