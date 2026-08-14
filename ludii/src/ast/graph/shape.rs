//! Graph shape ludemes (Language Reference 4.9-4.10): plain geometric outlines and repeating
//! patterns, as opposed to the named tilings in [`crate::ast::graph::generator`].

use crate::ast::common::Poly;
use crate::ast::graph::generator::DiagonalsType;
use crate::ast::located::LBox;
use crate::ast::numeric::dim::DimFunction;

/// `(rectangle ...)` (4.9.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub rows: LBox<DimFunction>,
    pub columns: Option<LBox<DimFunction>>,
    pub diagonals: Option<DiagonalsType>,
}

/// `(regular ...)` (4.9.2): a regular polygon (or, with the `Star` flag, a star polygon).
#[derive(Debug, Clone, PartialEq)]
pub struct Regular {
    pub star: bool,
    pub sides: LBox<DimFunction>,
}

/// `(repeat ...)` (4.9.3): repeats one or more shapes across a grid of rows/columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Repeat {
    pub rows: LBox<DimFunction>,
    pub columns: LBox<DimFunction>,
    pub step: Vec<(f64, f64)>,
    pub shapes: Vec<Poly>,
}

/// `(spiral ...)` (4.9.4): a board based on a spiral tiling, e.g. the Mehen board.
#[derive(Debug, Clone, PartialEq)]
pub struct Spiral {
    pub turns: LBox<DimFunction>,
    pub sites: LBox<DimFunction>,
    pub clockwise: Option<bool>,
}

/// `(wedge ...)` (4.9.5): a triangular wedge, one vertex at the top and three along the
/// bottom -- used to add triangular arms to Alquerque-style boards.
#[derive(Debug, Clone, PartialEq)]
pub struct Wedge {
    pub rows: LBox<DimFunction>,
    pub columns: Option<LBox<DimFunction>>,
}

/// `concentricShapeType` (4.10.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcentricShapeType {
    Square,
    Triangle,
    Hexagon,
    Target,
}

/// How the rings of a [`Concentric`] board are specified.
#[derive(Debug, Clone, PartialEq)]
pub enum ConcentricSpec {
    Shape(ConcentricShapeType),
    Sides(LBox<DimFunction>),
    CellsPerRing(Vec<LBox<DimFunction>>),
}

/// `(concentric ...)` (4.10.1): a board tiled from concentric rings, e.g. Morris/Merels
/// boards.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Concentric {
    pub spec: Option<ConcentricSpec>,
    pub rings: Option<LBox<DimFunction>>,
    pub steps: Option<LBox<DimFunction>>,
    pub midpoints: Option<bool>,
    pub join_midpoints: Option<bool>,
    pub join_corners: Option<bool>,
    pub stagger: Option<bool>,
}
