//! Graph operator ludemes (Language Reference 4.11): transform, combine, or modify board
//! graphs built by [`crate::ast::graph::generator`] or [`crate::ast::graph::shape`] ludemes.

use crate::ast::common::Poly;
use crate::ast::graph::GraphFunction;
use crate::ast::located::LBox;
use crate::ast::numeric::dim::DimFunction;
use crate::ast::numeric::float::FloatFunction;
use crate::ast::types::SiteType;

/// A 2D point whose coordinates are computed by [`FloatFunction`]s.
pub type Point2F = (LBox<FloatFunction>, LBox<FloatFunction>);

/// How the edges added by [`Add`] are specified: as explicit endpoint locations, or as index
/// pairs into the graph's existing/newly-added vertices.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeSpec {
    Points(Vec<(Point2F, Point2F)>),
    Indices(Vec<(LBox<DimFunction>, LBox<DimFunction>)>),
}

/// How the faces added by [`Add`] are specified: as explicit vertex locations, or as index
/// lists into the graph's existing/newly-added vertices.
#[derive(Debug, Clone, PartialEq)]
pub enum CellSpec {
    Points(Vec<Vec<Point2F>>),
    Indices(Vec<Vec<LBox<DimFunction>>>),
}

/// `(add ...)` (4.11.1): adds vertices, edges, and/or faces to a graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Add {
    pub base: Option<LBox<GraphFunction>>,
    pub vertices: Vec<Point2F>,
    pub edges: Option<EdgeSpec>,
    /// Curved edges: each entry is the list of points (endpoints and tangents) defining one
    /// curve.
    pub edges_curved: Vec<Vec<Point2F>>,
    pub cells: Option<CellSpec>,
    pub connect: Option<bool>,
}

/// `(clip ...)` (4.11.2): clips a graph to a polygon.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub graph: LBox<GraphFunction>,
    pub region: Poly,
}

/// `(complete ...)` (4.11.3): creates an edge between every pair of vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct Complete {
    pub graph: LBox<GraphFunction>,
    pub each_cell: Option<bool>,
}

/// `(dual ...)` (4.11.4): the weak dual of a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Dual(pub LBox<GraphFunction>);

/// `(hole ...)` (4.11.5): cuts a polygonal hole in a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Hole {
    pub graph: LBox<GraphFunction>,
    pub region: Poly,
}

/// `(intersect ...)` (4.11.6): the intersection of two or more graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Intersect {
    pub graphs: Vec<LBox<GraphFunction>>,
}

/// `(keep ...)` (4.11.7): keeps only the part of a graph within a polygon.
#[derive(Debug, Clone, PartialEq)]
pub struct Keep {
    pub graph: LBox<GraphFunction>,
    pub region: Poly,
}

/// `(layers ...)` (4.11.8): stacks multiple copies of a graph for 3D games.
#[derive(Debug, Clone, PartialEq)]
pub struct Layers {
    pub count: LBox<DimFunction>,
    pub graph: LBox<GraphFunction>,
}

/// `(makeFaces ...)` (4.11.9): recreates all non-overlapping faces of a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct MakeFaces(pub LBox<GraphFunction>);

/// `(merge ...)` (4.11.10): overlays two or more graphs, merging incident vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct Merge {
    pub graphs: Vec<LBox<GraphFunction>>,
    pub connect: Option<bool>,
}

/// `(recoordinate ...)` (4.11.11): regenerates coordinate labels for a graph's elements.
#[derive(Debug, Clone, PartialEq)]
pub struct Recoordinate {
    pub site_types: Vec<SiteType>,
    pub graph: LBox<GraphFunction>,
}

/// How the elements removed by [`Remove`] are specified: as explicit coordinates, or as
/// indices into the existing graph.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveCells {
    Points(Vec<Vec<(f64, f64)>>),
    Indices(Vec<LBox<DimFunction>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RemoveEdges {
    Points(Vec<((f64, f64), (f64, f64))>),
    Indices(Vec<(LBox<DimFunction>, LBox<DimFunction>)>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RemoveVertices {
    Points(Vec<(f64, f64)>),
    Indices(Vec<LBox<DimFunction>>),
}

/// `(remove ...)` (4.11.12): removes vertices, edges, and/or faces from a graph, either by
/// coordinate/index, or by clipping to a polygonal hole.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Remove {
    pub graph: Option<LBox<GraphFunction>>,
    pub cells: Option<RemoveCells>,
    pub edges: Option<RemoveEdges>,
    pub vertices: Option<RemoveVertices>,
    pub region: Option<Poly>,
    pub trim_edges: Option<bool>,
}

/// `(renumber ...)` (4.11.13): renumbers a graph's vertices into sequential order.
#[derive(Debug, Clone, PartialEq)]
pub struct Renumber {
    pub site_types: Vec<SiteType>,
    pub graph: LBox<GraphFunction>,
}

/// `(rotate ...)` (4.11.14): rotates a graph about its midpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotate {
    pub degrees: LBox<FloatFunction>,
    pub graph: LBox<GraphFunction>,
}

/// `(scale ...)` (4.11.15): scales a graph along each axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    pub x: LBox<FloatFunction>,
    pub y: Option<LBox<FloatFunction>>,
    pub z: Option<LBox<FloatFunction>>,
    pub graph: LBox<GraphFunction>,
}

/// `(shift ...)` (4.11.16): translates a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Shift {
    pub x: LBox<FloatFunction>,
    pub y: LBox<FloatFunction>,
    pub z: Option<LBox<FloatFunction>>,
    pub graph: LBox<GraphFunction>,
}

/// `(skew ...)` (4.11.17): skews a graph by a given amount (1.0 gives a 45-degree skew).
#[derive(Debug, Clone, PartialEq)]
pub struct Skew {
    pub amount: f64,
    pub graph: LBox<GraphFunction>,
}

/// `(splitCrossings ...)` (4.11.18): splits edge crossings, adding a vertex at each crossing
/// point.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitCrossings(pub LBox<GraphFunction>);

/// `(subdivide ...)` (4.11.19): subdivides faces about their midpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Subdivide {
    pub graph: LBox<GraphFunction>,
    pub min: Option<LBox<DimFunction>>,
}

/// `(trim ...)` (4.11.20): removes orphan vertices and edges.
#[derive(Debug, Clone, PartialEq)]
pub struct Trim(pub LBox<GraphFunction>);

/// `(union ...)` (4.11.21): the union of two or more graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Union {
    pub graphs: Vec<LBox<GraphFunction>>,
    pub connect: Option<bool>,
}
