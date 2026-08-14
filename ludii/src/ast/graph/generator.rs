//! Graph generator ludemes based on a named tiling (Language Reference 4.1-4.8): each builds a
//! board graph either from row/column dimensions, or from a bounding polygon/side-length list.

use crate::ast::common::Poly;
use crate::ast::located::LBox;
use crate::ast::numeric::dim::DimFunction;

/// Either a fixed number of rows/columns, or a polygon/side-length outline -- the two ways
/// most tiling generators accept their extent.
#[derive(Debug, Clone, PartialEq)]
pub enum Extent {
    Dims(LBox<DimFunction>, Option<LBox<DimFunction>>),
    Poly(Poly),
    Sides(Vec<LBox<DimFunction>>),
}

/// `brickShapeType` (4.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickShapeType {
    Square,
    Rectangle,
    Diamond,
    Prism,
    Spiral,
    Limping,
}

/// `(brick ...)` (4.1.1): a board on a 1x2 rectangular brick tiling.
#[derive(Debug, Clone, PartialEq)]
pub struct Brick {
    pub shape: Option<BrickShapeType>,
    pub rows: LBox<DimFunction>,
    pub columns: Option<LBox<DimFunction>>,
    pub trim: Option<bool>,
}

/// `(celtic ...)` (4.2.1): a board based on Celtic knotwork.
#[derive(Debug, Clone, PartialEq)]
pub struct Celtic {
    pub extent: Extent,
}

/// `hexShapeType` (4.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexShapeType {
    NoShape,
    Square,
    Rectangle,
    Diamond,
    Triangle,
    Hexagon,
    Star,
    Limping,
    Prism,
}

/// `(hex ...)` (4.3.1): a board on a hexagonal tiling.
#[derive(Debug, Clone, PartialEq)]
pub struct Hex {
    pub shape: Option<HexShapeType>,
    pub extent: Extent,
}

/// `(quadhex ...)` (4.4.1): a hexagon tessellated by quadrilaterals, as used for Three Player
/// Chess.
#[derive(Debug, Clone, PartialEq)]
pub struct Quadhex {
    pub layers: LBox<DimFunction>,
    pub thirds: Option<bool>,
}

/// `diagonalsType` (4.5.1): how to handle diagonal relations on square-based tilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalsType {
    Implied,
    Solid,
    Alternating,
    Concentric,
    Radiating,
}

/// `squareShapeType` (4.5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquareShapeType {
    NoShape,
    Square,
    Rectangle,
    Diamond,
    Limping,
}

/// How a square-tiled board handles diagonals or pyramidal stacking.
#[derive(Debug, Clone, PartialEq)]
pub enum SquareModifier {
    Diagonals(DiagonalsType),
    Pyramidal(bool),
}

/// `(square ...)` (4.5.2): a board on a square tiling.
#[derive(Debug, Clone, PartialEq)]
pub struct Square {
    pub shape: Option<SquareShapeType>,
    pub extent: Extent,
    pub modifier: Option<SquareModifier>,
}

/// `tilingType` (4.6.2): known non-regular tilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingType {
    T31212,
    T3464,
    T488,
    T33434,
    T33336,
    T33344,
    T3636,
    T4612,
    /// Tiling 3.3.3.3.3.3,3.3.4.3.4.
    T333333_33434,
}

/// `(tiling ...)` (4.6.1): a board graph built from a known tiling type and size.
#[derive(Debug, Clone, PartialEq)]
pub struct Tiling {
    pub tiling_type: TilingType,
    pub extent: Extent,
}

/// `tiling3464ShapeType` (4.7.1): known shapes for the rhombitrihexahedral (3.4.6.4) tiling,
/// e.g. as used for the Kensington board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tiling3464ShapeType {
    Custom,
    Square,
    Rectangle,
    Diamond,
    Prism,
    Triangle,
    Hexagon,
    Star,
    Limping,
}

/// `triShapeType` (4.8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriShapeType {
    Square,
    Rectangle,
    Diamond,
    Triangle,
    Hexagon,
    Star,
    Limping,
    Prism,
}

/// `(tri ...)` (4.8.1): a board on a triangular tiling.
#[derive(Debug, Clone, PartialEq)]
pub struct Tri {
    pub shape: Option<TriShapeType>,
    pub extent: Extent,
}
