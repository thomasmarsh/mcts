//! The cross-cutting enumerated "Types" ludemes (Language Reference chapter 16), plus the
//! direction and turtle-step vocabularies from chapter 15.1 that they depend on. These are
//! constant values (`UpperCamelCase` in `.lud` source) referenced as plain keyword arguments
//! throughout the rest of the grammar, e.g. `(piece "Pawn" P1)` uses [`RoleType::P1`].

/// The owner/role of a piece of equipment, or the subject of a rule (`roleType`, 16.3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoleType {
    Neutral,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
    P16,
    Team1,
    Team2,
    Team3,
    Team4,
    Team5,
    Team6,
    Team7,
    Team8,
    Team9,
    Team10,
    Team11,
    Team12,
    Team13,
    Team14,
    Team15,
    Team16,
    TeamMover,
    Each,
    Shared,
    All,
    Mover,
    Next,
    Prev,
    NonMover,
    Enemy,
    Friend,
    Ally,
    Player,
}

/// The type of graph element a site refers to (`siteType`, 16.1.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SiteType {
    Vertex,
    Edge,
    Cell,
}

/// A single "turtle graphics" step used to describe walks through the board graph
/// (`stepType`, 16.1.10). A walk is a sequence of these, e.g. `{ F F R F }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepType {
    /// Forward a step.
    F,
    /// Turn left a step.
    L,
    /// Turn right a step.
    R,
}

/// A turtle-graphics walk, e.g. `{ F F R F }`.
pub type Walk = Vec<StepType>;

/// Categories of absolute (board-relative, not player-relative) directions (15.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsoluteDirection {
    All,
    Angled,
    Adjacent,
    Axial,
    Orthogonal,
    Diagonal,
    OffDiagonal,
    SameLayer,
    Upward,
    Downward,
    Rotational,
    Base,
    Support,
    N,
    E,
    S,
    W,
    NE,
    SE,
    NW,
    SW,
    NNW,
    WNW,
    WSW,
    SSW,
    SSE,
    ESE,
    ENE,
    NNE,
    CW,
    CCW,
    In,
    Out,
    U,
    UN,
    UNE,
    UE,
    USE,
    US,
    USW,
    UW,
    UNW,
    D,
    DN,
    DNE,
    DE,
    DSE,
    DS,
    DSW,
    DW,
    DNW,
}

/// The 16-point compass rose, used e.g. by `(sites Side ...)` (15.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompassDirection {
    N,
    NNE,
    NE,
    ENE,
    E,
    ESE,
    SE,
    SSE,
    S,
    SSW,
    SW,
    WSW,
    W,
    WNW,
    NW,
    NNW,
}

/// Directions relative to a player's facing (15.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativeDirection {
    Forward,
    Backward,
    Rightward,
    Leftward,
    Forwards,
    Backwards,
    Rightwards,
    Leftwards,
    FL,
    FLL,
    FLLL,
    BL,
    BLL,
    BLLL,
    FR,
    FRR,
    FRRR,
    BR,
    BRR,
    BRRR,
    SameDirection,
    OppositeDirection,
}

/// Whether to read a stack from the bottom or the top (`stackDirection`, 15.1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StackDirection {
    FromBottom,
    FromTop,
}

/// The outcome of a game for a player/team (`resultType`, 16.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultType {
    Win,
    Loss,
    Draw,
    Tie,
    Abandon,
    Crash,
}

/// The mode of play (`modeType`, 16.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModeType {
    Alternating,
    Simultaneous,
    Simulation,
}

/// When to perform certain tests or actions within a turn (`whenType`, 16.3.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhenType {
    StartOfTurn,
    EndOfTurn,
}

/// Types of state repetition that a `(no Repeat ...)` meta rule can forbid (16.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepetitionType {
    SituationalInTurn,
    PositionalInTurn,
    Positional,
    Situational,
}

/// The result applied when all players pass in succession (`passEndType`, 16.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassEndType {
    Draw,
    NoEnd,
}

/// Which previous state `(prev ...)` refers to (16.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrevType {
    Mover,
    MoverLastTurn,
}

/// Which aspect of a site/component is hidden from a player (`hiddenData`, 16.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HiddenData {
    What,
    Who,
    State,
    Count,
    Rotation,
    Value,
}

/// How two graph elements relate to one another (`relationType`, 16.1.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    Orthogonal,
    Diagonal,
    OffDiagonal,
    Adjacent,
    All,
}

/// A known board shape (`shapeType`, 16.1.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeType {
    NoShape,
    Custom,
    Square,
    Rectangle,
    Triangle,
    Hexagon,
    Cross,
    Diamond,
    Prism,
    Quadrilateral,
    Rhombus,
    Wheel,
    Circle,
    Spiral,
    Wedge,
    Star,
    Limping,
    Regular,
    Polygon,
}

/// The type of a store on a Mancala-style board (`storeType`, 16.1.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreType {
    None,
    Outer,
    Inner,
}

/// Supported tilings for boardless containers (`tilingBoardlessType`, 16.1.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TilingBoardlessType {
    Square,
    Triangular,
    Hexagonal,
}

/// Known board tilings (`basisType`, 16.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisType {
    NoBasis,
    Triangular,
    Square,
    Hexagonal,
    T33336,
    T33344,
    T33434,
    T3464,
    T3636,
    T4612,
    T488,
    T31212,
    /// Tiling 3.3.3.3.3.3,3.3.4.3.4.
    T333333_33434,
    SquarePyramidal,
    HexagonalPyramidal,
    Concentric,
    Circle,
    Spiral,
    Dual,
    Brick,
    Mesh,
    Morris,
    Celtic,
    QuadHex,
}

/// A named landmark site on the board (`landmarkType`, 16.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LandmarkType {
    CentreSite,
    LeftSite,
    RightSite,
    Topsite,
    BottomSite,
    FirstSite,
}

/// The kind of variable used in a deduction puzzle (`puzzleElementType`, 16.1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PuzzleElementType {
    Cell,
    Edge,
    Vertex,
    Hint,
}

/// Regions that change during play (`regionTypeDynamic`, 16.1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionTypeDynamic {
    Empty,
    NotEmpty,
    Own,
    NotOwn,
    Enemy,
    NotEnemy,
}

/// Predefined, static regions of the board (`regionTypeStatic`, 16.1.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionTypeStatic {
    Rows,
    Columns,
    AllDirections,
    HintRegions,
    Layers,
    Diagonals,
    SubGrids,
    Regions,
    Vertices,
    Corners,
    Sides,
    SidesNoCorners,
}

/// The rank of a playing card (`cardType`, 16.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardType {
    Joker,
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
}

/// The suit of a playing card (`suitType`, 16.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuitType {
    Clubs,
    Spades,
    Diamonds,
    Hearts,
}

/// Which kind of component a `(deal ...)` rule distributes (`dealableType`, 16.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DealableType {
    Dominoes,
    Cards,
}
