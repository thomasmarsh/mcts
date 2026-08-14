//! Direction functions (Language Reference chapter 13): ludemes returning a set of directions,
//! converting between player-relative and absolute/compass directions as needed.

use crate::ast::boolean::BooleanFunction;
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;
use crate::ast::types::{AbsoluteDirection, RelationType, RelativeDirection, SiteType};

/// Any ludeme that computes a set of directions.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectionFunction {
    /// `(directions (<absoluteDirection> | {<absoluteDirection>}))` (13.2.1).
    Absolute(Vec<AbsoluteDirection>),
    /// `(directions ([<relativeDirection>] | [{<relativeDirection>}]) of:... bySite:...)`
    /// (13.2.1): converted to absolute directions relative to a player's facing.
    Relative {
        directions: Vec<RelativeDirection>,
        of: Option<RelationType>,
        by_site: Option<bool>,
    },
    /// `(directions <siteType> from:<int> to:<int>)` (13.2.1): the direction from one site to
    /// another.
    Between {
        site_type: SiteType,
        from: LBox<IntFunction>,
        to: LBox<IntFunction>,
    },
    /// `(directions Random <direction> num:<int>)` (13.2.1).
    Random {
        source: LBox<DirectionFunction>,
        num: LBox<IntFunction>,
    },
    /// `(difference <direction> <direction>)` (13.1.1).
    Difference(LBox<DirectionFunction>, LBox<DirectionFunction>),
    /// `(if <boolean> <direction> <direction>)` (13.3.1).
    If {
        condition: LBox<BooleanFunction>,
        then: LBox<DirectionFunction>,
        otherwise: LBox<DirectionFunction>,
    },
    /// `(union <direction> <direction>)` (13.4.1).
    Union(LBox<DirectionFunction>, LBox<DirectionFunction>),
}
