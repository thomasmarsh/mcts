//! Range functions (Language Reference chapter 14): an inclusive lower/upper integer bound,
//! e.g. for capping bets or step distances.

use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;

/// Any ludeme that computes an inclusive integer range.
#[derive(Debug, Clone, PartialEq)]
pub enum RangeFunction {
    /// `(range <int> [<int>])` (14.1.1): `max` defaults to `min` when absent.
    Range {
        min: LBox<IntFunction>,
        max: Option<LBox<IntFunction>>,
    },
    /// `(exact <int>)` (14.2.1): a range containing exactly one value.
    Exact(LBox<IntFunction>),
    /// `(max <int>)` (14.2.2): a range with only an upper bound specified.
    Max(LBox<IntFunction>),
    /// `(min <int>)` (14.2.3): a range with only a lower bound specified.
    Min(LBox<IntFunction>),
}
