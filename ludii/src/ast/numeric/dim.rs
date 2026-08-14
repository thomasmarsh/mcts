//! Dimension functions (Language Reference chapter 5): small integer-valued math expressions
//! used for board dimensions (e.g. `(square 8)`, `(hex Diamond 11)`).

use crate::ast::located::LBox;

/// Any ludeme that computes a board-dimension integer.
#[derive(Debug, Clone, PartialEq)]
pub enum DimFunction {
    Int(i64),
    Abs(LBox<DimFunction>),
    Add(Vec<LBox<DimFunction>>),
    Div(LBox<DimFunction>, LBox<DimFunction>),
    Max(LBox<DimFunction>, LBox<DimFunction>),
    Min(LBox<DimFunction>, LBox<DimFunction>),
    Mul(Vec<LBox<DimFunction>>),
    Pow(LBox<DimFunction>, LBox<DimFunction>),
    Sub(LBox<DimFunction>, LBox<DimFunction>),
}
