//! Float functions (Language Reference chapter 6): floating-point-valued expressions, used
//! e.g. for graph rotation/scale amounts and match payoffs.

use crate::ast::boolean::BooleanFunction;
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;

/// The source value converted by `(toFloat ...)`.
#[derive(Debug, Clone, PartialEq)]
pub enum ToFloatSource {
    Bool(LBox<BooleanFunction>),
    Int(LBox<IntFunction>),
}

/// Any ludeme that computes a floating-point value.
#[derive(Debug, Clone, PartialEq)]
pub enum FloatFunction {
    Float(f64),
    ToFloat(ToFloatSource),
    Abs(LBox<FloatFunction>),
    Add(Vec<LBox<FloatFunction>>),
    Cos(LBox<FloatFunction>),
    Div(LBox<FloatFunction>, LBox<FloatFunction>),
    Exp(LBox<FloatFunction>),
    Log(LBox<FloatFunction>),
    Log10(LBox<FloatFunction>),
    Max(LBox<FloatFunction>, LBox<FloatFunction>),
    Min(LBox<FloatFunction>, LBox<FloatFunction>),
    Mul(Vec<LBox<FloatFunction>>),
    Pow(LBox<FloatFunction>, LBox<FloatFunction>),
    Sin(LBox<FloatFunction>),
    Sqrt(LBox<FloatFunction>),
    Sub(LBox<FloatFunction>, LBox<FloatFunction>),
    Tan(LBox<FloatFunction>),
}
