//! JSON-serializable search configuration, split by strategy axis.
//!
//! Each axis module owns its specification enum and runtime dispatcher;
//! [`search`] combines the four resolved components into a runnable search.

mod backprop;
mod codec;
mod final_action;
mod search;
mod select;
mod simulate;

pub use backprop::*;
pub use final_action::*;
pub use search::*;
pub use select::*;
pub use simulate::*;

#[cfg(test)]
mod tests;
