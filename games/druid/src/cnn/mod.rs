//! CNN value+policy evaluation for Druid. The network itself is `grid-cnn` (geometry as data);
//! this module holds what is specific to the game: the input encoding and the action-id mapping.

pub mod encode;
