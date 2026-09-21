//! CNN value+policy evaluation for Gonnect: the input encoding, the batched search oracle, the
//! self-play driver and its shard format, and a play-time search agent. The network itself is
//! `grid-cnn` (geometry as data); nothing in it knows the rules.

pub mod agent;
pub mod config;
pub mod encode;
pub mod oracle;
pub mod search;
pub mod selfplay;
pub mod shard;
