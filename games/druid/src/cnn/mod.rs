//! CNN value+policy evaluation for Druid. The network itself is `grid-cnn` (geometry as data);
//! this module holds what is specific to the game: the input encoding and the action-id mapping
//! (always built), and, behind the `cnn` feature, the batched search oracle, the self-play
//! driver and its shard format, and a play-time search agent.

pub mod encode;

#[cfg(feature = "cnn")]
pub mod agent;
#[cfg(feature = "cnn")]
pub mod config;
#[cfg(feature = "cnn")]
pub mod oracle;
#[cfg(feature = "cnn")]
pub mod search;
#[cfg(feature = "cnn")]
pub mod selfplay;
#[cfg(feature = "cnn")]
pub mod shard;

/// Board sizes that have a compiled-in network agent (`CnnAgent<N>` is const-generic). To add a
/// size, add it here and as an arm of [`with_board_size!`]; a test checks the two agree.
pub const SUPPORTED_SIZES: &[usize] = &[5, 7, 9, 10];

/// Runs `$body` with the const `$n` bound to the runtime board size `$size`, or `$fallback`
/// (with the unsupported size bound to `$other`) when no network agent is compiled in for it.
#[macro_export]
macro_rules! with_board_size {
    ($size:expr, $n:ident => $body:expr, $other:ident => $fallback:expr $(,)?) => {
        match $size {
            5 => {
                const $n: usize = 5;
                $body
            }
            7 => {
                const $n: usize = 7;
                $body
            }
            9 => {
                const $n: usize = 9;
                $body
            }
            10 => {
                const $n: usize = 10;
                $body
            }
            $other => $fallback,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::SUPPORTED_SIZES;

    #[test]
    fn with_board_size_dispatches_exactly_the_supported_sizes() {
        for &size in SUPPORTED_SIZES {
            assert_eq!(crate::with_board_size!(size, N => N, _o => 0), size);
        }
        assert_eq!(crate::with_board_size!(8usize, N => N, o => o + 1000), 1008);
    }
}
