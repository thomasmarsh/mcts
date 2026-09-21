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

/// Board sizes that have a compiled-in network agent (`CnnAgent<N>` is const-generic). To add a
/// size, add it here and as an arm of [`with_board_size!`]; a test checks the two agree.
pub const SUPPORTED_SIZES: &[usize] = &[7, 9];

/// Runs `$body` with the const `$n` bound to the runtime board size `$size`, or `$fallback`
/// (with the unsupported size bound to `$other`) when no network agent is compiled in for it.
#[macro_export]
macro_rules! with_board_size {
    ($size:expr, $n:ident => $body:expr, $other:ident => $fallback:expr $(,)?) => {
        match $size {
            7 => {
                const $n: usize = 7;
                $body
            }
            9 => {
                const $n: usize = 9;
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
