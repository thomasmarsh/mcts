/// The raw word array backing a `Board`. Implemented for `u64` (a single
/// word, replacing `game_core::bitboard::BitBoard`'s backing store) and
/// `[u64; WORDS]` (replacing `game_core::bigbitboard::BigBitBoard`'s),
/// letting `Board` implement every bit-level and shift/flood operation
/// exactly once, generically over the word count, instead of hand-duplicated
/// per-type.
pub trait Storage: Copy + Send + 'static {
    const CAPACITY_WORDS: usize;

    fn zero() -> Self;
    fn word(&self, index: usize) -> u64;
    fn word_mut(&mut self, index: usize) -> &mut u64;
}

impl Storage for u64 {
    const CAPACITY_WORDS: usize = 1;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn word(&self, index: usize) -> u64 {
        debug_assert_eq!(index, 0);
        *self
    }

    #[inline(always)]
    fn word_mut(&mut self, index: usize) -> &mut u64 {
        debug_assert_eq!(index, 0);
        self
    }
}

impl<const WORDS: usize> Storage for [u64; WORDS] {
    const CAPACITY_WORDS: usize = WORDS;

    #[inline(always)]
    fn zero() -> Self {
        [0; WORDS]
    }

    #[inline(always)]
    fn word(&self, index: usize) -> u64 {
        self[index]
    }

    #[inline(always)]
    fn word_mut(&mut self, index: usize) -> &mut u64 {
        &mut self[index]
    }
}
