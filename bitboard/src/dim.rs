/// A board dimension (row or column count) that is either known at compile
/// time (`Const<N>`) or carried as a runtime value (`Dyn`). `Board` is
/// generic over this so the same shift/flood/wall logic serves both a
/// fixed-size game board and Gonnect/AtariGo's single runtime-sized board,
/// without duplicating the implementation.
pub trait Dim: Copy + std::fmt::Debug {
    fn get(self) -> usize;

    /// Reconstructs a dimension from a runtime length, e.g. read back off
    /// the wire by `Board`'s `Deserialize` impl. `Const<N>` has no runtime
    /// state to restore, so it ignores `len` and always produces `N` --
    /// callers that need to catch a mismatched `len` (as `Deserialize` does)
    /// must check `.get()` against it themselves. `Dyn` just carries the
    /// length through.
    fn from_len(len: usize) -> Self;

    /// Whether `Board::wall_words` should memoize its result in the
    /// process-wide cache. `Dyn` boards (e.g. Gonnect/AtariGo's single
    /// `Board<[u64; 6], Dyn, Dyn>` monomorphization serving `3..=19`) hit
    /// many distinct sizes from one monomorphization, so caching saves real
    /// recomputation. `Const<N>` boards are already one monomorphization
    /// *per size* -- every call ever made against that type has the same
    /// `rows`/`cols` -- so routing through the shared
    /// `Mutex<HashMap<..>>`/`TypeId`/`downcast_ref` machinery buys nothing
    /// over just recomputing the mask directly (a plain loop of at most
    /// `rows`/`cols` iterations, no synchronization), and costs a lock plus
    /// a hash lookup on every call instead.
    const CACHE_WALLS: bool;
}

/// Compile-time-known dimension, matching the const-generic size a game's
/// board type is instantiated with. A zero-sized type: `N` lives entirely in
/// the type, so `get()` should optimize down to the same constant a raw
/// `usize` const generic would produce.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Const<const N: usize>;

impl<const N: usize> Dim for Const<N> {
    #[inline(always)]
    fn get(self) -> usize {
        N
    }

    fn from_len(_len: usize) -> Self {
        Const
    }

    const CACHE_WALLS: bool = false;
}

/// Runtime-known dimension. Lets a single monomorphization of `Board` (and
/// everything built on it) serve every board size, e.g. Gonnect/AtariGo's
/// `3..=19`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dyn(pub usize);

impl Dim for Dyn {
    #[inline(always)]
    fn get(self) -> usize {
        self.0
    }

    fn from_len(len: usize) -> Self {
        Dyn(len)
    }

    const CACHE_WALLS: bool = true;
}
