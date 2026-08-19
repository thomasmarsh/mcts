/// A board dimension (row or column count) that is either known at compile
/// time (`Const<N>`) or carried as a runtime value (`Dyn`). `Board` is
/// generic over this so the same shift/flood/wall logic serves both a
/// fixed-size game board and Gonnect/AtariGo's single runtime-sized board,
/// without duplicating the implementation.
pub trait Dim: Copy {
    fn get(self) -> usize;

    /// Reconstructs a dimension from a runtime length, e.g. read back off
    /// the wire by `Board`'s `Deserialize` impl. `Const<N>` has no runtime
    /// state to restore, so it ignores `len` and always produces `N` --
    /// callers that need to catch a mismatched `len` (as `Deserialize` does)
    /// must check `.get()` against it themselves. `Dyn` just carries the
    /// length through.
    fn from_len(len: usize) -> Self;
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
}
