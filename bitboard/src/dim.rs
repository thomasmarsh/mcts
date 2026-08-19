/// A board dimension (row or column count) that is either known at compile
/// time (`Const<N>`) or carried as a runtime value (`Dyn`). `Board` is
/// generic over this so the same shift/flood/wall logic serves both a
/// fixed-size game board and Gonnect/AtariGo's single runtime-sized board,
/// without duplicating the implementation.
pub trait Dim: Copy {
    fn get(self) -> usize;
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
}
