"""Minimal local stub covering only the scrambled-Sobol engine this project uses."""

from collections.abc import Sequence

class _Points:
    def tolist(self) -> list[list[float]]: ...

class Sobol:
    def __init__(
        self,
        d: int,
        *,
        scramble: bool = ...,
        bits: int | None = ...,
        rng: int | None = ...,
        seed: int | None = ...,
        optimization: str | None = ...,
    ) -> None: ...
    def random(self, n: int = ...) -> _Points: ...
    def random_base2(self, m: int) -> _Points: ...
    def reset(self) -> Sobol: ...
    def fast_forward(self, n: int) -> Sobol: ...

__all__ = ["Sobol"]

def scale(
    sample: Sequence[Sequence[float]],
    l_bounds: Sequence[float],
    u_bounds: Sequence[float],
    *,
    reverse: bool = ...,
) -> _Points: ...
