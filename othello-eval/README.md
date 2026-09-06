# othello-eval

numpy reader for the fixed-width position records written by
`game-othello dump`, for offline learned-evaluation work. `uv`-managed,
numpy-only.

```sh
cd othello-eval
uv sync
uv run pytest        # includes a byte-exact Rust<->Python round-trip
uv run pyright
uv run ruff check .
```

## Record format

22 bytes, packed little-endian, one per self-play position:

| field  | type   | bytes | meaning |
|--------|--------|-------|---------|
| black  | u64 LE | 8     | black disc bitboard (bit `i` = square `i`, row-major from a1) |
| white  | u64 LE | 8     | white disc bitboard |
| side   | u8     | 1     | 0 = black to move, 1 = white to move |
| ply    | u8     | 1     | discs on board minus 4 (0 at the opening) |
| target | f32 LE | 4     | final result from the side-to-move perspective: +1 win, −1 loss, 0 draw |

```python
from othello_eval import load_positions
rows = load_positions("positions.bin")   # structured np.ndarray, itemsize 22
rows["black"], rows["target"], ...
```

## Producing a dump

```sh
cargo run --release -p game-othello -- dump \
    --out positions.bin --games 100000 --seed 0 --label outcome \
    --manifest positions.json      # optional JSON sidecar (round-trip golden)
```

`--label outcome` (uniform-random self-play, outcome labels) is the only
mode currently implemented; `treestrap` / `root_value` search-labelled
harvest modes are not built yet.
