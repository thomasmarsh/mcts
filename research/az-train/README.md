# az-train

Self-play trainer for a Gumbel-style AlphaZero loop over the small games in
this workspace: read `game-<kind> dump` records, refit an evaluator, repeat
one generation at a time.

Current scope: read **v2** records, fit a value head by least-squares, write
a flat `f32` `weights.bin`. Two heads: `--head ntuple` (default) is the
541-weight tic-tac-toe n-tuple head; `--head linear` is the 19-weight linear
head. No policy head yet (the recorded policy tail is carried through but not
trained against), no neural net.

## Layout

- `az_train.records` -- the v2 record codec (`decode_records` /
  `encode_records` / `load_positions`) and `me_opp_planes` featurisation.
  Records are variable-width (fixed 11-byte head + `(u8, f32)` policy tail),
  so this walks them sequentially rather than `np.fromfile`.
- `az_train.model` -- the 19-weight linear value head: `[bias, me[0..9],
  opp[0..9]]`, score squashed through `tanh`. This is the exact layout the
  Rust `Evaluator` that consumes `weights.bin` reads.
- `az_train.ntuple` -- the n-tuple value head: a bias plus one `3^k` weight
  table per structural line (8 tuples of 3 cells) and per 2x2 square (4
  tuples of 4 cells), base-3 indexed by `{empty, mover, opponent}` per cell,
  score squashed through `tanh`. Flat layout `[bias, line[0..8]*27,
  square[0..4]*81]` = 541 f32, mirrored in `game_ttt::valuenet::NTupleValueNet`.
- `az_train.train` -- the `az-train` CLI, one generation's refit.

## Use

```sh
uv sync
uv run pytest                      # includes a byte-exact round-trip vs `game-ttt dump`
cargo run -p game-ttt -- dump --out shards/gen_0.bin --games 300 --seed 0
uv run az-train --head ntuple --positions shards/gen_0.bin --out weights/gen_1.bin
```
