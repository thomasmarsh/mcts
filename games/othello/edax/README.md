# Edax — external Othello strength yardstick

Edax ([`abulmo/edax-reversi`](https://github.com/abulmo/edax-reversi)) is an
open-source alpha-beta Othello engine with a learned pattern evaluation,
superhuman on 8×8. It is used here as an *external* strength reference for
Othello work: `games/othello/examples/edax_match.rs` reports our engine's
strength as "Edax level N" instead of as a delta between two of our own
configs.

## Build

```sh
games/othello/edax/build-edax.sh          # clone + build + fetch weights + self-test
games/othello/edax/build-edax.sh --force  # wipe vendor/ and redo from scratch
```

Everything lands under `vendor/` (gitignored):

| path | what |
|---|---|
| `vendor/` | full clone of the upstream repo |
| `vendor/bin/mEdax-native` | the engine, built `-O3 -flto -march=native` for the host |
| `vendor/data/eval.dat` | pattern-evaluation weights (~13.3 MB) |

- **Pinned commit:** `14f048c05ddfa385b6bf954a9c2905bbe677e9d3` (tag `v4.6`).
- **Build line (Apple Silicon, macOS):**
  `make -C vendor/src build ARCH=native OS=osx`
  (single `clang -std=c17 -O3 -flto -ffast-math -march=native -mdynamic-no-pic all.c`).
- **Weights:** extracted from the `edax-4.6-linux-x86.tar.gz` release asset
  (`data/eval.dat` is platform-independent, so the linux archive is used to
  avoid needing a 7-Zip extractor for the separate `eval.7z` asset).
  sha256 `f8b2299612d9fa4414157e70e932636e33111c2602d0c2fc382a7d90ef21b792`.
  The `v4.4`-era `eval.dat` (same 13,952,436-byte size) also loads into this
  build and plays; `v4.6`'s is used for exactness.

## Protocol the match harness depends on

`games/othello/examples/edax_match.rs` drives `mEdax-native` over its native
line protocol. Invocation:

```
mEdax-native -q -book-usage off -eval-file <dir>/data/eval.dat -level <N>
```

- `-q` silences the per-move board dump; the `Edax plays` line is still printed.
- `-book-usage off` — no opening book (a book would confound the fixed-level
  ladder). The binary still prints a harmless `New book …` line at startup and
  a `Cannot open file: data/book.dat` on exit; both go to stderr and are
  ignored.

Per-move exchange:

| send | Edax replies |
|---|---|
| `mode 3` | *(nothing — sets manual mode: no auto-play, no pondering)* |
| `setboard <64><space><side>` | *(nothing)* |
| `go` | `Edax plays <MOVE>` |

- **Board string:** 64 chars in row-major order from A1 (`string[i]` is bit
  index `i` in `game_othello::State` — no transposition). `X` = black disc,
  `O` = white disc, `-` = empty, *regardless of side to move*. Trailing token
  after a space: `X` = black to move, `O` = white to move. See
  `game_othello::edax` and its `cargo test --lib` coverage.
- **Move reply:** `<file><rank>` upper-cased (e.g. `D3`), or `PA` for a pass.

### The one gotcha

Edax **aborts an in-progress `go` search the instant another line is queued
on its stdin** (so a GUI can interrupt it). The driver must therefore write
`setboard` + `go`, then *read the `Edax plays` reply back* before writing
anything else — never let `quit` (or the next `setboard`) sit in the pipe
during a search. Piping a whole command script at once silently yields no
move. `build-edax.sh`'s self-test spaces its writes with `sleep`s for the
same reason; `edax_match.rs` instead reads each reply before its next write.
