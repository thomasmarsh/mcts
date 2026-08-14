# Roadmap

## Terminal goal

A game description, written in this project's own surface syntax, compiles into a Rust crate
checked in under repo-root `games/<name>/` — implementing `mcts::game::Game` (plus zobrist/display
glue, matching every hand-written `games/*` crate's existing shape) — indistinguishable in the
workspace from a crate a person wrote by hand. Once that arrow exists, growing the game corpus
stops being bespoke Rust work per game and becomes "write surface syntax, run the compiler,"
checked against an oracle the way every game already is.

This document exists to keep individual sessions — which are, by design, free to pick whichever
piece of this looks most tractable or most forced by the current game — pointed at that one arrow,
and to hold the list of decisions that have to be made once, not re-litigated per session. Depth on
any phase belongs in `DESIGN.md`/`COMPLETENESS.md`, not here; this stays short on purpose.

## Cut from scope: the `.lud`/`ast`/`elaborate` pipeline

**Decision, made:** no further work loads or lowers `.lud` source. It was useful early on as a
second, independent proof for Tic-Tac-Toe/Hex, but the "translate `.lud` by understanding, don't
parse it mechanically" methodology (`DESIGN.md`'s "Goal") never needed `ast`/`parse`/`elaborate` as
*compiler* infrastructure — Y already proved a game start to finish without touching it. Carrying it
forward costs real weight (a second AST, a `define`-expansion gap that was never closed, two
fixtures — `tests/oracle.rs`/`tests/hex_oracle.rs` — pinned to `lud/*.lud` instead of the real
`style-c/sexpr/*.gdls` frontend) for a cross-check `style_c`'s own hand-built-`Program` tests already
provide more directly. `database-1/lud/games/` and `LudiiLanguageReference/` stay as reference
material for translation-by-understanding — reading a `.lud` file to figure out what a game does is
still the intended way to source new corpus games; only the *code path* that loads and lowers `.lud`
is retired.

Concrete cleanup (small, mechanical, do it early so it stops being live surface area to keep
compiling): retarget `tests/oracle.rs`/`tests/hex_oracle.rs` to lower `style-c/sexpr/tic-tac-toe.gdls`/
`hex.gdls` instead of `lud/*.lud` (same oracle-comparison structure, same games, just swap the
`Program`-construction side — `tests/y_oracle.rs` already shows the pattern), then delete
`src/ast/`, `src/parse/`, `src/elaborate/`, `src/core/lower.rs`'s `.lud`-shaped path, and `lud/`.

## Work breakdown structure

Each phase names its exit test — the thing that has to actually pass, not just be designed — since
that's what stops a phase from quietly turning into permanent design discussion (see `HISTORY.md`
for how long that happened to the surface-syntax question already).

1. **Retire the `.lud` pipeline. Done.** `src/ast/`, `src/elaborate/`, `src/core/lower.rs`, and
   `lud/` are deleted; `tests/oracle.rs`/`hex_oracle.rs` now build their `Program`s from
   `style-c/sexpr/*.gdls` via `style_c::parse_game`, the same pattern `tests/y_oracle.rs` already
   used. `src/parse/` (the generic s-expr reader) stayed — `style_c` depends on it — along with the
   small `Located`/`Span` span-tracking types it needed (moved from `ast::located` into
   `parse::located`, since `ast` no longer exists to own them) and `core::lower::all_occupied`
   (moved to `core::mod`, its only remaining caller being `style_c`).
2. **Freeze the parseable surface syntax.** *Decision:* is `style_c`'s existing sexpr rendering
   promoted to *the* real surface syntax (it already parses, already lowers, already has 3 proven
   fixtures), or does this project commit to building a real lexer/parser for the human-facing Style
   C grammar (`tak.md`) now that it's had six-plus rounds of review? Either answer is fine, but
   codegen in phase 4 needs one frozen input format to target, not two live candidates. *Exit test:*
   one grammar, with a parser, that every fixture used from here on is written in.
3. **Core IR completeness floor.** Not all of `DESIGN.md`'s wishlist — just what the *next* one or
   two codegen targets actually need beyond what Rect/Hex placement games already exercise (a
   `state`/effects layer is the likely first forcing case; `has_cycle` if Havannah is the next game
   rather than something state-free). Grow this the same "one real game forces it" way Region algebra
   already grew, not speculatively. *Exit test:* interpreter-level (still no codegen) proof of
   whichever primitive was added, on a real game.
4. **Backend codegen: Core IR → Rust source. First implementation landed, scoped to `Rect`.**
   *Decisions, made:* a generated crate implements `mcts::game::Game` plus a zobrist table and
   `Display`, the same shape every hand-written `games/*` crate already has (see `games/ttt-gen`);
   generation is an offline step whose output is checked in (`src/bin/codegen.rs`, piped through
   `rustfmt`), not a `build.rs`/proc-macro step; `Rect` went first. `src/codegen/rect.rs` only
   lowers the `Region`/`BoolExpr` shapes Tic-Tac-Toe's `Program` actually uses
   (`Occupied`/`Union`/`Complement`/`Sites`, `Contains`/`Any`) — `Intersect`/`Shift`/`Adjacent`/
   `Flood`/`Connects` (needed by Hex/Y) and `Topology::Hex` itself are still unimplemented, real
   next steps for phase 6, not attempted speculatively. *Exit test, passed:* `tests/
   ttt_gen_vs_interp.rs` round-trips `games/ttt-gen`'s `Game` impl against `core::interp`'s own
   oracle-tested `Program` evaluation, move by move.
5. **First full pipeline proof: Tic-Tac-Toe. Done.** Through the whole chain end to end: surface
   syntax (`style-c/sexpr/tic-tac-toe.gdls`) → Core IR (`style_c::parse_game`) → generated crate
   (`src/bin/codegen.rs`) → checked into `games/ttt-gen/src/lib.rs`, wired into the root workspace
   `Cargo.toml` as `game-ttt-gen`. *Exit test, passed:* `tests/ttt_gen_oracle.rs` walks
   `games/ttt-gen` and hand-written `games/ttt` through the same move sequences via
   `mcts::game::Game` on both sides and asserts legal moves/terminal-ness/winner agree at every
   step — the same kind of oracle check `games/ttt` already gets, run directly against the
   hand-built crate rather than only transitively through the interpreter. `cargo clippy`/`fmt`
   clean on both `gdl` and `games/ttt-gen`; `games/ttt-gen` is a first-class workspace member,
   playable by `mcts`/`game-host` like any other game crate (not yet actually wired into
   `game-host`'s game registry — that's ordinary per-game plumbing, not a pipeline question).
6. **Second game through codegen (Hex or Y).** Proves the generator isn't overfit to the trivial
   case, the same discipline the interpreter's own Tic-Tac-Toe→Hex bootstrap already followed.
   *Exit test:* same as phase 5, different topology.
7. **Decide generated-vs-hand-written coexistence.** Once codegen is real: do generated crates
   replace the hand-written ones they duplicate (`games/ttt` becomes generated), or live alongside
   them as a separate, clearly-labeled set? Either is fine; pick deliberately once there's a real
   generated crate to look at, not before.
8. **Steady state: corpus growth via the compiler.** Once phases 1-6 hold, adding a game is "write
   surface syntax for it, run the compiler, check the oracle" — `DESIGN.md`'s corpus table
   (Havannah, Abalone, Lines of Action, Margo, Amazons) becomes the backlog, picked one at a time,
   no longer gated on infrastructure decisions.

## Open decisions, consolidated

Threaded through the phases above; listed together so a session can check what's still open without
reading the whole WBS:

- Surface syntax: promote `style_c` sexpr to canonical, or build the Style C human grammar's parser?
- Core IR growth order: what's the minimal floor before codegen is worth starting (vs. `DESIGN.md`'s
  full wishlist)?
- ~~Codegen mechanism: checked-in generated source vs. build-time generation.~~ **Decided** (phase
  4): checked-in, via `src/bin/codegen.rs`.
- ~~Generated-crate integration: what exactly must a generated `games/<name>` crate implement to be
  a first-class workspace member (`Game`, zobrist, display, symmetry)?~~ **Decided for `Rect`**
  (phase 4/5, see `games/ttt-gen`): `Game`, a plain (non-symmetry-aware) zobrist table, `Display`.
  Symmetry-aware hashing (`games/ttt`'s own D4 `HashedPosition`) stays a hand-written-crate-only
  optimization for now — `Program` has no way to declare a topology's symmetry group yet, and no
  codegen'd game has forced that gap open.
- Generated vs. hand-written: replace or coexist, once there's a real generated crate to judge by.
  `games/ttt-gen` exists alongside `games/ttt` today (coexist) — revisit once a second generated
  game exists to judge the question by more than one data point, per this file's own "decide
  deliberately once there's a real generated crate to look at" framing.
- Effects/state layer (Freyd-category split, `DESIGN.md`'s Categorical structure section): still
  design-only — first real game that needs `state` forces landing it in `core::mod`, not before.

## Explicitly out of scope

GPU backend, a general category-theory framework/library, graph-shaped (non-tileable) boards,
mid-game player joining — all already flagged in `DESIGN.md`'s "Open problems"/"Non-goals" as
deferred-until-forced, unchanged by this roadmap. Style-syntax bikeshedding beyond what phase 2
needs to freeze: judged to have run its course (see `HISTORY.md`'s "pivot to descriptive complexity"
session note) — further review only if codegen work actually surfaces a real ambiguity, not
speculatively.
