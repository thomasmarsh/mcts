# GDL → Core IR

## Goal

Compile game descriptions to optimized Rust bitboard implementations (and eventually GPU kernels).
The intended primary source language is a small, human-authored **typed functional/equational
surface syntax** ("Style C") over `Region`/`Raster`/`Site`-typed board values — see `DESIGN.md` for
the full Core IR spec (value types, Region algebra, design principles) and the "Prior design
directions" section below for how Style C's syntax got to its current shape.

Ludii's `.lud` corpus (`database-1/lud/games/`, ~1650 real games) is **spec and oracle, not a
compilation target**: a person or an LLM reads a `.lud` game and writes the equivalent program in
this project's own authoring language by understanding what the game does, then checks it against
an existing `games/*` Rust implementation or a from-scratch reference test. `.lud`'s ludeme layer is
operationally specified (`then`/`apply`/`moveAgain`/`remember` describe effect sequences, not
relations), so mechanically parsing it can't converge on a small combinator set — see `DESIGN.md`'s
"Goal"/"Translating `.lud`" sections for why.

## Current status

**What actually compiles today:** `src/style_c/mod.rs`, the crate's one frontend onto
[`core::Program`], parsing a direct s-expression rendering of `core::Program`'s own shape — parens
for calls, no ludeme vocabulary, not an attempt at Style C's planned human-friendly notation — and
lowering straight to `Program`. Its load-bearing fixtures are `style-c/sexpr/*.gdls`; each is checked
(`include_str!`'d test) against an independent oracle or hand-built `Program` value, not just
against itself.

**Games proven end-to-end**, via `style_c` + an independent oracle each:

| Game | Topology | Fixture | Oracle/check |
|---|---|---|---|
| Tic-Tac-Toe | `Rect` 3×3 | `style-c/sexpr/tic-tac-toe.gdls` | `tests/oracle.rs` (cross-checked against `games/ttt`), hand-built-`Program` equality check |
| Hex | `Hex { Rhombus }` | `style-c/sexpr/hex.gdls` | `tests/hex_oracle.rs`, hand-built-`Program` equality check |
| Y | `Hex { Triangle }` | `style-c/sexpr/y.gdls` | `tests/y_oracle.rs`, hand-built-`Program` equality check |

**The `.lud`/`ast`/`elaborate` pipeline this project used to also have has been retired and
deleted** (`ROADMAP.md`'s phase 1, done) — it used to independently re-prove Tic-Tac-Toe/Hex by
lowering `lud/Tic-Tac-Toe.lud`/`lud/Hex.lud` through Ludii's own ludeme AST, but per `ROADMAP.md`
that was baggage relative to the terminal goal (a real Core-IR-to-Rust backend): `.lud` source
stays valuable as spec/oracle material to read by hand (`database-1/lud/games/`), but no code in
this crate loads or lowers it anymore. `src/parse/` (the generic s-expression reader) survives —
`style_c` reuses it — even though it originated as that pipeline's lexer.

**Backend codegen (`ROADMAP.md`'s phase 4) has a first real implementation:** `src/codegen/rect.rs`
lowers a `Topology::Rect` `Program` into the text of a standalone Rust `games/*` crate (a `Game`
impl, zobrist hashing, `Display` — the same shape every hand-written game crate has), rather than
interpreting it. `src/bin/codegen.rs` is the offline driver (`cargo run -p gdl --bin codegen --
<sexpr-path> <StructName> <"Game Name">`, output piped through `rustfmt`); its checked-in output
for Tic-Tac-Toe is `games/ttt-gen/src/lib.rs`, wired into the root workspace as `game-ttt-gen` —
`ROADMAP.md`'s phase 5 proof game. Three tests cross-check the three independent implementations of
Tic-Tac-Toe now in this repo: `tests/oracle.rs` (`style_c` + `core::interp` vs. hand-written
`games/ttt`), `tests/ttt_gen_vs_interp.rs` (`games/ttt-gen` vs. `core::interp`, phase 4's own exit
test read literally), and `tests/ttt_gen_oracle.rs` (`games/ttt-gen` vs. hand-written `games/ttt`
directly, via the `mcts::game::Game` trait on both sides — phase 5's exit test, and the "point of
comparison to the hand-built `games/ttt`" this was built for). Scoped narrowly on purpose, per
`DESIGN.md`'s "grow from real lowerings": `codegen::rect` only lowers the `Region`/`BoolExpr`
variants Tic-Tac-Toe's `Program` actually uses (`Occupied`/`Union`/`Complement`/`Sites`,
`Contains`/`Any`) and returns an error on anything else (`Intersect`/`Shift`/`Adjacent`/`Flood`,
`Connects`) rather than guessing at an unforced lowering; `Topology::Hex` has no codegen backend
yet either. Hex/Y stay on `core::interp` until a second game is deliberately routed through codegen
(`ROADMAP.md`'s phase 6).

**`style-c/`'s two other kinds of content are design exploration, not compiled or checked:**

- `style-c/games/*.gdl`, `style-c/games/tak.md`, and the numbered `style-c/0{1..5}-*.gdl` fragments
  are hand-written in the *human-facing* Style C surface syntax — the language `DESIGN.md`'s pipeline
  still lists as "not yet built." Nothing here has a lexer/parser; see `style-c/README.md` for what
  each file demonstrates.
- **`tak.md` is on the current syntax** (`guard`, primed `field'`/`out'` bindings, square-bracket
  `Tak[N]` template instantiation, `fold`/`fixpoint` split, `def` not `rule`) after several rounds of
  live syntax review — see "Prior design directions" below. **The rest predate that review and
  haven't been refreshed**: `games/{tic-tac-toe,hex,tak,kuhn-poker,sprouts,sylver-coinage,ghost}.gdl`
  and `01-check-safety.gdl` through `04-chess-pawn-template.gdl` still use older spellings (`rule`
  instead of `def`, bare `if`/`then` instead of `guard`, `<...>` instead of `[...]` for templates,
  etc.). `05-havannah-cycle.gdl` is deliberately exempt: it documents `has_cycle` as a Core primop
  call plus a reference definition of its semantics, not authoring-surface code, so it isn't subject
  to the syntax-currency question at all.

## Docs map

- **`ROADMAP.md`** — the work breakdown structure and sequencing from here to the terminal goal
  above, plus the list of open decisions each phase depends on. Read this before picking what to
  work on next; the sections below are current-state background, not sequencing.
- **`DESIGN.md`** — Core IR itself: value types, Region/Raster algebra, design principles, the
  Core/Stdlib/Extern builtin tiers, topology model, backend lowering, and the representative game
  corpus (what's already covered, what's worth adding next and why).
- **`COMPLETENESS.md`** — the completeness/universality claim, stated as a conjecture precise enough
  to prove or refute: Core IR's fragment is exactly FO(LFP) (= PTIME by Immerman–Vardi) over ordered
  finite structures. Primitive-by-primitive classification of what's confirmed vs. still open.
- **`EVALUATION.md`** — a snapshot comparison of Style C against Stanford GDL and Ludii on four axes
  (primitive correctness/composability, concision, provability, universality).
- **`style-c/README.md`** — what each file under `style-c/` demonstrates, plus the Style C surface
  syntax's own evolution (the Alloy-notation overcorrection and its fix, the temporal-refinement
  `state'`/`always`/`once` design, and six-plus rounds of live syntax review against `tak.md`).
- **`HISTORY.md`** — this file's own prior contents: the full chronological session-note log,
  archived rather than deleted. Consult it for the reasoning behind a decision this README now just
  states as fact — not needed to understand current state.

## Prior design directions (see `HISTORY.md` for the full reasoning)

The authoring-surface design went through two real pivots before settling on Style C, and Style C's
own syntax then went through several more rounds once basic semantics were settled:

1. **Mechanical `.lud` parsing → declarative "Relational GDL"** (Datalog-shaped) → **Style C**
   (typed functional/equational). A five-case design spike (Chess's check-safety filter, Go's
   suicide rule and superko, Chess's piece-template macros, Havannah's cycle check) found flat
   Horn-clause Datalog never reached the right term directly — every case needed a bespoke
   non-`:-` modifier bolted on, and one case (`has_cycle`) was outright wrong under naive Datalog
   until it borrowed typed-state discipline from elsewhere. Style C reaches every case as directly
   as a point-free categorical rendering, with no new vocabulary to teach (`let`, ordinary function
   definitions, generics). `DESIGN.md`'s "Relational GDL" section is kept, marked superseded, for
   what survives from it (intensional primitives, the categorical desugaring target).
2. **Style C's notation itself, corrected twice**: it first drifted into "Rust with game nouns," was
   over-corrected into literal Alloy notation, then fixed by borrowing Alloy's *semantics* (primed
   `state'`, `always`/`once` temporal operators, declarative field-transition bindings) while keeping
   this project's own notation for everything else (`table` for lookups, indexed `state`, `guard` for
   preconditions, square-bracket template instantiation). Six-plus rounds of live review against
   `tak.md` (see `style-c/README.md`) settled `def` vs. `rule`, `fixpoint` vs. `fold`, and `guard`
   vs. bare `if`/`then`.
3. **Universality claim re-grounded**: rather than claim to "preserve universality" the way GDL
   (unrestricted-but-uncompilable) or Ludii (after-the-fact corpus coverage) do, `COMPLETENESS.md`
   states a sharper, provable target — Core IR's fragment is exactly FO(LFP) — matching this
   project's own first-order, `bounded_fixpoint`-not-general-recursion design principles instead of
   contradicting them.

A lexer/parser for Style C's own surface grammar is explicitly not being built right now — several
rounds of syntax review left it unstable, and effort was deliberately redirected toward the
completeness conjecture and real `core::mod`/`core::interp` Rust work (the `s-expr -> Core IR`
frontend above) instead. Revisit once `style-c/games/tak.md`'s syntax has stabilized further and the
other `style-c/games/*.gdl` files have been brought up to match it.

## What's next

See `ROADMAP.md` for sequencing. Short version: retire the `.lud` pipeline, freeze one surface
syntax, then build the Core-IR-to-Rust backend that's currently 0% built — proving that pipeline on
Tic-Tac-Toe and a second game matters more right now than growing corpus breadth (Havannah, etc.),
which is steady-state work for *after* the compiler is real.
