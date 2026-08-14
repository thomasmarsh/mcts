# Style C reference cases

Standalone, self-contained fragments and full games in the typed functional/equational surface
syntax ("Style C" in the top-level `HISTORY.md`'s design-spike write-up), saved here as reference
artifacts in their own right rather than left as inline snippets in a session-note history. There
is no lexer/parser for this syntax yet (see the top-level `HISTORY.md`'s "Next session charter") --
these are hand-written, not machine-checked.

`games/` holds complete, runnable-shaped games (a full `game "Name" { ... }` block); the
numbered files below are isolated mechanic fragments transcribing one hard case each from the
design spike, not complete games -- see each file's header comment for which `.lud` source it's
from and, where relevant, what larger game it'd need to be embedded in to actually run.
`tic-tac-toe.gdl`/`hex.gdl` are the base-layer sanity check, `tak.gdl` pushes on a still-spatial
board, and `kuhn-poker.gdl`/`sprouts.gdl`/`sylver-coinage.gdl`/`ghost.gdl` each drop a different
piece of "spatial bitboard game" (hidden info, fixed topology, a board at all, in that order) to
see what's left of the grammar without it.

`sexpr/` is a different thing entirely, not more of the above: it holds real, checked, *parseable*
source for `src/style_c/mod.rs`, this project's one real frontend onto `core::Program` (see the
top-level `README.md`'s "Current status" and `DESIGN.md`'s Pipeline section) -- a direct
s-expression rendering of Core IR, distinct from this directory's still-hand-written, unparsed
Style C notation. `sexpr/tic-tac-toe.gdls` and `sexpr/hex.gdls` are load-bearing test fixtures
(`include_str!`'d by `src/style_c/mod.rs`'s tests), each checked against a hand-built `Program`
value -- treat them the same as any other checked-in fixture, not scratch files.

| File | Game / `.lud` source | Demonstrates |
|---|---|---|
| `games/tic-tac-toe.gdl` | Tic-Tac-Toe, full game | base declarative layer only -- no `then`/`state`/`invariant`/templates needed |
| `games/hex.gdl` | Hex, full game | same base layer plus named `regions` and `connects` |
| `games/tak.gdl` | not `.lud`-sourced, full game, board-size-parametrized 3x3-8x8 | superseded (surface syntax) by `games/tak.md`, see the top-level HISTORY.md; findings below still stand |
| `games/tak.md` | literate rewrite of `games/tak.gdl` (was `games/tak-relational.gdl` through round 6, moved to Markdown afterward) | same findings, in this project's own domain-native notation instead of borrowed Rust/Alloy syntax -- see the top-level HISTORY.md's "Style C was leaking Rust" session note |
| `games/kuhn-poker.gdl` | not `.lud`-sourced, full game (card) | `topology = None`, private/epistemic per-player state, `chance` moves, path-dependent outcome -- see below |
| `games/sprouts.gdl` | not `.lud`-sourced, full game (graph) | mutable/growing topology, unbounded-domain Raster, expensive geometric oracle predicate -- see below |
| `games/sylver-coinage.gdl` | not `.lud`-sourced, full game (math) | `topology = None`, unbounded `Set` state, oracle-as-`bounded_fixpoint`, non-constructive termination -- see below |
| `games/ghost.gdl` | not `.lud`-sourced, full game (word) | growing `Seq` state, dictionary oracle, legal-but-suicidal terminal move -- see below |
| `01-check-safety.gdl` | `Chess.lud:166` | top-level `invariant: always`, primed `state'` |
| `02-suicide-rule.gdl` | `Go.lud:35` | same `invariant: always` construct, confirms it generalizes |
| `03-superko.gdl` | `Go.lud`, `(meta (no Repeat))` | past-temporal `once`, no bespoke history builtin |
| `04-chess-pawn-template.gdl` | `Chess.lud:126-137` | `template def`, compile-time generics |
| `05-havannah-cycle.gdl` | `Havannah.lud:13`, `(is Loop)` | `has_cycle` as a Core primop (real usage is one call); a bounded-`fixpoint` derivation kept only as a reference definition, not authoring surface -- see the top-level `HISTORY.md`'s session note |

`games/tic-tac-toe.gdl` and `games/hex.gdl` are the sanity check the previous session's charter
asked for: neither needs any of the machinery the five case fragments exist to exercise, which is
the point -- it confirms the grammar isn't overfit to the five hard cases. They match how small
`core::Program` already is for both games (see each file's header comment) -- a strict superset of
what those already-proven Core programs needed, not a rewrite of them.

## Temporal refinement: Alloy-style `state'`/`always`/`once`, replacing `ifAfterwards`

The previous session's grammar (top-level `HISTORY.md`) had `ifAfterwards: P` as a per-move
guard clause -- workable, but flagged as feeling ad hoc: a bespoke keyword bolted onto whichever
move rules happened to need a one-step lookahead. Cases 01-03 here replace it with vocabulary
borrowed directly from Alloy 6's temporal mechanics (Electrum), not TLA+ (an earlier framing of
this same idea considered TLA+'s `state`/`state'` naming, which Alloy 6 also uses, but the
operators below are specifically Alloy's):

- **`state'`** is a primed reference to the hypothetical state after the move currently being
  legality-checked (`next(state, this)`, board-shaped only -- see the boundary discussion in
  the top-level `HISTORY.md`). It can appear in any expression, not just a dedicated guard
  clause, so `ifAfterwards` is no longer a separate keyword; an ordinary expression that happens
  to reference `state'` *is* the one-step-lookahead case.
- **`invariant: always P`** is a new top-level game declaration (alongside `moves`/`terminal`/
  `outcome`), automatically intersected into *every* move's legality: a move `m` is legal only if,
  in addition to its own `to`/`if` conditions, `P[state' := next(state, m)]` holds for every
  declared invariant. This is the more consequential change: cases 1 and 2 from the design spike
  (Chess check-safety, Go's suicide rule) turn out to be the *same* construct as case 3 (Go's
  superko) once `always` exists as a top-level declaration -- all three are genuine, standing
  game invariants ("the mover is never left in check", "a group always retains a liberty", "no
  position repeats"), not three separate per-move special cases. That wasn't visible with
  `ifAfterwards` attached move-by-move; stating it as a top-level `invariant` is what surfaces it.
- **`once P`** (Alloy's past-eventually, "P held at some point up to and including now") is what
  case 3 uses instead of a bespoke `visited` builtin or an author-declared `state history:
  Set<Hash>` field with a manual `insert` effect: `once(board = state'.board)` says positional
  superko directly. Because `once` ranges only over the trace *committed so far*, and `state'`
  is by definition not yet part of that trace, there's no self-referential "does checking-after-
  inserting make membership trivially true" hazard to reason about case by case the way there
  would be with a hand-threaded set -- that used to be a bespoke rule this project had to state
  and justify (the earlier "`next` is board-only" carve-out specifically existed to block this
  hazard); it now falls out of `once`'s ordinary finite-trace semantics for free.
- **Deliberately not adopted: future-directed operators beyond one-step `state'`** (`eventually`,
  `until`, `after` used forward, `historically`'s future dual). A game only ever computes one
  committed move at a time; a claim about the game's *future* trace (which hasn't been played,
  and whose length isn't bounded the way its past is) has no compileable meaning for a system
  that never model-checks unbounded traces. `once`/`historically` (Alloy's *past* operators)
  don't have this problem -- the past of a game trace played so far is always finite and already
  committed, exactly the same footing the `Set<Hash>` backend design already stood on. This is a
  sharper line than the previous session's "skip eventually" call: it's not that temporal
  operators in general are out of scope, it's specifically that *forward* ones are, because
  nothing about this project's bounded, per-move compilation model gives them an operational
  reading.

Backend consequence: `once(P)` where `P` references board-shaped state lowers to the same
mechanism the top-level `HISTORY.md`'s "Unbounded auxiliary state" section already sketched (a
heap-backed, amortized-O(1) hash set keyed on the Zobrist hash already derived for Core state) --
that design is unchanged by this session, just now motivated as the general lowering for the
`once` operator over `Region`/`Raster`-typed values, rather than a per-game special case the
author had to wire up by hand.

Cases 04/05 are unaffected by this refinement (templates and bounded fixpoints don't involve
one-step lookahead or trace history), included here for completeness so all five design-spike
cases live together as one set of artifacts rather than split across a session note and a
separate directory.

## `games/tak.gdl`: a sixth, more complicated pathological example

Explicitly a "pro forma" design exploration (session request), not required to parse against the
grammar above or lower to any existing `core::Program` shape -- unlike the five cases, Tak has no
`.lud` source anywhere in this repo, and is written directly from the published ruleset plus this
repo's own hand-written `games/tak/src/lib.rs`. Board-size-parametrized (3x3 through 8x8, with a
real piece-reserve table per size) on purpose, since that's what forces the most interesting new
ground: none of the five cases needed a compile-time quantity that varies with a topology
parameter. Five findings, each flagged inline in the file where it first appears:

- **`const fn` and `template game<const N: Int>`**: a *second* kind of template parameter beyond
  the five-case README's `fn(...) -> ...`-kinded ones. The earlier generics writeup's rule ("a
  parameter needs `<...>` iff its type is a function kind") was correct as far as it went, but
  incomplete -- it implicitly assumed a template parameter is always a missing *function*, because
  nothing before Tak needed a missing *board-size-dependent constant* (piece/capstone reserve,
  packed-cell bit width). Same monomorphization story (specialize away before Core sees it, one
  independent copy per instantiation), extended to a second parameter kind, and applied one level
  up -- a whole `game { }` block, not just one `rule`, needs the same treatment once board size
  itself is a free parameter.
- **Per-player indexed state** (`state reserve: [Int; players]`). Case 3's `history` was a single
  global field; Tak's reserve/capstone counts are the first case needing one slot per player. Not
  resolved here -- just surfaced as a real gap the hardened grammar's `StateDecl` doesn't cover.
- **Named, composable effect blocks** (`effect rule apply_spread(...) = { ... }`), not just
  template-supplied ones. The spread move's effect (pop a sub-stack, walk it along a line,
  conditionally flatten a wall) is the first `then`-shaped logic complex enough that inlining it
  stops being legible -- the same pressure that made case 5 give `fixpoint` an explicit typed
  state signature instead of a bare `Region`, now applied to the effect-block layer.
  `legal_spread_path`, alongside it, is an ordinary `rule` -- the wall/capstone blocking check
  needed a piece-*kind*-dependent branch inside a bounded per-cell loop, which no earlier case's
  `then` block needed (they were all straight-line).
- **Disjunctive `connects`**: Tak's road win completes across *either* edge-pair orientation, not
  one fixed `(Region, Region)` pair the way Hex's win condition is. First real use of
  DESIGN.md's `project` combinator with something built around it (`road_region`), not just cited
  as a target the way `games/tak/src/lib.rs`'s existing precedent is described in DESIGN.md today.
- **`count_where` promoted from aspirational to load-bearing.** DESIGN.md's "Control and
  aggregation" combinator list only has `any`/`all`/`for_each`; `count_where` was only ever
  *mentioned*, in the "Move caching" section, as an example of what a backend might
  incrementalize -- never actually required by a game. Tak's flat-count win is a second real
  corpus data point (beyond Tanbo's territory count) for DESIGN.md's already-flagged
  "Scoring/payoff aggregation" open problem, and the first case in this project to actually force
  `count_where` to exist as a real combinator rather than a backend-optimization aside.

## Four more pathological cases: card, graph, math, word

Tak was still, fundamentally, a spatial game on a bitboard -- everything it forced (const
generics, indexed state, named effect blocks, disjunctive `connects`, `count_where`) was new
*vocabulary* layered on the same Region/Raster foundation every earlier case shared. These four
go further: each picks a game genre with essentially nothing in common with Ludii's board-game
corpus, specifically to find out whether Core IR's declarative layer means anything once there's
no spatial board to build on top of. Same license as `games/tak.gdl` -- pro forma design
exploration, none `.lud`-sourced, none required to parse or lower:

- **`games/kuhn-poker.gdl` (card game).** Kuhn poker is the standard minimal test case in the
  extensive-form-game/CFR/ISMCTS literature (3-card deck, 1 card dealt to each player, one
  betting round) -- picked over a "bigger" card game specifically because it's small enough to
  transcribe in full while still forcing the real issue: **hidden information**. Every earlier
  case, Tak's per-player `reserve`/`caps` included, assumed `state` is fully observed by the
  engine and by both players. Kuhn needs `private[p]: Card` -- state that exists but is
  *epistemically scoped*: player p's own move/rule declarations may read `private[p]` but never
  `private[q]`, a static scoping rule with no earlier precedent, relaxed only for `outcome`
  (showdown is exactly the moment private state becomes public). It also needs a `chance` move
  kind (dealing is nobody's decision, so it doesn't fit `move`'s player-indexed shape at all), a
  new base value type (`Card`, ordered, drawn from a `Deck`), and an `outcome` that depends on
  *which move ended the hand* (fold vs. showdown), not just on a snapshot of final state the way
  every earlier `outcome` could get away with.
- **`games/sprouts.gdl` (graph game).** Picked over a fixed-graph game (Shannon Switching, Node
  Kayles) specifically because Sprouts's board *grows* every move and its legality is a *global
  topological* property (no crossing curves), which stresses the Topology/Region split harder
  than "just support an arbitrary graph" would. This is the one finding in this batch that cuts
  against a stated DESIGN.md principle rather than just extending the vocabulary: "topology as a
  compile-time type parameter" assumed topology is fixed once and state varies at runtime on top
  of it; Sprouts's vertex/edge set is itself runtime-growing `state`, with no static `N` a
  `template game<const N>` (Tak's mechanism) could parametrize over. Its move legality also needs
  `no_crossing`, an "oracle" predicate -- a call out to real computational geometry that Region
  algebra has no vocabulary for and no static cost bound for, unlike every earlier
  `bounded_fixpoint`'s `max_iters`.
- **`games/sylver-coinage.gdl` (math game).** No board, no topology, no spatial concept
  whatsoever -- `state` is a single unbounded `Set<Int>` of named numbers, and a move's legality
  is numerical-semigroup non-membership. Two findings: first, unlike Sprouts's `no_crossing`,
  this game's oracle predicate (`in_semigroup`) turns out to reduce cleanly to an ordinary
  `bounded_fixpoint` with `max_iters = k` -- not every oracle-shaped predicate this batch needed
  is genuinely new machinery, and it's worth recording which ones aren't. Second, and more
  interesting: nothing in `state` bounds how long the game can run before someone is forced to
  name 1 -- Sylver Coinage's finiteness is a real but *non-constructive* theorem (Hutchings), the
  first case in the corpus where "the game terminates" isn't something the grammar can express or
  check, only something its author has to take on faith from the literature.
- **`games/ghost.gdl` (word game).** `state` is a growing, order-sensitive `Seq<Letter>` (as
  opposed to Sylver Coinage's unordered `Set<Int>` -- both grow by one element per move, but only
  one of them needs a notion of "prefix of"). Legality and termination both depend on
  `dictionary_has_prefix`/`dictionary_has_word`, an oracle over externally supplied data with no
  algebraic reduction the way Sylver Coinage's did -- the middle of this batch's oracle-cost
  spectrum, cheaper than Sprouts's open-ended geometric search (a fixed 26-letter alphabet bounds
  branching at every step) but not a `bounded_fixpoint` instance either.

Two cross-cutting findings, visible only once these four sit next to each other:

- **`topology = None` needs to be a first-class, legal value**, not an implicit default nothing
  earlier ever exercised -- Kuhn poker, Sylver Coinage, and Ghost all have zero spatial
  structure, and all three needed to state that explicitly rather than lower into some degenerate
  1x1 board.
- **"Legal-but-suicidal" is a third terminal-move shape**, distinct from "the move that ends the
  game" and "the move that's illegal" (the only two shapes every earlier case needed). Sylver
  Coinage's naming-1 and Ghost's word-completion are independently-arrived-at instances of the
  identical pattern: a move that is always legal, always played by `mover`, and always loses for
  `mover` -- both games needed `outcome: Outcome = ... Win(opponent(mover))`, the first cases in
  the corpus where `outcome` isn't simply `Win(mover)`. No new grammar construct is required for
  this (it's an ordinary `terminal`/`outcome` pair), but it's a pattern worth naming rather than
  re-deriving per game.
