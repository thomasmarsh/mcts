# Core IR design

## Goal

Compile game descriptions to optimized Rust bitboard implementations (and eventually GPU
kernels). The primary source language is a small **typed functional/equational language** (see
"Relational GDL: superseded..." further down, and `HISTORY.md`'s design-spike write-up) over
`Region`/`Raster`/`Site`-typed objects: named `let`-bound values (including hypothetical
"next state" values, replacing GDL-style unification), ordinary function definitions with
pattern matching, a restricted typed `then { }` effect block, and templates as compile-time
generics — not the flat Horn-clause/Datalog-shaped grammar this doc originally proposed (see
below for why that was retracted as the authoring surface). Whichever concrete grammar this
becomes, the underlying semantics stay the same: a function body's conjunction is a join (a
bitboard AND under a `Region` encoding), disjunction is a union (OR), stratified negation is a
complement, and bounded recursion is a fixpoint over a statically-bounded number of iterations.
This language *is* Core IR's surface syntax — there is no separate elaboration step recovering
declarative structure from an operational description, because the source is already
declarative.

Ludii's `.lud` corpus (`database-1/lud/games/`, ~1650 real games) is not compiled by this
pipeline. It's **spec and oracle**: the real rules of a game, cross-checked against an existing
`games/*` Rust implementation or a from-scratch reference test where one doesn't exist (the
methodology `tests/hex_oracle.rs` already established). A person or an LLM reads a `.lud` game
(plus its known-define expansions — see "Translating `.lud`" below) and writes the equivalent
program in this project's own authoring language by understanding what the game does, not by
mechanically parsing Ludii's ludeme syntax. This distinction matters because Ludii's ludeme layer
is **operationally specified**: `then`, `apply`, `moveAgain`, `remember` describe a *sequence of
effects*, not a *relation to be computed*. No amount of syntax-directed translation turns an
effect sequence into a declarative expression in general — that's decompilation, an open-ended,
per-idiom reverse-engineering problem, which is exactly why this project used to also have a
mechanical `ast::*`/`parse`/`elaborate` pipeline lowering `.lud` source into Core IR directly (it
worked, and covered `Tic-Tac-Toe`/`Hex`), but had to special-case a new operational shape roughly
once per game and was never going to converge on a small combinator set by itself. Per
`ROADMAP.md`'s decision, that pipeline has since been deleted outright rather than kept as a
bootstrap — nothing loads or lowers `.lud` source in code anymore; `crate::style_c` is this
project's one frontend onto `core::Program`, and `.lud` is read by a person, never parsed by this
crate.

Core itself — the value types, Region algebra, Raster ops, and backend lowering below — is
mostly unchanged by this pivot: it was already trying to be the declarative, referentially
transparent language the authoring surface now targets directly, rather than a translation target
`elaborate/` had to be steered toward game by game.

Core is scoped *up* from a concrete corpus of real games (see below), not *down* from lambda
calculus or category theory — including the categorical structure introduced further down, which
is adopted only where a corpus game has already forced the problem it solves, not speculatively.

## Pipeline

Primary path — the typed functional/equational surface is authored directly, close to Core IR already:

```
typed functional/equational source  (NEW: grammar not yet designed -- next session per HISTORY.md)
  -> lex/parse -> rule AST
  -> rule AST -> Core IR             (NEW: mostly direct -- named values already are Region-algebra/effect expressions)
  -> Core IR -> Core IR (optimized)  (NEW: algebraic rewrite passes, justified by the categorical structure below)
  -> Core IR -> backend primops      (NEW: per-topology lowering, a functor to a concrete backend category)
  -> backend primops -> Rust source  (NEW: codegen, mirrors games/*/src/lib.rs by hand today)
```

Translation path — offline, by a human or LLM, not part of the compiler:

```
.lud text (database-1/lud/games/, the spec)
  -> understand the game's real rules (cross-reference known-define expansions, see below)
  -> write the equivalent program in the typed functional/equational surface
  -> verify against an oracle (an existing games/* crate, or a from-scratch reference test)
  -> source, checked in
```

This project used to also have a mechanical pipeline (`.lud text -> lex -> s-expr -> ast::* ->
Core IR`, via `parse`/`elaborate`) that independently re-proved `Tic-Tac-Toe`/`Hex`. Per
`ROADMAP.md`'s decision it has been deleted outright, not kept as a bootstrap — see "Goal" above
for why syntax-directed translation from `.lud`'s operationally-specified ludemes doesn't scale the
way translation-by-understanding does, and why the cross-check it provided wasn't worth the weight
of a second AST once `style_c`'s own hand-built-`Program` tests covered the same ground more
directly.

**`src/style_c/mod.rs` is this project's one real frontend onto `core::Program`** (see
`README.md`'s "Current status"): it targets the primary path's `s-expr -> Core IR` arrow directly,
skipping the "typed functional/equational source -> lex/parse -> rule AST" stages above entirely
rather than waiting for that surface grammar to stabilize — it reuses `parse::sexpr`'s reader
(which was never actually Ludii-specific, just used that way originally) against its own
s-expression vocabulary that mirrors `core::Program`/`Region`/`BoolExpr`'s own shape instead of
Ludii's ludeme names, and lowers straight to `Program` with no intermediate typed AST. This makes
"grammar not yet designed" true only of the human-facing surface syntax annotation above, not of
the arrows after it: those now have a real, tested, growing implementation (`style-c/sexpr/*.gdls`,
checked against hand-built `Program` values and independent oracles). A pretty-printer from Style
C's eventual concrete syntax down to this s-expression form remains a plausible way to fill in the
still-missing first arrow later, or `ROADMAP.md`'s phase 2 may instead promote this sexpr form to
the canonical surface syntax outright; neither is required to keep growing Core IR/backend coverage
in the meantime.

Each arrow above is a separate, independently testable pass. Core IR should be *constructible and
checkable by hand* (as a Rust value, not just parsed from the authoring surface's source) so backend
lowering and optimization passes can be tested against hand-built Core programs before a given
rule shape's parser support exists.

## Translating `.lud`: `define` bodies are required knowledge, not surface noise

Ludii's `(define "Name" body)` mechanism (Language Reference ch. 20, Appendix B) is not
Java-class-hierarchy leakage into the surface syntax — it's a genuine small-core-plus-derived-forms
design, the same shape as Scheme's core special forms plus a library of derived expressions. A
"known define" expands, by textual substitution with positional parameters (`#1`, `#2`, ...), into
ordinary compositional ludeme syntax:

```
(define "NoMoves" (if (no Moves Next) (result Next #1)))
(define "HopCapture"
  (move Hop (between if:(is Enemy (who at:(between))) (apply (remove (between))))
            (to if:(is Empty (to)))))
```

`parse/sexpr.rs`'s generic reader recognizes a quoted-string call head as `Head::Define(String)` (a
"known define" invocation, 20.4) at the syntax level, but nothing in this project resolves what a
call like `("BlockWin")` or `("ReachWin" (sites Mover) Mover)` actually expands to — that expansion
has to happen in a translator's head (or an LLM's), reading the real `.lud`/`def/` source directly,
since no code here loads `.lud` at all (see "Goal" above) to do it mechanically.

This is not a hypothetical gap. Across a representative sample of the corpus (games this project
has actually read closely so far):

| File | Known-defines used |
|---|---|
| `Amazons.lud` | `BlockWin` |
| `Breakthrough.lud` | `TwoPlayersNorthSouth`, `StepForwardToEmpty`, `IsEnemyAt`, `ReachWin` |
| `Lines of Action.lud` | `IsEnemyAt`, `IsFriendAt`, `MoveTo` |
| `Minishogi.lud` | `BlockWin`, `InPromotionZone`, `IsEnemyAt`, `IsFriendAt`, `IsInCheck`, `NextCannotMove`, `OnePawnPerColumn`, `Promote`, `SameTurn`, `ShogiGold`, `SlideMove`, `StepMove`, `TwoPlayersNorthSouth` |

Of the roughly 14 distinct names above, only three (`IsInCheck`, `SameTurn`, `StepForwardToEmpty`)
are documented in this repo's bundled `LudiiLanguageReference/` (Appendix B lists 28 known defines
total — a curated sample of Ludii's real `def/` library, not the full set shipped with the actual
Ludii distribution). The rest — `BlockWin`, `ReachWin`, `IsEnemyAt`, `IsFriendAt`, `MoveTo`,
`TwoPlayersNorthSouth`, and everything Minishogi-specific — have no expansion available anywhere in
this repo yet. **Sourcing the real `.def` bodies (from Ludii's public distribution, or reconstructed
by hand from a game's known rules where the source can't be found — the same treatment
options/templates already get) is on the critical path for every remaining corpus game except the
two already done (`Tic-Tac-Toe`, `Hex`), neither of which happens to use a known define.**

`database-1/` (the full Ludii games database: `database-1/lud/games/` has all ~1650 real `.lud`
files, plus a `ludiiGames.sql` dump of Ludii's own game/concept/ruleset metadata) confirms this
gap rather than closing it: it has zero `.def` files, and the SQL dump's `Ludemes`/
`DefineLudemeplexes` tables are a concept taxonomy/analytics catalog (game classification,
ludeme-frequency stats for Ludii's own research) — `ReachWin`, for instance, appears there only as
a one-line human-readable description ("Win in reaching a region"), not as expandable source. The
actual `def/` bodies live in Ludii's own source distribution and still need to be sourced from
there. What `database-1/` *does* give this project: the real, un-concretized source for every
corpus game to read by hand, and a much larger pool to eventually pick "worth adding" games from
than the handful sketched in the corpus table further down.

Whoever translates a `.lud` game into this project's authoring surface — a person or an LLM — needs the real
`define` bodies to read the source correctly; guessing what `("BlockWin")` or `("ReachWin"
(sites Mover) Mover)` mean and translating the guess is exactly the kind of silent-wrongness bug
an oracle test exists to catch, so verify against real definition text (or the oracle) rather than
trusting a plausible-looking guess. A mechanical `s-expr -> s-expr` expansion pass (`Head::Define`
call sites replaced by their body with `#N` positional substitution, recursively) is still worth
building *if* `.lud`-parsing work is ever revived — e.g. to hand an LLM translator pre-expanded
source instead of requiring it to independently know Ludii's `def/` library — but it's no longer
required, load-bearing infrastructure the way it was scoped last session.

## Design principles

- **First-order, not full lambda calculus.** Ludii ludemes are closer to
  typed macro expansion (`(define ...)`) plus a fixed set of aggregate
  combinators (`forEach`, `count`, `all`, `if`) than genuine higher-order
  functions with closures. Core has `let`-bindings and named combinators,
  not first-class functions. If a game genuinely needs recursion (unbounded
  chain captures — see Congo's Monkey below) that's modeled as a bounded
  fixpoint combinator over regions, not general recursion. Templates
  (Chess's `ChessPawn`-style piece parametrization — see the design spike in
  `HISTORY.md`) are compile-time generics, monomorphized per call site before
  Core ever sees them, not first-class function values passed at runtime —
  same principle, applied to the template layer specifically.
  **Scope correction from the design spike:** "statically bounded" is a claim
  about `Region`/`Raster` board-state values (bounded by board geometry,
  known at compile time) — it does *not* automatically extend to auxiliary
  effect-state objects the Freyd-category layer below threads through moves.
  Congo's `river_since` is a small per-cell scalar and fits this bound; Go's
  positional-superko history (an unbounded, per-ply-growing `Set<Hash>` —
  see Open Problems) does not, and needs its own backend story.
- **Referentially transparent, à la Halide.** A Core expression describes
  *what* a region/board value is, not *how* to compute it incrementally.
  Scheduling/incrementalization (e.g. maintaining a running occupancy
  bitboard instead of recomputing `union` of piece boards every call) is a
  backend concern, analogous to how Halide separates the algorithm from its
  schedule. This is what makes a second backend (GPU, eventually) plausible
  without redesigning Core.
- **A whole-value state update (`field' = expr`) is not a whole-value copy at runtime.** The
  design-spike surface syntax (`HISTORY.md`'s Style C) writes every state transition as a pure
  function of the *entire* prior value — `board' = push(board, s, v)` denotationally replaces the
  whole board, not one cell. That's the same Halide-style algorithm/schedule separation as the
  principle above, applied to *updates* rather than to computing a value: the expression says
  *what* changed, and lowering it to "materialize a whole new board" is one valid schedule, not
  the only one, and not the one any real backend should pick. This is tractable specifically
  because the effect vocabulary (`push`/`pop`/`set`/`insert`, `fold`/`bounded_fixpoint`) is a
  **closed, first-order set of primitives**, not arbitrary user-defined functions — so, unlike
  general-purpose functional-update compilation (Haskell/ML, which need real escape/alias
  analysis to prove in-place mutation is safe), each primitive can carry its own known "touched
  sites" shape as part of its own definition, and those shapes compose structurally through
  `let`/`if`/`fold` without any whole-program analysis. Every bounded-iteration construct
  (`fold`'s known-length sequence, `bounded_fixpoint`'s `max_iters`) already guarantees the
  touched-site set is statically *bounded in size* even when its members are move-parameter-
  dependent (not statically enumerable in value) — so the real backend contract is "every effect
  body's touched-site set is small and enumerable at move-generation time," not "exactly one
  site." Two lowering strategies follow from this, not in tension with each other: (1) static
  delta composition, where the compiler derives a move's touched-site set (or a small
  parametrized shape of it) directly from the primitives it calls, the same way `MoveGen`'s
  `Region` already describes a move's *legality* shape; (2) runtime make/unmake, where the search
  backend never materializes `board'` as a new struct at all — it writes the delta in place and
  pushes an undo record, exactly how essentially every high-performance game-search engine already
  handles tree exploration (this is also why MCTS doesn't actually want an immutable "new struct
  per node" model regardless: sibling branches from the same parent state need make/unmake or
  copy-on-write, not literal persistent-value sharing). The discipline this imposes: **every new
  effect primitive added to the grammar later must ship with its own known touched-site/lowering
  rule at the point it's added** — this can't be assumed to "probably work by analogy" the way
  expressiveness gaps get resolved elsewhere in this doc; an effect primitive with an unbounded or
  unknown touch shape would silently break the whole backend contract, not just be slow.
- **Topology is a type parameter, not a special case.** The same region
  algebra (union, shift, flood, adjacency, connects) is defined once, over
  an abstract `Topology`, and each concrete topology (rectangular grid, hex
  grid, pyramidal stack, raster-of-small-ints) supplies its own direction
  set, wall masks, and shift/support primitives. Elaboration and
  optimization passes never need to know which concrete topology they're
  compiling for.
- **Grow the combinator set from real lowerings.** Every combinator in this
  doc is justified by at least one game in the corpus below that needs it.
  Resist adding a combinator "because Ludii has a ludeme for it" until a
  corpus game actually forces the lowering.
- **Growing the primitive set is not growing the authoring grammar.** A corpus game that forces a
  new backend-native combinator (Havannah's `has_cycle`, joining `flood`/`connects`/`adjacent`/
  `shift`) doesn't automatically force new authoring-surface syntax to go with it -- it's expected
  and fine for Core to carry a large, hand-written instruction set over bitboards/hexboards, the
  same way a CPU's ISA has instructions no compiler derives from smaller ones in ordinary user
  code. A primitive only forces a *grammar* change when some game needs to *express* something the
  existing surface constructs can't say at all (a new value type, a new control construct); needing
  an efficient *implementation* of an existing primitive is a backend concern, not a surface-syntax
  one. Confirmed by `has_cycle`: a hand-derived `fixpoint` definition is useful as a reference/spec
  (pinning down intended semantics precisely enough to check a backend against, the same role
  `tests/hex_oracle.rs`'s BFS oracle plays for `flood6`), not as something the grammar must be able
  to express as ordinary authoring-surface code -- see `style-c/05-havannah-cycle.gdl` and the
  top-level `HISTORY.md`'s session note.
- **Promote to a composable primitive once a second dedicated special case
  appears — don't wait for a third to force it.** `core::EndRule` growing a
  second hardcoded, non-composable variant (`Connected`, copying `Line`'s
  shape wholesale instead of expressing both as composable Region-algebra/
  Boolean expressions — see "Already covered" below) is exactly the failure
  this principle exists to prevent: two data points already show the
  pattern (an end rule is really "some Boolean/Region predicate over the
  board is true"), and Y's three-edge win was the third — landed by
  generalizing `connects`'s operand pair to an arbitrary list (see "Already
  covered" below), not by adding a third dedicated variant. "Grow from
  real lowerings" means grow the *combinator design* from evidence already
  in hand, not let ungeneralized special cases silently accumulate because
  no single session's charter happened to ask for the refactor.

## Standard library: Core, Stdlib, and Extern

An audit of every builtin-looking identifier called across `style-c/`'s corpus (all games plus the
five numbered pathological-case fragments) found roughly fifty names with no declared status:
some need real per-topology backend lowering, some are pure algebra expressible in the surface
language itself, and a few are genuinely opaque calls to something outside the language entirely
(a dictionary, a planar-geometry check, an RNG). Nothing distinguishes these three shapes today,
and that ambiguity had already produced real bugs, not just unlabeled examples — `05-havannah-
cycle.gdl` defines `has_cycle` from scratch via `fixpoint`/`frontier`/`adjacent` even though this
doc's own Region-algebra table already lists `has_cycle` as a *core primitive*; `tak.md`
called a `pop(r, s, n)` three-argument form nothing in this doc ever specified (the documented
signature is the one-site `pop(r: Raster, s: Site): Raster`); `sprouts.gdl` reads "whose turn it is"
as `to_move`, `kuhn-poker.gdl`/`tak.md` read it as `mover`, and this doc's own Game-state
combinators table calls it `current_player` — three names for one concept; and
`tak.md`'s `legal_spread_path` called `carried_top_is(from, drops, i, Capstone)`, a
helper never defined anywhere, an outright dangling reference rather than a design question.

Going forward, every name used in a `.gdl`/`.gdls` file must resolve to exactly one of three tiers:

- **Core** — backend-native primitives needing real per-topology lowering (bit shifts, packed-cell
  arithmetic, Zobrist-hash bookkeeping) that the surface language cannot express in terms of
  anything smaller. This is everything already in the Region algebra / Raster ops / Control and
  aggregation / Game-state combinators tables below, plus the temporal operators
  (`state'`/`once`/`always`), plus the corrections this section makes.
- **Stdlib** — ordinary `def`/`template def` bodies, written *in* this project's own surface
  language over Core primitives, checked into a real file rather than invented ad hoc per game.
  Nothing here needs compiler-special-cased backend work; it's authored the same way a game author
  would write `has_road` or `flat_count`. (No stdlib file exists yet — the worked pass below is
  the first real candidate content for one.)
- **Extern** — a genuinely foreign call this language will never be able to express, declared with
  a new `extern def name(params): Type` form: no body, just a typed signature the backend treats
  as an opaque call into host code. `geometric_oracle` (`sprouts.gdl`'s planar-curve intersection
  check), `dictionary_has_prefix`/`dictionary_has_word` (`ghost.gdl`), and `shuffle`/`draw2`
  (`kuhn-poker.gdl`'s card dealing) are the corpus's only real candidates so far — every one of them
  is already flagged inline, in prose, as "black-box" or "externally supplied"; `extern def` just
  gives that existing intent an actual keyword instead of leaving an extern call syntactically
  indistinguishable from a typo or an unfinished `def`. Left open: `shuffle`/`draw2` are also
  *nondeterministic*, which `extern def` alone doesn't flag — whether chance moves need a purity/
  determinism tag on top of `extern`, or whether that's adequately covered by `chance` already
  being its own move kind (see `games/kuhn-poker.gdl`), is undecided; no corpus game has forced a
  second nondeterministic extern call yet to check the two designs against each other.

### Worked pass: every builtin `games/tak.md` uses

Tak is the only file currently on post-review syntax (`guard`, primed `field'`/`out'` bindings,
`Tak[N]` square-bracket instantiation — see the HISTORY.md session notes), so it's the only file
this pass reclassifies. `01`-`05` and `kuhn-poker.gdl`/`sprouts.gdl`/`sylver-coinage.gdl`/`ghost.gdl`
still predate those rounds; their builtin surface is real evidence for the *existence* of gaps
(the `mover`/`to_move`/`current_player` and `len`/`length` collisions above both came from them)
but reclassifying their calls one by one isn't worth doing until they get the same syntax refresh
`tak.md` already had — noted as a followup, not attempted here.

| Name as used in Tak | Tier | Disposition |
|---|---|---|
| `top`, `height`, `push`, `project`, `connects` | Core | Unchanged — already specified below. |
| `pop(r, s, n)` | Core | **New second overload.** The existing `pop(r: Raster, s: Site): Raster` (single top-of-stack pop) stays; adding `pop(r: Raster, s: Site, n: Int): Stack<Value>` (pop `n` pieces as an ordered carry) as a distinct, documented signature rather than an unspecified extra argument. |
| `shift(from, dir, i + 1)` | Core | **Renamed to `walk(site: Site, dir: Direction, n: Int): Site`.** This was silently overloading the name of the existing Region-algebra `shift(dir: Direction): Region -> Region` (shifts a whole region one step) with an unrelated Site-to-Site stepping operation Core never had a name for. Same underlying per-topology adjacency knowledge as `shift`, but a different operand and result type — worth its own name, not an overload. Applied in `tak.md` (both call sites, in `apply_spread` and `legal_spread_path`). |
| `count_where(region, pred)` | Core | Promoted from aspirational (this doc previously only *mentioned* it, in "Move caching", as an example of what a backend might incrementalize) to a real fourth member of the Control and aggregation table, alongside `any`/`all`/`for_each`. Needs the same backend-internal cell-by-cell access `count` already has — not expressible as a Stdlib `def` over the existing combinator set. |
| `is_full(board)` | Core | **New.** A `Raster`-level "every cell holds a value" check, parallel to Region's `is_empty` — needs the backend's own empty-cell sentinel, which is exactly the kind of thing a Stdlib `def` can't see. |
| `opponent(mover)` | Core | **New.** Added to Game-state combinators: `opponent(p: Player): Player`. |
| `mover` | Core | Confirmed as the canonical name for "whose turn it is." `current_player` (this doc's Game-state combinators table) and `to_move` (`sprouts.gdl`) are retired in its favor — `mover` is what every current-syntax file already uses, and reads more like game vocabulary than either alternative. |
| `min`, `sum`, `len`, `range`/`a..b` | Core | Ordinary scalar/collection primitives — type-generic arithmetic, not game-domain library content, so Core rather than Stdlib even though they're trivial. `len` is confirmed as the canonical sequence-length name; `length` (`ghost.gdl`) is retired in its favor. |
| `side(North)` etc. | Core | Needs a value type this doc never listed: `Edge` (already implied by `connects(edge_a, edge_b: Edge): Region -> Bool`'s own signature, and by `regions P1 = (side(NE), side(SW))` in the hardened grammar — just never added to the Value types table). `side(dir: Direction): Edge` is the constructor. |
| `sites(Empty)`, `sites(Occupied(mover))` | Core | Confirmed as direct sugar over `core::Region`'s existing leaf variants (`Occupied`/`Complement`/`Sites`) — the syntactic front door onto a `Region` value, not a separate library function. |
| `take(carried, drops[i], from: i)` | Stdlib | The one real Stdlib candidate this pass found: `take(s: Stack<T>, n: Int, from: Int): Stack<T>` is an ordinary sub-run extraction, definable over a small new Core `Stack<T>` value kind (needs only `len`/`nth`) rather than needing its own backend lowering. |
| `carried_top_is(from, drops, i, Capstone)` | — | **Not a design question — a bug.** Never defined anywhere in the file. Fixed directly (see below): the predicate it was standing in for ("is the single piece landing on the last cell a Capstone") is already answerable from `top(board, from).kind == Capstone`, since Tak's pickup/drop order means the final single-piece drop is always the original stack's top piece — no helper needed at all. |

Two items surfaced by the audit but not resolved by this pass, flagged the same way `Player`/`Team`
was in the HISTORY.md session notes — real, but with no forcing case yet:

- **`Raster` cell `Value`'s shape.** Every game so far accesses it with named fields (`.owner`,
  `.kind`) rather than positional ones (`.0`, `.1`), but nothing formally specifies whether `Value`
  is an anonymous tuple, a per-game closed record type, or something else. Needs its own pass once
  a second `Raster`-typed game (Shibumi's `Pyramid` generalization, or a real corpus `Raster` game
  beyond Tak) exists to check a design against.
- **`has_cycle`.** Kept as a Core primitive (unchanged from its original placement below) — it
  needs real backend-level flood-plus-parent-tracking for performance the way `05-havannah-
  cycle.gdl`'s own `fixpoint` rendering doesn't attempt. That file's version should be re-labeled a
  *reference definition* (what a correct Core `has_cycle` lowering must agree with, not a
  competing definition) the next time `05-havannah-cycle.gdl` gets its own syntax-refresh pass —
  not changed blind here, per the same discipline this section applies to the other stale files.

## Relational GDL: superseded as the primary authoring language

**Status: superseded, kept for its intensional-primitive design points.** A design-spike session
(see `HISTORY.md`) hand-transcribed five pathological cases (Chess's `ifAfterwards` check-safety
filter, Go's `ifAfterwards` suicide-rule filter, Go's positional-superko `(meta (no Repeat))`,
Chess's `ChessPawn` piece-template composition, Havannah's `has_cycle`) in three candidate surface
syntaxes: flat Horn-clause Datalog (this section's original proposal), point-free categorical
notation, and a typed functional/equational style designed from scratch (named `let`-bindings
instead of unification, ordinary function definitions, a restricted typed `then { }` effect block,
templates as compile-time generics). All three converge on the same underlying Core term in every
case, but flat Horn-clause conjunction never reached it *directly* — every case needed either a
bespoke non-`:-` modifier (`ifAfterwards:`, reused for the superko case) grafted onto plain
conjunction, or (Havannah's `has_cycle`) produced an outright incorrect term (naive transitive
closure over an undirected adjacency relation is trivially "true" for any edge) until it borrowed
the categorical style's discipline of writing the threaded state's type down explicitly. The typed
functional/equational style reached every case as directly as the categorical style while needing no
new vocabulary (`let`, generics, and a named accumulator loop, not `guard`/`Tr`/`≜`) — see
`HISTORY.md`'s spike write-up for all five cases in all three styles and the full reasoning.

**Consequence:** this section's core claim ("Horn-clause logic is sugar over the categorical core")
is retracted as a description of the primary authoring surface — a human should not be asked to
write Prolog-style unification bodies for these rules. What survives from this section: `adjacent`/
`shift`/`flood`/`connects` as **intensional, compiler-known primitives** (not exploded ground-fact
tables) was never actually a Datalog-specific idea and carries over unchanged to whatever surface
replaces this section; conjunction-as-join/disjunction-as-union/negation-as-complement likewise
remain true statements about Core's *semantics*, just no longer a description of what a human types.
The categorical structure below is unaffected — it was never proposed as the authoring surface, and
the spike confirmed it as the correct desugaring *target*. The next session designs the typed
functional/equational style's actual grammar (Style C in `HISTORY.md`'s spike) as the real primary
authoring language replacing this section.

A game is a finite set of typed relations over `Region`/`Raster`/`Site`/`Player` objects (the
Core value types below), and rules define new relations by Horn-clause-style bodies over them —
the same shape as Stanford GDL (Prolog/Datalog), but with `adjacent`/`shift`/`flood`/`connects`
as **intensional, compiler-known primitives** parameterized by `Topology` instead of exploded
ground-fact tables, which is what made vanilla GDL engines fail to scale to real boards:

```
legal(add(P, S))        :- turn(P), empty(S).
legal(hop(P, From, To)) :- piece(P, From), enemy(P, Between),
                           adjacent(From, Between), opposite(From, Between, To),
                           empty(To).
next_occupied(P, To)    :- legal(hop(P, From, To)).
```

Conjunction in a rule body is a join — under a `Region` encoding, a bitwise AND; disjunction
across rule instances is a union (OR); stratified negation is a complement. This lowering is
syntax-directed and compositional, not pattern-matching over operational idioms: the source-level
semantics already *is* the target-level semantics.

State transition (`next`) is a **pure function of `(state, move)`** — which facts flip — never a
sequence of effects. There is no `then`, `apply`, or `moveAgain`: a Ludii-style chain capture is
instead the least fixpoint of a bounded recursive relation (see `bounded_fixpoint` below and the
"Categorical structure" section), the same mechanism that covers flood-fill territory and
Havannah's `has_cycle`. Bounded recursion is genuinely native here (Datalog's stratified
least-fixpoint semantics is decidable and well understood), not a bolted-on escape hatch the way
it would be layered onto an ordinary functional core.

This section was deliberately a sketch, not a grammar — and per the "Status" note above, it's now a
retired sketch: the design spike found flat Horn-clause bodies didn't reach the pathological cases
directly, and the actual syntax/type system work moves to the typed functional/equational style
instead (see `HISTORY.md`'s next-session charter), checked the same way this section always intended
— against real corpus games, starting with `Tic-Tac-Toe`/`Hex`'s existing working Core programs and
oracles — per this doc's "grow from real lowerings" principle.

## Categorical structure (applied narrowly, where it resolves a named problem)

Core's Region algebra already wants to be a well-behaved algebraic structure — union is a
commutative monoid, `shift` should distribute over `union`, etc. (see "Non-goals" below on *not*
building a general categorical framework up front). Two places in this doc had an unresolved
shape that a specific piece of category theory resolves exactly, so they're adopted now, narrowly,
rather than left ad hoc:

**Pure Region algebra is a cartesian category; effects are a separate, non-cartesian layer
(Freyd categories).** `Region` values support both genuine duplication (reused freely across
multiple rule bodies — the thing that makes CSE and "referentially transparent" *true* rather than
just asserted) and a genuine merge (`union`), because `Region` is a commutative-idempotent-monoid-
valued type, not a linear resource — fan-out is safe here in a way it isn't in general
symmetric-monoidal settings. That's the pure half. The effectful half — history state threaded
across moves (Congo's `river_since`, ko rules; previously the unresolved "History-dependent state"
open problem) and the `then`/`apply`/`remember` idioms encountered when translating a `.lud`
game — is a Freyd category: an identity-on-objects functor from the pure cartesian category into a
premonoidal "actions" category whose morphisms can be sequenced but can't be freely reordered or
duplicated the way pure `Region` expressions can. Concretely: Core's auxiliary state lives in this
premonoidal layer, composed with — but never implicitly commuted past — the pure Region-algebra
layer. Not yet implemented in `core::mod`, which still only has ad hoc extra scalars, but no
longer an open *design* question.

**`bounded_fixpoint` is a bounded trace.** A traced monoidal category's trace operator
(`Tr: Hom(A⊗X, B⊗X) -> Hom(A,B)`, feeding an output wire back into an input) is the standard
categorical shape for "feedback" in a wiring diagram, and it's the same shape `bounded_fixpoint`
already has informally: Congo's chain capture, Tak's spread-move, and Havannah's `has_cycle` (a
flood/reachability fixpoint over the adjacency relation, not a bespoke new opcode) are all the same
feedback construction at different bounds. Adopting the trace framing doesn't change
`bounded_fixpoint`'s Rust shape today, but it's the thing to check before writing an optimizer
pass over it: trace axioms (naturality, yanking, superposing) are exactly the laws that would
justify fusing or reordering two bounded fixpoints, rather than that being argued fresh per game.
**Confirmed, not just hypothesized, by the design spike** (see `HISTORY.md`): Havannah's `has_cycle`
specifically forced the threaded state object to be `(Region, Raster<Direction>)` rather than bare
`Region` — a naive bare-`Region` transitive-closure rendering is not just weaker, it's outright
wrong (every edge in an undirected adjacency relation falsely looks like a 2-cycle). The trace
framing generalizes cleanly over the *type* of the threaded state object; bare-`Region` flood and
`(Region, Raster<Direction>)` `has_cycle` are genuinely the same construction at two different state
types, not two different constructions that happen to rhyme.

**Status: confirmed as Core's real desugaring target, not the authoring surface.** The same spike
also confirmed `guard(P) : A -> A` (a restriction-category-style partial identity, defined only
where `P` holds — the categorical reading of `ifAfterwards:`) belongs in this Freyd-category story:
`ifAfterwards:P` is `guard(P ∘ next)` fanned around a move candidate, for both Chess's check-safety
filter and Go's suicide-rule filter, and Go's positional-superko rule reuses the identical
construction with `P` reading membership in the effects-layer history set. See "Relational GDL"
above — this section stays as Core's internal semantics/optimizer-law foundation; a human should not
be asked to write `guard`/`Tr`/`∘` by hand, and the spike found a typed functional/equational surface
(named `let`s, ordinary function definitions) reaches the same terms without that vocabulary.

Deliberately not adopted yet: a general symmetric-monoidal-category *library* (an actual `Cat`
abstraction backend lowering is written against — though informally, backend lowering already is
a functor to a concrete backend category, one per `Topology` variant), or Petri-net semantics for
move concurrency (a real, correct connection — Petri nets and free symmetric monoidal categories
correspond via Meseguer–Montanari — but nothing in the corpus needs genuine concurrent/
simultaneous moves yet).

**Universality target: descriptive complexity, not unrestricted universality.** `HISTORY.md`'s
GDL/Ludii evaluation session concluded this project should not claim universality the way GDL
(unrestricted logic programming) or Ludii (an after-the-fact corpus-coverage proof over an ad hoc
ludeme set) do — this doc's own "First-order, not full lambda calculus" principle and
`bounded_fixpoint`-over-general-recursion choice already rule that out by design. `COMPLETENESS.md`
states the intended replacement precisely: a conjecture that Core IR's fragment (Region algebra +
`bounded_fixpoint` + bounded `fold` + the temporal layer) is exactly FO(LFP), which by
Immerman–Vardi equals PTIME on ordered finite structures (an assumption this repo already satisfies
for free, since `BitBoard`'s row-major indexing already fixes a canonical `Site` order). See that
document for the primitive-by-primitive classification, the rule-complexity-vs-game-complexity
distinction that must not be conflated, and what remains to turn the conjecture into a real proof.

## Topology model

```rust
enum Topology {
    /// N x M grid, single-bit occupancy per cell, 4/8-way + diagonal
    /// adjacency. Backend: `BitBoard<N, M>` (N*M <= 64) or
    /// `BigBitBoard<N, M, WORDS>` (N*M > 64) -- games/game-core/src/{bitboard,bigbitboard}.rs.
    Rect { rows: usize, cols: usize },

    /// Hex grid, 6-way adjacency, axial coordinates packed into a
    /// rectangular (rhombus) or triangular bit layout depending on shape.
    /// Backend: NEW, no precedent in games/ yet.
    Hex { shape: HexShape },

    /// Stack of shrinking square layers with a 4-cell support constraint
    /// (a piece at level k+1 requires all 4 cells beneath it at level k).
    /// Backend: NEW generalization of games/shibumi/src/lib.rs, which today
    /// hand-rolls a single WIDTH=4/STACK_LEVELS=4 instance.
    Pyramid { base: usize, levels: usize },

    /// Rect-shaped grid where each cell holds a small packed integer
    /// (piece kind + owner + stack height + per-slot color bits), not a
    /// single bit. Backend: NEW generalization of games/tak/src/lib.rs's
    /// hand-rolled `cells: [u64; 36]` packed-word-per-cell encoding.
    Raster { rows: usize, cols: usize, cell_bits: u32 },
}

enum HexShape {
    Rhombus { side: usize },   // Hex
    Triangle { side: usize },  // Y
    Hexagon { side: usize },   // Havannah
}
```

`Rect` and `Pyramid`'s support-constraint arithmetic and `Raster`'s
packed-cell arithmetic already have working, tested Rust precedent in this
repo (`games/game-core/src/bitboard.rs`, `bigbitboard.rs`,
`games/shibumi/src/lib.rs`, `games/tak/src/lib.rs`). `Hex` had none when this
sketch was written — it's since grown real code for two of `HexShape`'s three
variants (`core::hex::Hex`/`HexShape`, `Rhombus` proven by Hex, `Triangle` by
Y): both turned out to need **no new bit layout or backend at all**, just the
same `side x side` `BitBoard<N, N>` `Rect` already uses, with `Triangle`
additionally masking legal moves down to `row + col < side` via a new
`Region::Intersect` combinator (see "Region algebra" below) — a triangular
board is a bounded subset of the same infinite hex lattice a rhombus board
also samples from, not a different coordinate system. `Hexagon` (Havannah)
remains real, unstarted design work — see "Backend lowering" below for why
it's the hard case the other two weren't.

## Core IR

### Value types

| Type | Meaning | Existing precedent |
|---|---|---|
| `Region<T>` | bitset over topology `T`'s cells | `BitBoard`/`BigBitBoard` |
| `Raster<T>` | small-int-per-cell over topology `T` | Tak's `cells: [u64; 36]` |
| `Site<T>` | a single cell index | `BitBoard::from_index` argument |
| `Direction<T>` | one of `T`'s adjacency directions | `bitboard::Direction` (4-way today; needs a 6-way variant for `Hex`) |
| `Player` | turn owner | `PlayerIndex` |
| `Bool`, `Int` | ordinary scalars | — |
| `Edge<T>` | one of `T`'s named board edges, as used by `connects` | `side(dir: Direction): Edge`, per "Standard library" below |
| `Stack<T>` | an ordered run of `T` values popped together off a `Raster` cell | `pop`'s 3-arg overload, per "Standard library" below |

### Region algebra (topology-generic)

```
union, intersect, complement, difference : Region -> Region -> Region
shift(dir: Direction)                    : Region -> Region
flood(seed: Region, conn: Connectivity)  : Region -> Region
adjacent(conn: Connectivity)             : Region -> Region
is_empty, count, member(site)            : Region -> Bool | Int
connects(edges: [Edge])                  : Region -> Bool
has_cycle                                : Region -> Bool
```

`shift`, `flood`, `adjacent`, and `connects` are direct Core-level names for
what `games/game-core/src/bitboard.rs` already implements per-direction
(`shift_north`/`shift_northeast`/etc., `flood4`/`flood8`,
`adjacency_mask`, `connects_walls4`/`has_opposite_connection4`). Backend
lowering for `Rect` is close to the identity mapping — that's intentional;
it's the existing, proven code, and it's the strongest evidence the
combinator set is *not* overfit to one topology, since the same names must
also make sense for `Hex`.

**Implemented in `core::mod`/`core::interp`** (`shift`/`flood`/`adjacent`/`intersect` as real
`Region` variants plus `connects` as a `BoolExpr` variant, all proven against Tic-Tac-Toe/Hex/Y —
see the "Already covered" table's Hex/Y entries and `HISTORY.md`'s session notes). One correction
this pass made to the signature above: `flood`'s `seed` is `Region`-valued, not `Site`-valued as an
earlier draft of this table had it — a connectivity check seeds from a whole board edge (potentially
several sites, e.g. Hex's `(sites Side NE)`), not a single site, which `flood6`'s own signature
(`game_core::bitboard::BitBoard::flood6(self, seed: Self)`) already reflected before this table did.
A second correction, from Y: `connects`'s operand list is `[Edge]`, not a fixed `(edge_a, edge_b)`
pair — `core::Program.player_regions: Vec<Vec<Region>>` and `core::interp::eval_bool`'s
`BoolExpr::Connects` arm both generalize the same way, flooding from the first entry and checking
the result intersects every remaining one; Hex's two-edge win and Y's three-edge win are now two
lengths of the same list, not two shapes. `intersect` (`Region::Intersect`) is also new, forced by
Y's triangular board: `(sites Empty)` there means "empty AND inside the triangle," not just
"empty," since a triangle's valid sites are a proper subset of the `side x side` grid its
`core::hex::Hex::valid_sites` is carved out of — see "Already covered" below.

`has_cycle` is new — no existing game needs it (see Havannah below, the game
that forces it in). This session confirmed (not just asserted) that `core::interp::bounded_fixpoint`
— the generic trace function `Region::Flood` is one instantiation of — can hold `has_cycle`'s
richer simultaneous `(visited, parent, cycle)` state via its `Aux` type parameter, without a second
IR node shape; see `COMPLETENESS.md`'s primitive table and `core::interp`'s
`has_cycle_shape_holds_a_parent_and_cycle_flag` test. `has_cycle` itself is still not landed as a
`Region`/`BoolExpr`/`Program` primitive.

### Raster ops (stacking topologies: `Pyramid`, `Raster`)

```
top(r: Raster, s: Site)          : Value           // top-of-stack piece kind+owner
height(r: Raster, s: Site)       : Int
push(r: Raster, s: Site, v: Value) : Raster
pop(r: Raster, s: Site)          : Raster           // pop the single top value
pop(r: Raster, s: Site, n: Int)  : Stack<Value>     // pop the top n values as an ordered carry
is_full(r: Raster)               : Bool             // every cell holds a value -- Raster's `is_empty`
support(t: Topology, s: Site)    : Region           // cells that must be occupied to place at s
project(r: Raster, pred: Value -> Bool) : Region    // e.g. "cells whose top piece belongs to player X"
```

`pop`'s two-argument and three-argument forms are genuinely different operations sharing a name by
overload, not one operation with an optional argument: the two-argument form removes and returns
the top single `Value`, leaving a smaller `Raster`; the three-argument form removes and returns an
ordered `Stack<Value>` of the top `n` values, for a move (Tak's spread) that carries a sub-stack
away from its origin cell before redistributing it elsewhere. `is_full` needs the same backend
empty-cell-sentinel knowledge `is_empty` already does — see "Standard library" above.

`support` generalizes Shibumi's hand-rolled `MASKS`/`index()` pair
(`games/shibumi/src/lib.rs:19,68-73`) into a topology-level primitive so
`Pyramid { base: 5, levels: 5 }` (Margo-sized) gets the same support-mask
table generation `Pyramid { base: 4, levels: 4 }` (Shibumi) gets today,
instead of a second hand-written `MASKS` constant.

`project` is how Region algebra composes with stacking games: Tak's road
connectivity is computed by flood-filling a `Region` derived from
`project(cells, |v| v.owner == player)` — i.e. Raster games don't need a
second connectivity implementation, they need one extraction step down to
`Region` and then reuse the same `flood`/`connects` combinators as `Rect`
games.

### Control and aggregation

```
let x = e in body
if c then e1 else e2
for_each(region: Region, body: Site -> Effect) : [Effect]
any(region: Region, pred: Site -> Bool)  : Bool
all(region: Region, pred: Site -> Bool)  : Bool
count_where(region: Region, pred: Value -> Bool) : Int
fold(seed: Acc, iter: Region | Range | [T], step: (Acc, T) -> Acc) : Acc
bounded_fixpoint(seed: State, step: State -> Option<State>, max_iters: Int) : State
min, max, sum, len, range(a: Int, b: Int)   : ordinary scalar/collection ops, generic over Int/[Int]
```

`count_where` needs the same backend-internal cell-by-cell traversal `count`/`any`/`all` already
have, so it's a fifth member of this table rather than something a Stdlib `def` could build out of
the others — see "Standard library" above for why this was previously only aspirational.

`fold` is an ordinary call, not a bespoke block-header special form the way an earlier
`style-c/games/tak.md` draft had it (`fold out = seed for i in iter { ... out' = expr }`)
— that draft read as unwarranted Alloy-adjacent sugar in live review, and concretely gave a
pre-existing `def` nowhere to plug in as `step`, unlike `any`/`all`/`project` above, which already
take their predicate as an ordinary lambda argument.

**On `pred`/`step` arguments and "no first-class functions":** every function-typed parameter in
this table (`any`/`all`/`count_where`/`for_each`/`fold`/`bounded_fixpoint`'s `pred`/`step`, and
`project`'s `pred` in Raster ops above) is filled by an inline lambda, lexically scoped at its own
call site, applied immediately by the one combinator it's passed to, never stored in `state`,
returned from a `def`, or passed on to a second combinator. That's a real pattern, worth naming
once rather than leaving implicit at each call site: call it a **second-class lambda** — it reads
like a closure and captures enclosing bindings the way one does (`project(board, |v| v.owner ==
p && ...)` closes over `road_region`'s own `p` parameter), but the compiler can always inline or
lambda-lift it away at the single point it's used, so it never needs a real runtime function value,
an environment record, or an escape/alias analysis. This is *not* an exception to "First-order, not
full lambda calculus" below — it's the precise thing that principle was already relying on for
`any`/`all`/`project`, just never spelled out as its own rule until `fold`'s redesign made the gap
concrete. Consequence for "can I pass a `def` directly instead of a lambda": yes, exactly when that
`def`'s own parameter list already covers everything the step needs (no free variables to close
over) — e.g. `fold(seed, iter, my_step)` where `my_step(acc: Acc, x: T): Acc` is an ordinary
top-level `def`. When it needs to reference the caller's local bindings (as `apply_spread`'s `fold`
step needs `from`/`dir`/`carried`), write it inline as a lambda instead; that's not a new limitation
`fold` introduces, it's the same constraint `any`/`all`/`project`'s predicates already had.

`bounded_fixpoint` is the one concession to genuine, unbounded-shape "recursion" (as opposed to
`fold`'s statically-known-length walk): Congo's Monkey multi-jump chain capture and Havannah's
cycle check are both repeated application of a step function until convergence, not a fixed number
of iterations known before the walk starts. Modeling it as a capped fixpoint keeps Core first-order
(`max_iters` is always a static bound derivable from board size) while still covering the real
cases in the corpus that looked recursion-shaped at first glance.

### Game-state combinators

```
mover          : Player   // whose turn it is now -- canonical name, see "Standard library" above
opponent(p: Player) : Player
turn, phase    : Int
score(p: Player) : Int
end_if(cond: Bool, result: Outcome)
```

## Backend lowering

Core → backend is one pass per `Topology` variant. Only `Pyramid`/`Raster` and `Hex { Hexagon }`
still lack a working target to lower onto — `Rect` and both proven `Hex` shapes already have one:

- **`Rect`, N×M ≤ 64** → `BitBoard<N, M>` (`games/game-core/src/bitboard.rs`).
  Direct mapping, already proven by 9 games.
- **`Rect`, N×M > 64** → `BigBitBoard<N, M, WORDS>`
  (`games/game-core/src/bigbitboard.rs`), used today for Tanbo's 19×19
  board. Same combinator set, carried-shift word arithmetic instead of
  single-`u64` shifts.
- **`Pyramid`** → generalize `games/shibumi/src/lib.rs`'s packed-`u32`,
  triangular-number `index()` scheme to arbitrary `base`/`levels`, with
  `support_mask` computed the same way `BitBoard::WALL_LUT` is: a `const fn`
  table built once per topology instantiation instead of Shibumi's literal
  `MASKS: [u32; 3]`.
- **`Raster`** → generalize Tak's packed-cell-word scheme
  (`games/tak/src/lib.rs`'s `cells: [u64; N*N]`, `pack`/`kind`/`color_bit`
  helpers) to a topology-parameterized `cell_bits` layout.
- **`Hex { Rhombus }`/`Hex { Triangle }`** → **no new backend needed, turned out.** Both reuse
  `BitBoard<N, N>` directly: `Rhombus` (Hex) uses every site, `Triangle` (Y) masks legal moves down
  to `row + col < side` via `Region::Intersect` — six-way adjacency (`flood6`) is unchanged between
  the two, since a triangular board is a bounded subset of the same lattice a rhombus board also
  samples from, not a different coordinate system. See `core::hex::Hex`/`HexShape`'s module doc and
  "Already covered" below. `Hexagon` (Havannah) is still the hard case, and this finding doesn't
  extend to it by analogy: a hexagon-shaped hex board doesn't tile a rectangle or triangle cleanly,
  so it likely needs its own coordinate packing rather than reusing `Rhombus`'s. This remains real,
  unstarted design work — treat it as its own milestone before attempting Havannah.

### Slider move generation (queen/rook/bishop rays): a worked comparison for Amazons

Amazons (see "Worth adding" below) forces a queen slide-to-first-blocker move shape on a 10×10 board — the
first corpus candidate where naive move generation (ray-cast one cell at a time per direction) is a real
*efficiency* question, not just a correctness one, per that table's own framing. Three candidate techniques
exist in the wider bitboard-programming literature; this pass checked each against this project's actual
`BitBoard`/`BigBitBoard` shape (row-major `to_index(row, col) = row*M + col`) rather than citing them from
memory, since the multi-word `BigBitBoard` boundary turned out to matter more than expected.

- **Kogge-Stone-style doubling flood — already proven, in production.** `games/othello/src/lib.rs`'s
  `flood_left`/`flood_right` (repeated shift-and-OR by doubling distances, masked against a wall guard each
  step) already computes exactly this shape — "flood along a direction until blocked" — for Othello's
  slide-and-flip, generalizing directly onto this doc's own `shift(dir)`/`Direction` Region-algebra
  primitives above. It needs zero precomputed tables, and composes with `BigBitBoard` for free (its `shift`
  already handles cross-word carries — `games/game-core/src/bigbitboard.rs`). **Recommended as the first
  implementation for Amazons'** queen move (and arrow-shot, same ray geometry): per "Move caching" below's
  own "get a correct backend that recomputes every predicate every ply first" principle, reach for something
  fancier only once profiling shows this is a bottleneck at 10×10 scale.
- **Hyperbola Quintessence (`(o ^ (o - 2*s)) & mask`, plus a reversed-word pass for the opposite direction)**
  — verified correct, but with a hard boundary at exactly this project's own `BitBoard`/`BigBitBoard` split.
  Checked against a naive ray-cast oracle over 5,000 random `(square, occupancy)` trials each: **5000/5000**
  matched on an 8×8 board (fits one `u64`, same as `BitBoard<N,M>` for `N*M <= 64`); only **1784/5000**
  matched on a 10×10 board (100 cells, doesn't fit one word) using a naive single-word bit-reversal. The
  failure is real, not incidental: `BitBoard::reverse_bits` (`games/game-core/src/bitboard.rs:266`) already
  exists and is exactly right for the `BitBoard` case, but a correct 10×10 version needs a genuine multi-word
  bit-reversal (reverse word order *and* reverse bits within each word) plus ripple-borrow subtraction across
  words for the `o - 2s` half — real, unbuilt `BigBitBoard` primitives, not a free generalization of the
  single-word trick. Worth keeping in mind for a future `Rect` game that stays under 64 cells and has a hot
  single-line connectivity/slide check; not recommended for Amazons as-is.
- **Magic bitboards (per-square multiplicative hash into a precomputed attack table)** — the search itself
  works fine at Amazons' scale; an initial worry that a *combined* queen ray-mask would exceed 64 bits on a
  10×10 board was checked and was wrong. Measured mask sizes (edge-inclusive, no far-square-exclusion
  optimization):

  | | 8×8 | 10×10 |
  |---|---|---|
  | rook (both axes) | 14 bits, *every square, always* — `(N-1)+(M-1)`, provably position-invariant since opposite-direction ray lengths always sum to the full dimension | 18 bits, every square |
  | bishop | 7–13 bits | 9–17 bits |
  | queen (rook ∪ bishop) | 21–27 bits | 27–35 bits |

  35 bits fits a single `u64` multiply comfortably even for Amazons' combined queen mask — the real obstacle
  is per-square table *size* if a naive dense queen table were built (`2^35` entries), which is why real
  chess engines never build one: separate rook-magics (≤18 bits here) and bishop-magics (≤17 bits) unioned
  at query time keeps every per-square table ≤ 262,144 entries. Magic numbers for a handful of representative
  8×8 squares were found and verified collision-free against every occupancy subset of their mask (e.g. rook
  corner: `0x020C644040008000` in 7,914 random tries; bishop center: `0x0001004002040002` in 149 tries) — the
  search itself converges fast. **These belong in the Core→backend lowering pass, generated once per template
  instantiation, not searched at runtime**: board size is already a compile-time template parameter (the same
  `[const N: Int]` monomorphization `tak.md`'s `piece_reserve(N)`/`stack_bits(N)` tables use), so the mask
  shapes and a valid magic are fully determined the moment a game like `Amazons[10]` is instantiated — codegen
  should run the (seeded, deterministic) search once and emit the found constants as a `static` array literal
  in the generated source, the same way `shibumi`'s `MASKS: [u32; 3]` is a checked-in constant today, not
  something recomputed at process startup.

De Bruijn sequences (a classic bitscan technique: `index = table[((bb & -bb) * DEBRUIJN64) >> 58]`) were also
considered and are **not worth adopting**: `BitBoard`'s existing `trailing_zeros()`/`leading_zeros()`
(`games/game-core/src/bitboard.rs:261-263`) already lower to a single hardware `TZCNT`/`LZCNT` instruction via
Rust's intrinsics on every target this project cares about. The multiply-and-lookup trick was a workaround for
languages/eras without that intrinsic; it adds a table and an extra multiply for no benefit here.

## Cross-cutting backend concerns: hashing, symmetry, caching, evaluation

Once Core state is a uniform, typed `(Site, Value)` space instead of a
per-game hand-rolled struct, four things that `games/*` currently
hand-writes separately per game become *derivable* from the same
`Topology`/state description, once, in the backend — not four more things
to hand-design per compiled game.

### Zobrist hashing

Generalizes `mcts::zobrist::LazyZobristTable<N>` (`mcts/src/zobrist.rs`):
today each game picks its own `N` and its own mapping from board state to
table index by hand (see `games/ttt`, `games/othello`). Once Core knows a
topology's site count and each site's value cardinality (1 bit for `Region`,
`cell_bits` for `Raster`, occupancy × piece-kind for `Pyramid`), the table
size and the `(site, value) -> table index` mapping are both derivable at
compile time, and Core's `apply`-effect combinators (`push`/`pop`/region
set/clear) already say exactly which `(site, value)` pairs flip on a given
move — so generated `apply` can XOR in the incremental update for free,
rather than a human re-deriving which table entries a given move touches
for each new game.

### Symmetry

`games/game-core/src/symmetry.rs`'s `D4Symmetry<S>` is hand-verified for
square `Rect` boards only (used by `othello`, `traffic-lights`). Ludii
itself computes board *geometry* automatically (graph/tiling structure,
adjacency, directions — see "General Board Geometry", Piette et al.), which
is exactly the structure a symmetry-group deriver would consume, but
nothing in the published system suggests it exposes or exploits
*automorphism-group* symmetry (equivalence classes of positions/moves under
D4 etc.) as an engine feature the way `D4Symmetry` does for hashing/search
here. That reads as a real gap Core can close, not prior art to port.

The derivation itself generalizes `D4Symmetry`: a topology's geometric
automorphism group (D4 for square `Rect`, the smaller Klein four-group
`{id, H, V, HV}` for non-square `Rect`, D6/D12 for `Hex { Hexagon }`, a
per-level D4 for square-based `Pyramid`, etc.) is computable from the
`Topology` descriptor alone. **The hazard**: geometric symmetry of the grid
is necessary but not sufficient — the *rules* must also be symmetric under
the transform. Congo's board is a square grid (D4-geometric) but the river
occupies one absolute rank and the starting position isn't reflection
symmetric across it, so naively applying `D4Symmetry`-style canonicalization
would be unsound. Correct derivation has to intersect the topology's
geometric automorphism group with the subgroup that also fixes every
absolute site/region the `equipment`/`rules` reference by name (river rank,
home ranks, etc.) — deriving the *geometric* group is the easy half.

### Move caching

A backend optimization pass, not a Core-level concept — this is the
existing "referential transparency lets the backend choose incremental vs.
recompute" design principle, made concrete. `games/druid/src/movecache.rs`
is the precedent: `MoveCandidates` maintains per-anchor legality bits
incrementally instead of recomputing `State::*_legal_at` for every anchor on
every call to `generate_actions`. The general version: any Core aggregate
(`any`, `all`, `count_where`, `for_each`) evaluated once per ply over a
`Region` that changes by a small, move-local delta is a candidate for
incrementalization, *if* the backend can bound the predicate's "reach" (how
far from a touched site its value could change) — Druid's lintel legality
only depends on a 3-cell neighborhood, which is exactly what makes its cache
cheap to invalidate correctly.

This is explicitly tier-2 work: get a correct backend that recomputes every
predicate every ply first (that's what proving the pipeline end-to-end on
tic-tac-toe should target — nothing about a 9-cell board needs caching),
and only add incrementalization once profiling a real, larger compiled game
shows it matters.

### Heuristics, analysis, and position evaluation

No new Core combinators needed yet — piece count, mobility count, and
region-distance-to-goal are already expressible with the Region algebra and
Raster ops above. What's missing is a place in the *pipeline*: tag a
compiled Core program with a role — `MoveGen | Terminal | Heuristic |
Analysis` — sharing one expression language, differing only in what a
program is allowed to return (`Region`/effects for `MoveGen`, `Bool` for
`Terminal`, `Int`/score for `Heuristic`) and where codegen wires the result
(a `Heuristic` program plugs into `mcts`'s minimax/AB eval or MCTS rollout
policy hooks, not the `Game` trait's move/apply/terminal methods). Deferred
past the first proof-of-chain slice — there's no heuristic worth writing
for tic-tac-toe.

## Representative game corpus

The existing `games/` crates already cover a good spread of `Rect`/`Pyramid`
mechanics. What's genuinely missing — and what makes "stranger" games worth
adding deliberately — is anything that forces a *new* topology or a new
Region-algebra primitive, rather than just another game that reuses `Rect`
+ flood-fill. Each entry below is now a target for translation-by-understanding into this
project's authoring surface (see above), verified against an oracle — not a shape for
`elaborate/` to grow to cover — but which primitive/topology each game forces is unchanged by
that shift.

`database-1/lud/games/` is the full, real, un-concretized Ludii games database (~1650 files, source
of record) -- read by hand (or by an LLM) to translate a game, per the "Goal"/"Translating `.lud`"
sections above. This project used to also keep a small `lud/` directory of hand-concretized `.lud`
fixtures (fixed board size, options/templates resolved) that a now-deleted mechanical pipeline
loaded at test time; per `ROADMAP.md`'s decision that directory and its load-bearing role are gone
-- every game below is instead checked against `style-c/sexpr/*.gdls` fixtures and an oracle, the
same way `y` already was before this cleanup.

### Already covered (existing `games/*`)

| Game | Topology | What it already proves |
|---|---|---|
| `ttt`, `bid_ttt` | `Rect` (small) | baseline placement + terminal check |
| `othello` | `Rect` 8×8 | directional slide-and-flip (`Direction`-indexed shifts) |
| `atarigo` | `Rect` (big, via `BigBitBoard`) | `flood4`/`check_go_move` capture logic |
| `breakthrough`, `knightthrough` | `Rect` | piece-specific move geometry (pawn diagonal capture, knight leap) |
| `congo` | `Rect`, mailbox+bitboard hybrid | terrain-conditioned movement (river), *bounded* multi-jump chain capture (Monkey), per-square history state (`river_since` drowning counter) |
| `tanbo` | `Rect` 19×19 (`BigBitBoard`) | large-board flood-fill, territory scoring |
| `shibumi` | `Pyramid` (base 4, levels 4) | support-constrained stacking |
| `tak` | `Raster` (packed stack-per-cell) | dynamic connectivity over a *derived* region (stack tops), variable-length spread moves |
| `gonnect` | `Rect`, big board | edge-to-edge `connects` win condition (Go-Connect hybrid) |
| `druid` | `Rect` | fixed small-template placement (`Piece::Lintel`, a rigid 3-cell shape in 2 orientations) — a narrow case of shape-placement, not full polyomino rotation |
| `nim`, `count`, `unit`, `null`, `traffic-lights` | non-bitboard / trivial | harness/test games, not board-topology-relevant |
| `hex` (`style-c/sexpr/hex.gdls`, Core-interpreted via `style_c`, no `games/hex` crate) | `Hex { Rhombus }` | axial-into-rectangle packing (reuses `BitBoard<N, N>` directly — see below), six-way adjacency (`flood6`), edge-to-edge `BoolExpr::Connects` |
| `y` (`style-c/sexpr/y.gdls`, Core-interpreted via `style_c`) | `Hex { Triangle }` | triangular masking of the same `Rhombus` grid/adjacency (`Region::Intersect` against `Hex::valid_sites`), three-edge `connects` (generalizes `player_regions`/`BoolExpr::Connects` from a fixed pair to an arbitrary list) |

Hex turned out *not* to need a new bit layout or backend at all: axial coordinates `(col, row)`
packed into the same row-major `BitBoard<N, N>` indexing `Rect` already uses, with adjacency
restricted to six of `Rect`'s eight queen-move directions (N/S/E/W plus one diagonal pair —
`game_core::bitboard::BitBoard::flood6`), turns out to be a complete, correct hex-rhombus
topology. `core::Topology` is now a real enum (`Rect`/`Hex`) rather than the single hardcoded
`Rect` field the first session left it as, and `core::Region` grew a `Sites(Vec<usize>)` leaf (a
static site list, e.g. a board edge) alongside `Occupied`/`Union`/`Complement` — and, per the
"Design principles" corollary above (flagged as due once `EndRule::Connected` became a second
dedicated, non-composable variant, the same shape `EndRule::Line` already was), `connects`/`flood`
are now real, composable Region-algebra combinators: `core::Region` also has `Shift`/`Adjacent`/
`Flood` variants, `core::BoolExpr::Connects` replaces the old `EndRule::Connected` Rust variant,
and `core::interp` evaluates `Region::Flood` via a generic `bounded_fixpoint` trace function
instead of calling `BitBoard::flood6` directly. `EndRule` itself is now just `{ condition:
BoolExpr }` — Tic-Tac-Toe's line-win and Hex's connectivity-win are two different `BoolExpr` values,
not two Rust enum variants. See `HISTORY.md`'s session note for what this confirmed (and one real
open gap it surfaced: `connects`'s edge operands are still looked up per-mover by the interpreter,
not embedded as literal `BoolExpr` operands — see that variant's doc comment) and
`COMPLETENESS.md`'s primitive table for how it checks against the FO(LFP) upper-bound conjecture.

Y (`style-c/sexpr/y.gdls`) confirmed the coordinate-packing prediction wrong in the useful direction:
a `Hex { Triangle }` board turned out *not* to need "coordinate packing that isn't just Hex's
rhombus with corners chopped" (see the now-superseded "Worth adding" rationale below) — it's
literally the same `side x side` grid and six-way adjacency, restricted to `row + col < side` via a
new `Region::Intersect` combinator masking `(sites Empty)` against `core::hex::Hex::valid_sites`.
The three-edge win did force the predicted generalization: `core::Program.player_regions` is now
`Vec<Vec<Region>>` (was `Vec<(Region, Region)>`) and `core::interp::eval_bool`'s `BoolExpr::Connects`
arm floods from the first entry and checks the result intersects every remaining one — Hex's
two-edge win and Y's three-edge win are two lengths of the same list, not two `BoolExpr` shapes.
Y is deliberately **not** pushed through the `.lud`/`ast`/`elaborate` pipeline the way Hex was —
consistent with this doc's own "Goal" section framing that pipeline as legacy, not where new corpus
games should grow — so there's no `lud/Y.lud` hand-concretized fixture; the authoring surface is
directly `style_c`'s sexpr frontend, checked against a from-scratch hand-rolled BFS oracle
(`tests/y_oracle.rs`, the same methodology `tests/hex_oracle.rs` established) plus a hand-built
`Program` equality check (`style_c::tests::y_matches_a_hand_built_program`), per this doc's own
"Core IR should be constructible and checkable by hand" principle.

### Worth adding, and why

| Game | New topology/primitive it forces | Rationale |
|---|---|---|
| **Havannah** | `Hex { Hexagon }`, **`has_cycle`** (ring win condition), plus bridge/fork win conditions (connectivity to 2 vs 3+ distinct edge-or-corner classes) | The one remaining game in this list that needs a genuinely new Region-algebra primitive (`has_cycle`) rather than just a new topology instance. Also the hardest hex board to coordinate-pack — unlike `Rhombus`/`Triangle`, a hexagon doesn't tile a rectangle or triangle cleanly (see "Backend lowering" above). |
| **Margo** | `Pyramid { base: 5, levels: 5 }` (or whatever its actual base/levels are — confirm rules before implementing) | Same mechanic family as Shibumi at a different size; the point is proving `support` is a real topology-level primitive and not something Shibumi's `MASKS` happened to get away with hardcoding once. |
| **Abalone** | `Rect`, but a **push-along-a-line** move (a run of up to 3 own pieces shoves adjacent opponent pieces off the board edge) | New move-generation shape: not a single-cell placement or single-piece slide, but a line-shaped multi-cell effect with off-board elimination. Tests whether `for_each`/`shift`-composition is expressive enough for line pushes without a new combinator. |
| **Amazons** | `Rect`, large board, queen-move-then-burn-a-square (shrinking the legal region every move, on both players' behalf) | Tests whether `Region` composition handles a *monotonically shrinking* shared obstacle set cleanly, and whether move generation over a large empty region (queen moves on 10×10) is efficient in the lowered form, not just correct. See "Slider move generation" under Backend lowering above for a worked comparison of three candidate techniques (Kogge-Stone flood, Hyperbola Quintessence, magic bitboards) against this project's actual `BitBoard`/`BigBitBoard` shape. |
| **Lines of Action** | `Rect`, **dynamic-range sliding** (a piece moves exactly as many squares as there are pieces, of either color, on its current line) | The one case in this corpus where move *distance* is a runtime-computed value (`count` along a line) rather than a static direction/offset — stresses whether Core's `shift` combinator needs a variable-distance form or whether `count` + iteration already covers it. |

Recommended order if you want to build the corpus incrementally rather than
all at once: Hex and Y are done — **Havannah** next (the last, hardest hex
board, now that `Rhombus`/`Triangle` have derisked the rest of the topology),
then **Abalone or Lines of Action**
(cheapest new move-shape stress test on a topology you already have), with
**Margo** any time (cheap, mostly validates `Pyramid` generalization) and
**Amazons** last (no new primitive, but a real performance/scale check).

## Open problems (not resolved by this sketch)

- ~~History-dependent state beyond the board itself~~ — resolved in design by the Freyd-category
  split above (pure `Region` algebra vs. a premonoidal effects layer for Congo's `river_since`,
  ko rules, etc.). Not yet implemented in `core::mod`, which still only has ad hoc extra scalars —
  that's real remaining work, just no longer an open design question.
- **Unbounded auxiliary effect-state (positional superko).** Go's `(meta (no Repeat))` needs a
  `Set<Hash>` of every past state, checked by membership and grown by one entry per ply — the design
  spike (`HISTORY.md`) confirmed the Freyd-category effects layer is agnostic to the threaded state
  object's *type* (a scalar and a growing set are both just "some object"), but that's a different
  claim from "Core stays first-order/statically bounded" (this doc's "Design principles" above,
  currently justified by *board-size* bounds: `max_iters` "always a static bound derivable from board
  size"). `Region`/`Raster` board-state values stay statically bounded; auxiliary effect-state objects
  like a superko history set do not, and need a backend representation Core doesn't have yet — a real
  hash set with amortized-O(1) membership, not a fixed-width bitset the way Congo's `river_since`
  scalar array is. Undesigned past this framing; a concrete data structure and its cost per ply are
  open.
- **Scoring/payoff aggregation.** Tanbo's territory count and similar
  area-scoring rules need a `Region -> Int` reduction family beyond the
  `count`/`any`/`all` sketched above (e.g. "count of maximal empty regions
  bordered by exactly one color"). Not yet designed.
- **Graph-based boards.** A minority of Ludii games use genuinely irregular
  (non-tileable) boards that don't fit `Rect`/`Hex`/`Pyramid`/`Raster` at
  all. Deliberately out of scope until a concrete corpus game needs it —
  don't add a `Topology::Graph` speculatively.
- **GPU backend.** Referential transparency is meant to make this possible
  later, but nothing here has been validated against an actual GPU lowering
  yet. Treat as a long-term consequence of the design, not a near-term
  target.

## Non-goals

- Full lambda calculus with closures/first-class functions — meaning function *values* that can be
  stored in `state`, returned from a `def`, or passed on to a second combinator rather than applied
  immediately by the one combinator they're written for. What the corpus *does* need, and already
  has (`any`/`all`/`count_where`/`for_each`/`fold`/`bounded_fixpoint`/`project`'s `pred`/`step`
  arguments), is the narrower "second-class lambda" pattern in the "Control and aggregation"
  section above: inline, lexically-scoped, immediately-applied, and always compilable away by
  inlining at its one call site. That's not an exception carved out of this non-goal, it's the
  precise boundary the non-goal is drawing.
- A general category-theoretic *framework* (an actual `Cat`/functor abstraction in the codebase,
  Petri-net move-concurrency semantics, etc.) up front. The two specific categorical applications
  adopted above (Freyd-category effect split, trace-shaped `bounded_fixpoint`) aren't exceptions to
  this principle — they're corpus-forced: Congo already needed history state, Hex/Y/Havannah
  already need bounded connectivity fixpoints. A general framework beyond what those two problems
  need is still deferred until a further corpus game forces more of it.
