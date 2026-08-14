# Evaluation: Style C vs. Stanford GDL and Ludii

A first-principles assessment, prompted directly by the operator asking how this project's
authoring surface (`style-c/`, see `HISTORY.md`'s design-spike history and `DESIGN.md`'s Core IR
spec) is actually faring against the two established general game description languages, against
four criteria the operator named: (1) a small set of correct, composable primitives, (2)
expressiveness/concision, (3) provability guarantees useful for compilation, (4) preserving
universality the way Ludii's own completeness paper argues for. Snapshot as of `tak.md`/`DESIGN.md`
in their current state — not re-litigated as those documents evolve.

## What each system is actually optimizing for

The three aren't peers on one axis — they're answers to different questions, and that framing
matters for reading every comparison below:

- **Stanford GDL**: "What's the minimal logic that's *definitely* general enough?" Answer:
  stratified Datalog over ground facts. Universal by construction (it's unrestricted logic
  programming), at the cost of having no board/topology primitives at all — `cell(1,1,x)` is a fact
  like any other, so every engine has to *rediscover* adjacency, symmetry, and board structure from
  raw ground rules before it can run fast. Propnet compilation exists specifically to claw back the
  structure the language itself throws away.
- **Ludii**: "What's broad enough to cover every game we can find, built incrementally?" Answer:
  ~400 ludeme classes grown bottom-up from a large corpus, with `define` giving real macro reuse
  (Chess in 60 lines). Concise and battle-tested across ~1650 real games, but operationally
  specified (`then`/`apply`/`moveAgain`/`remember` are effect sequences, not relations) and shaped
  by its own Java class hierarchy — universality had to be *proven after the fact*, because nothing
  about the design process guaranteed it going in.
- **Style C**: "What's the smallest algebra that's provably compilable to bitboards, that a human
  can also read?" A genuinely different target than either — closer to a typed DSL with a
  complexity-bounded core than a general game-description logic.

## 1. Small, correct, composable primitives — clear win

Region algebra (`union`/`shift`/`flood`/`adjacent`/`connects`/`has_cycle`), Raster ops, and
`fold`/`bounded_fixpoint` are a genuinely small set, and — unlike Ludii's ludeme pile — there is
real cross-game reuse evidence: the same fixpoint shape covers Congo's chain-capture, Tak's spread,
and Havannah's cycle check (`DESIGN.md`'s corpus table). The design spike's case 5
(`05-havannah-cycle.gdl`) caught a *real bug* this way — naive transitive closure over undirected
adjacency is wrong — specifically because the primitive's threaded-state type was written down
before the logic was. GDL has no equivalent: `adjacent`/`flood` aren't primitives, they're exploded
per-game fact tables. Ludii has macro reuse but no comparable guarantee that two ludemes claiming
similar behavior actually share underlying algebraic laws.

## 2. Expressive and concise — competitive, with a real caveat

Tak's full ruleset (`style-c/games/tak.md`) is comparable in density to Ludii's macro-heavy games,
and dramatically ahead of what raw GDL would need (GDL has no `Stack`/`Raster` type — Tak-style
stacking would be brutal in ground facts). But: Ludii's conciseness claim is backed by ~1650 real
games and GDL's by decades of live competition play; this project's is backed by two fully-proven
games (Tic-Tac-Toe, Hex) through an actual working pipeline, plus hand-transcribed markdown for five
hard cases and one file (`tak.md`) no parser has ever ingested. The surface syntax is also still
visibly discovering itself — six-plus rounds of live review on `tak.md` alone (`rule`→`def`,
`ifAfterwards`→`invariant: always`/`state'`/`once`, bare `if`/`then`→`guard`, `fold`
block-header→ordinary call, angle brackets→square brackets). Healthy iteration, not a flaw — but
"how concise are we" is currently a claim about hand-transcribed examples, not measured output from
a working compiler.

## 3. Provability guarantees for compilation — strongest asset, least finished

The Freyd-category split (pure `Region` algebra as cartesian/freely-duplicable vs. effects as
premonoidal/sequenced-non-duplicable) and `bounded_fixpoint`-as-trace (`DESIGN.md`'s "Categorical
structure") are real formal devices licensing specific optimizer moves — CSE is sound on the pure
side; trace axioms (naturality, yanking, superposing) would justify fixpoint fusion/reordering.
Neither GDL nor Ludii has anything like this. GDL's propnet compiler is an engineered discovery
process, not something derived from stated laws of the source language. Ludii has essentially no
optimizing-compilation story at all — it interprets ludeme trees with caching, never compiles to
specialized per-game bitboard code.

But `DESIGN.md` says it plainly: the effects/Freyd layer is "not yet implemented in `core::mod`,
which still only has ad hoc extra scalars." The guarantee is currently a *design commitment*, not a
checked property of running code. See `COMPLETENESS.md` for the concrete next step here — grounding
this claim in descriptive complexity, and implementing enough of Core IR to check it against.

## 4. Universality — the sharpest tension, worth naming directly

GDL is universal because it's unrestricted logic programming. Ludii needed a whole paper because its
ad hoc ludeme pile had to be shown, after the fact, to cover everything. This project's own design
principles explicitly *decline* that target: "First-order, not full lambda calculus,"
`bounded_fixpoint` instead of general recursion, statically-bounded `Region`/`Raster` state, and an
Open Problems list naming exactly the two things this rules out — unbounded/dynamic topology
(`sprouts.gdl`, deferred) and truly unbounded auxiliary state (superko's `Set<Hash>`, sketched only at
the backend level). That's a real, considered trade — buying provability/compilability by *refusing*
general recursion and dynamic structure — but it means "preserve universality" the way GDL or Ludii
mean it isn't actually the goal, and claiming it as a fourth win alongside the other three would be
soft under scrutiny.

The better move, and the one this project is now acting on: don't chase GDL/Ludii-style unrestricted
universality at all. Ground the claim in **descriptive complexity** instead — `bounded_fixpoint` over
first-order region algebra is structurally close to fixpoint logic, and Immerman–Vardi says fixpoint
logic over *ordered* finite structures captures exactly PTIME. If Core IR's fragment can be shown to
correspond to that fragment, the resulting claim — "universal for every finite-state game decision
procedure computable in polynomial time in board size" — is sharper and more useful to a compiler
than either predecessor's: stronger than Ludii's empirical "we checked it against our corpus," and
more informative than GDL's unrestricted-but-uncompilable universality, which says nothing about
worst-case cost. See `COMPLETENESS.md` for the working conjecture.

## Concrete weaknesses to track

- **Effects layer unimplemented.** The provability story's premonoidal half is design-only.
- **No real parser/lowering beyond two games.** Everything past Tic-Tac-Toe/Hex is hand-transcribed
  markdown, not machine-checked. (Deliberately not being fixed next — see `HISTORY.md`'s current
  charter; the language is judged too unstable to freeze into a parser yet.)
- **Open items sitting exactly where universality would break**: `Raster` cell `Value`'s shape
  (tuple vs. record — undecided), determinism tagging for `extern` chance calls (undecided — GDL-II
  already has a peer-reviewed formal treatment of randomness/imperfect information via `sees`/
  `random` roles worth borrowing *semantics*, not notation, from, per this project's own "borrow
  semantics freely" rule), and dynamic-topology/mid-game-joining games (structurally deferred, no
  forcing example yet).
- **Corpus breadth is the biggest empirical gap vs. both competitors** — not a design flaw, just the
  honest current state (2 proven games vs. Ludii's ~1650 and GDL's competition history).

## Recommendation this session acted on

Redirect effort from further surface-syntax review toward (1) formalizing the completeness
conjecture above and (2) extending `core::mod` itself — making `flood`/`connects`/`has_cycle` real
composable Region-algebra primitives instead of dedicated `EndRule` variants, which `DESIGN.md`'s own
"promote once a second special case appears" principle already flags as due — deferring the Style C
parser until the language is judged stable enough to be worth freezing into a grammar. See
`HISTORY.md`'s current "Next session charter."
