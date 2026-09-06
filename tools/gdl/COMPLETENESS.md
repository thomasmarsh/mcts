# Completeness via descriptive complexity: the Immerman–Vardi conjecture

**Status: conjecture, under construction -- upper bound now has real code behind two of its three
hardest rows.** `core::mod`/`core::interp` grew real, composable `flood`/`adjacent`/`shift`
Region-algebra combinators and a generic `bounded_fixpoint` trace function (Tic-Tac-Toe's line-win
and Hex's edge-to-edge connectivity-win are now two `BoolExpr` values built from them, not two
dedicated Rust variants -- see `DESIGN.md`'s "Design principles" corollary and `HISTORY.md`'s
session note). See the primitive-by-primitive table below for what that run confirmed for `flood`/
`connects`, and what it could only confirm at the *shape* level (not landed) for `has_cycle`.

This document states a completeness/universality claim
for Core IR precisely enough to be provable or refutable, and lays out what proving it actually
requires. It replaces "preserve universality" (`EVALUATION.md`'s criterion 4) as this project's
universality target, for reasons that document explains: neither Stanford GDL's unrestricted
universality nor Ludii's after-the-fact corpus-coverage proof is the right thing to imitate, because
neither gives a compiler anything to work with. A tight complexity-class characterization does.

## The claim, informally

**Core IR's expressible fragment — Region algebra + `bounded_fixpoint` + `fold`/bounded aggregation +
the temporal layer (`state'`/`always`/`once`) — is exactly first-order logic with a least-fixpoint
operator (FO(LFP)) over ordered finite relational structures, which by the Immerman–Vardi theorem
is exactly PTIME.**

Two independent-but-related things follow if this holds:

1. **Upper bound (soundness):** every Core IR program computes a function decidable in time
   polynomial in board size. A real backend guarantee, not "seems fast on the corpus so far" — every
   generated program has a *provable* polynomial worst case.
2. **Lower bound (completeness):** every polynomial-time-decidable predicate over the board
   structure is expressible using Core's primitive set. Nothing has been left out that a real game's
   rule engine could actually need — "universal," but for a precisely bounded and useful class, not
   universal in GDL's unrestricted sense.

The upper bound is the cheap direction and should be established first, primitive by primitive. The
lower bound is genuinely open research work — see "What's needed" below.

## Formal setup: games as ordered finite relational structures

The Immerman–Vardi theorem needs the structures it ranges over to carry a built-in linear order
(without one, LFP is strictly weaker than PTIME — the classic counterexample is that plain, unordered
LFP cannot express "the structure has even cardinality"). This is usually the awkward part of
applying the theorem to a new domain. It isn't here, because every backend representation in this
repo already imposes one:

- **`Site`**: `BitBoard`'s row-major indexing already fixes a canonical total order on cells. No new
  assumption — this order already exists and is load-bearing in every existing `games/*` crate.
- **`Player`**: already an ordered, small finite domain (`P0`, `P1`, ...) — turn order is itself a
  linear order over `Player`.
- **`Ply`/time** (for the temporal layer, `state'`/`always`/`once`): ordered by move sequence,
  inherently — a game trace is a finite ordered structure by construction, not something that needs
  an order bolted on.

So the vocabulary is: a universe of `Site` (ordered), a small finite `Player` domain (ordered), a
family of binary relations `Adjacent_d(s, s')` for each direction `d` in the topology, a relation or
function `Occupied(s, p)` / cell-`Value` function for `Raster` topologies, and — for the temporal
layer — a second sort `Ply` (ordered) with a relation tying board state to a given ply. With order
available, successor, addition, and multiplication are all LFP-definable from it (a standard
descriptive-complexity construction) — so nothing about `count`/`sum`/`min`/`max` needs to be
smuggled in as a separate primitive outside the FO(LFP) fragment; they fall out of order + LFP for
free, which is a useful sanity check that Core's existing "ordinary scalar/collection ops" (`DESIGN.md`'s
Control and aggregation table) aren't secretly reaching outside the target fragment.

## Primitive-by-primitive mapping

| Core IR primitive | FO(LFP) classification | Why |
|---|---|---|
| `union`, `intersect`, `complement`, `difference`, `is_empty`, `member` | **FO** | Quantifier-free/plain Boolean combinations over the `Site` sort. |
| `count` | **FO(order)**, i.e. still within FO given the built-in order | Counting up to a linear bound is expressible with order; general sum/count over relations is the standard order-definable-arithmetic construction. |
| `shift(dir)`, `adjacent(conn)` | **FO** | A single relational join against `Adjacent_d`. **Implemented, not just classified**: `Region::Shift`/`Region::Adjacent` in `core::mod`, evaluated by `core::interp::{shift,adjacent}` as direct, unmodified calls onto the existing `BitBoard::shift_*`/one-expression-per-direction-fold pattern -- see `core::interp::adjacent`'s doc comment for why folding over a direction list (rather than a hand-written `\|=` sequence) structurally can't reintroduce the compounding-shift bug `flood6` once had. |
| `flood(seed, conn)` | **LFP, not FO** | The canonical textbook example motivating LFP: reachability is not first-order expressible over general graphs (no fixed quantifier-depth formula computes transitive closure uniformly in graph size), but is exactly the smallest fixpoint of "reachable in one more step." **Implemented, not just classified**: `Region::Flood`, evaluated by `core::interp::flood` as `core::interp::bounded_fixpoint`'s `Aux = ()` instantiation with `max_iters` set to the flooded region's own cell count -- literally the "monotone operator over an n-element domain stabilizes within n iterations" fact this table's `bounded_fixpoint` row cites, not a separately-chosen bound. Replaces what was a direct, non-composable `BitBoard::flood6` call in Hex's end rule; `region_flood_matches_direct_flood6_call` in `core::interp`'s tests pins the two down as identical. |
| `connects(edge_a, edge_b)` | **LFP + FO** | ∃ a flooded cell touching both edge sets — an FO wrapper around one `flood`. **Implemented, not just classified**: `BoolExpr::Connects`, evaluated as one `flood` call plus `BitBoard::intersects` -- Hex's `(is Connected Mover)` end rule lowers to this (`core::lower::lower_end`), no longer a dedicated `EndRule::Connected` Rust variant. One real gap this pass surfaced: `edge_a`/`edge_b` themselves aren't embedded in the `BoolExpr` term the way `connects`'s signature above suggests -- `Program.end` is shared across every player while `Program.player_regions` varies per player, so the interpreter still looks the pair up per mover at eval time (see `BoolExpr::Connects`'s doc comment in `core::mod`). That's a real, unresolved seam between "Region algebra is a cartesian category with genuine literal operands" and "some operands are actually `mover`-relative runtime lookups" -- the same seam `HISTORY.md`'s Style C surface resolves with a `regions(mover)` expression-level accessor, which this interpreter-only pass deliberately didn't build (no parser this session). |
| `has_cycle` | **LFP, simultaneous** | Needs a richer threaded state — `(visited: Region, parent: Raster<Direction>)`, per the design spike's finding in `HISTORY.md` (a naive bare-`Region` version is not just weaker, it's *wrong*, since every edge of an undirected adjacency relation trivially "reaches" itself backward). This is *simultaneous* least-fixpoint induction — several relations (`visited`, `parent`, a `cycle` flag) bootstrapped together — a standard, expressiveness-preserving extension of plain single-relation LFP, not something that needs more than FO(LFP) can give. **Shape confirmed, not landed**: `core::interp::bounded_fixpoint` is generic over an auxiliary threaded state `Aux` specifically so `flood`'s `Aux = ()` case and `has_cycle`'s richer case are the same node shape, not two. `has_cycle_shape_holds_a_parent_and_cycle_flag` (`core::interp`'s tests) instantiates `Aux = (HashMap<Site, Direction>, bool)` -- a real parent map plus a cycle flag, threaded together through the same fixpoint call -- against a hand-verifiable 4-cycle/path pair, confirming the *shape* holds by actually compiling and running it, not by prose assertion alone. `has_cycle` itself is still not a `Region`/`BoolExpr`/`Program` primitive (no `EndRule` lowers to it, no `Raster<Direction>` value type exists) -- landing it is unchanged future work, per `HISTORY.md`'s session note. |
| `fold(seed, iterable, step)` over a statically-bounded-length sequence | **FO(order)** — no LFP needed | Bounded iteration over a length known before the walk starts is exactly what plain first-order logic (with order/counting) already captures — see "A finding this conjecture retroactively explains," below. |
| `bounded_fixpoint(seed, step, max_iters)` | **LFP, directly** | This *is* the LFP operator: `step` is a monotone operator on relations, and the LFP theorem guarantees a monotone operator over subsets of an `n`-element domain stabilizes within `n` iterations — which is exactly `DESIGN.md`'s own justification for `max_iters` ("always a static bound derivable from board size"), not an independently-chosen safety margin. |
| `state'` (primed next-state reference) | **FO** | An ordinary field read one step ahead on the `Ply` sort, given `Ply`'s order. |
| `once(P)` | **FO(order) over `Ply`** | Past-eventually — a bounded existential over `Ply ≤ current` — is first-order given `Ply`'s built-in order. Consistent with Kamp's theorem (LTL over finite/ordered traces corresponds to FO over the trace's order), which is independent supporting precedent for treating the temporal layer as "just FO over a second ordered sort," not a separate modal system needing its own semantics. |
| `invariant: always P` | **Needs its own treatment — flagged, not yet classified** | Unlike `once`, this is not obviously a *query evaluated at an instant* — as used, it's a legality-generation-time restriction (intersected into every move's legality, `HISTORY.md`'s Alloy-refinement session note), not a temporal formula checked against a trace. Whether it's "just" FO-universal-over-future-`Ply` or something that needs separate handling because it constrains move generation rather than answering a yes/no question is open; don't assume it by analogy to `once` without checking. |

### A finding this conjecture retroactively explains

Round 2 of the live syntax review (`HISTORY.md`) split `fixpoint` into two constructs — `fixpoint`
(genuine convergence, unknown iteration count) and `fold` (deterministic, statically-known-length
walk) — purely because the merged version "read badly." Nobody involved was thinking about
descriptive complexity at the time. It turns out that split is *exactly* the FO/LFP expressiveness
boundary: `fold`'s bounded, length-known-in-advance iteration is squarely FO(order); `fixpoint`'s
unknown-iteration-count convergence is squarely LFP and not FO. A readability complaint independently
rediscovered a real complexity-theoretic distinction. This is strong circumstantial evidence the
design is already well-aligned with the target formalism, and it's worth treating this kind of
"doesn't read well" signal (already flagged twice in `HISTORY.md`'s own session notes, for `fixpoint`/
`fold` and separately for `guard`) as a plausible symptom of a real semantic seam, not just
unfamiliarity — a pattern worth remembering when the next primitive feels wrong to write.

## Crucial disambiguation: rule complexity vs. game complexity

This conjecture is about the complexity of evaluating **one query** — is this move legal? is the game
over? what does `next` compute? — as a function of board size. Call this **rule complexity**. It says
nothing about, and is completely orthogonal to, **game complexity**: the complexity of determining
the game-theoretic *value* (who wins under optimal play) as board size grows as a parameter. Well-known
results in combinatorial game theory show generalized versions of real games are typically far above
PTIME under that measure — generalized Hex is PSPACE-complete, generalized Go and generalized Chess
are EXPTIME-complete under standard formalizations. A Core IR program for Hex having PTIME rule
complexity says nothing about Hex's PSPACE-complete game complexity, and doesn't need to — an MCTS
engine needs the *rule engine* (legal/next/terminal) to be fast per call regardless of how hard the
underlying game is to solve; that's the only claim this conjecture makes. Conflating the two would be
a real error in any write-up of this result, so it has to stay explicit every time the claim is
restated, not just here.

## Relationship to GDL and Ludii, sharpened

- **GDL.** Stratified Datalog evaluated inflationarily is, as a matter of established database
  theory, already within PTIME — this is well-trodden ground in finite model theory, not a new
  insight. GDL's designers picked Datalog for decidability/tractability reasons, not because they set
  out to hit exactly PTIME as a design target. This project's potential contribution isn't
  "discovering fixpoint logic is a good idea for games" — it's that a compilable, human-authorable
  board-game DSL maps onto FO(LFP) *by construction*, primitive by primitive, checkable up front,
  rather than needing after-the-fact verification against an existing pile of constructs the way a
  Datalog-shaped GDL engine's practical tractability is usually argued.
- **Ludii.** Has no comparable characterization. Its own universality argument is presumably closer
  to a Turing-completeness/simulation argument over the ludeme set — "at least as expressive as
  anything computable" — which is a fundamentally different, much weaker kind of guarantee than a
  tight complexity-class characterization: Turing-completeness says nothing about worst-case cost,
  which is exactly what a compiler needs to know and exactly what this conjecture, if proven, would
  give for free.

## What's needed to move from conjecture to theorem

1. **Formal vocabulary definition.** A page of model theory: `Site`/`Player`/`Ply` sorts, the
   relations each `Topology` variant supplies, and an explicit statement of the order each sort
   carries. Mechanical, should be done first.
2. **Upper-bound proof.** Primitive-by-primitive FO(LFP)-expressibility, per the table above —
   the "cheap" direction, and the one worth finishing even if the lower bound stalls, since it's a
   real, checkable backend guarantee on its own (every Core program has provably polynomial cost).
3. **Lower-bound argument, or an honest scoping-down.** Either (a) show Core's primitives can
   simulate the standard FO(LFP) normal form (a formula built from atomic relations, Boolean
   connectives, and a single outermost LFP application — closure-under-composition results in the
   descriptive-complexity literature already give this normal form; the work is showing Core's
   primitive set can express an *arbitrary* FO step function for `bounded_fixpoint`, which mostly
   reduces to whether Region algebra + Raster ops are FO-complete over the vocabulary above), or
   (b) find the actual gap (the `invariant: always` case above is the leading suspect) and state a
   narrower, still-useful completeness claim rather than force a false one through.
4. **Decide the rigor target.** Whether this is meant to become a citable, submittable proof, or is
   primarily an internal design-confidence exercise that stays informal, changes how much of (2)/(3)
   is worth fully discharging before moving on to corpus/implementation work.

## Where this connects to concrete Core IR work

The upper-bound direction (step 2) is best checked against real code, not just written down in the
abstract — which is also exactly the "promote to a composable primitive" work `DESIGN.md`'s corpus
table already flagged as due, and which has now happened: `EndRule::Line`/`EndRule::Connected` are
unified into real, composable `flood`/`connects`/`adjacent`/`shift` Region-algebra combinators,
backed by a real generic `bounded_fixpoint` trace function in `core::interp` (`Region::Flood` is
its `core::mod` IR node). This was simultaneously (a) the concrete Rust work `DESIGN.md`'s own
"promote once a second special case appears" principle already called for, and (b) the first real
implementation this conjecture's LFP classification of `flood`/`connects`/`has_cycle` needed to be
checked against — see the primitive-by-primitive table above for what survived contact with actual
code unchanged (`flood`/`connects`'s LFP/FO shape), what surfaced a real, still-open gap
(`connects`'s `edge_a`/`edge_b` operands are mover-relative runtime lookups, not literal `BoolExpr`
operands, until a real authoring-surface parser exists to bind `mover` as an expression-level
value), and what could only be checked at the shape level rather than landed (`has_cycle`'s
simultaneous `(visited, parent, cycle)` induction, confirmed via a generic, non-`Program`-wired
`bounded_fixpoint` instantiation). None of this touches the lower-bound direction (step 3), which
is still fully open.
