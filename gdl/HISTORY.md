# Development history

The full session-by-session log this project's `README.md` used to be. Kept verbatim as an archive
of the reasoning behind decisions that are now just stated as fact in `README.md`/`DESIGN.md` (why
Relational GDL was retracted, why Style C's syntax went through six-plus rounds of review, why
`Hex { Triangle }` turned out not to need a new backend, etc.) — useful when the *why* behind a
current design point is in question, not needed to understand the project's current state. Start at
`README.md` for that; come here only when a "see the session note" pointer sends you here, or when
digging into the reasoning behind something `README.md`'s "Prior design directions" section
summarizes.

Entries are in chronological order (oldest first, as they were written); nothing below has been
edited for accuracy against the current codebase — see `README.md` and `DESIGN.md` for that.

---

# Summary

## Last Session

Grew the Core IR corpus past Tic-Tac-Toe by proving Hex end to end, per the previous session's
charter. Hex forced a real second topology (`Hex { Rhombus }`) and a new region-algebra
primitive (six-way flood/connectivity) rather than just reusing `Rect` + line-win.

- `lud/Hex.lud`: the user supplied the real Ludii source (full option-templated board size, swap
  rule, misère variant); concretized to a fixed 3x3 board, no swap rule, standard win — the same
  treatment `Tic-Tac-Toe.lud`'s macro call got last session, since option/template resolution is
  out of scope for elaboration.
- `game_core::bitboard::BitBoard`: added `flood6` (six-way flood fill, seeded from an entire
  region rather than a single index, since a Hex connectivity check seeds from a whole board
  edge). Building it surfaced a real latent bug pattern: splitting a flood's directional shifts
  across multiple `|=` statements (as `flood8` already does) lets one direction's shift compound
  off a *previous statement's* unmasked result within the same loop iteration, bridging through a
  cell not actually in the flooded region. That's invisible for `flood8` (any such bridge lands
  on a cell that's already a legitimate 8-way neighbor anyway) but a real correctness bug for
  `flood6`, which deliberately excludes one diagonal. Fixed by combining all six shifts into a
  single statement, matching `flood4`'s existing (correct) pattern; documented the invariant in
  both functions' doc comments. `flood8` itself was left alone — the bug is latent/harmless there,
  and fixing unrelated, unrequested pre-existing code was out of scope this session.
- `core::hex::Hex`: a `side x side` rhombus topology. Axial coordinates pack into the same
  row-major `BitBoard<N, N>` indexing `Rect` already uses, with adjacency restricted to six of
  `Rect`'s eight queen-move directions (N/S/E/W plus the northeast/southwest diagonal) — so Hex
  needed *no* new bit layout or backend, just a restricted direction set. `Hex::edge_for_compass`
  maps `.lud`'s diagonal compass edge names (NE/SE/SW/NW) onto the rhombus's four straight edges
  via a documented, self-consistent convention (not a claim of matching real Ludii's rendered
  geometry, since there's no `games/hex` crate to check against).
- `core::mod`: `Program.topology` is now a real `Topology` enum (`Rect`/`Hex`) instead of a
  hardcoded `Rect` field; `Region` grew a `Sites(Vec<usize>)` static-list leaf; `EndRule` is now
  an enum (`Line`/`Connected`) instead of a single struct; `Program` grew `player_regions` for
  Hex's per-player named board-edge pairs. `connects`/`flood` are *not* yet generic Region-algebra
  combinators — `EndRule::Connected` stays a dedicated, non-composable variant, matching how
  `EndRule::Line` already worked, since nothing yet needs more than one `(Region, Region)` pair
  per player.
- `elaborate/`: extended `graph.rs` (`(hex Diamond <dim>)`), `equipment.rs` (`(regions <roleType>
  {...})`), `region.rs` (`(sites Side <compassDirection>)`), `boolean.rs` (`(is Connected
  <roleType>)`) — each restricted to exactly the shape Hex's `.lud` uses, same scoping convention
  as every other elaborate module.
- `tests/hex_oracle.rs`: a from-scratch hand-rolled connectivity oracle (plain BFS over an
  explicit six-neighbor delta list, recomputed independently rather than calling into
  `core::hex`/`flood6` at all) replaying fixed move sequences against the interpreter — including
  a case that specifically distinguishes the two square diagonals (one hex-connected, one not)
  and a full-board fill where the winning connection is only completed on the final move.

96 tests pass (85 lib + 5 ttt-oracle + 6 hex-oracle in `ludii`, 56 in `game-core`, all up from 75
in `ludii`/53 in `game-core`): `cargo test -p ludii -p game-core`, `cargo clippy --workspace
--exclude mcts-bench --exclude game-host --all-targets`, and `cargo fmt -p ludii -p game-core --
--check` are all clean. Full-workspace `cargo test --lib --workspace --exclude mcts-bench
--exclude game-host` also passes (the two exclusions are the same pre-existing, unrelated
environment failures noted last session — a missing system `duckdb` library and a game-host
subprocess test needing a pre-built release binary — neither touched by this session).

## Suggested commit message:

Prove Hex end to end: a second Core IR topology and six-way connectivity

Concretize the user-supplied lud/Hex.lud to a fixed 3x3/no-swap/standard-win
instance, add flood6 to game_core::bitboard (fixing a latent
directional-shift-compounding bug along the way, harmless for flood8 but real
for six-way adjacency), generalize core::{Topology,Region,EndRule,Program} to
carry a second topology and an edge-to-edge connectivity end rule, extend
elaborate/ just far enough to cover Hex's equipment/region/boolean shapes, and
check in a from-scratch hand-rolled BFS oracle test independent of the
interpreter's own flood6 path.

## Session note: `database-1/` added, `lud/` fixtures recovered, `define` gap scoped

This session was design discussion, not corpus work, but it changed real repo state:

- `database-1/` (Ludii's full games database — `database-1/lud/games/` has all ~1650 real `.lud`
  files, plus `ludiiGames.sql`, a dump of Ludii's own concept/ruleset/game metadata) was added,
  replacing the previous `lud/` directory outright. `lud/` was never git-tracked, and the swap
  deleted the hand-concretized fixtures `tests/{oracle,hex_oracle}.rs`, `src/core/{interp,lower}.rs`,
  `src/elaborate/game.rs`, and `src/parse/sexpr.rs` load via `include_str!` — `cargo test -p ludii`
  stopped compiling. Recovered `Tic-Tac-Toe.lud`, `Hex.lud`, `Breakthrough.lud`, `Havannah.lud`,
  `Amazons.lud` verbatim (their content was read earlier in this same session, before the swap);
  `Minishogi.lud`, `Spargo.lud`, `Lines of Action.lud`, `Y.lud` (never read verbatim this session)
  were re-copied from `database-1/lud/games/` instead of reconstructed from memory. All 96 tests
  pass again. `DESIGN.md` now documents the `lud/` (hand-concretized, load-bearing fixtures) vs.
  `database-1/lud/games/` (full raw corpus, source of record) split explicitly — don't let a future
  "add more games" pass overwrite `lud/` wholesale again.
- Checked whether `database-1/` closes the `def/`-library gap (Ludii's `(define ...)` macro
  bodies — see `DESIGN.md`'s new "Macro expansion" section): it doesn't. Zero `.def` files, and the
  SQL dump's `Ludemes`/`DefineLudemeplexes` tables are a concept-taxonomy/analytics catalog (e.g.
  `ReachWin` appears only as the one-line description "Win in reaching a region"), not expandable
  source. The real `.def` bodies still need to come from Ludii's own source distribution.

## Session note: the compilation-target architecture pivoted, twice

This session was entirely design discussion (no corpus/Core Rust work), and it substantially
changed `DESIGN.md`'s direction from where the last few sessions left it:

- **First pivot**: `.lud`'s ludeme layer is operationally specified (`then`/`apply`/`moveAgain`/
  `remember` describe effect sequences, not relations), so mechanically parsing it can only ever
  special-case one operational idiom at a time and was never going to converge on a small
  combinator set. New direction: a small **relational GDL** (Datalog-shaped, `Region`-typed
  intensional primitives instead of Stanford GDL's exploded ground-fact tables) becomes the
  primary authoring language; `.lud`/`database-1` become spec-and-oracle, translated by a human
  or LLM and checked against an existing `games/*` crate or a hand-rolled reference, not compiled.
  The existing `ast`/`parse`/`elaborate`/Core pipeline still compiles and still proves
  `Tic-Tac-Toe`/`Hex` (96 tests pass) but is no longer where new corpus games should grow.
- **Second pivot**: flat Datalog regresses toward Stanford GDL's own well-known failure mode —
  verbose, no reusable/parametrized rule templates, which is exactly what Ludii's `define`-driven
  ludeme reuse avoids (real evidence this session: `Chess.lud` compiles to 60 lines specifically
  because of piece/end-condition macro templates like `("ChessPawn" "Pawn" ...)`/`("Checkmate"
  "King")`). Corrected direction: Horn-clause rule syntax is sugar over a **categorical core**
  (regular-category correspondence — conjunction = pullback, shared variables = fan-out+join),
  not a separate thing from it, plus a compile-time template/macro layer (specializes away before
  Core, no runtime closures, consistent with the existing "not full lambda calculus" principle)
  that's the actual fix for verbosity. A Freyd-category split (pure `Region` algebra vs. a
  premonoidal effects layer) resolved the previously-open "history-dependent state" problem, and
  `bounded_fixpoint` was reframed as a bounded trace, unifying chain-capture/flood/`has_cycle`
  under one construction instead of three.
- **Corpus survey** (`Chess.lud`, `Go.lud`, `Shakhmaty (Early Modern).lud`) stress-tested that
  synthesis against real, canonical, unavoidable games and found it incomplete in two concrete
  ways — see the charter below — plus a corpus-selection caution: some `.lud` files (the Shakhmaty
  one) bundle several complete, unrelated rulesets under one filename via the option mechanism
  (`(option "Variant" ... (item "..." <11000-line game body> ...))`), so "options are simple to
  resolve by hand" doesn't generalize to every file.

`DESIGN.md`'s "Relational GDL" and "Categorical structure" sections are now marked **provisional**
— the core architectural claim ("logic is sugar over the categorical core, and a template layer
fixes verbosity without reintroducing Ludii's operational semantics") hasn't been checked against
a real pathological case yet. That check is the next session.

## Session note: design spike — Horn-clause vs. categorical vs. a third, from-scratch style

This session was the design spike the previous charter called for: no Rust changes, no
parser/grammar, no further corpus survey — five pathological cases from last session's `Chess.lud`/
`Go.lud`/`Havannah.lud` survey, hand-written in candidate surface syntaxes and checked for
convergence. Mid-session the scope grew by one axis on request: rather than testing only "Datalog
sugar vs. categorical core," also design a third style from first principles — ignore both Datalog
and Ludii precedent, ask what a human would actually want to write, subject to the same compilation
constraints — since Stanford GDL (state-as-exploded-fluent-table) and Ludii (concise but
operationally specified, hostile to hand-authoring) are each disqualified for a different reason and
neither Horn clauses nor point-free notation is obviously the missing third option.

All three styles below share Core's existing value types (`Region`/`Raster`/`Site`/`Direction`/
`Player`/`Bool`/`Int`) and the Freyd-category pure/effect split — they differ only in surface
notation, not in what they're allowed to express.

**Style A — Datalog/Horn-clause** (`legal(X) :- body.`, unification variables, conjunction=join).

**Style B — categorical/point-free** (explicit `∘`/`;`/`⊗`, `Tr` for trace/fixpoint, no unification
variables, no named intermediate values).

**Style C — typed functional/equational**, the new one: named `let`-bound values instead of
unification, ordinary function definitions (`rule name(args): Type = expr`) instead of Horn-clause
bodies, an explicit `then { ... }` block for the effect layer (typed, not Ludii's unrestricted
operational sequencing — an effect block can only call the fixed `set`/`remove`/`insert`/`push`/pop`
primitives Core already lists), and templates written as ordinary compile-time generics (monomorphized
per call site, ANY reader who's used Rust generics or C++ templates already has the right mental
model — no category-theory exponential-object subtlety, no macro-hygiene notation to invent).

### Case 1+2: Chess's check-safety filter / Go's suicide-rule filter

Both are `ifAfterwards:P` — legality gated on a predicate over the *hypothetical* state one move
ahead, discarded if the move isn't actually played. Two canonical games, same shape (`(not (IsInCheck
"King" Mover))` in `Chess.lud:166`, `("HasFreedom" Orthogonal)` in `Go.lud:35`), so per this doc's own
"unify once two data points share a shape" principle, one construct must cover both.

- **A.** Plain Horn-clause conjunction has no way to name "the state after a not-yet-committed move"
  — Datalog has no function application. The only way to express it without inventing a keyword is to
  make `State` an explicit argument threaded through *every* relation (`legal(S, M) :- candidate(S,
  M), next(S, M, S'), not in_check(S', king, mover(S)).`), i.e. reify `next` as an intensional
  builtin relation (consistent with treating `adjacent`/`flood` as compiler-known primitives rather
  than exploded tables) but pay for it with situation-calculus-style state-threading through every
  other rule in the program — exactly the GDL cost this doc's own "Relational GDL" section was
  written to avoid. The escape hatch is a **non-Horn modifier bolted onto the clause**:
  `legal(M) :- candidate(M) ifAfterwards: safe(next(state, M)).` — sugar for a specific wiring, not
  sugar for ordinary conjunction.
- **B.** Native and unforced: `next : State ⊗ Move -> State` is already a pure morphism per this doc's
  "Categorical structure" section; `ifAfterwards(P) = P ∘ next`. Legality is `guard(safe ∘ next) ;
  candidate` where `guard : (A -> Bool) -> A -> A` is the standard "partial identity"/assert
  combinator from restriction-category or Markov-category semantics (the categorical shape of
  `observe` in probabilistic programming). No new keyword — it's fan-out (keep the candidate move,
  also compute `next` from it), guard on the second output, project the first back out.
- **C.** Also native, and arguably the most legible of the three because it needs *no new concept at
  all*, just `let`:
  ```
  rule legal(m: Move): Bool =
      is_candidate(m) && safe(let s' = next(state, m) in s')

  rule safe(s: State): Bool = !is_in_check(s, King, mover(state))
  ```
  "The state after this move" is just a value with a name, the same way any imperative or functional
  programmer already reads "compute the hypothetical, then test it." Nothing about *evaluating* a
  hypothetical next state ever needed Horn-clause unification or a `Tr`/wiring-diagram construct in
  the first place — those two styles were solving a self-imposed notational problem.

**Verdict:** all three converge on the same underlying Core term (`guard`/`let`-tested composition of
`next` with a pure predicate). B and C reach it directly from their own primitives; A needs a bespoke
non-Horn keyword grafted onto naked conjunction, which is itself evidence that "Horn-clause body
conjunction" was never actually the whole of Datalog's proposed sugar layer — `ifAfterwards:` was
always going to be a distinguished modifier, not derived from `:-`.

### Case 3: Go's positional superko, `(meta (no Repeat))`

Legality gated on membership in an *unbounded, growing* set of past states (Zobrist-hash the state,
check + insert on every ply) — not a small threaded scalar the way Congo's `river_since` is.

- **A.** `legal(M) :- candidate(M), next(state, M, S'), not seen(hash(S')).` — reuses the same
  `ifAfterwards`-shaped modifier from case 1/2, with `seen` reading from accumulated effect state
  instead of the board. No new Horn-clause machinery beyond what case 1/2 already needed.
- **B.** `guard(not ∘ member(history) ∘ hash) ∘ next`, same shape as case 1/2's `guard ∘ next`, with
  `history` read from the Freyd-category effects object threaded alongside `state`; committing a move
  appends an `insert(hash(state'))` effect afterward — sequenced, non-duplicable, which is exactly why
  it belongs in the premonoidal layer and not pure `Region` algebra.
- **C.** `state history: Set<Hash> = {}` declared alongside `state board: Region`, referenced as an
  ordinary typed value: `rule legal(m: Move): Bool = is_candidate(m) && !history.contains(hash(next(state, m)))`,
  with `then { history.insert(hash(state)) }` on the move's effect tail.

**Verdict:** confirms the case 1/2 construct generalizes (all three styles reuse it unmodified) — but
surfaces a real correction to this doc's own principle, not just a syntax note: **the categorical
"auxiliary state lives in the Freyd effects layer" story is agnostic to the state object's size, but
Core's "first-order, statically bounded" design principle is currently justified by *board-size*
bounds (`max_iters` "always a static bound derivable from board size") and does not automatically
extend to a growing `Set<Hash>`.** These are two different claims that happened to look like one.
`Region`/`Raster` board-state values stay statically bounded; auxiliary effect-state objects (history
logs, superko sets) do not, and need a backend representation Core doesn't have yet (a real hash set
with amortized membership, not a fixed-width bitset the way Congo's `river_since` scalar array is).
Flagged in `DESIGN.md`'s Open Problems below — this is new, undesigned backend work, not something
the categorical framing already solved by being agnostic to object type.

### Case 4: Chess's `ChessPawn` piece-template macro composition

`("ChessPawn" "Pawn" (or "InitialPawnMove" "EnPassant") (then (and ("ReplayInMovingOn" (sites Mover
"Promotion")) (set Counter))))` (`Chess.lud:126-137`) — a template invoked once per piece kind, taking
extra move alternatives and an effectful tail as parameters. (Caveat: `ChessPawn` itself is a known
Ludii `define` whose body isn't sourced in this repo — see `DESIGN.md`'s "Translating `.lud`" section
— so what's tested here is the *shape* of template parametrization the call site forces, not a claim
about `ChessPawn`'s exact undocumented body.)

- **A.** A rule *template*, expanded at compile time via the same positional `#1`/`#2` textual
  substitution real Ludii `define`s already use (`DESIGN.md:82-94`) — parameters can themselves be
  rule-body fragments, but substitution is syntactic (AST splicing before Core ever sees it), not a
  higher-order relation:
  ```
  template ChessPawn(piece, extra_moves, tail) {
    move(piece, M) :- step_forward_to_empty(piece, M).
    move(piece, M) :- diagonal_capture(piece, M).
    move(piece, M) :- extra_moves(M).
    after_move(piece, S, S') :- tail(S, S').
  }
  ChessPawn("Pawn", or(InitialPawnMove, EnPassant), and(ReplayInMovingOn(sites(Mover, "Promotion")), SetCounter))
  ```
- **B.** Point-free notation *wants* this to be genuinely higher-order — `ChessPawn : (Move ->
  Effect) -> (State -> Effect) -> Piece -> Morphism` looks like an ordinary function taking morphisms
  as arguments, which needs the ambient category to be Cartesian *closed* (an internal-hom/exponential
  object) — precisely the thing this doc's Non-goals section rules out ("full lambda calculus with
  closures... nothing in the corpus has needed it"). The honest resolution is the same as A's: `≜`
  (meta-level, textual, expanded before compilation) is not `=` (semantic equality of two already-built
  Core morphisms) — `ChessPawn(extra, tail) ≜ (step_forward ∪ diagonal_capture ∪ extra) ; tail` — but
  that discipline has to be stated explicitly, because nothing about point-free notation itself signals
  "this operator is a macro, not a category-internal combinator."
- **C.** An ordinary compile-time generic, monomorphized per call site — the mental model every reader
  who's used Rust generics or C++ templates already owns, with zero new notation invented:
  ```
  template rule chess_pawn<Extra: fn() -> MoveSet, Tail: fn(State) -> State>(piece: Piece): MoveSet =
      step_forward_to_empty(piece) | diagonal_capture(piece) | Extra()
      then Tail
  
  chess_pawn::<InitialPawnMoveOrEnPassant, ReplayThenSetCounter>(Pawn)
  ```

**Verdict:** all three specialize away to the same Core term, confirming the template layer doesn't
leak *as a compilation matter* — but A and B both require inventing and explaining a notational
distinction (`#1`-substitution-is-not-application; `≜`-is-not-`=`) that C gets for free by borrowing a
concept (monomorphized generics) virtually every target reader already has internalized from a
mainstream language. Point-free (B) is the *most* exposed of the three here: without the `≜`/`=`
discipline stated up front, it actively implies Core has exponentials it doesn't.

### Case 5: Havannah's `has_cycle` (`(is Loop)` in `Havannah.lud:13`)

A stone group contains a cycle (encircles ≥1 cell) — the concrete test of "bounded trace unifies
chain-capture/flood/`has_cycle`" (`DESIGN.md`'s Categorical structure section).

- **First attempt, A (and it's wrong — a real finding, not a style preference):** the tempting Datalog
  rendering reuses plain transitive closure, the same template as flood/`connects`:
  ```
  reach(X, Z) :- adj_in_group(X, Y), reach(Y, Z).
  has_cycle :- adj_in_group(X, Y), reach(Y, X).
  ```
  This is **actively incorrect** for an undirected adjacency relation: `adj_in_group` is symmetric, so
  `adj_in_group(X, Y)` already implies `reach(Y, X)` via the single edge traversed backward — every
  edge in the group falsely reports a cycle. Fixing it needs the traversal to remember *where it came
  from* and reject only a genuine, non-parent back-edge:
  ```
  visited(S, none) :- seed(S).
  visited(S2, S)   :- visited(S, _), adj_in_group(S, S2), not visited(S2, _).
  has_cycle        :- visited(S2, P1), visited(S2, P2), P1 != P2.
  ```
  (`visited` here is deliberately multi-valued — a site reachable via two different, mutually
  non-adjacent parents *is* the cycle witness — not an imperative DFS spanning tree needing an
  evaluation-order guarantee Datalog's set-at-a-time semantics doesn't give for free.)
- **B.** `bounded_fixpoint`'s threaded state must be `(visited: Region, parent: Raster<Direction>)`,
  not bare `Region` — the same richer object the corrected Datalog version above needed once the naive
  attempt failed. `has_cycle = detect_backedge ∘ Tr(step)`, same trace shape as flood, parametrized by
  a bigger state object and a different morphism composed onto the trace's output. Because point-free
  notation forces the state object's *type* to be written down explicitly before anything else, it
  can't accidentally regress to the too-weak bare-`Region` version the way A's first attempt did.
- **C.**
  ```
  rule has_cycle(group: Region): Bool =
      fixpoint (visited: Region = seed, parent: Raster<Direction> = empty, cycle: Bool = false)
        step(v, p, c) = for n in frontier(v) adjacent-in group:
            if n in v && p[n] != reverse(dir_to(n, frontier_site)) then c := true
            else v := v | n; p[n] := dir_to(n, frontier_site)
        until no_change or max_iters(|group|)
      in cycle
  ```
  Same `(Region, Raster<Direction>)` state as B, but written as an ordinary named-accumulator loop —
  legible without knowing what a trace operator is, and it's close to a direct transcription of Core's
  own already-sketched `bounded_fixpoint(seed, step, max_iters)` signature (`DESIGN.md`'s "Control and
  aggregation" section), not a new idiom.

**Verdict:** the Core-level claim holds — `has_cycle` really is the same bounded-trace construction as
flood/chain-capture, generalized over the threaded state's type — but naive Horn-clause style does
*not* reach it for free; it reaches an actively wrong term unless the author already knows to add
parent-tracking, information that B and C's typed-state-up-front discipline supplies automatically and
A's untyped tuple-of-arguments style does not. This is the sharpest divergence of the five cases.

### Overall verdict

The two-way question the previous charter posed — "does Horn-clause desugar cleanly into the
categorical core, or diverge" — undersold how much work the Horn-clause side needed in every single
case: cases 1/2/3 all required a non-conjunction modifier bolted onto `:-` bodies (arguably no longer
"just Datalog"), and case 5's naive rendering was outright incorrect until it borrowed the categorical
style's explicit state-typing. Every case converges on the same Core term eventually, so "logic is
sugar over the categorical core" isn't false — but Style A never got there by naked unification alone;
it got there by re-deriving ad hoc pieces of Style B's discipline case by case.

Style C (typed functional/equational, `let`-bound hypotheticals, typed effect blocks, monomorphized
generics) reached every case as directly as Style B, using notation with no new concepts to teach —
`let`, ordinary function definitions, generics, a named accumulator loop are already how most working
programmers think, whereas Style B's `guard`/`Tr`/`≜` vocabulary requires importing restriction-category
and trace-monoidal-category concepts that add real reading friction for a human author even where they
resolve a case cleanly. Style A (flat Horn-clause Datalog) is now the weakest of the three candidates:
every one of its wins in these five cases was Datalog notation dressed around an idea (guard-after-`next`,
typed multi-valued back-edge state) that isn't actually Datalog's own.

**Recommendation, superseding the previous charter's implicit framing:** stop treating "Horn-clause
sugar over a categorical core" as the two-axis question. The categorical structure (Freyd-category
effect split, `bounded_fixpoint`-as-trace) stays exactly where `DESIGN.md` already scoped it — Core's
internal semantics and the source of optimizer-law justification (trace axioms licensing fixpoint
fusion/reordering) — not something a human author writes by hand. The primary **authoring** surface
should be Style C: a small typed functional/equational language directly over Core's value types
(`let`, named function definitions, pattern matching, a restricted typed `then { }` effect block, and
compile-time generics for templates), which desugars into the same categorical Core Style B already
targets, without asking a game author to think in either Prolog unification or category theory.
`DESIGN.md`'s "Relational GDL" section needs correcting to reflect this, not extending.

## Next session charter: harden Style C into an actual grammar, checked against the same five cases

Goal: now that Style C (typed functional/equational surface, see above) is the recommended primary
authoring language, design its real grammar and type system — informally, still markdown/pseudocode
plus maybe an EBNF sketch, but precise enough that all five pathological cases above, plus
`Tic-Tac-Toe`/`Hex` (which already have working Core programs and oracles to check a rewrite against),
can be transcribed without further invented notation. Concretely:

- Nail down the effect-block story: what exactly can appear inside `then { }` (Core's fixed
  `set`/`remove`/`insert`/`push`/`pop` primitives plus what else, if anything), and how it lowers to
  the Freyd-category premonoidal layer B already describes — this is the one place Style C's fragments
  above were still hand-wavy about the boundary.
- Design the concrete backend representation for unbounded auxiliary effect-state (case 3's history
  `Set<Hash>` — flagged as a new, undesigned open problem this session, see `DESIGN.md` update) at
  least at a sketch level: what data structure, what it costs per ply, whether it's viable inside the
  existing bitboard-oriented backend story at all.
- Decide how `template rule` generics get checked at "compile" (translation) time — full monomorphization
  per call site is assumed above but not yet specified: what happens if two call sites would require
  incompatible specializations, is there any generic bound/constraint syntax, etc.
- Transcribe `Tic-Tac-Toe` and `Hex` into Style C as a sanity check that the grammar isn't overfit to
  the five pathological cases and still reads well on the two simple, already-proven games.

Explicit non-goals: still no parser implementation, still no changes to existing `ast`/`parse`/
`elaborate`/Core Rust code; no further corpus survey beyond the seven games already in hand
(`Tic-Tac-Toe`, `Hex`, plus the five pathological-case sources) unless the grammar design hits a real
ambiguity nothing here resolves.

## Session note: Style C hardened — grammar, effect-block boundary, unbounded state, generics

This session was the charter above: no Rust changes, no parser, no new corpus games — harden Style C
(the typed functional/equational surface the last session recommended) into an actual grammar and type
system, precise enough to transcribe all five pathological cases plus `Tic-Tac-Toe`/`Hex` without
inventing further notation along the way. It's still markdown/pseudocode plus an EBNF sketch, not a
parser — per the charter's own non-goals.

### Grammar

```
Game        := "game" String "{" TopologyDecl PlayersDecl StateDecl* RegionsDecl*
                 MoveDecl+ RuleDecl* TemplateDecl* TerminalDecl OutcomeDecl "}"

TopologyDecl := "topology" "=" TopologyExpr
TopologyExpr := "Rect" "{" "rows" ":" Int "," "cols" ":" Int "}"
              | "Hex" "{" "shape" ":" HexShape "}"
HexShape    := "Rhombus" "{" "side" ":" Int "}" | "Triangle" "{" "side" ":" Int "}"
              | "Hexagon" "{" "side" ":" Int "}"

PlayersDecl := "players" "=" Int
StateDecl   := "state" Ident ":" Type ["=" Expr]
RegionsDecl := "regions" PlayerRef "=" "(" RegionExpr ("," RegionExpr)* ")"

Type        := "Region" | "Raster" | "Site" | "Direction" | "Player" | "Bool" | "Int"
              | "Set" "<" Type ">" | "Move" | "State" | "Outcome"
              | "fn" "(" (Type ("," Type)*)? ")" "->" Type   -- function *kind*, template params only

MoveDecl    := "move" Ident "(" ParamList? ")" "to" RegionExpr
               ["if" Expr]              -- ordinary legality guard, reads current state only
               ["ifAfterwards" Expr]    -- guard over the hypothetical post-move *board* (see below)
               ["then" EffectBlock]     -- this move's committed effect tail

RuleDecl    := "rule" Ident "(" ParamList? ")" ":" Type "=" Expr
TemplateDecl:= "template" "rule" Ident "<" TParam ("," TParam)* ">"
                 "(" ParamList? ")" ":" Type "=" Expr
TParam      := Ident ":" Type            -- Type here is always an "fn(...) -> ..." kind

TerminalDecl := "terminal" ":" "Bool" "=" Expr
OutcomeDecl  := "outcome" ":" "Outcome" "=" Expr

Expr        := Literal | Ident | FieldAccess | Call | BinOp | UnOp
              | "let" Ident "=" Expr "in" Expr
              | "if" Expr "then" Expr "else" Expr
              | "any" "(" RegionExpr "," Lambda1 ")" | "all" "(" RegionExpr "," Lambda1 ")"
              | Fixpoint

Fixpoint    := "fixpoint" "(" FixVar ("," FixVar)* ")"
                 Ident "(" Ident ("," Ident)* ")" "=" Block
               "until" Expr
               "in" Expr
FixVar      := Ident ":" Type "=" Expr

EffectBlock := "{" EffectStmt* "}"
EffectStmt  := EffectPrim "(" Expr ("," Expr)* ")" ";"
              | "let" Ident "=" Expr "in"
              | "if" Expr Block ["else" Block]
              | "for" Ident "in" RegionExpr Block
              | Ident                      -- a spliced-in template effect-block parameter
EffectPrim  := "set" | "place" | "remove" | "insert" | "push" | "pop"
```

`moves`/`terminal`/`outcome` are the three fixed program *roles* `DESIGN.md`'s "Heuristics, analysis,
and position evaluation" section already sketched (`MoveGen | Terminal | Heuristic | Analysis`) — a
`game { }` block just declares all of them together instead of tagging separately compiled programs,
since a single `.lud`-sourced game always needs all three at once. A `Heuristic`/`Analysis` role isn't
in the grammar above because no case (pathological or corpus) has needed one yet — same "grow from real
lowerings" discipline `DESIGN.md` already states, applied to the grammar itself.

### The effect-block boundary, nailed down: `next` is board-only, by construction

The charter flagged this as the one place Style C's earlier sketches were hand-wavy: what exactly is
inside `then { }`, and how does `ifAfterwards`'s "hypothetical next state" interact with it? The
resolution falls directly out of the Freyd-category split `DESIGN.md` already committed to, once stated
precisely:

**`next(state, m)` is defined to return only the *pure*, board-shaped part of state (`Region`/`Raster`/
`Int` fields) — never a hypothetical value for any `Set<Hash>`-typed auxiliary effect-state field.**
Concretely, `next` runs a *copy* of `state` through move `m`'s own `then { }` block (so `ifAfterwards`
never needs a second, separately-authored description of what the move does — it's always the same
`then` block, run non-committingly) but only the board-shaped output is observable from the result;
any `insert`/`set` touching a `Set<...>`-typed field during that hypothetical run is discarded along
with the rest of the copy. This isn't an arbitrary restriction — it's forced by the Freyd-category
story itself: aux effect-state lives in the premonoidal ("actions") layer specifically *because* it
can't be freely duplicated or discarded the way pure `Region` values can (`DESIGN.md`'s "Categorical
structure" section). "Compute an aux-state update hypothetically, then throw it away" is exactly the
operation that layer exists to disallow; a `next` that pretended to return a hypothetical `Set<Hash>`
would silently reintroduce the discardability the split was built to rule out. So `ifAfterwards`
predicates can read `next(state, m)`'s board fields freely, and can read the *real*, currently-committed
aux-state fields (never a hypothetical one) — which is exactly the mix case 3 below needs.

This also resolves an ordering hazard that looked real while drafting: does checking superko membership
against `history` before or after this move's own `insert(history, ...)` (in its `then` block) change
the answer? It doesn't, and not by luck — `ifAfterwards`'s `history.contains(...)` refers to the *real*,
pre-move `history` (an ordinary field read, no `next` involved), while `hash(next(state, m))` refers
only to the hypothetical *board*. The two never touch the same copy, so there's no self-referential
"does inserting make membership trivially true" trap to reason about case by case.

`then { }` itself is a straight-line effect-primitive sequence (`set`/`place`/`remove` for `Region`,
`push`/`pop` for `Raster`, `insert` for `Set<T>`, plus `let`/`if`/`for` control flow — `for` is always
statically bounded, since it ranges over a `Region`, which is bounded by board geometry per `DESIGN.md`'s
"Design principles"), composed via `;` (ordered, non-reorderable composition in the premonoidal layer,
matching `DESIGN.md`'s "sequenced, non-duplicable" language for this exact case). A template effect-block
parameter (case 4 below) splices in as a bare statement — its argument was already specialized away by
monomorphization before this block is elaborated, so by the time Core sees it, it's just more inlined
effect-primitive statements, not a call.

### Unbounded auxiliary state (superko `history`): backend sketch

`DESIGN.md`'s open problem — Go's positional superko needs a `Set<Hash>` that grows by one entry per
ply, which the "statically bounded" design principle (justified by *board-size* bounds) doesn't cover.
Sketch, at the level the charter asked for (a data structure, its cost per ply, whether it fits the
existing bitboard-oriented backend):

- **Representation**: an ordinary open-addressing hash table (e.g. a Robin-Hood or Swiss-table style
  layout, matching what `hashbrown`/`rustc-hash` already give for free in Rust) keyed on the `u64`
  Zobrist hash `DESIGN.md`'s "Cross-cutting backend concerns" section already derives for every board
  state — `history`'s keys are exactly the same hashes the incremental Zobrist-update machinery already
  computes per move, so no second hashing scheme is needed. This is genuinely a new Core value *kind*,
  distinct from `Region`/`Raster`/`Int` board state: an `Aux<Set<Hash>>` field in the generated per-game
  `State` struct, heap-allocated and resizable, sitting alongside the fixed-width bitboard fields rather
  than packed into them.
- **Cost per ply**: `insert` and `contains` are both amortized O(1); the table resizes (doubling,
  amortized) as the game gets deep. Total memory is O(plies played so far), not O(board size) or
  O(reachable states) — bounded by however long the actual game runs, same as any other per-move history
  a game engine already keeps (move list, undo stack), not a new asymptotic-cost class for the engine.
- **Fits the existing backend story**: yes, as an *addition* alongside the bitboard fields, not a
  replacement for any of them — `Region`/`Raster` board state stays fixed-width and cache-friendly;
  `history` is the one field per applicable game that's heap-backed. `Rect`/`Hex`/`Pyramid`/`Raster`
  topologies are unaffected; this is a state-shape concern (an aux field a `Set<Hash>`-typed `state`
  declaration lowers to), not a topology-lowering concern, so it composes with every topology variant
  the same way.

### Generics: what a template parameter is, and why "incompatible specializations" can't happen

The charter asked how `template rule` checking works at translation time. The grammar above gives a
syntactic answer, not a judgment call: **a parameter must be a template (`<...>`) parameter if and only
if its type is a function kind (`fn(...) -> ...`); every other parameter (`Region`, `Player`, `Site`,
`Int`, `Bool`, ...) is an ordinary runtime argument.** This isn't a style preference — Core has no
closures or first-class functions (`DESIGN.md`'s "First-order, not full lambda calculus" principle), so
a function-*valued* argument can only ever be resolved by substituting a concretely-named rule/effect
block at each call site, before Core sees the body at all. There's no other way to pass "a predicate" or
"an effect tail" into a shared body under that constraint.

Checking is just ordinary elaboration, run once per call site: substitute the call site's concrete
named function/effect-block for each `fn(...) -> ...`-kinded parameter (textual splicing, same
positional-substitution mechanism real Ludii `define`s already use), then elaborate the resulting
monomorphized body as an ordinary, generic-free rule. A type error inside the substituted body is an
ordinary type error reported at that call site — there's no separate "generics-checking" pass.

**"What if two call sites need incompatible specializations" doesn't arise**, and it's worth stating
why rather than leaving it as an assumption: because each call site produces its own independent,
monomorphized copy of the template body (no shared vtable, no cross-call-site unification the way a
Rust/C++ generic function's single compiled body would need), two call sites simply can never conflict
with each other — each either type-checks on its own or doesn't. Code duplication from this is a
non-issue at this project's scale (a handful of templates per game, not a shared library compiled once
for many callers). No bound/constraint syntax beyond the function-type kind itself is needed:
`Extra: fn() -> MoveSet` *is* the complete contract a template parameter can have, since Core doesn't
have a trait/interface system for that type to range over — consistent with "first-order, not full
lambda calculus," applied to the template layer specifically, same conclusion `DESIGN.md` already drew.

### The five pathological cases, in the hardened grammar

Compact renderings — the case-by-case reasoning for *why* these shapes are right was already settled
last session; this is just checking the finalized grammar actually reaches them with no further
invented notation.

**Cases 1+2 (Chess check-safety / Go suicide-rule, both `ifAfterwards`)** — `next` is now a built-in
tied to the move's own `then` block (see above), so there's no separate hand-written `next` function
the way the previous sketch had:

```
move StepOrCapture(from: Site, to: Site) to candidate_targets(from)
  ifAfterwards !is_in_check(next(state, this).occupied(King, mover), mover)
  then { move_piece(from, to) }
```

**Case 3 (Go superko)** — `history` stays a plain field read, `next` stays board-only, per the boundary
resolution above:

```
state history: Set<Hash> = {}

move Add(s: Site) to sites(Empty)
  ifAfterwards !history.contains(hash(next(state, this)))
  then {
    place(board, mover, s)
    insert(history, hash(board))
  }
```

**Case 4 (Chess `ChessPawn` template)** — unchanged in shape from last session's sketch, now typed
against the generics rule above (`Extra`/`Tail` are `fn(...) -> ...`-kinded, `piece` is an ordinary
runtime `Player`-or-`Site` argument, not a template parameter):

```
template rule chess_pawn<Extra: fn() -> Region, Tail: fn() -> EffectBlock>(piece: Site): Region =
    step_forward_to_empty(piece) | diagonal_capture(piece) | Extra()

move PawnMove(piece: Site, to: Site) to chess_pawn::<InitialPawnMoveOrEnPassant, ReplayThenSetCounter>(piece)
  then { move_piece(piece, to); ReplayThenSetCounter }
```

**Case 5 (Havannah `has_cycle`)** — matches the finalized `Fixpoint` production directly:

```
rule has_cycle(group: Region): Bool =
    fixpoint (visited: Region = seed(group), parent: Raster<Direction> = empty, cycle: Bool = false)
      step(v, p, c) = {
        for n in frontier(v, group) {
          if member(v, n) && p[n] != reverse(dir_to(n, frontier_site(v, n)))
            then { c := true }
            else { v := place(v, n); p := push(p, n, dir_to(n, frontier_site(v, n))) }
        }
      }
    until no_change(v) || max_iters(count(group))
    in cycle
```

### Sanity check: `Tic-Tac-Toe` and `Hex` transcribed

The point of this check is that the grammar shouldn't be overfit to the five hard cases — and it isn't:
neither game below needs `then`, `state`, `ifAfterwards`, templates, or `fixpoint` at all, just the base
declarative layer (`topology`/`players`/`regions`/`moves`/`terminal`/`outcome`), matching how small
`core::Program` already is for both (`Region` with `Occupied`/`Union`/`Complement`/`Sites`,
`MoveGen { to: Region }`, `EndRule::{Line,Connected}`, `player_regions`) — the hardened grammar is a
strict superset of what these two already-proven Core programs needed, not a rewrite of them:

```
game "Tic-Tac-Toe" {
  topology = Rect { rows: 3, cols: 3 }
  players  = 2

  moves: Region = sites(Empty)

  terminal: Bool = has_line(occupied(mover), length: 3)
  outcome: Outcome = Win(mover)
}
```

```
game "Hex" {
  topology = Hex { shape: Rhombus { side: 3 } }
  players  = 2

  regions P1 = (side(NE), side(SW))
  regions P2 = (side(NW), side(SE))

  moves: Region = sites(Empty)

  terminal: Bool = connects(occupied(mover), regions(mover).0, regions(mover).1)
  outcome: Outcome = Win(mover)
}
```

`connects` appears here as the general two-`Edge` combinator `DESIGN.md`'s Region-algebra table already
listed (`connects(edge_a, edge_b: Edge): Region -> Bool`) — the grammar is written against that intended
combinator set, not against `core::EndRule::Connected`'s current dedicated-variant implementation status
(already flagged as due for unification in the "Already covered" table; unaffected by this session).

## Session note: Alloy-style temporal refinement — `state'`/`always`/`once` replace `ifAfterwards`

Prompted by a live design discussion (not a fresh corpus pass): `ifAfterwards:` in the grammar
above was flagged as feeling ad hoc — a bespoke per-move guard keyword rather than a real piece of
vocabulary. The fix borrows directly from Alloy 6's temporal mechanics (Electrum): primed `state'`
for one-step lookahead (subsuming `ifAfterwards` as an ordinary expression rather than a dedicated
clause), a new top-level `invariant: always P` declaration (intersected into every move's legality
automatically), and Alloy's past-eventually operator `once` for case 3's superko check in place of
a bespoke `visited` history builtin. The finding worth keeping: cases 1, 2, and 3 above (Chess
check-safety, Go's suicide rule, Go's superko) turn out to be the *same* top-level `invariant:
always` construct once it exists as a standing declaration rather than a per-move modifier — that
wasn't visible while `ifAfterwards` was attached move-by-move.

The five cases are now rewritten against this refinement and saved as standalone, indelible
artifacts in `style-c/` (`style-c/01-check-safety.sc` through `style-c/05-havannah-cycle.sc`, plus
`style-c/README.md` for the full reasoning and the grammar delta) rather than kept only as inline
snippets in this file's session-note history. The grammar section above (`MoveDecl`'s
`ifAfterwards` clause) is left as-is, superseded but not rewritten, per this project's existing
convention (`DESIGN.md`'s "Relational GDL: superseded" section) of marking supersession explicitly
rather than editing history away. A future grammar pass should fold `state'`/`invariant: always`/
`once` into the main EBNF in place of `ifAfterwards`, once the next session's parser work (below)
makes it load-bearing rather than descriptive.

## Session note: Style C was leaking Rust — and a first fix overcorrected into leaking Alloy instead

Prompted by a live design critique, not a corpus pass: the grammar above (and every `style-c/`
artifact written against it, including the four card/graph/math/word cases in
`style-c/README.md`) had drifted into being "Rust with game nouns" rather than a syntax actually
targeted at this domain — the same kind of leak this project's own `DESIGN.md` already criticizes
Ludii for (a surface grammar shaped by its host implementation language rather than by the
problem). The first revision attempt fixed this by transliterating into literal Alloy notation
(`module`/`open`, dot-join and box-join chains, `<:` domain restriction, `~` relational transpose,
`abstract sig`/`extends`, `no`/`univ`) — which turned out to be the identical mistake pointed a
different direction: swapping one borrowed dialect for another borrowed dialect, still not a
syntax *derived from this domain*. Caught before landing (`games/tak-alloy.sc` was written, judged
to have overcorrected, and deleted rather than kept as a superseded artifact — it was never
complete or shown before the correction, so there was nothing to preserve). The standing rule
going forward: **borrow semantics freely, keep notation our own.** Concretely, what's worth taking
from Alloy is a small number of *ideas* — declarative relational state transitions (no mutation
statements), the primed-name convention (`field'` = value after this transition), and the bounded
temporal operators (`always`/`once`, already adopted in the previous session note above) — not
Alloy's specific *operators* for expressing them. Those operators are Alloy's own accidental
complexity, earned by being a general first-order relational logic with a model finder behind it;
this project has no model finder and a much narrower domain (bounded board games), so there's no
reason its notation should look like Alloy's any more than it should look like Rust's. Revised,
domain-native fixes for the same five leaks:

- **`const fn ... match n { 3 => 10, ... }`** — a `table` declaration: a plain finite compile-time
  lookup (`table piece_reserve(n: Int): Int = { 3: 10, 4: 15, 5: 21, 6: 30, 7: 40, 8: 50 }`), not
  Rust's general-purpose `match`/`const fn` (which imply arbitrary pattern matching and control
  flow this never needed) and not Alloy's dot-join-over-a-union-of-arrow-tuples trick either
  (`n.(3->10 + 4->15 + ...)`, correct but unreadable to anyone who hasn't used Alloy). A lookup
  table is exactly what this construct *is*; it should say so.
- **`state reserve: [Int; players]`** — an indexed-state declaration binder,
  `state reserve[p: Player]: Int = piece_reserve(N)`, extending this grammar's own existing
  `state name: Type = init` form with an index rather than switching to Rust's array-type syntax
  *or* Alloy's `Player -> one Int` relation-plus-domain-restriction idiom. This is genuinely new
  notation, but it's ours: shaped for exactly one recurring need (one value per member of a small
  enumerable domain — players today, plausibly piece kinds or regions later), not a general
  container or relation type imported wholesale from somewhere else.
- **`Set<Int>`, `Seq<Letter>`** — kept close to the Alloy-influenced fix (`set Int`, `seq Letter`
  as multiplicity keywords ahead of a base type, no angle brackets) since this one was never
  really Alloy jargon to begin with — "a set of Int" and "a sequence of Letter" read as plain
  English, not as a borrowed dialect. Worth noting explicitly: not every similarity to Alloy is a
  leak that needs re-litigating; the test is whether the notation is legible on this domain's own
  terms, not whether some other language happens to spell it the same way.
- **Imperative `then { push(...); set(...); for i in ... { if ... { } } }`** — `then` blocks
  become a set of `field' = expr` bindings (declarative, one per changed field) rather than a
  statement sequence, reusing this grammar's own already-established `push`/`pop`/`set`/`insert`
  *names* as ordinary pure functions (`board' = set(board, s, (kind, mover))`) instead of
  replacing them with Alloy's `++`/`+`/`-` override operators — the verbs were already
  domain-appropriate game vocabulary ("push a stone," "set a cell"), the only thing wrong was
  using them as commands instead of as functions returning a new value. Anywhere a `then` block
  needs to walk a bounded sequence (Tak's spread), use this grammar's own existing `fixpoint`
  construct and `let`/`if`/`any`/`all` combinators (already established in the hardened EBNF)
  rather than an ad hoc `for {}` loop (Rust's) or Alloy's `|`-pipe quantifier/let syntax (also
  foreign) — both were reaching for something this grammar already had a spelling for.
- **`enum PieceKind { Flat, Wall, Capstone }`** — withdrawn as a finding entirely. On reflection
  this was never a Rust-specific leak (`enum { A, B, C }` for a closed alternative set is close to
  notation-neutral across most typed languages, not distinctively Rust's), and Alloy's own
  alternative for the same idea (`abstract sig` + `extends`) is *more* foreign to a game-rules
  reader, not less — replacing it would itself have been an unforced Alloy-shaped substitution,
  exactly the mistake this note is about avoiding. Left as-is.

Templates (`template rule<T: fn() -> Region>` / `template game<const N: Int>`) are also left
as-is: angle-bracket generic-parameter syntax is close to universal across typed languages at this
point (C++/Rust/Java/TS/Swift all spell it this way), not a distinctively Rust import, so there's
no obviously-more-domain-native alternative notation to switch to the way there was for lookup
tables or per-player state. The trait-bound-flavored *constraint* syntax (`T: fn() -> Region`)
stays under review, not because it's wrong, but because nothing in this batch of cases actually
exercised whether a simpler unconstrained form would do — a question for whenever the parser work
below makes template resolution load-bearing rather than descriptive.

The Tak session's "named, composable effect blocks" and "per-player indexed state" findings still
mostly dissolve under this revision, same conclusion as the discarded Alloy draft reached: once
effect logic is an ordinary named `rule`/`effect rule` returning a value (nothing Rust-flavored
about naming and reusing a function) and per-player state is an ordinary indexed-state
declaration, neither needed a new grammar construct — they only *looked* new because they were
being expressed with imperative statements and arrays instead. `count_where` is the one place the
discarded draft's insight is worth keeping even though its notation isn't: whatever this
grammar's own comprehension/counting notation ends up being, the underlying point stands that
"count where" is a special case of a more general aggregation-over-a-predicate idea, not a
bespoke combinator earning its own name.

`style-c/games/tak-relational.sc` is the actual proof-of-concept, rewriting `games/tak.sc` against
these revised, domain-native rules; `games/tak.sc` itself stays in place, header-flagged
superseded-by, per this project's mark-don't-delete convention. The other eight `style-c/` files
are **not yet rewritten** — still scoped as one worked example for review before a full pass. The
`state'`/`invariant: always`/`once` temporal layer needs no changes; it was already right (Alloy
*semantics*, this project's own notation, exactly the balance this whole note is arguing for). The
"Next session charter" below is still written against the pre-revision grammar — worth revisiting
before starting that work, once/if the rest of `style-c/` gets the same treatment.

## Session note: second round of live syntax review on `tak-relational.sc`

Live, line-by-line review of `style-c/games/tak-relational.sc` from a PLT-literate reader,
prompted by the operator's framing that surface syntax is worth nailing down for human legibility
before working back up from core semantics. Decisions, in the order raised:

- **Template instantiation switches from angle brackets to square brackets** (`Tak[N]`, not
  `Tak<N>`/Rust's `Tak::<N>` turbofish), following Go's precedent. Applied in
  `tak-relational.sc`; **not yet back-ported** to the main EBNF (`TemplateDecl`/`TParam` in the
  hardened-grammar section above) or to `04-chess-pawn-template.sc`, which still uses `<...>` —
  syntax is deliberately iterating fast on the one living proof-of-concept file rather than
  keeping every artifact in sync on every round, per the same reasoning that scoped the original
  Alloy-leak fix to one file first.
- **`template` overlaps with C++ templates and OCaml's parametric modules/functors** — noted, not
  acted on. OCaml's module-as-abstraction is a plausible right long-term semantic target (imports,
  namespacing, and parametric templating living in one unified world), but OCaml's own collision
  between "module the compilation unit" and "module the PLT idea" makes it a genuinely alien
  surface, which is exactly why `module` stays deliberately unused as a keyword even where
  `template` is doing module-shaped work. Recorded as a real future direction — this project will
  need some import/namespacing story eventually, and unifying it with `template`'s parametrization
  the way OCaml unifies modules and functors would be nice — explicitly not a day-1 requirement.
- **Full sum types, not bare enumerations, supported out of the gate.** `enum` already covers
  payload-carrying variants, not just unit variants — `Outcome` (`Win(Player)`/`Draw`) was already
  being used this way as an implicit builtin throughout every earlier case without ever being
  declared; `tak-relational.sc` now declares it explicitly with the same `enum` keyword
  `PieceKind` uses, demonstrating one construct covers both shapes rather than needing a separate
  "sum type" declaration form.
- **`fixpoint` was confusing — and turned out to be a real design bug, not just unfamiliar
  notation.** `tak-relational.sc`'s first draft reused `fixpoint` for the bounded walk over Tak's
  spread `drops`, and it read badly even to a reviewer who already knows what a fixpoint is
  computationally. On inspection, that usage was actually wrong: `fixpoint` promises *convergence*
  semantics (repeat until no more change, `max_iters` only as a safety valve) — exactly what
  `05-havannah-cycle.sc`'s cycle check legitimately needs, since it can't know in advance how many
  steps until the visited set stabilizes. Tak's spread never had that shape: `drops` has a
  statically known length before the walk starts, so there's no convergence question, only "apply
  one step per element of an already-bounded sequence, threading an accumulator" — an ordinary
  fold. **Split into two constructs**: `fixpoint` stays for genuine least/greatest-fixed-point
  iteration (unknown iteration count, convergence- or `max_iters`-terminated); a new `fold i in
  <sequence> with acc = <init> { <body-expr> }` covers deterministic, known-length walks, with the
  body's trailing expression becoming the next `acc` (no `:=`, consistent with `then` blocks never
  using mutation statements). Reflects a general lesson worth keeping: notation that "reads badly"
  is sometimes actually a semantic conflation wearing a syntax problem's clothes, not just
  unfamiliar spelling — worth checking which one it is before just re-wording the same construct.
- **`const fn` dropped.** Nothing about a function like `stack_bits` needs to declare its own
  const-evaluability; only the call site does (here, inside a `template game "Tak"[const N: Int]`
  binding, where `N` is compile-time-known by construction). const-ness is now inferred from
  whether a given call's arguments are themselves compile-time constants, not declared on the
  function — an ordinary `rule` covers both runtime- and compile-time-evaluable pure functions
  uniformly. `table` (the earlier fix for `const fn ... match`) is unaffected — it stays a
  distinct form not because of const-ness but because "this is a literal finite lookup" is worth
  saying regardless of when it's evaluated.
- **`state` indexed-declaration notation (`state reserve[p: Player]: Int = ...`) confirmed as-is**
  — no change requested, called out as a good fit.
- **`move` notation flagged as premature to redesign.** The reviewer noted `move`'s `if` guards
  will likely get much more complex in real games (Chess's full pawn-move disjunction, case 4, was
  named as a harder example already in this corpus) and wondered whether an expansive multi-clause
  guard deserves its own first-class `move`-body primitive (e.g. a `where`-clause list) rather than
  an ordinary boolean-expression `if`. Explicitly deferred, not decided — by the reviewer's own
  framing, judging this needs more move-heavy examples than this corpus currently has, not a
  redesign forced through on one case. Flagged inline in `tak-relational.sc` at `move Place` as an
  open question for a future round.
- **`let`/`in` confirmed, unchanged** — read as convenient and appropriately regular even though
  it's visibly ML/Alloy-flavored; possible future sugar noted as a maybe, not requested now.
- **`if`/`then`/`else` kept as the primitive, with new guard-arm sugar for cascading chains.** A
  leading-`|` list of `condition -> value` arms (Haskell-guard/OCaml-match-arm flavored, exact
  syntax as proposed live), first match wins, `otherwise` required as the catch-all — sugar only,
  desugars directly to nested `if`/`else if`/`else`. Applied to `tak-relational.sc`'s `outcome`
  (the case that prompted it, a priority-ordered list of mutually exclusive conditions ending in a
  default); left as plain binary `if`/`else` inside `apply_spread`, since there's no cascading
  chain there to flatten.

`tak-relational.sc` is updated in place (not versioned into a third file) — these are refinements
of the same living proof-of-concept, not a new supersession event the way the Rust→Alloy→relational
progression was; nothing here contradicts an earlier decision, it's the same decision made more
precise. Its trailing "Scorecard" comment is updated to track both this round's findings
(`fixpoint`/`fold` split, `const fn` removal) alongside the original five.

## Session note: third round — `move`'s bare `if`/`then` was a footgun, not just unfamiliar syntax

Direct pushback on the second round's "`move` notation flagged as premature to redesign" call: the
reviewer pointed out `move`'s `if COND then { effect }` doesn't read well specifically because it's
an `if`/`then` with no `else` — and objected on principle to bare `if`/`then` anywhere, not just in
this one spot, calling it a footgun. Right on both counts, and the second one sharpens the first:
`move`'s `if`/`then` was never actually a value-producing conditional in the first place, it was a
*precondition* — "this move only exists when COND holds" — wearing `if`/`then` as a borrowed
costume, with no `else` because "the move doesn't exist" isn't a value any `else` branch could
produce. That's a category error (a boolean filter and a value expression aren't the same kind of
thing) as much as a readability one, and it's exactly the kind of thing that invites a reader or
future editor to write a *real* value-producing `if` the same way, no `else`, and get it wrong.

Fix, demonstrated in `style-c/games/tak-relational.sc`'s three `move` declarations:

- **`guard`** is `move`'s dedicated precondition keyword — boolean-only, produces no value, exactly
  where a valueless conditional belongs. A single condition sits inline (`guard turn < players`);
  multiple conditions list one per line beneath `guard`, implicitly conjoined (AND) the way a
  Haskell guard list or a Prolog clause body already reads, no `&&` needed to chain top-level
  conditions (an individual line can still use `&&`/`||` internally for a compound condition, as
  `Place`'s two ownership-check lines do).
- **`if`/`then`/`else` becomes a hard, project-wide rule: always total, `else` always required, no
  exceptions anywhere in the grammar.** With `guard` covering every precondition, there's never a
  legitimate reason left to write a bare, valueless `if` — so the rule can be absolute rather than
  "usually total, except at the top of a move," which is exactly the exception that was the
  footgun.
- **`then { }` is gone from `move` entirely.** A blank line separates the `guard` block from the
  effect bindings (unchanged `field' = expr` list from the previous round, just without the wrapper
  braces or trailing semicolons — newline is the statement separator now, `guard` and the blank
  line already delimit the blocks without needing braces too).

This also directly resolves the second round's deferred "should an expansive guard be a
first-class `move`-body primitive?" question — the reviewer's own answer, arrived at from the
readability complaint rather than decided in the abstract, is yes, and `guard` is that primitive.
Worth noting as a pattern for this whole review process: "doesn't read well" turned out to be a
reliable signal of an actual design problem twice in a row now (`fixpoint`/`fold` in round two,
`guard` here) — worth taking that complaint literally and looking for the conflation underneath it
rather than treating it as pure bikeshedding.

## Session note: fourth round — `field'` unified into `fold`, range sugar, and a `Player`/`Team`/`Outcome` framework

Three threads from the same live-review conversation, of different weights:

**1. Backend-lowering worry, resolved and recorded in `DESIGN.md`.** The reviewer asked whether
whole-value effect syntax (`board' = push(board, s, v)`, denotationally replacing the entire
board) is hurting us relative to GDL's apparent per-relation locality or Ludii's direct mutation —
i.e. will the compiler actually be able to derive "this sets one bit"? Answer: no, and not by
luck — recorded as a new "Design principles" bullet in `DESIGN.md` (right after the existing
"Referentially transparent, à la Halide" bullet, since it's the same separation of algorithm from
schedule, applied to updates instead of to computing a value). The short version: the effect
vocabulary is a *closed, first-order primitive set* (not arbitrary user-defined functions), so
each primitive's touched-site shape is known by construction and composes structurally, without
needing the general escape/alias analysis a real functional language would; every bounded-
iteration construct (`fold`, `bounded_fixpoint`) already guarantees that touched-site set is
statically *bounded in size* even when its members are move-parameter-dependent. GDL's locality
turns out not to be a counterexample — real GDL engines compile to propnets for exactly this
reason, they don't get it for free either — and Ludii avoids the problem only by not having a
declarative surface at all, which is the leak this whole spike exists to avoid. `DESIGN.md` now
also states the discipline this requires going forward: every new effect primitive must ship with
its own known touched-site rule at the point it's added, not by assumed analogy.

**2. `Player`/`Team`/`Outcome` — a framework, not yet a syntax change.** The reviewer pushed on
whether `players = N` / `Player` / `P0`/`P1` are principled or just convenient for the games
written so far — specifically raising partnership/team games, variable player counts, and
mid-game elimination, and noting `Outcome` needs to ultimately be interpretable as utilities by
the engine, not just a friendly enum. Working through it: `Player` is best understood as a
**static, compile-time-sized domain exactly like `Site`** — fixed at instantiation (today via
`players = N`; for a variable-player-count game, the identical `template game[const N: Int]`
mechanism Tak already uses for board size, no new construct). Under that framing, every concern
raised turns out to be an instance of a pattern this session already established:
- **Mid-game elimination/dropout** dissolves into ordinary indexed `state` — `state active[p:
  Player]: Bool`, ordinary state-transition logic to skip inactive players in turn order. No new
  vocabulary; `Player`-the-domain never shrinks, only a per-player property changes, symmetric to
  how `state board: ...` is indexed `state` over the static domain `Site`.
- **Partnerships/teams** need one genuinely new (but small) piece: a `team: Player -> Team`
  mapping, defaulting to the identity map (each player their own team) when undeclared, which
  recovers every game written so far for free. `Outcome`'s constructors should then generalize
  from `Win(Player)` to `Win(Team)` (a plain player is just a singleton team).
- **`Outcome` and engine-visible utilities**: `Outcome` (`Win`/`Draw`/whatever a given game
  declares) should stay an ordinary user-declared `enum`, per this session's earlier sum-types
  decision — it should *not* become compiler-magic just because the engine needs utilities out of
  it. What the MCTS backend actually consumes is a `Player -> Real` (or `Team -> Real`) utility
  vector; a given game's `Outcome` enum is that game's convenient sugar for producing one. The
  common shapes (zero-sum win/draw/lose, symmetric team win) are good candidates for a small
  standard-library convention once the import/namespacing mechanism flagged in the previous round
  exists, rather than hardcoded compiler knowledge of the names `Win`/`Draw`.
- **Mid-game joining** (a genuinely open/growing player set, not just elimination) is structurally
  the same problem as `sprouts.sc`'s dynamic topology, and stays flagged as a real, harder,
  deferred case — no existing corpus game forces it.

None of this is applied to `tak-relational.sc` this round — Tak is 2-player, zero-sum, no
elimination, no variable player count, so there's no forcing example for `Team`/utility-mapping
syntax yet, and inventing it without one would break "grow the combinator set from real
lowerings," the same discipline this whole review has stuck to throughout (per-player state,
named effect blocks, etc. all only got real syntax once a real game forced it). A partnership card
game (Bridge/Euchre-shaped) is a good future addition to the card-game corpus specifically to force
this concretely, the way Kuhn Poker forced hidden information.

**3. Two small, low-risk syntax fixes, applied to `tak-relational.sc`:**
- **`a..b` sugar for `range(a, b)`.** Purely notational; `range` stays the underlying primitive.
- **`fold`'s body now uses the explicit primed-accumulator convention (`out' = expr`)** instead of
  an implicit trailing-expression return, and the header leads with the accumulator
  (`fold out = b for i in 0..len(drops)`) rather than trailing it after a `with` clause. This was
  a second, independent "doesn't read well" complaint about `fold` even after the round-two
  `fixpoint`/`fold` split — and again turned out to be a real inconsistency, not just
  unfamiliarity: `fold`'s body was the one place in the file that didn't use the same `name' =
  expr` idiom `then` blocks and `invariant` already use everywhere else, relying instead on
  positional "last expression is the return value." One consistent primed-name convention now
  covers all three. `05-havannah-cycle.sc`'s own `fixpoint` syntax likely deserves the same
  treatment eventually, flagged as a followup rather than redesigned blind — that file isn't in
  front of us this round.

## Next session charter: implement the Style C parser and Core lowering, re-proving `Tic-Tac-Toe`/`Hex`

**Status: superseded, not executed.** Session after session kept finding real problems in Style C's
surface syntax instead (`rule`→`def`, `ifAfterwards`→`invariant: always`/`state'`/`once`, bare
`if`/`then`→`guard`, `fold`'s block-header form→an ordinary call, angle brackets→square brackets —
see the session notes below), so this charter was never actually picked up. A later session note
("Evaluation vs. GDL and Ludii, and a pivot to descriptive complexity", near the end of this file)
judges the syntax churn to have run its useful course and deliberately deprioritizes parser work
further — see that note and `EVALUATION.md`/`COMPLETENESS.md` for why. Left in place rather than
deleted, per this project's mark-don't-delete convention; it accurately describes what was planned
at the time.

Goal: turn the grammar above from a design document into working Rust, per `DESIGN.md`'s pipeline
diagram (`typed source -> lex/parse -> rule AST -> Core IR`) — the first "NEW" stages actually get
built, not just designed. Concretely:

- Lexer/parser for the grammar above (a new `src/style_c/` or similar, alongside — not replacing —
  the existing `ast`/`parse`/`elaborate` pipeline, which stays as-is per `DESIGN.md`'s "still a working
  bootstrap" framing).
- A rule-AST-to-`core::Program` lowering pass for exactly the declarative subset the `Tic-Tac-Toe`/`Hex`
  transcriptions use (`topology`/`players`/`regions`/`moves`/`terminal`/`outcome`, no `then`/`state`/
  `invariant`/templates/`fixpoint` yet) — this is the fastest path to an actual regression test: parse
  `style-c/games/tic-tac-toe.sc` and `style-c/games/hex.sc` (checked-in fixtures, not just inline
  markdown snippets), lower to `core::Program`, and check the result against the same
  `tests/oracle.rs`/`tests/hex_oracle.rs` fixtures the existing `ast`-based pipeline already passes.
- Explicit non-goal for next session: don't yet implement `then`/`state`/`ifAfterwards`/templates/
  `fixpoint` lowering — those need real `core::Program`/`core::EndRule` extensions this session
  deliberately didn't design (extending `core::mod` itself is Rust work, out of scope for a grammar-only
  session), so building their parser support first would produce AST nodes with nowhere to lower to.
  Land the declarative subset end-to-end first, matching this project's own "prove the pipeline on
  tic-tac-toe before adding capability" bootstrap order.

## Session note: `rule` renamed to `def`

Prompted by a live design question, not a corpus pass: every use of the `rule` keyword across the
whole `style-c/` corpus (`stack_bits`, `road_region`, `has_road`, `is_word`, `in_semigroup`, `apply_spread`,
`has_cycle`, ...) turned out to be an ordinary named pure function, nothing Horn-clause-shaped. That's
not an accident of which examples happened to get written -- Relational GDL (this doc's original
Datalog-sugar proposal) was already retracted as the authoring surface a few sessions back (see
"Relational GDL: superseded" above), so `rule` had been carrying a name left over from a design that
no longer exists. Worse, the name collided with the plain-English sense of "a game rule" in a way
that was actively misleading: `has_road` reads naturally as a rule, `stack_bits` (a bit-packing
helper) does not, and the keyword covered both identically. Per this project's own "don't invent a
distinction without a forcing case" discipline, there's no live case in the corpus that needs `rule`
to mean anything other than "named pure function" -- so it's renamed to `def` (and `template rule` to
`template def`), a shorter, more neutral name matching the ML/Lisp-family register the rest of the
grammar already borrows from (`let`/`in`, pattern matching).

Applied mechanically across every living `style-c/` artifact (`01`-`05`, `games/{tic-tac-toe,hex,
tak-relational,kuhn-poker,sprouts,sylver-coinage,ghost}.sc`, `style-c/README.md`'s reference table).
`games/tak.sc` is left untouched -- it's already marked superseded and frozen per this project's
mark-don't-delete convention, and rewriting its body would misrepresent what that historical artifact
actually said at the time. The grammar EBNF in this file's own "Style C hardened" session note above
is left as-is too, same reasoning as every other supersession here: it already predates several later
decisions (`state'`/`invariant: always`/`once` replacing `ifAfterwards`, `guard` replacing bare
`if`/`then`, `Tak[N]` square-bracket instantiation) and isn't the place to patch in one more delta:
`RuleDecl`/`TemplateDecl` there still read `rule`, and stay that way until a real grammar-hardening
pass consolidates all of these deltas at once, per that section's own already-stated followup.

## Session note: Core / Stdlib / Extern -- the builtin surface gets tiered, and Tak's gets audited

Prompted directly by a question about the standard library, not a corpus pass: `style-c/`'s ~50
builtin-looking identifiers had no declared status (core primitive vs. surface-language-expressible
library sugar vs. genuinely-foreign oracle call), and that ambiguity had already produced real bugs
-- `has_cycle` defined twice, disagreeing with itself, at two different layers; `pop` called with an
undocumented third argument; three spellings of "whose turn it is" (`mover`/`to_move`/
`current_player`); and a dangling `carried_top_is` reference in `tak-relational.sc` that was never a
design question at all, just an unfinished call. `DESIGN.md`'s new "Standard library: Core, Stdlib,
and Extern" section (inserted after "Design principles") establishes the three-tier split plus a new
`extern def name(params): Type` declaration form for foreign calls (`geometric_oracle`,
`dictionary_has_prefix`/`dictionary_has_word`, `shuffle`/`draw2` -- all already flagged in prose as
black-box, now with an actual keyword instead of being syntactically indistinguishable from a typo),
and does one full worked pass reclassifying every builtin `games/tak-relational.sc` actually uses --
the only file on current (`guard`/primed-field/`Tak[N]`) syntax, so the only one this pass touched.
Concrete outcomes: `shift`'s Site-stepping overload split out into its own `walk` primitive; `pop`
gained a documented `Stack<Value>`-returning 3-arg form; `is_full`/`count_where`/`opponent` promoted
to real Core primitives; `mover`/`len` confirmed canonical over `current_player`/`to_move`/`length`;
and the dangling `carried_top_is` call fixed in place (`top(board, from).kind == Capstone` already
says what it was reaching for). `01`-`05` and `kuhn-poker.sc`/`sprouts.sc`/`sylver-coinage.sc`/
`ghost.sc` still predate the `guard`/`state'`/`Tak[N]` syntax rounds and were deliberately *not*
reclassified this session -- they're real evidence the naming collisions above are project-wide, not
Tak-specific, but fixing their builtin surface has to wait for the same syntax refresh
`tak-relational.sc` already got, per this project's established one-file-at-a-time discipline.

## Session note: `fold` becomes an ordinary combinator call, not a block-header special form

Live-review pushback, same session: `fold`'s syntax (`fold out = seed for i in iter { ... out' =
expr }`) read as unwarranted Alloy-adjacent sugar, and concretely gave a pre-existing `def` nowhere
to plug in as the step function, unlike `any`/`all`/`project`, which already take their predicate
as an ordinary lambda argument. Fixed by making `fold(seed, iterable, step)` an ordinary call, with
`step` an inline `|acc, elem| body` lambda in exactly the same shape `any`/`all`/`project` already
use. That shape -- lexically-scoped, immediately-applied, never stored or returned -- turns out to
already be pervasive in this grammar without ever being named as its own thing; `DESIGN.md`'s
"Control and aggregation" section now calls it a **second-class lambda** and states the rule
precisely: it reads like a closure and captures enclosing bindings the way one does, but the
compiler can always inline it away at its one call site, so it's not the "full lambda calculus with
closures" this doc's own Non-goals section excludes -- that non-goal (also corrected this session)
was stated more broadly than what the corpus actually relies on. Direct answer to "can I pass a
`def` instead of writing the lambda inline": yes, whenever that `def`'s parameter list already
covers everything the step needs, with no free variables left to close over; otherwise write it
inline, the same constraint `any`/`all`/`project`'s predicates already had.

This also retires round 4's `out' = ...` primed-accumulator convention for `fold` specifically: once
`fold` is an ordinary call, its lambda's trailing expression is already unambiguously the return
value, the same as every other lambda in the grammar, so the priming was patching a symptom of
`fold` not being a real call rather than a genuine gap. Applied to `apply_spread` in
`games/tak-relational.sc`, the file's inline comments, and its scorecard. `05-havannah-cycle.sc`'s
`fixpoint` likely has the identical block-header problem, but per this project's one-file-at-a-time
discipline it isn't touched this round -- still an open followup, not assumed fixed by analogy.

## Session note: `tak.md`, the numbered cases swept, and `kuhn-poker.sc` refreshed -- charter executed early

Prompted directly by the operator judging strict one-file-at-a-time too conservative for the
`01`-`05` fragments specifically ("cheap to update. Do it now") -- this session both executes the
kuhn-poker charter above and goes further than it, in one pass:

**`games/tak-relational.sc` moved to `games/tak.md`.** The file had become as much a worked-example
write-up (six rounds of live syntax review, each with real reasoning worth keeping) as it was
source; Markdown lets that reasoning read as prose next to the fenced code it explains, instead of
competing with it as `//`-block comments. Content preserved, reorganized into headed sections
matching the game's own structure (lookup tables, per-player state, placing, spreading, road win,
instantiation) with a "Revision history" section replacing the trailing scorecard comment. Every
reference to the old filename updated across `DESIGN.md` and `style-c/README.md`'s reference table;
references inside older, already-committed session notes above (rounds 1-6) are left naming
`tak-relational.sc`, since they're accurately describing what the file was called at the time --
same "don't edit history away" reasoning applied everywhere else in this doc.

**`01`-`05` swept.** `01-check-safety.sc`/`02-suicide-rule.sc`/`03-superko.sc` turned out to already
be on current syntax -- each is a single `invariant: always ...` expression with no `rule`, no bare
`if`/`then`, no templates, so there was nothing stale to fix. `04-chess-pawn-template.sc` got a real
refresh: square brackets for template params/instantiation (`chess_pawn[...]`, not `chess_pawn<...>`/
`::<...>`), and its `move`'s effect body converted from a `then { statement; statement }` block to a
`field' = expr` binding, which forced `Tail`'s kind to change from `fn() -> EffectBlock` (a spliceable
statement list) to `fn(Region) -> Region` (an ordinary pure function), matching how every other
effect in this grammar works now. **`05-havannah-cycle.sc` deliberately not touched, and flagged as
genuinely not cheap, unlike the other four:** its `fixpoint` step body still uses `for`+`:=` mutation
statements, which is real stale notation (retired project-wide back in the "Style C was leaking Rust"
session), but the honest fix isn't a rename -- `fixpoint`'s block-header form has the identical
"no argument slot for a pre-existing `def`" problem `fold` had before round 6, and fixing *that*
needs some way to thread multi-value state (`visited`/`parent`/`cycle` together) through an
ordinary call, which this project has never resolved (tuples exist informally as anonymous
positional values -- `(kind, mover)` in `games/tak.md`, `.0`/`.1` in `games/hex.sc` -- but
destructuring-bind syntax doesn't). Attempting a mechanical fix risked silently changing the
cycle-detection algorithm's actual behavior at an edge case (an already-visited neighbor reached via
its own immediate parent) rather than just re-spelling it. Left as its own follow-up rather than
forced through under this session's "cheap" framing.

**`games/kuhn-poker.sc` refreshed, both charter questions resolved:**

- **Private, epistemically-scoped state survives `guard`/primed-field cleanly, no changes needed to
  either.** `state private[p: Player]: Card` (indexed, and the corpus's first *uninitialized*
  `state` -- the hardened grammar's `StateDecl` already makes `= Expr` optional) reads and writes
  fine under the existing conventions; the epistemic-scoping rule (player p's own declarations may
  reference `private[p]`, never `private[q]`) is a read-access restriction the syntax layer doesn't
  need to enforce or even be aware of.
- **`extern def` gets its first live call site, and the determinism-tag question stays open, now
  with a concrete example instead of an abstract one.** `extern def shuffle(deck: [Card]): [Card]`
  and `extern def draw2(deck: [Card]): (Card, Card)`, consumed from a new `chance Deal()` body. That
  body also forced a small, real grammar extension: a `let dealt = draw2(shuffle(...)) in` prefix
  ahead of the primed bindings, the first move/chance body needing one shared local computation
  feeding multiple fields (`private'[P0]`/`private'[P1]` both need to come from the *same* shuffle,
  not two independent draws) rather than one independent expression per field -- `let` itself wasn't
  new (already used inside `def` bodies), just where it's allowed to appear. `Outcome` declared
  explicitly (`enum Outcome { Win(Player) }`, no `Draw` -- Kuhn's 3-card deck has no ties), closing
  the same gap `tak.md`'s own `enum Outcome` declaration closed.
- **`fold`'s closure-call shape stays untested against a non-spatial game.** Kuhn poker needed no
  `fold`-shaped loop, so this question is still open -- unresolved by omission, not by a negative
  result.

Explicit non-goals, still deferred: `sprouts.sc`/`sylver-coinage.sc`/`ghost.sc` untouched this
session; `05-havannah-cycle.sc` untouched, now with a concrete reason rather than just precedent;
the `Raster` cell `Value` tuple-vs-record shape stays open (`kuhn-poker.sc` has no `Raster`, so it
still can't force that question).

## Session note: evaluation vs. GDL and Ludii, and a pivot to descriptive complexity

Prompted directly by the operator asking for a first-principles evaluation of Style C against
Stanford GDL and Ludii, on four axes: a small set of correct composable primitives, expressiveness/
concision, provability guarantees useful for compilation, and preserving universality the way
Ludii's own completeness work argues for. Full assessment saved as `EVALUATION.md` rather than kept
only in conversation — it's a snapshot judgment worth being able to re-check later, not throwaway
discussion.

Headline finding: the first three axes hold up well (real cross-game primitive reuse
`flood`/`connects`/`has_cycle`/`bounded_fixpoint` already demonstrate that GDL and Ludii don't;
`tak.md` is competitively concise; the Freyd-category/trace framing gives a real optimizer-law
foundation neither prior system has). The fourth does not hold up as stated — this project's own
design principles (first-order, `bounded_fixpoint` instead of general recursion, statically-bounded
`Region`/`Raster` state) *deliberately decline* GDL/Ludii-style unrestricted universality in exchange
for provability, so claiming to "preserve universality" the way Ludii's paper means it would be soft
under scrutiny. `EVALUATION.md` recommends re-grounding the claim in descriptive complexity instead:
Immerman–Vardi (FO(LFP) = PTIME on ordered finite structures) gives a sharper, provable completeness
target — "universal for every finite-state game predicate decidable in polynomial time in board
size" — that neither predecessor can make, since GDL's universality is unrestricted-but-uncompilable
and Ludii's is an empirical, after-the-fact corpus-coverage argument, not a complexity-class
characterization. `COMPLETENESS.md` states this conjecture precisely: a primitive-by-primitive
FO/LFP classification of every Core IR construct, the ordered-structure setup (already free here,
since `BitBoard`'s row-major indexing already fixes a canonical `Site` order), the rule-complexity-
vs-game-complexity distinction that must not get conflated, and what's actually needed to turn the
conjecture into a real proof.

One finding worth flagging on its own: the primitive-by-primitive mapping in `COMPLETENESS.md`
retroactively explains round 2's `fixpoint`/`fold` split (`README.md`, above) as *exactly* the FO
vs. LFP expressiveness boundary — `fold`'s statically-known-length walk is FO(order), `fixpoint`'s
unknown-iteration-count convergence is genuinely LFP — even though nobody was thinking about
descriptive complexity when that split was made purely for readability. Real independent evidence
the design is already well-aligned with the target formalism, not a coincidence manufactured to fit
the conjecture after the fact.

Decision, acted on directly in this session: redirect effort away from further surface-syntax
review (six-plus rounds on `tak.md` alone is judged enough for now) and toward (1) hardening the
completeness conjecture and (2) real `core::mod` implementation work — deliberately *not* the Style
C parser, which needs a stable grammar to be worth freezing and the grammar is judged not stable
enough yet. `DESIGN.md`'s existing "Style C parser" charter (`README.md`, above) is marked
superseded rather than deleted, per this project's usual convention.

## Next session charter: unify `flood`/`connects`/`has_cycle` into real Region-algebra combinators, and check the FO(LFP) upper bound against them

Goal: real Rust work in `core::mod`/`core::interp`, no parser, no new corpus game beyond
`Tic-Tac-Toe`/`Hex` (both already proven) — the two are deliberately combined into one charter
because they're the same piece of work looked at from two directions: `DESIGN.md`'s own "promote to
a composable primitive once a second dedicated special case appears" principle already flags
`EndRule::Line`/`EndRule::Connected` as due for unification (two hardcoded, non-composable variants
is the trigger, not a hypothetical third), and that unification is also the first real implementation
`COMPLETENESS.md`'s FO(LFP) upper-bound classification of `flood`/`connects`/`has_cycle` needs to be
checked against, rather than staying a claim about a design document.

Concretely:

- Add `flood`/`connects`/`adjacent`/`shift` as real, composable `core::Region`-algebra combinators
  (not `EndRule`-only special cases) and a real `bounded_fixpoint`/trace IR node they lower to, per
  `DESIGN.md`'s Region algebra table and `COMPLETENESS.md`'s classification of `bounded_fixpoint` as
  the LFP operator directly.
- Re-express `Tic-Tac-Toe`'s line-win and `Hex`'s edge-to-edge connectivity win as instances of the
  new combinators rather than the existing dedicated `EndRule::Line`/`EndRule::Connected` variants,
  and confirm `tests/oracle.rs`/`tests/hex_oracle.rs` still pass unchanged — this is the regression
  safety net, not a new game.
- While doing this, check `COMPLETENESS.md`'s table entries for `flood`/`connects`/`has_cycle`
  concretely: does the new `bounded_fixpoint` IR node's shape actually support the *simultaneous*
  multi-relation induction `has_cycle` needs (`(visited, parent, cycle)` threaded together per the
  design spike), or does building it surface a real gap in either the IR design or the conjecture's
  classification? Either answer is useful and worth recording either way — this charter is explicitly
  not committed to `has_cycle` itself landing this session (no game in the currently-proven corpus
  forces it), only to the `bounded_fixpoint` node being shaped so it *could*.
- Write up whatever the implementation surfaces as a `COMPLETENESS.md` update — confirming the
  primitive-by-primitive table, or correcting it — rather than letting the document drift out of sync
  with what `core::mod` actually does.

Explicit non-goals: no Style C lexer/parser (`src/style_c/` or equivalent) — still judged premature,
per this session's pivot above; no new corpus game (`Y`, `Havannah`, etc.) — the point is checking
the conjecture against already-proven ground truth, not expanding coverage; no attempt at
`COMPLETENESS.md`'s lower-bound (completeness) direction yet — that's a harder, more open-ended
research question properly scoped to its own session once the upper-bound direction has real code
behind it, not something to force through opportunistically while doing IR refactoring work.

## Session note: `flood`/`connects`/`adjacent`/`shift` landed as real Region-algebra combinators; `has_cycle`'s fixpoint shape confirmed, not landed

This session was the charter above, executed as scoped: real `core::mod`/`core::interp` Rust work,
no parser, no new corpus game. `cargo test -p ludii -p game-core` is 92 lib + 6 hex-oracle + 5
oracle tests, all green (up from 85 lib + 6 + 5 last session — the new tests are direct,
hand-built-Program-style coverage of the new combinators themselves, not just indirect coverage via
Tic-Tac-Toe/Hex); `cargo clippy --workspace --exclude mcts-bench --exclude game-host --all-targets`
and `cargo fmt -p ludii -- --check` are clean.

- **`core::Region` grew three real combinators**: `Shift { region, dir }`, `Adjacent { region,
  conn }`, and `Flood { region, seed, conn }`, backed by a new `core::Direction` (all eight
  queen-move directions, naming `BitBoard::shift_*` directly) and `core::Connectivity`
  (`Four`/`Six`/`Eight`, one per existing `BitBoard::flood{4,6,8}`). `core::interp::adjacent` folds
  every direction's shift of the *same* input region into one expression before OR-ing — the same
  single-statement discipline `flood6`'s doc comment already called for, now structural (a `fold`
  over a direction list) rather than something a future combinator's author has to remember by
  convention.
- **`core::interp::bounded_fixpoint`** is a new generic trace function (`DESIGN.md`'s "Categorical
  structure" `Tr` operator, made real): iterates a step function from a seed, unioning into an
  accumulator until it stops growing or a board-size-derived `max_iters` is hit. `Region::Flood`
  is its `Aux = ()` instantiation (`step` just unions in `adjacent`); this replaced a direct,
  non-composable `BitBoard::flood6` call in Hex's end-rule evaluation with the same underlying bit
  operations expressed as a real composable node — `region_flood_matches_direct_flood6_call`
  checks the two produce identical results.
- **`core::EndRule`/`core::BoolExpr` replace `EndRule::Line`/`EndRule::Connected`**, per `DESIGN.md`'s
  own "an end rule is really 'some Boolean/Region predicate over the board is true'" framing:
  `EndRule` is now just `{ condition: BoolExpr }`, and `BoolExpr` has `Contains`/`Connects`/`Any`.
  Tic-Tac-Toe's line-win lowers to `Any` of one `Contains(Sites(line))` per candidate line
  (`Rect::lines`, unchanged); Hex's connectivity-win lowers to `BoolExpr::Connects { conn: Six }`,
  evaluated as one `flood` call plus an intersection test. Both are genuinely different `BoolExpr`
  values now, not two Rust enum variants `interp::State::winner` had to pattern-match by name.
- **A real gap this surfaced, not just a refactor**: `connects(edge_a, edge_b)`'s two operands
  aren't embedded as literal `Region` values inside `BoolExpr::Connects` — `Program.end` is shared
  across every player while `Program.player_regions` varies per player, so the interpreter still
  looks the edge pair up per mover at eval time (`State::winner` computes `edges` the same way it
  already computed `board` from `last_mover`), the same "runtime binding the interpreter supplies,
  not a value inside the AST" restriction `DESIGN.md`'s non-goals already place on `mover` itself.
  Closing this for real needs an expression-level `mover`/`regions(mover)` accessor — i.e. a real
  authoring-surface parser — which is exactly why this session's non-goals excluded one.
  `COMPLETENESS.md`'s `connects` row now documents this as a genuine, still-open seam rather than
  something the refactor quietly papered over.
- **A second correction surfaced by writing real code against `DESIGN.md`'s own table**: `flood`'s
  documented signature (`flood(seed: Site, conn: Connectivity)`) was already stale — a connectivity
  check seeds from a whole board edge (Hex's `(sites Side NE)`, potentially several sites), not one
  site, which `BitBoard::flood6(self, seed: Self)`'s own signature already reflected before this
  table did. Fixed to `flood(seed: Region, conn: Connectivity)`.
- **`has_cycle`'s fixpoint shape confirmed, deliberately not landed**: `core::interp::bounded_fixpoint`
  is generic over an auxiliary threaded state `Aux`, specifically so `flood`'s bare-`Region` case and
  `has_cycle`'s richer simultaneous case (`README.md`'s design-spike case 5: `visited`/`parent`/
  `cycle` bootstrapped together) are one node shape, not two. `has_cycle_shape_holds_a_parent_and_cycle_flag`
  instantiates `Aux = (HashMap<Site, Direction>, bool)` against a hand-verifiable positive/negative
  pair (a 2x2 board under four-way adjacency is itself a 4-cycle; a 3-cell sub-board of the same
  board is a path) and asserts on the recovered parent map, not just the cycle flag — confirming
  both halves of the simultaneous state are real, threaded data, not one live field plus one
  vestigial one. This is a real, compiling, passing check of the *shape*, per the charter's own
  scoping — `has_cycle` is still not a `Region`/`BoolExpr`/`Program` primitive; no `EndRule` lowers
  to it, and no `Raster<Direction>` value type exists yet for a real (non-`HashMap`-backed) backend
  implementation to use.
- **`COMPLETENESS.md` updated** with what actually held (`flood`/`connects`'s LFP/FO classification,
  confirmed unchanged by real code) and what didn't fully resolve (`connects`'s mover-relative
  operands; `has_cycle` landing) — see that document's primitive-by-primitive table and its "Where
  this connects to concrete Core IR work" closing section, both revised rather than left describing
  a pre-implementation state.

## Next session charter: not yet written

The natural next steps this session's findings point at — a `regions(mover)`/`mover`-as-value
authoring-surface construct to close the `connects` operand gap, landing `has_cycle` for real
(needs a `Raster<Direction>` value type plus a real backend, not just a `HashMap`-backed test), or
finally starting the Style C lexer/parser now that Region algebra has real code behind it — aren't
charter-written yet; pick one deliberately next session rather than defaulting to the first item in
this list.

## Session note: `05-havannah-cycle.sc` resolved by treating `has_cycle` as a primop, not by generalizing `fixpoint`

Prompted directly by the operator, mid-task-selection for this session: the `tak.md`-sweep
session's "genuinely not cheap" flag on `05-havannah-cycle.sc` (its `fixpoint` still used the
pre-round-6 block-header form and retired `for`/`:=` mutation, and fixing it properly looked like
it needed a general way to thread multi-value state through an ordinary call — tuple
destructuring-bind, which this project has never designed) turns out to dissolve once stated as a
scope question rather than a syntax question: **it's expected and fine for Core to carry a large,
hand-written instruction set over bitboards/hexboards, the same way a CPU's ISA has instructions no
compiler derives from smaller ones in user code.** `has_cycle` was already declared a Core
primitive in `DESIGN.md`'s Region algebra table, sitting alongside `flood`/`connects`/`adjacent`/
`shift` — the open question was never really "how does a game author write `has_cycle` from
scratch," it was "what does a game author write to *use* it," and the answer is just
`has_cycle(group)`.

Consequence: `style-c/05-havannah-cycle.sc` is updated in place (not superseded/replaced) — its
header now states explicitly that `has_cycle` is a primop call in real Style C source, and the
`fixpoint` derivation underneath is a *reference definition* (documents the semantics a correct
backend lowering must agree with, the same role `tests/hex_oracle.rs`'s hand-rolled oracle already
plays for `flood6`) rather than authoring-surface code, and is therefore exempt from the grammar's
usual rules — it doesn't need a syntax-refresh pass at all, since it was never claiming to be
parseable Style C in the first place. This retires the dangling followup from the `tak.md`-sweep
session note above without generalizing the grammar: no tuple-destructuring-bind, no multi-value
`fixpoint`-threading construct is added, and none is expected to be needed until some other corpus
game forces one independently of `has_cycle`.

This doesn't change `DESIGN.md`'s standing "grow the combinator set from real lowerings" principle
— it sharpens a corollary already implicit in it, now stated explicitly as its own bullet: growing
the *primitive* set (more backend instructions) and growing the *authoring grammar* (more surface
constructs) are different axes, and a game forcing the former doesn't automatically force the
latter. `style-c/README.md`'s reference table updated to match.

## Session note: real Rust work begins — a second, independent `s-expr -> Core IR` frontend, bypassing Style C's never-built lexer

Prompted directly by the operator: "we need to start getting these compiling," with a concrete
three-stage split of `DESIGN.md`'s pipeline diagram (`surface syntax -> s-expr -> Core IR ->
backend`) — leave the first arrow (a real lexer/parser for Style C's `def`/`guard`/`fixpoint`/...
notation) unimplemented for now, since several sessions of syntax review already left that grammar
unstable and the "pivot to descriptive complexity" session note above deliberately deprioritized
it; but stop treating that as a blocker for the second and third arrows. The key realization: the
existing `s-expr -> Core IR` path (`parse::sexpr` -> `ast::*`/`elaborate::*` -> `core::lower`) is
real, working code — but it exists *only* for Ludii's own ludeme-shaped s-expressions
(`(is Line 3)`, `(move Add (to (sites Empty)))`, ...), restricted to exactly what `Tic-Tac-Toe.lud`/
`Hex.lud` need. Nothing about `parse::sexpr`'s reader itself (parens for calls, `{}` for lists,
ordinary literals) is Ludii-specific; a *second*, independent `s-expr -> Core IR` lowering — reusing
the same reader but skipping `ast`/`elaborate` entirely — can target a direct parenthesized
rendering of `core::Program`/`Region`/`BoolExpr`'s own Rust shape instead of Ludii's ludeme
vocabulary. That sidesteps designing and stabilizing a second lexer/grammar altogether: nothing
about Style C's planned human-friendly notation (`def`, primed fields, `guard`, ...) needs to exist
yet for a game's declarative subset to be written down and actually compiled today.

**Landed**: `src/style_c/mod.rs`, a new top-level module (registered in `src/lib.rs`, independent of
`ast`/`elaborate` by design — see its own module doc for why). Its grammar is close to 1:1 with
`core::Program`'s existing Rust shape (`(game "Name" (topology (rect 3 3)) (players 2)
(moves (sites Empty)) (end (has_line 3)))`, etc.) plus two small derived forms already legitimate
elsewhere in this project's own discipline: `(has_line <n>)` expands via the same `Rect::lines`
helper `core::lower::lower_end` already calls for `.lud`'s `(is Line n)`, and `(sites Empty)`
reuses `core::lower::all_occupied` (now `pub(crate)`) for the same "every unoccupied site" shape
`core::lower::lower_move_gen` already produces. `(side <compass>)` (Hex-only, inside `(regions
...)`) duplicates `core::hex::Hex::edge_for_compass`'s NE/SE/SW/NW mapping in miniature rather than
calling it, specifically to avoid pulling in `crate::ast::types::CompassDirection` — keeping this
frontend's independence from the old pipeline real rather than nominal.

**Proof**: `style-c/sexpr/tic-tac-toe.sc` and `style-c/sexpr/hex.sc`, two new load-bearing fixtures
(the same discipline `lud/*.lud` already has), each lowered through the new frontend and asserted
equal (`assert_eq!` on the whole `Program` value) against the *same* game lowered through the
existing `.lud` pipeline — the identical regression pattern `core::interp`'s own
`manual_program_matches_lowered_one`/`manual_hex_program_matches_lowered_one` tests already
established for hand-built `Program` values, just with a real parser producing the hand side now
instead of Rust literals. Both pass, confirming the two independent frontends agree on the same
Core IR target. `cargo test -p ludii -p game-core` is 96 lib tests (up from 92) + 6 hex-oracle + 5
oracle, all green; `cargo clippy -p ludii --all-targets` and `cargo fmt -p ludii -- --check` are
clean; full-workspace `cargo test --lib --workspace --exclude mcts-bench --exclude game-host`
passes unchanged.

**Explicitly not done, on purpose**: no `has_cycle`/`state`/`fold`/`fixpoint`/templates/effect
blocks in the new grammar yet — this session only covers the same declarative subset
`core::Program` already has real fields for (topology/players/moves/end/regions), matching this
project's own "prove the pipeline on tic-tac-toe/hex before adding capability" bootstrap order
applied to a second frontend. Growing `core::mod`'s IR itself (raster ops, `state`, `has_cycle` as
a real primitive per the session note above, ...) is real Rust work this new frontend will need to
track as it happens, not something solved by this session. "Backend lowering" per the operator's
own three-way split is `core::interp` — a tree-walking interpreter binding `Program` to a concrete
`BitBoard`, not codegen to Rust source (`DESIGN.md`'s "backend primops -> Rust source" arrow) — the
operator's "partly done" was accurate on both counts: real for `Rect`/`Hex`, and real only as an
interpreter, not a compiler, so far.

## Session note: Y proven -- `Hex { Triangle }` needs no new backend, `connects` generalizes to N edges

Third corpus game, per `DESIGN.md`'s recommended order (Hex, Y, then Havannah) and the operator's
own framing for this session: pick Y and push `core::mod`/`style_c` forward together, one real
capability at a time, rather than speculatively growing the grammar further. Two real capabilities
landed, both forced directly by Y's own shape, neither spent on anything speculative:

- **`Region::Intersect`**, a new `core::Region` combinator (`core::mod`, `core::interp::eval_region`).
  Forced by the triangular board: `(sites Empty)` has to mean "empty AND inside the triangle," not
  just "empty," since a `Hex { Triangle }` board's valid sites are a proper subset of the
  `side x side` grid a `Hex { Rhombus }` board fills completely.
- **`core::Program.player_regions` generalized from `Vec<(Region, Region)>` to `Vec<Vec<Region>>`**,
  and `BoolExpr::Connects`'s interpretation with it (`core::interp::eval_bool`): flood from the
  first named region, then check the result intersects every remaining one. Hex's two-edge
  edge-to-edge win and Y's three-edge win are now two lengths of the same list, not two `BoolExpr`
  shapes -- exactly the generalization `DESIGN.md`'s own "promote to a composable primitive once a
  second dedicated special case appears" principle already flagged as due ("Y's three-edge win is
  about to be a third [data point]"), now landed rather than predicted.

**The topology itself needed no new backend at all**, which wasn't obvious going in --
`DESIGN.md`'s corpus table previously called Y's coordinate packing "not just Hex's rhombus with
corners chopped." It turned out to be exactly that: `core::hex::Hex` grew a `HexShape` field
(`Rhombus`/`Triangle`) and `Hex::valid_sites()` (every site for `Rhombus`, `row + col < side` for
`Triangle`) plus a `TriangleEdge` enum (`Bottom`/`Left`/`Hypotenuse`, sharing exactly one corner
site per pair, checked directly in `core::hex`'s own tests) -- the same `side x side`
`BitBoard<N, N>` and six-way adjacency (`flood6`) Hex already proved, just masked smaller. No new
coordinate system, no new shift/flood backend code.

`style_c` grew in step, not ahead of it: `(hex_triangle <side>)` (a new `Topology` shape; renamed
from an initial `hex-triangle` after the lexer turned out not to treat `-` as an identifier
character -- `is_ident_continue` is alphanumeric-or-underscore only, so a hyphenated head silently
mis-tokenized into three separate tokens rather than erroring loudly), `(tri_side Bottom | Left |
Hypotenuse)` (a `(regions ...)` endpoint, parallel to `(side <compass>)` for `Rhombus`), `(intersect
a b)` (the new combinator, exposed directly for testability, not just used internally), and
`(regions player edge...)` generalized from a fixed two-endpoint clause to a variable-length list.
`(sites Empty)` on a `Hex { Triangle }` topology now automatically wraps the usual
complement-of-occupied in `Region::Intersect(_, Sites(valid_sites))`; every other topology is
unaffected (`match`ed to the identity case), so no existing `Program` value changed.

`style-c/sexpr/y.sc` is the new load-bearing fixture: a fixed side-4 triangular board (10 valid
sites out of the 16-cell grid it's carved from), `(end (connects Six))`, and *identical* three-edge
`(regions ...)` lists for both players -- unlike Hex, where each player owns a different edge pair,
Y's win condition doesn't distinguish edge ownership at all, only which player's stones connect
them.

**Deliberately not pushed through the `.lud`/`ast`/`elaborate` pipeline**, unlike Hex's own session
-- consistent with `DESIGN.md`'s current "Goal" framing of that pipeline as a working bootstrap, not
where new corpus games should grow. There's consequently no hand-concretized `lud/Y.lud` fixture;
`database-1/lud/games/Y.lud`'s real, option-templated source (board size 3-19, standard/misère end
rules) stayed the spec, read directly rather than lowered. Verification instead follows this
project's "translation path" methodology precisely: `style_c::tests::y_matches_a_hand_built_program`
(a hand-built `Program` equality check, the same "Core IR should be constructible and checkable by
hand" discipline `core::interp`'s own manual-Program tests use) plus `tests/y_oracle.rs` (a
from-scratch, independent BFS oracle mirroring `tests/hex_oracle.rs`'s method exactly --
`YOracle::neighbors`/`winner` never call into `core::hex`/`core::interp` at all), including a
touching-only-two-of-three-edges negative case specifically checking the new N-ary `Connects` logic
doesn't regress to "any edge is enough."

108 tests pass in `ludii` (102 lib -- up from 92 -- + 6 hex-oracle + 5 oracle + 5 new y-oracle), all
up from 96 lib+hex+oracle last session: `cargo test -p ludii -p game-core`, `cargo clippy -p ludii
--all-targets`, and `cargo fmt -p ludii -- --check` are all clean. Full-workspace
`cargo test --lib --workspace --exclude mcts-bench --exclude game-host` also passes unchanged.

`DESIGN.md` updated to match: the `Topology model` section's `Hex`/`HexShape` sketch now notes two
of its three variants are real; the Region algebra table's `connects` signature is now
`connects(edges: [Edge])`; "Backend lowering" no longer calls `Hex` "no precedent, new backend" (only
`Hex { Hexagon }` still is); the corpus tables move Y from "Worth adding" into "Already covered" and
update the recommended build order (Havannah next, now the last and hardest hex board rather than
one of two remaining).

## Suggested commit message:

Prove Y end to end: a triangular Hex topology and N-ary `connects`

Add `core::Region::Intersect` and generalize `Program.player_regions`/
`BoolExpr::Connects` from a fixed two-edge pair to an arbitrary-length list --
both forced directly by Y's triangular board and three-edge win, landing
`DESIGN.md`'s own previously-predicted `EndRule` generalization. `Hex {
Triangle }` needed no new backend: it reuses `Rhombus`'s `BitBoard<N, N>`
grid and six-way adjacency unchanged, masked to `row + col < side`. Grow
`style_c`'s sexpr frontend in step (`hex_triangle`, `tri_side`, `intersect`,
variable-length `regions`), add `style-c/sexpr/y.sc`, and verify against a
hand-built `Program` plus a from-scratch BFS oracle (`tests/y_oracle.rs`)
rather than the legacy `.lud`/`ast`/`elaborate` pipeline, per `DESIGN.md`'s
own framing of that pipeline as bootstrap, not where new corpus games grow.


## Session note: first Core-IR-to-Rust codegen, proven on Tic-Tac-Toe against the hand-built `games/ttt`

Prompted directly by the operator: "take the next step in building the core IR to rust pipeline,"
with Tic-Tac-Toe named explicitly as "a point of comparison to the hand-built `games/ttt`" --
`ROADMAP.md`'s phases 4 ("Backend codegen: Core IR → Rust source," 0% built until now) and 5
("First full pipeline proof: Tic-Tac-Toe"), tackled together since phase 5 is phase 4's own first
real exercise.

**Landed**: `src/codegen/` (`mod.rs` dispatches on `Program.topology`; `rect.rs` is the only real
backend). Unlike `core::interp` -- a generic tree-walking evaluator that re-walks a `Program`'s
`Region`/`BoolExpr` trees every call, deliberately kept as the slow, obviously-correct oracle --
`codegen::rect::generate` lowers a `Program` once, at generation time, into the *text* of an
ordinary standalone Rust source file: a `Player` enum, `Move`/`Position` types built on
`game_core::bitboard::BitBoard<N, M>`, an incrementally-hashed `HashedPosition`, and a real
`mcts::game::Game` impl -- the same shape every hand-written `games/*` crate already has. Per
`ROADMAP.md`'s phase 4 decision (made this session): generation is an offline step
(`src/bin/codegen.rs`, `cargo run -p ludii --bin codegen -- <sexpr> <StructName> <"Game Name">`,
output piped through `rustfmt` so the checked-in result passes this repo's own `cargo fmt --check`),
not a `build.rs`/proc-macro step -- reviewable and debuggable the same way every other crate in this
workspace is source-controlled.

Scoped narrowly on purpose, matching `DESIGN.md`'s "grow from real lowerings" principle applied to
a *second* backend, not just the first: `region_expr`/`bool_expr` only lower the `Region`/`BoolExpr`
variants Tic-Tac-Toe's own `Program` actually uses (`Occupied`/`Union`/`Complement`/`Sites` for the
move generator, `Contains`/`Any` for the end rule) and return `codegen::Error` -- not a panic, not a
silent wrong lowering -- on anything else (`Intersect`/`Shift`/`Adjacent`/`Flood`, `Connects`).
Those are exactly what Hex/Y's `Program`s need and don't have a `Rect`-codegen lowering yet; neither
does `Topology::Hex` itself. Real next steps for `ROADMAP.md`'s phase 6, not attempted
speculatively here.

**Proof, checked into `games/ttt-gen`** (a new workspace member, `game-ttt-gen`): the generated
`games/ttt-gen/src/lib.rs` is `codegen::rect::generate`'s literal output for
`style-c/sexpr/tic-tac-toe.sc`, regenerating byte-for-byte identical (confirmed by re-running the
binary and diffing). Two new oracle tests close the loop from both directions named in the two
exit tests `ROADMAP.md` already specified:

- `tests/ttt_gen_vs_interp.rs` -- phase 4's exit test, read literally ("a generated crate compiles
  and its `Game` impl round-trips through the interpreter's own oracle tests for the same game"):
  walks `tests/oracle.rs`'s same move sequences through `games/ttt-gen`'s `Position` and
  `core::interp::State<3, 3>` directly, asserting legal moves and winner agree at every step.
- `tests/ttt_gen_oracle.rs` -- phase 5's exit test ("the generated crate passes the same kind of
  oracle check `games/ttt` already gets") and the operator's own framing: walks the same sequences
  through `games/ttt-gen::TicTacToe` and hand-written `games/ttt::TicTacToe` via the
  `mcts::game::Game` trait on *both* sides (not just `Position` methods), so it exercises the
  generated crate's actual `Game` impl -- `generate_actions`/`apply`/`is_terminal`/`winner` -- against
  the hand-built one end to end. One wrinkle: `games/ttt::Game::winner` panics
  (`unreachable!()`) if called on a non-terminal state, so the comparison only calls `winner` on
  both sides once both agree the position is terminal.

This repo now has three independent, cross-checked implementations of Tic-Tac-Toe's rules
(`core::interp` + `style_c`, `games/ttt-gen`, hand-written `games/ttt`), pairwise checked against
each other by `tests/oracle.rs`, `tests/ttt_gen_vs_interp.rs`, and `tests/ttt_gen_oracle.rs`.

**Deliberate representation differences from `games/ttt`**, called out in `games/ttt-gen/src/
lib.rs`'s own doc comment rather than treated as a gap to close: one `BitBoard` per player here
(matching `core::interp::State`'s own representation) vs. `games/ttt`'s packed 2-bit-per-cell
`u32`; a plain incrementally-XORed zobrist hash here vs. `games/ttt`'s D4-symmetry-aware
`HashedPosition` (`Program` has no way to declare a topology's symmetry group yet, and no codegen'd
game has forced that gap open -- not attempted speculatively); `'A'`/`'B'`-per-player-index display
characters here vs. `games/ttt`'s game-specific `'X'`/`'O'`, since generic codegen has no
game-specific vocabulary to draw display characters from. `ROADMAP.md`'s phase 5 exit test asks for
behavioral parity via an oracle check, not byte-identical memory layout -- these differences are
exactly the kind of backend/schedule choice `DESIGN.md`'s "referentially transparent, à la Halide"
principle says shouldn't be forced to match a hand-tuned implementation's specific choices.

Two small `rustc`/clippy lints shaped the codegen templates directly, worth recording since they'll
recur for the next game routed through codegen: `-D unused-parens` rejects a redundant enclosing
`(...)` specifically in a few syntactic positions (a `let` value, a block's tail) -- `BoolExpr::Any`
no longer wraps its `||`-joined operands in parens at all (every combinator at this level is `||`,
so grouping was never semantically load-bearing); and `-D clippy::double_parens` caught
`Region::Complement` wrapping an already-self-parenthesizing `Region::Union` a second time --
fixed by having `Complement` never add its own parens, relying on every other `region_expr` arm
already being unambiguous on its own (an atom, or already parenthesized).

`cargo test -p ludii -p game-ttt-gen -p game-ttt`, `cargo clippy -p ludii -p game-ttt-gen
--all-targets`, and `cargo fmt -p ludii -p game-ttt-gen -- --check` are all clean; full-workspace
`cargo test --lib --workspace --exclude mcts-bench --exclude game-host` passes unchanged (168 lib
tests across the touched crates plus every other workspace member).

## Suggested commit message:

Land the first Core-IR-to-Rust codegen backend, proven on Tic-Tac-Toe

Add `src/codegen/rect.rs` (`Program` -> Rust source, scoped to the
`Region`/`BoolExpr` shapes Tic-Tac-Toe's `Program` actually uses) and
`src/bin/codegen.rs` (the offline generator driver, output piped through
`rustfmt`) -- `ROADMAP.md`'s phase 4, previously 0% built. Check in its
output for Tic-Tac-Toe as a new workspace member, `games/ttt-gen`
(`ROADMAP.md`'s phase 5 proof game), and add two oracle tests:
`tests/ttt_gen_vs_interp.rs` (generated crate vs. `core::interp`, phase 4's
exit test) and `tests/ttt_gen_oracle.rs` (generated crate vs. hand-written
`games/ttt` via `mcts::game::Game` on both sides, phase 5's exit test and
the operator's own requested point of comparison). `Hex`/`Connects`-shaped
end rules still route through `core::interp` only -- real work for
`ROADMAP.md`'s phase 6, not attempted speculatively here.
