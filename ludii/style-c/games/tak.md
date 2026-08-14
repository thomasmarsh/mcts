# Tak

Proof-of-concept rewrite of `games/tak.sc`, per the top-level `HISTORY.md`'s "Style C was leaking
Rust -- and a first fix overcorrected into leaking Alloy instead" session note. Same game, same
license (pro forma, not required to parse or lower to any existing `core::Program` shape) -- the
point of this file is the *syntax*, not new rules content.

Moved from `games/tak-relational.sc` to this literate form because the file had become as much a
worked-example write-up (six rounds of live syntax review, each with real reasoning worth keeping)
as it was source -- Markdown lets that reasoning read as prose next to the code it explains, rather
than as `//`-block comments competing with the code for visual weight. The code fragments below are
not `.sc` in their own right; read them as excerpts of one `Tak` game, in source order.

Two governing rules apply throughout, both established in live review and unchanged since:

- **Borrow semantics freely, keep notation this project's own.** Declarative field-transition
  bindings, no mutation statements, the primed `field'` convention; reach for another language's
  notation only where nothing native does the job.
- **`if` is always a total, `else`-required value expression, everywhere in this grammar, no
  exceptions.** Preconditions get their own keyword (`guard`, on `move`), so there's never a reason
  to write a valueless `if`.

## Piece-reserve lookup tables

`table` is a plain finite compile-time lookup -- not Rust's `const fn`/`match` (which imply
arbitrary control flow this never needed), and not a relation-algebra trick borrowed from somewhere
else either. This *is* what the construct is; it says so.

```sc
table piece_reserve(n: Int): Int = { 3: 10, 4: 15, 5: 21, 6: 30, 7: 40, 8: 50 }
table capstone_reserve(n: Int): Int = { 3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2 }
```

## Bit-width helper

`const fn` was dropped in round 2 of the leak review: nothing about `stack_bits` itself needs to
declare that it's const-evaluable, only the *call site* below does (inside `template game
"Tak"[const N: Int]`'s `topology` binding, where `N` is compile-time-known by construction).
Const-ness is inferred from whether a given call's arguments are themselves compile-time constants,
not declared on the function -- an ordinary `def` covers both runtime- and compile-time-evaluable
pure functions uniformly. `table` above stays a distinct form regardless -- not for const-ness, but
because "this is a literal finite lookup" is worth saying whether or not it happens to be
const-evaluable.

`rule` was renamed to `def` project-wide (see the top-level `HISTORY.md`'s session note): every use
of the keyword across this corpus is an ordinary named pure function -- there's no surviving case
where it carries genuine Horn-clause/logic-rule semantics now that Relational GDL is retracted as
the authoring surface, so the name was a vestige of retired terminology that also collided with the
plain-English sense of "a game rule" (true for `has_road` below, false for `stack_bits` right here)
-- a sharper ambiguity than a neutral name carries.

```sc
def stack_bits(n: Int): Int = 2 * (piece_reserve(n) + capstone_reserve(n))
```

## Piece kind and outcome

`enum` was withdrawn from the leak inventory (see the `HISTORY.md` session note) -- this was never
Rust-specific notation, so it's unchanged from `games/tak.sc`.

```sc
enum PieceKind { Flat, Wall, Capstone }
```

Full sum types, not bare C-style enumerations: variants may carry payload, same `enum` keyword.
`Outcome` was already being used with payload-carrying constructors (`Win(mover)`) throughout every
earlier case without ever being declared; making it explicit here demonstrates `enum` already
covers both unit variants (`PieceKind`, above) and payload variants, out of the gate, no separate
"sum type" construct needed.

```sc
enum Outcome { Win(Player), Draw }
```

## The game

Square brackets, not angle brackets, for template parameters/instantiation (`Tak[N]`, not `Tak<N>`
or Rust's `Tak::<N>` turbofish) -- Go's precedent for generic instantiation syntax. Worth recording,
not just applying: `template` here genuinely overlaps with both C++ templates and OCaml's parametric
modules/functors. OCaml's module-as-abstraction feels like the right semantic target longer-term --
imports/namespacing and parametric templating living in one unified world the way OCaml's
`module`/functor system unifies them -- but OCaml's own collision between "module the
filesystem/compilation unit" and "module the PLT abstraction" makes it a genuinely alien surface to
non-PLT readers, which is exactly why `module` is deliberately avoided as a keyword here even though
`template` is doing module-shaped work. Flagged as a real future direction (this project will need
imports/namespacing eventually, and it'd be nice if that concept and `template`'s parametrization
turned out to be the same idea under the hood), explicitly not a day-1 requirement -- nothing below
depends on it.

```sc
template game "Tak"[const N: Int] {
  topology = Raster { rows: N, cols: N, cell_bits: stack_bits(N) }
  players  = 2
```

### Per-player state

New notation, but ours: an indexed-state declaration binder, extending this grammar's existing
`state name: Type = init` form with an index rather than reaching for Rust's array type (`[Int;
players]`) or Alloy's relation-plus-domain-restriction idiom (`Player -> one Int` / `Player <:
(...)`). Shaped for exactly one recurring need -- one value per member of a small enumerable domain
-- not a general container or relation type.

```sc
  state reserve[p: Player]: Int = piece_reserve(N)
  state caps[p: Player]:    Int = capstone_reserve(N)
```

### Placing

`guard`, not `if ... then { }`: an earlier round's `move` syntax used `if`/`then` for a legality
filter that never had, or needed, an `else` -- a bare, valueless "if" that only reads correctly once
you already know moves without a satisfied condition simply aren't generated. That's a footgun
(`if`/`then` looking optional-else everywhere invites writing a real value-producing conditional the
same way and forgetting the `else`), and it was also a category error: a precondition and a value
expression aren't the same kind of thing and shouldn't share a keyword. Fixed by giving
preconditions their own keyword, `guard`, and making the rule project-wide: **`if` is always an
expression and always requires `else`, full stop, no exceptions anywhere in this grammar.** `guard`
is boolean-only, produces no value, and is exactly where a bare, no-`else` conditional belongs. A
single condition can sit inline after `guard`; multiple conditions list one per line beneath it,
implicitly conjoined (AND) the way a Haskell guard list or Prolog clause body already reads -- no
`&&` needed to chain them, though an individual line can still use `||`/`&&` internally (see `Place`
below).

```sc
  move PlaceOpeningFlat(s: Site) to sites(Empty)
    guard turn < players

    board' = push(board, s, (Flat, opponent(mover)))

  move Place(kind: PieceKind, s: Site) to sites(Empty)
    guard
      turn >= players
      (kind != Capstone || caps[mover] > 0)
      (kind == Capstone || reserve[mover] > 0)

    board' = push(board, s, (kind, mover))
    caps'[mover]    = if kind == Capstone then caps[mover] - 1 else caps[mover]
    reserve'[mover] = if kind == Capstone then reserve[mover] else reserve[mover] - 1
```

The effect body is a set of `field' = expr` bindings (the semantic borrow from the temporal layer's
`state'`, applied consistently), not a statement sequence. The separator is a blank line between
`guard` and the bindings, and a newline between bindings -- no `then { }` wrapper and no semicolons,
since `guard` already delimits where the precondition list ends. `push`/`pop`/`set` stay ordinary
pure functions returning a new `Region`/`Raster` value.

### Spreading

```sc
  move Spread(from: Site, dir: Direction, drops: [Int]) to sites(Occupied(mover))
    guard
      top(board, from).owner == mover
      sum(drops) <= min(N, height(board, from))
      legal_spread_path(from, dir, drops)

    board' = apply_spread(board, from, dir, drops)
```

`fold`, not `fixpoint` (unchanged verdict from round 2): `fixpoint` promises *convergence* semantics
(repeat until no more change, `max_iters` only as a safety valve) -- exactly what
`05-havannah-cycle.sc`'s cycle check legitimately needs, since it can't know in advance how many
steps until the visited set stabilizes. Tak's spread has a statically known length before the walk
starts, so there's no convergence question, just "apply one step per element of an already-bounded
sequence, threading an accumulator" -- an ordinary fold.

**Round 6, superseding round 4's fix:** `fold` was still a bespoke block-header special form (`fold
out = b for i in ... { ... out' = ... }`), not an ordinary call -- and that, not just the missing
primed accumulator round 4 patched, was the real complaint raised in live review: a block-header
special form has no argument position a pre-existing `def` of the right shape could ever be plugged
into, unlike `any`/`all`/`project` below, which already take their predicate as an ordinary lambda
argument (`|v| ...`). Fixed by making `fold` an ordinary call, `fold(seed, iterable, step)`, with
`step` an inline lambda in exactly that same `|params| body` shape -- the same "second-class lambda"
pattern `any`/`all`/`project`/`bounded_fixpoint`'s `step` already use throughout this grammar (see
`DESIGN.md`'s "Control and aggregation" section): lexically scoped, immediately applied at its one
call site, never stored in `state` or returned from a `def`, so it compiles away by ordinary
inlining rather than needing a real runtime closure -- consistent with this project's "not full
lambda calculus" principle, not an exception to it. Once `fold` matches that shape, round 4's `out'
= ...` priming turns out to have been curing a symptom: an ordinary lambda's trailing expression is
unambiguously its return value the same way it already is for `any`/`all`/`project`'s lambdas, so no
special primed-name convention is needed once `fold` stops being a special form -- one fewer idiom
to remember, not a new one. (`fixpoint`'s own header/body syntax in `05-havannah-cycle.sc` likely
has the identical problem; still an open followup, not assumed fixed by analogy.)

`shift(from, dir, i + 1)` was renamed to `walk` (see `DESIGN.md`'s "Standard library" section): it
was silently overloading Region algebra's own `shift(dir): Region -> Region` (which shifts a whole
region one step) with an unrelated Site-to-Site stepping operation that never had its own name. Same
per-topology adjacency knowledge, different operand/result type -- worth a distinct name rather than
an overload.

```sc
  def apply_spread(b: Raster, from: Site, dir: Direction, drops: [Int]): Raster =
      let carried = pop(b, from, sum(drops)) in
      fold(b, 0..len(drops), |out, i| {
        let dest = walk(from, dir, i + 1) in
        let base = if top(out, dest).kind == Wall
                   then set(out, dest, (Flat, top(out, dest).owner))
                   else out
        in
        push(base, dest, take(carried, drops[i], from: i))
      })
```

`0..len(drops)`, not `range(0, len(drops))`: sugar for the same `range` primitive. `range` stays the
underlying primitive (`fold`/`all`/`any` all still take an ordinary iterable expression); `a..b` is
purely a spelling of `range(a, b)`.

The wall-flattening guard's last-drop check no longer calls a `carried_top_is` helper -- that name
was never defined anywhere in this file, a dangling reference rather than a design question (see
`DESIGN.md`'s "Standard library" section). Tak's pickup/drop order means the single piece landing on
the last cell is always the original stack's top piece, which `top(board, from)` already answers
directly with no helper needed.

```sc
  def legal_spread_path(from: Site, dir: Direction, drops: [Int]): Bool =
      all(0..len(drops), |i| {
        let dest = walk(from, dir, i + 1) in
        let is_last = i == len(drops) - 1 in
        top(board, dest).kind != Capstone
          && (top(board, dest).kind != Wall
              || (is_last && drops[i] == 1 && top(board, from).kind == Capstone))
      })
```

### Road win

Disjunctive `connects`, unchanged from `games/tak.sc` -- this was never Rust-flavored notation,
nothing to revise.

```sc
  def road_region(p: Player): Region =
      project(board, |v| v.owner == p && v.kind != Wall)

  def has_road(p: Player): Bool =
      connects(road_region(p), side(North), side(South))
      || connects(road_region(p), side(West), side(East))

  def out_of_pieces(p: Player): Bool = reserve[p] == 0 && caps[p] == 0
```

`count_where`, unchanged in name from `games/tak.sc` -- the discarded Alloy draft's insight (that
"count where" is a special case of a more general aggregation-over-a-predicate idea, not a bespoke
combinator) still stands even though its notation (`#{s: Site | ...}`) didn't survive the
correction. It's since been promoted from an open notation question to a real Core primitive (see
the Core/Stdlib/Extern audit in the revision history below).

```sc
  def flat_count(p: Player): Int =
      count_where(board, |v| v.owner == p && v.kind == Flat)

  terminal: Bool =
      has_road(mover) || out_of_pieces(P0) || out_of_pieces(P1) || is_full(board)
```

Guard-arm sugar over `if`/`then`/`else`-`if`: a leading-`|` list of `condition -> value` arms, first
match wins, `otherwise` required as the catch-all. Purely sugar -- desugars directly to `if
has_road(mover) then Win(mover) else if ... else Draw` -- kept alongside plain `if`/`then`/`else`
rather than replacing it (see `apply_spread` above, which uses a plain binary `if`/`else` because
there's no cascading chain to flatten there). The two forms aren't in tension: `if`/`then`/`else` is
the honest primitive, this is sugar for exactly the shape a scoring/outcome function usually has --
a priority-ordered list of mutually exclusive conditions ending in a default.

```sc
  outcome: Outcome =
    | has_road(mover)                 -> Win(mover)
    | flat_count(P0) > flat_count(P1) -> Win(P0)
    | flat_count(P1) > flat_count(P0) -> Win(P1)
    | otherwise                       -> Draw
}
```

## Instantiation

A concrete instantiation still needs a fixed board size, the same as any other template call site:

```sc
game "Tak5" = Tak[5]
```

## Revision history

Scorecard against `games/tak.sc`'s five original findings, after six rounds of live syntax review:

- **Const/template generics** survive, now spelled with square brackets per Go's precedent rather
  than angle brackets (`template game "Tak"[const N: Int]`).
- **Per-player indexed state** dissolves as a *gap*, but keeps its own genuinely new domain-native
  notation -- `state x[p: Player]: T`, neither Rust arrays nor an Alloy relation.
- **Named composable effect logic** dissolves entirely -- ordinary `def`, no new construct needed.
- **Disjunctive `connects`** was never a syntax problem.
- **`count_where`** was an open notation question for four rounds; resolved by round 5's audit (see
  below) by promoting it to a real Core primitive.

Findings from later rounds, none anticipated by the original five:

1. **`fixpoint`/`fold` split.** `fixpoint` was quietly covering two different bounded-iteration
   shapes (genuine convergence vs. a deterministic known-length walk) and needed to split in two.
2. **`const fn` dropped.** Unnecessary ceremony once const-ness is inferred from call-site argument
   constness instead of declared on the function.
3. **`guard` introduced.** `move`'s `if COND then { effect }` was conflating a valueless
   precondition with a value-producing conditional under one keyword; fixed by giving preconditions
   their own `guard` keyword and making `if`/`then`/`else` total everywhere, no exceptions.
4. **`fold`'s body gained explicit primed-accumulator convention** (`out' = ...`), superseded by
   round 6 below once `fold` became an ordinary call -- the priming turned out to be unnecessary,
   not just re-spelled -- plus `a..b` sugar for `range(a, b)`.
5. **Core/Stdlib/Extern builtin audit** (`DESIGN.md`'s "Standard library" section): `rule` renamed
   to `def` project-wide; `shift(site, dir, n)` renamed to `walk` to stop overloading Region
   algebra's `shift`; `pop` gained a documented 3-arg overload (`Stack<Value>`, not `Raster`);
   `is_full`/`count_where`/`opponent`/`min`/`max`/`sum`/`len`/`range` all formalized as Core
   primitives; `mover`/`len` confirmed as this project's canonical names over
   `current_player`/`to_move`/`length`; and `carried_top_is`, never defined anywhere in this file,
   turned out to be an outright dangling reference rather than a design question -- fixed by
   inlining the check it was standing in for (`top(board, from).kind == Capstone`), no helper
   needed at all.
6. **`fold` becomes an ordinary combinator call.** Live-review pushback on `fold`'s block-header
   special form (`fold out = b for i in ... { ... }`): it read as unwarranted Alloy-adjacent sugar,
   and concretely, it had no argument position a pre-existing `def` could be passed into the way one
   already can be to `any`/`all`/`project`. Fixed by making `fold(seed, iterable, step)` an ordinary
   call taking `step` as an inline `|acc, elem| body` lambda -- the same lexically-scoped,
   immediately-applied "second-class lambda" `any`/`all`/`project`/`bounded_fixpoint` already use,
   formalized in `DESIGN.md`'s "Control and aggregation" section. This also retires round 4's `out'
   = ...` primed-accumulator convention: an ordinary lambda's trailing expression already is its
   return value, the same as every other lambda in this grammar, so the special priming was patching
   a symptom of `fold` not being a real call, not a genuine notational gap.
