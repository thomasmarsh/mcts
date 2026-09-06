// Havannah.lud:13 -- (is Loop), a stone group encircling >= 1 cell.
//
// `has_cycle` is a Core primitive (`DESIGN.md`'s Region algebra table), not authoring-surface
// code a game writes out by hand. The real Havannah rule, in Style C, is just a call:
//
//   has_cycle(stone_group)
//
// Deliberate scope decision, not an unresolved gap: it's expected and fine for Core/the backend
// to carry a broad, hand-written instruction set over bitboards/hexboards -- `has_cycle` joins
// `flood`/`connects`/`adjacent`/`shift` as one more primop with its own backend lowering, the same
// way a CPU's ISA has instructions no compiler derives from smaller ones in ordinary user code. So
// the grammar does *not* need a general tuple-destructuring-bind or multi-value fixpoint-threading
// construct just to make this one case self-hosting in-language -- that generalization is deferred
// indefinitely, not merely "not yet designed": once `has_cycle` is accepted as a primitive rather
// than something every game must be able to re-derive from `fixpoint`, there's no forcing case
// left for it.
//
// What follows is therefore a *reference definition*, not Style C source: it documents the
// semantics a correct backend `has_cycle` lowering must agree with (the same role
// `tests/hex_oracle.rs`'s hand-rolled BFS oracle plays for `flood6`), and is exempt from the
// grammar's usual rules (declarative field-transitions, no mutation statements, `guard` for
// preconditions) -- it was never meant to parse, only to pin down intended behavior precisely
// enough to check an implementation against. Left in its original pre-round-6 notation (`fixpoint`
// block header, `for`/`:=` mutation) rather than chasing every later syntax round, since a
// reference definition has no obligation to track the authoring grammar's own churn -- but note
// that `fixpoint`, specifically, is not stale the way the block-header/mutation notation around it
// is: unlike Tak's spread (round 6 moved that to `fold`, since `drops` has a statically known
// length before the walk starts -- no convergence question, just one step per element of an
// already-bounded sequence), this walk's frontier grows adjacency-step by adjacency-step with no
// known-in-advance step count, stopping only when `visited` stops growing (`max_iters` purely a
// safety valve) -- exactly the genuine least-fixed-point shape `fixpoint` was carved out of `fold`
// to cover in round 2, which named this file's cycle check as the motivating case. Even a full
// syntax-refresh pass would keep `fixpoint` here, not convert it to `fold`. `visited`
// here is an ordinary local fixpoint accumulator (the flood-fill's frontier so far), not the
// trace-history sense 01-03's `once`/`state'` vocabulary uses -- there is no shared primitive
// between the two, just a name that would coincidentally clash if this project ever promotes
// trace-`once`-membership to a same-named builtin; flagged here so a future grammar pass doesn't
// silently conflate them.

def has_cycle_reference(group: Region): Bool =
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
