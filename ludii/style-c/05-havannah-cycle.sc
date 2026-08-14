// Havannah.lud:13 -- (is Loop), a stone group encircling >= 1 cell.
//
// Not a temporal-semantics case -- the bounded fixpoint construct (feedback within a single
// hypothetical evaluation, not a claim about the game's played-so-far trace) is unrelated to
// state'/always/once. Included for completeness alongside the other four cases. `visited` here
// is an ordinary local fixpoint accumulator (the flood-fill's frontier so far), not the
// trace-history sense any of 01-03's `once`/`state'` vocabulary uses -- there is no shared
// primitive between the two, just a name that would coincidentally clash if this project ever
// promotes trace-`once`-membership to a same-named builtin; flagged here so a future grammar
// pass doesn't silently conflate them.

def has_cycle(group: Region): Bool =
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
