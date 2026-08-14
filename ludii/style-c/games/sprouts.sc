// Sprouts -- pencil-and-paper topological game (Conway/Paterson). Start with n spots; a move
// draws a curve joining two spots (or a spot to itself), not crossing any existing curve or
// passing through any other spot, then places one new spot somewhere along the new curve. Each
// spot has 3 "lives" (curve-endpoints it can host); a spot with 0 lives left is dead. Last player
// able to move wins (normal play). No .lud source: Sprouts isn't a spatial game on a fixed grid
// at all -- it's the "graph game" pathological case, chosen over a fixed-graph game (e.g. Shannon
// Switching) specifically because its board *grows* and its legality is a *global topological*
// property, which stresses Core IR's Topology/Region split in a way a static arbitrary graph
// wouldn't. Pro forma, same license as `games/tak.sc`: not required to parse or lower.

game "Sprouts" {
  // --- NEW: a topology that is itself mutable game state, not a compile-time type parameter.
  // Every earlier case (including Tak's `template game<const N>`) picked topology size at
  // *instantiation* time and then held it fixed for the rest of the game -- DESIGN.md's "topology
  // as a type parameter" principle assumed exactly this split (topology: compile-time,
  // occupancy: runtime). Sprouts breaks the split at the root: the vertex and edge sets both grow
  // every single move, so there is no fixed `N` to parametrize a `template game` over the way
  // Tak's board size was. `Graph { dynamic: true }` is written here as a placeholder for "this
  // needs its own topology kind whose site set is a runtime-growing `state` field, not a
  // compile-time constant" -- an open question, not a resolved design.
  topology = Graph { dynamic: true, initial_nodes: 3 }
  players  = 2

  // A node's remaining lives (0..3). Like Sprouts's own node set, this Raster's *domain* grows
  // every move -- another break from every earlier Raster use (Tak's `board` was sized once, at
  // `stack_bits(N)` per cell, over a fixed `N x N` site set fixed at instantiation).
  state lives: Raster<Int> = initial_lives(3)   // three starting spots, 3 lives each

  // A move names two endpoints and a path between them through the plane; `path` is opaque here
  // (a sequence of intermediate points in some embedding) since Core IR has no representation for
  // planar curves at all -- there's no Region/Raster shape this fits into, unlike every previous
  // move's `to: Region`.
  move Connect(a: Site, b: Site, path: Curve, new_spot: Site)
    if is_alive(a) && lives[a] >= 1
       && is_alive(b) && lives[b] >= 1
       && (a != b || lives[a] >= 2)          // a self-loop uses 2 of the same spot's lives
       && no_crossing(path, drawn_curves)     // --- see below
    then {
      add_node(new_spot);
      add_edge(a, new_spot, first_half(path));
      add_edge(new_spot, b, second_half(path));
      set(lives[a], lives[a] - 1);
      set(lives[b], lives[b] - 1);
      set(lives[new_spot], 1);   // the new spot is born with 3 lives, 2 already spent on the split
      set(drawn_curves, insert(drawn_curves, path));
    }

  state drawn_curves: Set<Curve> = empty

  def is_alive(s: Site): Bool = lives[s] > 0

  // --- NEW: an "oracle" predicate -- a legality condition that isn't a Region/Raster algebra
  // expression at all but a call out to an external combinatorial check (here, planar-curve
  // non-crossing) that this project's Region algebra has no vocabulary for and no bound on cost
  // for. Contrast `games/ghost.sc`'s `is_prefix` (also an oracle, but a cheap bounded-alphabet
  // dictionary lookup) and `games/sylver-coinage.sc`'s `in_semigroup` (an oracle that *does*
  // reduce to an ordinary bounded_fixpoint) -- `no_crossing` is the expensive end of the same
  // pattern: verifying a new curve doesn't cross any of `drawn_curves` is a real computational-
  // geometry problem with no small static bound the way `bounded_fixpoint`'s `max_iters` always
  // had in every earlier case (Havannah's cycle check, Tak's spread carry limit).
  def no_crossing(path: Curve, existing: Set<Curve>): Bool = geometric_oracle(path, existing)

  // Terminal: no legal `Connect` move exists for *any* pair of alive spots and *any* embeddable
  // path -- an existential quantifier over an unbounded, continuous space of curves, not a finite
  // enumerable move list the way `sites(Empty)` was for Tic-Tac-Toe/Tak. Every earlier `terminal`
  // was a closed-form Bool expression over current state; this one is only well-defined relative
  // to "no witness exists," which Core IR has no native way to state short of literally trying
  // every candidate move, the same shape of problem `games/ghost.sc`'s dead-end check has, but
  // over a vastly larger candidate space.
  terminal: Bool = !exists_legal_move(Connect)

  // Normal play: the player who cannot move loses, i.e. the player who just moved (their
  // opponent, now stuck) wins -- outcome keyed off *whose turn it would be*, not off any board
  // predicate, structurally the same "outcome depends on whose turn got stuck" shape as
  // `games/sylver-coinage.sc`'s naming-1 loss and `games/ghost.sc`'s word-completion loss.
  outcome: Outcome = Win(opponent(to_move))
}
