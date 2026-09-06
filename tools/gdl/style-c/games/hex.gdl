// lud/Hex.lud -- full game, not a case fragment (fixed 3x3 board, no swap rule, standard win,
// same concretization treatment lud/Hex.lud itself already got -- option/template resolution is
// out of scope for this surface, same as for elaborate/). Same base declarative layer as
// tic-tac-toe.sc, plus named board-edge `regions` and `connects` in place of `has_line` --
// still no `then`/`state`/`invariant`/templates needed. `connects` is the general two-`Edge`
// combinator DESIGN.md's Region-algebra table already lists (`connects(edge_a, edge_b:
// Edge): Region -> Bool`), written against that intended combinator set rather than against
// `core::EndRule::Connected`'s current dedicated-variant implementation status (already flagged
// as due for unification in DESIGN.md's "Already covered" table; unaffected by this file).

game "Hex" {
  topology = Hex { shape: Rhombus { side: 3 } }
  players  = 2

  regions P1 = (side(NE), side(SW))
  regions P2 = (side(NW), side(SE))

  moves: Region = sites(Empty)

  terminal: Bool = connects(occupied(mover), regions(mover).0, regions(mover).1)
  outcome: Outcome = Win(mover)
}
