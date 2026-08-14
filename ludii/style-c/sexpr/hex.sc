// Hex, in style_c's direct s-expression encoding of Core IR (see src/style_c/mod.rs). A
// load-bearing fixture (include_str!'d by src/style_c/mod.rs's tests), checked against the same
// Program the existing .lud pipeline lowers lud/Hex.lud to -- not a new game, a second concrete
// syntax for one already-proven Core program.
//
// Compass points name a rhombus edge the way lud/Hex.lud's own (sites Side <compassDirection>)
// does -- see core::hex::Hex's doc comment for why NE/SE/SW/NW rather than the cardinal points.

(game "Hex"
  (topology (hex 3))
  (players 2)
  (moves (sites Empty))
  (end (connects Six))
  (regions 0 (side NE) (side SW))
  (regions 1 (side NW) (side SE)))
