// Tic-Tac-Toe, in style_c's direct s-expression encoding of Core IR (see src/style_c/mod.rs).
// A load-bearing fixture (include_str!'d by src/style_c/mod.rs's tests), checked against the
// same Program the existing .lud pipeline lowers lud/Tic-Tac-Toe.lud to -- not a new game, a
// second concrete syntax for one already-proven Core program.

(game "Tic-Tac-Toe"
  (topology (rect 3 3))
  (players 2)
  (moves (sites Empty))
  (end (has_line 3)))
