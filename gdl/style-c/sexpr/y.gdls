// Y, in style_c's direct s-expression encoding of Core IR (see src/style_c/mod.rs). Concretized
// to a fixed side-4 triangular board -- database-1/lud/games/Y.lud's real source is fully
// option-templated (board size 3-19, standard/misere end rules); this fixes both, the same
// treatment lud/Hex.lud's own session gave Hex's board size and swap rule, since option/template
// resolution is out of scope for this project's frontends (see DESIGN.md's "Translating .lud").
//
// Unlike Hex.lud (edge-to-edge, one player per pair of opposite sides), Y.lud's win condition is
// "(is Connected 3 Sides)": a single connected group of the mover's own stones touching all three
// board sides, the same three sides for every player. That's the forcing case for generalizing
// core::Program.player_regions/BoolExpr::Connects from a fixed (Region, Region) pair to an
// arbitrary-length list -- see core::mod's doc comments on both. Both players' (regions ...)
// clauses below are consequently identical lists, not a per-player pair the way Hex's are.
//
// The triangular board itself needs no new coordinate packing or backend -- see
// core::hex::Hex's module doc: a Hex { Triangle } board is the same side x side grid and
// six-way adjacency a Hex { Rhombus } board already uses, restricted to the upper-left triangular
// half (row + col < side). The only new Region-algebra primitive this forces is `intersect`
// (Region::Intersect), needed so "(sites Empty)" means "empty AND inside the triangle" rather
// than "empty" over the full side x side grid.

(game "Y"
  (topology (hex_triangle 4))
  (players 2)
  (moves (sites Empty))
  (end (connects Six))
  (regions 0 (tri_side Bottom) (tri_side Left) (tri_side Hypotenuse))
  (regions 1 (tri_side Bottom) (tri_side Left) (tri_side Hypotenuse)))
