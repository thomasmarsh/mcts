// Chess.lud:126-137 -- ("ChessPawn" "Pawn" (or "InitialPawnMove" "EnPassant") (then (and
// ("ReplayInMovingOn" (sites Mover "Promotion")) (set Counter)))), the piece-template
// macro-composition case: a template invoked once per piece kind, taking extra move
// alternatives and an effectful tail as compile-time parameters.
//
// Not a temporal-semantics case -- templates and compile-time generics are orthogonal to
// state'/always/once. Included here for completeness, since the five design-spike cases are
// meant to be read as one set. Caveat carried over from the design spike: `ChessPawn` itself is
// a known Ludii `define` whose real body isn't sourced in this repo (see DESIGN.md's
// "Translating `.lud`" section) -- what's demonstrated here is the *shape* of template
// parametrization the call site forces, not a claim about `ChessPawn`'s exact body.
//
// --- Syntax refresh (see HISTORY.md's Core/Stdlib/Extern and `rule`-to-`def` session notes):
// square brackets for template params/instantiation (`chess_pawn[...]`, not `chess_pawn<...>`/
// the `::<...>` turbofish), matching the `Tak[N]` precedent; `move`'s effect body is a `field' =
// expr` binding, not a `then { statement; statement }` block. `Tail`'s kind changes from
// `fn() -> EffectBlock` (a spliceable statement list -- the pre-refresh effect-block shape) to
// `fn(Region) -> Region` (an ordinary pure function over the post-move board), matching how
// every other effect in this grammar is now an ordinary value-returning function rather than a
// mutation sequence (see `games/tak.md`'s `apply_spread`).
template def chess_pawn[Extra: fn() -> Region, Tail: fn(Region) -> Region](piece: Site): Region =
    step_forward_to_empty(piece) | diagonal_capture(piece) | Extra()

move PawnMove(piece: Site, to: Site) to chess_pawn[InitialPawnMoveOrEnPassant, ReplayThenSetCounter](piece)

  board' = ReplayThenSetCounter(move_piece(board, piece, to))
