// Go.lud -- (meta (no Repeat)), positional superko: a move is illegal if it recreates any
// board position that has occurred at any earlier point in the same game.
//
// Demonstrates: Alloy's past-temporal operator `once` (past-eventually: "held at some point
// up to and including now") applied to a primed reference, instead of a bespoke `visited`
// history builtin or an author-declared `state history: Set<Hash>` field with a manual
// `insert` in a `then` block (both of which the previous session's design needed). `once`
// ranges only over the committed trace so far -- the hypothetical `state'` hasn't been added
// to it yet -- so this is correct with no separate "don't let a move see its own effect on
// history" carve-out to reason about; that used to be a bespoke rule this project had to state
// and justify, and now falls out of `once`'s ordinary trace semantics for free. See this
// directory's README for the backend story (still the same Set<Hash>-of-Zobrist-hashes sketch,
// now motivated as the general lowering for `once` over board-shaped values rather than a
// per-game special case).

invariant: always !once(board = state'.board)
