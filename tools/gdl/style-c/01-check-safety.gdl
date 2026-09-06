// Chess.lud:166 -- (not (IsInCheck "King" Mover)), the check-safety filter that applies to
// every move a player might make.
//
// Demonstrates: `state'` (Alloy-style primed next-state reference) plus a top-level
// `invariant: always` declaration, replacing the earlier `ifAfterwards:` per-move guard
// keyword. "The mover is never left in check by their own move" is a genuine standing game
// invariant, not a special case attached move-by-move -- see 02-suicide-rule.sc, which is the
// same construct applied to a different game, and this directory's README for why stating it
// as `invariant: always` (rather than per-move `ifAfterwards`) is what makes that visible.

invariant: always !is_in_check(state'.occupied(King, mover), mover)
