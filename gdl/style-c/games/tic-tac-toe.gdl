// lud/Tic-Tac-Toe.lud -- full game, not a case fragment. The sanity-check target for the
// grammar: no `then`, `state`, `invariant`/`state'`/`once`, or templates needed at all, just the
// base declarative layer (`topology`/`players`/`moves`/`terminal`/`outcome`). Matches how small
// `core::Program` already is for this game (`Region` with `Occupied`/`Union`/`Complement`,
// `MoveGen { to: Region }`, `EndRule::Line`) -- this transcription is a strict superset of what
// that already-proven Core program needed, not a rewrite of it.

game "Tic-Tac-Toe" {
  topology = Rect { rows: 3, cols: 3 }
  players  = 2

  moves: Region = sites(Empty)

  terminal: Bool = has_line(occupied(mover), length: 3)
  outcome: Outcome = Win(mover)
}
