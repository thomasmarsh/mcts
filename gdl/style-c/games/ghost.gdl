// Ghost -- players alternately append one letter to a shared fragment; a player who completes a
// real word (length >= min_length) loses, and a letter that would make the fragment not a prefix
// of any dictionary word is simply illegal to play (the classic variant also allows challenging a
// claimed-phony fragment; omitted here as an orthogonal complication, not a needed one for this
// case's purpose). No .lud source, no board, no topology, no numeric state -- the "word game"
// pathological case: state is a growing symbolic sequence, and legality/termination both depend
// on an external dictionary Core IR's value-type table has no representation for. Pro forma, same
// license as `games/tak.sc`.

game "Ghost" {
  topology = None
  players  = 2

  // --- NEW: a growing, ordered sequence-of-symbols state. Every earlier `state` field was
  // Region/Raster/Int/Bool/Set-typed (or, this batch, `Card`/`Graph`-typed) -- none was an
  // *ordered* structure where position matters and length is unbounded. `Seq<Letter>` is written
  // here as a placeholder for a Core value kind this project has never needed before; note it's
  // structurally close to `games/sylver-coinage.sc`'s `Set<Int>` (both unbounded, both grow by
  // one element per move) but a `Set` has no order and `member`/`insert` are its only real
  // operations, where `Seq` needs `append`/indexing/a notion of "prefix of."
  state fragment: Seq<Letter> = empty

  move Add(c: Letter)
    if is_prefix(append(fragment, c))
    then { set(fragment, append(fragment, c)) }

  // --- The middle of this batch's oracle-cost spectrum: cheaper than `games/sprouts.sc`'s
  // `no_crossing` (a fixed 26-letter alphabet bounds the branching factor at every step, so
  // enumerating "does some completion exist" is at worst a bounded trie walk, not an open-ended
  // geometric search), but not a `bounded_fixpoint` instance the way
  // `games/sylver-coinage.sc`'s `in_semigroup` turned out to be -- there's no algebraic reduction
  // of "is this a prefix of an English word" to Region/Raster/Set operations; it's an opaque
  // lookup against externally supplied data (the dictionary) that Core IR would have to treat as
  // a black-box, compile-time-constant oracle, the same status `no_crossing`'s planar-geometry
  // routine would need.
  def is_prefix(s: Seq<Letter>): Bool = dictionary_has_prefix(s)
  def is_word(s: Seq<Letter>):   Bool = dictionary_has_word(s) && length(s) >= min_length

  const min_length: Int = 4

  // --- Same "legal-but-suicidal move" shape as `games/sylver-coinage.sc`'s naming-1: completing
  // a real word is always a *legal* `Add` (it's still a valid prefix -- of itself), but it ends
  // the game and loses for whoever played it. Two independent games in this batch landing on the
  // identical outcome shape (`Lose(mover)` rather than every earlier case's `Win(mover)`) is
  // itself worth flagging as a corpus-level finding, not just a per-file one -- see
  // style-c/README.md.
  terminal: Bool = is_word(fragment) || !exists_legal_move(Add)
  outcome: Outcome =
      if is_word(fragment) then Win(opponent(mover))   // mover just completed a word: they lose
      else Win(mover)   // opponent is stuck with no legal letter to add: they lose instead
}
