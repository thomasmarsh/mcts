// SUPERSEDED (surface syntax only, findings below still stand) by `games/tak-relational.sc` --
// this file's syntax turned out to be Rust with game nouns (`const fn`/`match`, `[T; N]` arrays,
// `Set<T>`/`Seq<T>`, imperative `then { push/set/for }` blocks) rather than anything targeted at
// this domain, per the top-level README.md's "Style C was leaking Rust" session note. A first
// revision attempt overcorrected into transliterating literal Alloy notation instead
// (`<:`/`~`/dot-join chains/`abstract sig`/`no`/`univ`) -- caught and discarded before landing,
// see that session note's follow-up. Left in place rather than rewritten in place or deleted, per
// this project's mark-don't-delete convention -- most of the "new grammar needed" findings below
// turn out to evaporate once expressed in this project's own relational vocabulary (see
// `games/tak-relational.sc`); only the board-size-as-template-parameter finding survives.
//
// Tak, board-size-parametrized (3x3 through 8x8). No .lud source is transcribed here -- Tak
// isn't in this repo's `lud/`/`database-1/` corpus at all; this is written directly from the
// published ruleset and this repo's existing hand-written `games/tak/src/lib.rs`, per the
// session request for a "pro forma design exploration," explicitly NOT required to parse
// against the grammar in the top-level README.md or lower to any existing `core::Program`
// shape. Tak needed real extensions the five pathological cases never forced -- const generics,
// per-player indexed state, composable named effect blocks, a disjunctive `connects`, and
// `count_where` -- each flagged inline where it first appears, the same way the five cases'
// own session flagged the gaps *they* found. Piece reserve is the whole point of the exercise:
//
//   N          3   4   5   6   7   8
//   normal    10  15  21  30  40  50
//   capstone   0   0   1   1   2   2
//
// a genuine function of board size, unlike Tic-Tac-Toe's win-length or Hex's edge pairs, which
// are per-game constants that don't vary with a topology parameter.

// --- NEW: const fn -- compile-time-only functions over template Int parameters, evaluated at
// monomorphization time (not Core expressions; there is no runtime `match` on `Int` anywhere
// else in this grammar). Distinct from `template rule<T: fn(...) -> ...>`'s existing
// function-kinded generics (see the five-case README): those substitute a *named rule/effect
// block*, chosen once per call site; this substitutes a *value*, chosen once per game
// instantiation. Both are "resolved before Core ever sees the body," so both stay inside the
// "first-order, not full lambda calculus" principle -- but the earlier generics writeup's
// "only fn(...)->... parameters need <...>" rule was incomplete: it implicitly assumed a
// template parameter is always a missing *function*, because nothing before Tak needed a
// missing *board-size-dependent constant*. `const N: Int` is the same monomorphization
// mechanism at a second kind.
const fn piece_reserve(n: Int): Int =
    match n { 3 => 10, 4 => 15, 5 => 21, 6 => 30, 7 => 40, 8 => 50 }

const fn capstone_reserve(n: Int): Int =
    match n { 3 => 0, 4 => 0, 5 => 1, 6 => 1, 7 => 2, 8 => 2 }

// Exact bit layout is a backend concern per DESIGN.md's "Raster ops"/"Backend lowering"
// sections -- this const fn just states that the packed-per-cell width is itself a function of
// N (bounded by max stack height, which is bounded by total pieces in play, which is bounded by
// N via the table above), not a magic literal the way `games/tak/src/lib.rs`'s real `cells:
// [u64; 36]` hardcodes today for the one size it supports.
const fn stack_bits(n: Int): Int = 2 * (piece_reserve(n) + capstone_reserve(n)) // room for owner+kind per level, loosely

enum PieceKind { Flat, Wall, Capstone }

// --- NEW: `template game`. The five-case README's `template rule<...>` monomorphizes a single
// rule body per call site; this applies the identical idea one level up, to an entire `game
// { }` block. Nothing about the monomorphization *story* changes -- it's still "specialize away
// before Core sees it, one independent copy per instantiation, no shared vtable to conflict
// across instantiations" (see the generics section of the main README) -- only the granularity
// does. A concrete board size is still required to get an actual compiled game (last line,
// below), the same way `chess_pawn::<...>(Pawn)` needed a call site.
template game "Tak"<const N: Int> {
  topology = Raster { rows: N, cols: N, cell_bits: stack_bits(N) }
  players  = 2

  // --- NEW: state fields indexed by player. The hardened grammar's `StateDecl` only had bare
  // scalar/Region/Raster/Set fields (case 3's `history` was global, not per-player) -- Tak's
  // reserve counts are the first case that needs one slot per player. `[Int; players]` is
  // written informally here; a real hardening pass would need to decide whether this is sugar
  // for two scalar fields, a first-class indexed-state-array type, or something else.
  state reserve: [Int; players] = [piece_reserve(N), piece_reserve(N)]
  state caps:    [Int; players] = [capstone_reserve(N), capstone_reserve(N)]

  // Tak's own opening wrinkle: the first stone each player places is a *flat*, of their
  // *opponent's* colour -- an ordinary use of the existing `turn` game-state combinator
  // (DESIGN.md's "Game-state combinators"), no new vocabulary needed. Included mostly to show
  // the existing `turn`-gated `if` guard composes fine alongside everything new below.
  move PlaceOpeningFlat(s: Site) to sites(Empty)
    if turn < players
    then { push(board, s, (Flat, opponent(mover))) }

  move Place(kind: PieceKind, s: Site) to sites(Empty)
    if turn >= players
       && (kind != Capstone || caps[mover] > 0)
       && (kind == Capstone || reserve[mover] > 0)
    then {
      push(board, s, (kind, mover))
      if kind == Capstone
        then { set(caps[mover], caps[mover] - 1) }
        else { set(reserve[mover], reserve[mover] - 1) }
    }

  // --- Spread move: pick up up to `N` stones from `from` (carry limit = board size, per Tak's
  // real rules -- `bounded_fixpoint`'s "max_iters is always a static bound derivable from board
  // size" principle, DESIGN.md's "Control and aggregation" section, confirmed by a second real
  // case beyond Congo's Monkey), and drop them one cell at a time along `dir`, `drops[i]`-many
  // per cell. `drops` itself is board-size-bounded (`len(drops) <= N`, `sum(drops) <= N`), so
  // this stays inside "statically bounded," same as every other fixpoint/loop in this grammar.
  //
  // The wall-flattening rule -- a capstone moving *alone* onto a standing stone flattens it,
  // any other drop onto a standing stone or a drop of any kind onto a capstone is illegal -- is
  // the pathological part: it's a piece-*kind*-dependent branch inside a bounded per-cell loop,
  // something none of the five earlier cases needed (their `then` blocks were straight-line, no
  // conditional effect depending on what's already at the destination).
  move Spread(from: Site, dir: Direction, drops: [Int]) to sites(Occupied(mover))
    if top(board, from).owner == mover
       && sum(drops) <= min(N, height(board, from))
       && legal_spread_path(from, dir, drops)
    then { apply_spread(from, dir, drops) }

  // --- NEW: named, composable effect blocks ("effect rule"), not just template-supplied ones.
  // The five-case grammar's `EffectStmt` allowed splicing in a *template parameter* that had
  // already been bound to a concrete effect block (case 4), but had no way to name and reuse an
  // effect block declared independently the way `rule` already does for pure `Bool`/`Region`
  // values. Once a move's effect got this complex, writing it inline in `then { }` stopped
  // being legible -- this is the forcing function for that gap, the same way case 5 forced
  // `fixpoint` to have an explicitly typed threaded-state signature instead of a bare `Region`.
  effect rule apply_spread(from: Site, dir: Direction, drops: [Int]) = {
    let carried = pop(board, from, sum(drops)) in
    for i in range(0, len(drops)) {
      let dest = shift(from, dir, i + 1) in
      if top(board, dest).kind == Wall {
        set(board, dest, (Flat, top(board, dest).owner))  // capstone flattens the wall in place
      }
      push(board, dest, take(carried, drops[i], from: i))
    }
  }

  rule legal_spread_path(from: Site, dir: Direction, drops: [Int]): Bool =
    all(range(0, len(drops)), |i| {
      let dest = shift(from, dir, i + 1) in
      let is_last = i == len(drops) - 1 in
      top(board, dest).kind != Capstone
        && (top(board, dest).kind != Wall
            || (is_last && drops[i] == 1 && carried_top_is(from, drops, i, Capstone)))
    })

  // Road win: a connected group of the mover's own road-eligible pieces (flats + capstone tops,
  // never standing stones) touching *either* opposite edge pair -- unlike Hex's single
  // `(Region, Region)` pair, Tak's road can complete in either orientation, so this is the first
  // case that needs `connects` used disjunctively rather than as a single fixed pair. `project`
  // is exactly DESIGN.md's Raster-ops sketch's own worked example ("Tak's road connectivity is
  // computed by flood-filling a Region derived from project(cells, |v| v.owner == player)") --
  // this transcription is the first time that claim gets checked against an actual move/effect
  // layer around it, not just cited as a target.
  rule road_region(p: Player): Region =
      project(board, |v| v.owner == p && v.kind != Wall)

  rule has_road(p: Player): Bool =
      connects(road_region(p), side(North), side(South))
      || connects(road_region(p), side(West), side(East))

  // Flat win / draw: triggers when either player's full reserve (normal + capstone) is
  // exhausted, or the board fills up with no road completed -- majority flat-top count wins,
  // equal counts draw. `count_where` doesn't exist in the "Control and aggregation" combinator
  // list DESIGN.md/the hardened grammar settled on (only `any`/`all`/`for_each` are there) --
  // it's only ever been *mentioned* aspirationally, in DESIGN.md's "Move caching" section, as an
  // example of what a backend might incrementalize, never actually required by a game. This is
  // the forcing function that promotes it from aspirational to load-bearing, per DESIGN.md's own
  // "grow the combinator set from real lowerings" principle -- and it's exactly the "Scoring/
  // payoff aggregation" open problem DESIGN.md already flagged as undesigned, now with a
  // concrete second corpus game (beyond Tanbo's territory count) forcing the same gap.
  rule out_of_pieces(p: Player): Bool = reserve[p] == 0 && caps[p] == 0

  rule flat_count(p: Player): Int =
      count_where(board, |v| v.owner == p && v.kind == Flat)

  terminal: Bool =
      has_road(mover) || out_of_pieces(P0) || out_of_pieces(P1) || is_full(board)

  outcome: Outcome =
      if has_road(mover) then Win(mover)
      else if flat_count(P0) > flat_count(P1) then Win(P0)
      else if flat_count(P1) > flat_count(P0) then Win(P1)
      else Draw
}

// A concrete instantiation actually needs a fixed board size, same as any other template call
// site -- "Tak" alone isn't a compiled game any more than `chess_pawn<...>` alone is a move.
game "Tak5" = Tak::<5>
