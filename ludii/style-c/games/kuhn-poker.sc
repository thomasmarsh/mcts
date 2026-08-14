// Kuhn poker -- the canonical minimal imperfect-information game (3-card deck, 1 card each,
// single betting round), standard test case in the extensive-form-game/CFR/ISMCTS literature.
// No .lud source: card games with hidden information sit entirely outside this repo's `lud/`
// corpus and, more importantly, outside Core IR's foundational assumption that `state` is fully
// observed by the engine and by both players alike (every earlier case, including Tak's
// per-player `reserve`/`caps`, was public state -- visible to both players, just indexed by
// one). Written pro forma per the same license as `games/tak.sc`: not required to parse against
// the grammar or lower to any existing `core::Program` shape. This is the "card game" pathological
// case; findings flagged inline, cross-referenced in `style-c/README.md`.
//
// --- Syntax refresh (per README.md's "Next session charter: refresh games/kuhn-poker.sc"),
// bringing this file from its original pre-review syntax (bare `if COND then { }` moves, Rust
// array-typed `[Int; players]` state, an unlabeled implicit `Outcome`) up to what `games/tak.md`
// already has: `guard`, primed `field'` effect bindings, `state x[p: Player]: T` indexed-state
// declarations, and an explicit `enum Outcome`. Picked as the second file to refresh (over
// `sprouts.sc`/`sylver-coinage.sc`/`ghost.sc`) because it forces two things no other file could:
// a real `extern def` call (this file's `shuffle`/`draw2` are the corpus's only genuinely
// nondeterministic builtins -- see DESIGN.md's "Standard library" section), and private,
// epistemically-scoped state surviving the `guard`/primed-field conventions. Both are addressed
// below; neither forced a change to the conventions themselves -- they already generalized clean.

// --- NEW: a base value type that isn't Region/Raster/Int/Bool/Set. `Card` needs a total order
// (J < Q < K) for showdown comparison -- neither `Card` nor an ordinal `>` over `enum` variants
// has any precedent in DESIGN.md's value-type table, which was built entirely from spatial games.
// Still open, not resolved by this refresh: whether `enum` variants get ordinal comparison for
// free from declaration order (as used below), or whether that needs an explicit `def
// card_rank(c: Card): Int` lookup instead -- flagged, not decided, same as the original file left
// it.
enum Card { Jack, Queen, King }  // ordered: Jack < Queen < King

// --- `extern def`: this project's first live use (see DESIGN.md's "Standard library" section).
// `shuffle`/`draw2` are genuinely nondeterministic, not just externally-supplied-but-pure the way
// `sprouts.sc`'s `geometric_oracle`/`ghost.sc`'s `dictionary_has_prefix` are -- flagged there as
// an open question (does `extern def` need its own determinism/purity tag?), still open here too:
// nothing about the `extern def` declaration below distinguishes "opaque but deterministic" from
// "opaque and randomized," and this file doesn't resolve that, it just confirms the question is
// real by being the one call site that needs it.
extern def shuffle(deck: [Card]): [Card]
extern def draw2(deck: [Card]): (Card, Card)

// No `Draw` variant -- Kuhn's 3-card deck has no ties. Declared explicitly, closing the same gap
// `games/tak.md`'s own `enum Outcome` declaration closed: `Outcome` was already used with
// payload-carrying constructors here without ever being declared.
enum Outcome { Win(Player) }

game "KuhnPoker" {
  topology = None   // --- NEW: see games/sylver-coinage.sc for the fuller version of this finding;
                     // noted here too since it's Kuhn's first appearance, not Sylver's.
  players  = 2

  state pot: Int = 0

  // --- Indexed state, Tak's `state x[p: Player]: T = init` form (was the Rust-array-typed
  // `[Int; players]` before this refresh -- that notation was already retired everywhere else).
  state committed[p: Player]: Int = 1   // both players ante 1 before dealing

  // --- Private, per-player state. Tak's `reserve[mover]`/`caps[mover]` was per-player *indexed*
  // but fully public -- either player's `guard` could read either index. Kuhn's dealt card is
  // per-player *and* epistemically scoped: player p's own `move`/`def` declarations may reference
  // `private[p]`, but never `private[q]` for `q != p` -- a *static* scoping rule with no earlier
  // precedent (nothing before this needed to distinguish "state that exists" from "state this
  // particular def is allowed to read"). `outcome`, evaluated by the engine rather than by either
  // player, is exempt from the restriction: showdown is exactly the moment private state becomes
  // public, so `outcome` below reads both `private[P0]` and `private[P1]` freely. This survives
  // the `guard`/primed-field conventions with no changes needed to either -- scoping is a
  // read-access rule the syntax refresh didn't need to touch. Also the first indexed `state` with
  // no initializer (the hardened grammar's `StateDecl` already makes `= Expr` optional; nothing
  // has a value before `Deal()` sets it).
  state private[p: Player]: Card

  // --- `chance Deal()`: a transition whose actor is neither player, has no `guard` (chance moves
  // are never illegal), and whose effect is drawn from a distribution rather than chosen.
  // Distinguished from an ordinary `move` by keyword, not by a `guard turn == ...` the way Tak's
  // opening-flat wrinkle was -- a chance move isn't *any* player's decision to make, so folding it
  // into `move`'s player-indexed shape would be a category error, not just an awkward encoding.
  //
  // The `let ... in` before the primed bindings is new: every earlier move's effect body was a
  // flat list of independent `field' = expr` bindings, each computable straight from pre-move
  // state, but both `private'[P0]` and `private'[P1]` here need to come from the *same* shuffle
  // (drawing twice independently would let the two hands overlap). `let` was already established
  // inside ordinary `def` bodies (see `apply_spread` in `games/tak.md`); this is the same
  // construct, just the first time a move/chance body has needed a shared local computation ahead
  // of its bindings rather than one independent expression per field.
  chance Deal()
    let dealt = draw2(shuffle([Jack, Queen, King])) in

    private'[P0] = dealt.0
    private'[P1] = dealt.1
    // the third card is never dealt and never observed by either player -- there is no state
    // field for it at all, not even a hidden one; Core IR has never before had a value that
    // exists in the game's real-world referent but has no representation anywhere in `state`.

  // Player-to-act alternates starting with P0; `Bet`/`Check` are only legal on the first decision
  // of a betting round, `Call`/`Fold` only in response to an outstanding bet -- `history`-style
  // reasoning again (compare case 3's `once`), but over a much shorter, fully bounded trace (this
  // game is at most 3 plies of betting), so no unbounded backend is needed here the way superko's
  // was.
  move Check()
    guard turn == 0 || (turn == 1 && !bet_outstanding())
    // no state change -- passes the turn

  move Bet()
    guard !bet_outstanding() && turn < 2

    pot'              = pot + 1
    committed'[mover] = committed[mover] + 1

  move Call()
    guard bet_outstanding()

    pot'              = pot + 1
    committed'[mover] = committed[mover] + 1

  move Fold()
    guard bet_outstanding()
    // no state change -- terminal is keyed off the fold itself, see below

  def bet_outstanding(): Bool = committed[P0] != committed[P1]

  // --- Terminal: three distinct shapes bundled into one game for the first time -- a fold ends
  // the hand immediately (no showdown), two checks end it at a showdown, and a bet-then-call ends
  // it at a showdown. Tak's `terminal` was a flat disjunction of independent conditions; Kuhn's
  // needs the *history* of which branch was taken (fold vs showdown) to determine not just *that*
  // the game ended but *how* to score it -- outcome depends on path, not just on the final state
  // snapshot, which every earlier case's `outcome` could get away with ignoring.
  terminal: Bool = folded() || turn == 2 || (turn == 1 && bet_outstanding())

  def folded(): Bool = /* true once a Fold has been played this hand -- needs the same
                            once()-over-committed-trace machinery as case 3, applied to a move
                            name rather than a board equality */ once(last_move() == Fold)

  // --- Switched to the guard-arm `|` sugar for consistency with `games/tak.md`'s `outcome` --
  // same priority-ordered-conditions-ending-in-a-default shape, even though every branch here is
  // actually mutually exclusive on its own terms rather than needing the priority ordering
  // `flat_count` comparisons did.
  outcome: Outcome =
    | folded()                  -> Win(opponent(mover))
    | private[P0] > private[P1] -> Win(P0)
    | otherwise                 -> Win(P1)
}
