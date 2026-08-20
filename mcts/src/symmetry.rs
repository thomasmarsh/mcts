//! Symmetry-element bookkeeping shared by any search that canonicalizes
//! game states -- today MCTS (`strategies::mcts::node::real_action`,
//! `strategies::mcts::stack::NodeStack::incoming_syms`), and negamax's
//! symmetry-aware transposition table. Nothing here is MCTS-specific; it
//! only depends on `Game::canonical_representation`/`Transform`.

use crate::game::Game;
use crate::game::Real;
use crate::game::Transform;

/// The symmetry index relating `real_state` (an actual literal-board game
/// state -- never a canonicalized one) to its own canonical form: what a
/// caller needs to translate a canonical-orientation action (e.g. from a
/// `ChildArray` or transposition-table entry) back to the literal board via
/// `Game::invert_action`. `Transform::IDENTITY` for the root (whose own
/// action list is never canonicalized) and for any search mode that doesn't
/// share nodes/entries across paths (`canonicalizes` is `false`): plain
/// single-orientation search, with no transposition table at all.
///
/// `canonicalizes` must be `true` whenever a node/entry can be reached by
/// more than one real orientation -- not just explicit graph-search modes,
/// but also a legacy transposition table that shares a shape across whatever
/// real states hash-collide onto it exactly the same way. Passing `false`
/// there while a game's `Game::zobrist_hash` folds symmetry into its hash
/// (as ttt's `HashedPosition::hash` does) silently applies an action
/// generated in one real orientation against a different real orientation's
/// state.
///
/// Deliberately recomputed from `real_state` on every call rather than
/// cached anywhere on the incoming edge: a node/entry reached by more than
/// one real orientation (a transposition on that node's *parent*, not just
/// on the node itself) needs a different translation per path, since each
/// path's own real state canonicalizes via a different symmetry element in
/// general. A value cached at edge-creation time would silently keep
/// reflecting whichever path happened to create the edge first, which is
/// wrong for every other path that later reuses it.
pub fn incoming_sym<G: Game>(
    canonicalizes: bool,
    is_root: bool,
    real_state: Real<&G::S>,
) -> Transform {
    if canonicalizes && !is_root {
        G::canonical_representation(Real(real_state.0.clone())).1
    } else {
        Transform::IDENTITY
    }
}
