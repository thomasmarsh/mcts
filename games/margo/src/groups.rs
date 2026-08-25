//! Incremental group/liberty tracking for Margo, via union-find over the
//! pyramid's full touching-adjacency graph (same-level and cross-level
//! support/dependent edges alike -- see `pyramid::adjacency`'s module docs
//! for the geometry). Proven equivalent to `raster::Raster::flood`/
//! `count_liberties` by this module's own tests, and wired into `State` as
//! a real field (maintained through `Game::apply`) that `candidate_is_legal`
//! reads to answer most legality checks without ever flooding the board.
//!
//! # What counts as "active"
//!
//! `place`/`rebuild` treat every occupied cell as active, split only by
//! colour (`black`/`white`) -- matching `visible_boards`/
//! `raster::Raster::from_pyramid`, neither of which exclude buried or
//! zombie cells from connectivity: a buried piece still physically touches
//! its neighbours, and `visible_boards`'s own comment is explicit that a
//! zombie does too ("A zombie still physically touches its neighbours and
//! participates in group connectivity"). The zombie mask only ever affects
//! which captured members survive removal (`apply_captures`), never group
//! membership or liberties. This module matches that ground truth rather
//! than the buried/zombie exclusion an older design note for this game
//! describes, since `raster::Raster::flood` -- the oracle this module's
//! tests check against -- doesn't perform that exclusion either.
//!
//! # Liberties
//!
//! Each group's liberties are stored as a bitboard (a [`GoBoard`], the same
//! flat-indexed board `resolve_captures` uses) rather than an integer
//! count: merging two groups' liberties is a bitwise OR, which -- unlike
//! adding two liberty *counts* -- doesn't double-count a liberty point the
//! two groups already shared before the merge. The reported liberty count
//! ([`Groups::liberty_count`]) is that bitboard's popcount, computed on
//! demand.
//!
//! # Placement vs. capture
//!
//! [`Groups::place`] is a real incremental update, proportional to the
//! placed cell's own (small, constant) neighbour count: it unions the new
//! stone into same-colour neighbouring groups (merging liberties via OR),
//! then clears the placed cell's own bit from every distinct group --
//! same-colour or enemy -- that had it as a liberty. [`Groups::rebuild`]
//! recomputes an entire `Groups` from a `Cells` occupancy/colour pair from
//! scratch (by replaying every occupied cell through `place` against an
//! empty structure) -- proportional to board size, not to what changed.
//! Capture removal (and the rarer swap-rule recolouring) are handled by a
//! full `rebuild` rather than an incremental removal, since a union-find
//! structure has no cheap way to un-union a group when one of its members
//! disappears; captures are comparatively rare next to placements, so this
//! is an acceptable cost for now (see this module's own property test for
//! the intended call shape: `place` after every placement, `rebuild`
//! whenever that placement also captured something or the swap rule fired).

use bitboard::Adjacency;
use pyramid::TouchingAdjacency;

use crate::{go_board, ground_mask, Cells, GoBoard};

/// Union-find (disjoint-set, path compression + union by size) over a
/// base-`n` pyramid's flat cell indices, restricted to occupied cells (see
/// the module docs for why buried/zombie cells are not excluded). Each
/// root additionally owns the group's colour and a liberty bitboard.
#[derive(Clone, Debug)]
pub struct Groups {
    total: usize,
    parent: Vec<u32>,
    size: Vec<u32>,
    /// `Some(true)` = black, `Some(false)` = white, `None` = not active
    /// (empty cell).
    color: Vec<Option<bool>>,
    /// Valid only at a root's own index; stale/meaningless elsewhere.
    liberties: Vec<GoBoard>,
    ground: GoBoard,
}

impl Groups {
    /// An empty structure for a base-`n` board -- no cell active.
    pub fn new(n: usize) -> Self {
        let total = pyramid::total_cells(n);
        Groups {
            total,
            parent: (0..total as u32).collect(),
            size: vec![1u32; total],
            color: vec![None; total],
            liberties: vec![go_board(total); total],
            ground: ground_mask(n, total),
        }
    }

    /// Rebuilds a `Groups` from scratch against `occupied`/`black`'s
    /// current state -- an empty structure with every occupied cell
    /// replayed through [`place`](Self::place). Cost is proportional to
    /// board size (a fresh flood-equivalent walk), not to what changed
    /// since some earlier state -- see the module docs for when this is
    /// the right call vs. incremental `place`.
    pub fn rebuild(
        n: usize,
        occupied: &Cells,
        black: &Cells,
        adjacency: &TouchingAdjacency,
    ) -> Self {
        let mut groups = Self::new(n);
        for index in occupied.iter_set() {
            groups.place(index, black.get_index(index), adjacency);
        }
        groups
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        let mut cur = x;
        while self.parent[cur] as usize != root {
            let next = self.parent[cur] as usize;
            self.parent[cur] = root as u32;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u32;
        self.size[big] += self.size[small];
        // Union of liberty sets, not a sum of counts -- a bitwise OR can't
        // double-count a liberty the two groups already shared.
        let merged = self.liberties[big] | self.liberties[small];
        self.liberties[big] = merged;
    }

    /// Whether `index` currently holds an active (occupied) cell -- for
    /// inspection/tests.
    #[allow(dead_code)]
    pub fn is_active(&self, index: usize) -> bool {
        self.color[index].is_some()
    }

    /// `index`'s own colour (`None` if not active) -- every member of a
    /// group carries the same colour its root does (only same-colour cells
    /// ever union), so this needs no root lookup.
    pub fn color(&self, index: usize) -> Option<bool> {
        self.color[index]
    }

    /// Non-mutating root lookup: unlike [`find`](Self::find), performs no
    /// path compression, so it can run from a shared `&self` -- for
    /// read-only callers (e.g. speculative candidate-legality checks) that
    /// only have `&State`/`&Groups` access and must not mutate the real
    /// structure while probing it. Still bounded by union-by-size's
    /// `O(log total)` tree depth, just without the amortization a mutating
    /// `find` gets from compressing paths as it goes.
    fn find_readonly(&self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        root
    }

    /// `index`'s group's liberty bitboard (`index` must be active) --
    /// read-only counterpart to [`liberty_count`](Self::liberty_count) that
    /// returns the full bitboard rather than its popcount, for callers that
    /// need to test *which* cells are liberties (not just how many).
    pub fn liberties(&self, index: usize) -> GoBoard {
        debug_assert!(self.color[index].is_some());
        let root = self.find_readonly(index);
        self.liberties[root]
    }

    /// Places a stone of colour `black` at `index` (must not already be
    /// active). Unions it with every same-colour group touching it across
    /// the full adjacency graph (lateral and cross-level support/dependent
    /// edges alike -- whatever `adjacency` reports, matching
    /// `raster::Raster::flood`'s own cross-level handling), merging liberty
    /// bitboards via OR, and clears `index` itself from every distinct
    /// group -- same-colour or enemy -- that had it as a liberty. Cost is
    /// proportional to `index`'s own neighbour count (at most
    /// `bitboard::MAX_NEIGHBORS`), not to group or board size.
    pub fn place(&mut self, index: usize, black: bool, adjacency: &TouchingAdjacency) {
        debug_assert!(index < self.total);
        debug_assert!(self.color[index].is_none(), "cell already active");

        // `index` is no longer empty -- drop it from every distinct
        // group's liberties before this cell has a group of its own to
        // avoid re-adding it to itself.
        let mut touched_roots: Vec<usize> = Vec::new();
        for nb in adjacency.neighbors(index) {
            if self.color[nb].is_some() {
                let r = self.find(nb);
                if !touched_roots.contains(&r) {
                    touched_roots.push(r);
                }
            }
        }
        for &r in &touched_roots {
            self.liberties[r].clear_index(index);
        }

        // Activate `index` as its own singleton group, liberties seeded
        // from its own empty ground-level (level-0) neighbours -- higher
        // levels never contribute a liberty directly (see raster.rs's
        // `count_liberties` doc comment), and a piece's own supporters
        // (its only level-0 touching neighbours if it isn't level 0
        // itself) are always occupied by the time it can be placed, so
        // this naturally only picks up real liberties either way.
        self.color[index] = Some(black);
        self.parent[index] = index as u32;
        self.size[index] = 1;
        let mut libs = go_board(self.total);
        for nb in adjacency.neighbors(index) {
            if self.ground.get_index(nb) && self.color[nb].is_none() {
                libs.set_index(nb);
            }
        }
        self.liberties[index] = libs;

        // Union with every same-colour neighbouring group.
        for nb in adjacency.neighbors(index) {
            if self.color[nb] == Some(black) {
                self.union(index, nb);
            }
        }
    }

    /// The flat cell indices in the same group as `index` (`index` must be
    /// active), ascending. `O(total cells)` -- for inspection/tests, not
    /// the hot path this structure exists for.
    #[allow(dead_code)]
    pub fn group_members(&mut self, index: usize) -> Vec<usize> {
        debug_assert!(self.color[index].is_some());
        let root = self.find(index);
        (0..self.total)
            .filter(|&i| self.color[i].is_some() && self.find(i) == root)
            .collect()
    }

    /// The number of empty level-0 cells adjacent (via the full touching
    /// graph) to `index`'s group -- the popcount of the group's liberty
    /// bitboard, computed on demand.
    #[allow(dead_code)]
    pub fn liberty_count(&mut self, index: usize) -> usize {
        debug_assert!(self.color[index].is_some());
        let root = self.find(index);
        self.liberties[root].count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use bitboard::Dyn;
    use mcts::game::Game;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    use super::*;
    use crate::{raster, Action, Margo, Player, State};

    /// Plain black/white per-level raster masks (unlike `build_color_masks`,
    /// not relative to whichever player currently has the move) -- what
    /// `raster::Raster::flood` needs as its `color` argument to flood a
    /// specific coloured group.
    fn per_level_masks(
        n: usize,
        occupied: &Cells,
        black: &Cells,
    ) -> (Vec<raster::LevelBoard>, Vec<raster::LevelBoard>) {
        let mut black_m: Vec<raster::LevelBoard> = (0..n)
            .map(|_| raster::LevelBoard::new(Dyn(n), Dyn(n)))
            .collect();
        let mut white_m: Vec<raster::LevelBoard> = (0..n)
            .map(|_| raster::LevelBoard::new(Dyn(n), Dyn(n)))
            .collect();
        for idx in occupied.iter_set() {
            let (col, row, level) = occupied.to_coord(idx);
            let pos = row * n + col;
            if black.get_index(idx) {
                black_m[level].set_index(pos);
            } else {
                white_m[level].set_index(pos);
            }
        }
        (black_m, white_m)
    }

    /// Drives real legal play through `Margo::apply` while maintaining a
    /// `Groups` alongside it -- `place` after every placement, a full
    /// `rebuild` whenever that move also captured something (occupied
    /// count changed by anything other than the placed stone itself) or
    /// the swap rule fired -- and after every move checks every occupied,
    /// non-buried, non-zombie cell's group membership and liberty count
    /// against a fresh `raster::Raster::flood`/`count_liberties`, the
    /// ground truth this structure must agree with. Seeded and ply-capped,
    /// following `random_action_matches_generate_actions`'s shape rather
    /// than `random_play_smoke_test`'s unbounded one.
    fn groups_match_raster_ground_truth(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let adjacency = TouchingAdjacency::new(n);
        let mut groups = Groups::rebuild(n, &state.occupied, &state.black, &adjacency);
        let max_plies = state.total_cells() + 2;

        for _ in 0..max_plies {
            if Margo::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            Margo::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "n={n} seed={seed}: no legal moves on a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];

            let prev_occupied_count = state.occupied.count_ones();
            let mover_black = state.turn == Player::Black;
            state = Margo::apply(state, &action);

            match action {
                Action::Swap => {
                    groups = Groups::rebuild(n, &state.occupied, &state.black, &adjacency);
                }
                Action::Place(index, _) => {
                    let index = index as usize;
                    groups.place(index, mover_black, &adjacency);
                    if state.occupied.count_ones() != prev_occupied_count + 1 {
                        // A capture removed at least one cell (or turned
                        // one into a zombie, which doesn't itself require
                        // a rebuild -- see the module docs -- but the two
                        // aren't distinguished here since captures are
                        // rare enough that a rebuild either way is cheap).
                        groups = Groups::rebuild(n, &state.occupied, &state.black, &adjacency);
                    }
                }
            }

            assert_groups_match_raster(n, seed, &state, &mut groups);
        }
    }

    /// Checks `groups`' group membership/liberty count against a fresh
    /// raster ground truth for every occupied, non-buried, non-zombie cell
    /// in `state` -- the shared assertion body both
    /// `groups_match_raster_ground_truth` (a separately-threaded `Groups`)
    /// and `state_groups_match_raster_ground_truth` (`State`'s own `groups`
    /// field, maintained by `Margo::apply` itself) check against.
    fn assert_groups_match_raster(n: usize, seed: u64, state: &State, groups: &mut Groups) {
        let raster = raster::Raster::from_pyramid(n, &state.occupied);
        let (black_masks, white_masks) = per_level_masks(n, &state.occupied, &state.black);

        for index in state.occupied.iter_set() {
            let (col, row, level) = state.occupied.to_coord(index);
            if state.occupied.is_buried(col, row, level) || state.is_zombie(index) {
                continue;
            }
            let color_masks = if state.is_black(index) {
                &black_masks
            } else {
                &white_masks
            };
            let flood = raster.flood(col, row, level, color_masks);
            let expected_liberties = raster.count_liberties(&flood);

            let mut got_members = groups.group_members(index);
            got_members.sort_unstable();
            let mut expected_members: Vec<usize> = Vec::new();
            for (l, level_board) in flood.iter().enumerate() {
                for pos in level_board.iter_set() {
                    expected_members.push(state.occupied.index(pos % n, pos / n, l));
                }
            }
            expected_members.sort_unstable();

            assert_eq!(
                got_members, expected_members,
                "n={n} seed={seed}: group membership mismatch at cell {index} \
                 ({col},{row},L{level})"
            );
            assert_eq!(
                groups.liberty_count(index),
                expected_liberties,
                "n={n} seed={seed}: liberty count mismatch at cell {index} \
                 ({col},{row},L{level})"
            );
        }
    }

    /// Unlike `groups_match_raster_ground_truth` above, this drives real
    /// play through `Margo::apply` and checks `state.groups` -- the field
    /// `apply` itself maintains -- directly, with no separately-threaded
    /// shadow copy at all. A passing run here proves the maintenance logic
    /// survives being embedded in a cloned, tree-searched `State`, not just
    /// driven by this module's own test harness.
    fn state_groups_match_raster_ground_truth(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = state.total_cells() + 2;

        for _ in 0..max_plies {
            if Margo::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            Margo::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "n={n} seed={seed}: no legal moves on a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = Margo::apply(state, &action);

            let mut groups = state.groups.clone();
            assert_groups_match_raster(n, seed, &state, &mut groups);
        }
    }

    #[test]
    fn test_state_groups_match_raster_ground_truth() {
        for seed in 0..8 {
            state_groups_match_raster_ground_truth(4, seed);
        }
        for seed in 0..4 {
            state_groups_match_raster_ground_truth(7, seed);
        }
    }

    #[test]
    fn test_groups_match_raster_ground_truth() {
        for seed in 0..8 {
            groups_match_raster_ground_truth(4, seed);
        }
        for seed in 0..4 {
            groups_match_raster_ground_truth(7, seed);
        }
    }

    /// A placement that touches two previously-separate same-colour
    /// singleton groups purely through cross-level support edges (not a
    /// lateral one) must union them. `(0,0,0)` and `(1,1,0)` are diagonal
    /// to each other -- same-level diagonal pairs never touch -- so they
    /// start out genuinely separate; both are supporters of level-1 cell
    /// `(0,0)`, which touches all four of its supporters directly.
    #[test]
    fn placement_unions_two_groups_across_a_support_edge() {
        let n = 4;
        let adjacency = TouchingAdjacency::new(n);
        let mut groups = Groups::new(n);

        let a = pyramid::index(n, 0, 0, 0);
        let b = pyramid::index(n, 1, 1, 0);
        groups.place(a, false, &adjacency);
        groups.place(b, false, &adjacency);
        assert_eq!(groups.group_members(a), vec![a]);
        assert_eq!(groups.group_members(b), vec![b]);

        let connector = pyramid::index(n, 0, 0, 1);
        groups.place(connector, false, &adjacency);

        let mut members = groups.group_members(a);
        members.sort_unstable();
        let mut expected = vec![a, b, connector];
        expected.sort_unstable();
        assert_eq!(
            members, expected,
            "connector must union both previously-separate groups"
        );
        assert_eq!(groups.group_members(b), members);
    }

    /// A placement that closes the one shared liberty of two distinct,
    /// non-touching enemy groups at once must remove it from both --
    /// exercising the "every distinct enemy group" part of `place`'s
    /// liberty-clearing loop, not just a single group.
    #[test]
    fn placement_removes_a_liberty_from_two_distinct_enemy_groups() {
        let n = 4;
        let adjacency = TouchingAdjacency::new(n);
        let mut groups = Groups::new(n);

        let north = pyramid::index(n, 1, 0, 0);
        let south = pyramid::index(n, 1, 2, 0);
        groups.place(north, true, &adjacency);
        groups.place(south, true, &adjacency);
        let north_before = groups.liberty_count(north);
        let south_before = groups.liberty_count(south);

        let shared = pyramid::index(n, 1, 1, 0);
        groups.place(shared, false, &adjacency);

        assert_eq!(groups.liberty_count(north), north_before - 1);
        assert_eq!(groups.liberty_count(south), south_before - 1);
    }

    /// A capture's two possible outcomes for a member cell: one actually
    /// cleared from `occupied` must leave the union-find's connectivity
    /// entirely (it's simply no longer active); one that becomes a zombie
    /// (pinned, per `apply_captures`) stays in `occupied` and keeps its
    /// colour, so per this module's own ground-truth rule (see the module
    /// docs) it stays active and grouped exactly like any other occupied
    /// cell -- `apply_captures` never removes it from `occupied`/`black`,
    /// and `raster::Raster::flood`/`build_color_masks` don't filter it out
    /// either.
    #[test]
    fn rebuild_drops_a_removed_cell_but_keeps_a_zombie_grouped() {
        let n = 7;
        let adjacency = TouchingAdjacency::new(n);

        let mut occupied = Cells::new(Dyn(n));
        let mut black = Cells::new(Dyn(n));
        let pinned = occupied.index(0, 0, 0);
        let pin = occupied.index(0, 0, 1);
        let removed = occupied.index(3, 3, 0);
        occupied.set_index(pinned);
        occupied.set_index(pin);
        occupied.set_index(removed);
        black.set_index(pin);

        let before = Groups::rebuild(n, &occupied, &black, &adjacency);
        assert!(before.is_active(pinned));
        assert!(before.is_active(removed));

        // Simulate the post-capture occupancy `apply_captures` would
        // produce: `removed` cleared outright, `pinned` left in place
        // (still white, now a zombie -- the zombie mask itself plays no
        // part in this structure, see the module docs).
        let mut occupied_after = occupied;
        occupied_after.clear_index(removed);

        let after = Groups::rebuild(n, &occupied_after, &black, &adjacency);
        assert!(
            !after.is_active(removed),
            "a cell actually captured must leave the union-find entirely"
        );
        assert!(
            after.is_active(pinned),
            "a zombie is still occupied and still touches its neighbours, \
             so it stays active and grouped"
        );
    }
}
