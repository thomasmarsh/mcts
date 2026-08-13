// Regression tests for internal MCTS infrastructure that don't need a
// specific game -- they construct raw arena nodes / child arrays /
// transposition tables directly. Tests that exercise the search through a
// real game (Tic-Tac-Toe) live in `mcts-tests/tests/ttt_strategies.rs`.

#[test]
fn test_child_array_child_index_matches_creation_order() {
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();
    let ids: Vec<_> = (0..5).map(|i| index.insert(Node::new(0, i))).collect();

    let children = ChildArray::new(vec![10, 11, 12, 13, 14], 1);
    for &idx in [3usize, 0, 4, 1, 2].iter() {
        let resolved = children.get_or_create_child(idx, || ids[idx]);
        assert_eq!(resolved, ids[idx]);
    }

    for (idx, &id) in ids.iter().enumerate() {
        assert_eq!(
            children.child_index(id),
            idx,
            "child_index should invert get_or_create_child's id -> idx mapping"
        );
        assert_eq!(
            children.get_or_create_child(idx, || panic!("should not re-create")),
            id
        );
        assert_eq!(children.child_index(id), idx);
    }
}

#[test]
fn test_child_array_child_index_survives_concurrent_resolution() {
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use std::sync::Arc;

    for _ in 0..500 {
        let index: Arc<TreeIndex<u32>> = Arc::new(TreeIndex::new());
        let created_id = index.insert(Node::new(0, 0));
        let children = Arc::new(ChildArray::<u32>::new(vec![42], 1));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let children = Arc::clone(&children);
                scope.spawn(move || {
                    let id = children.get_or_create_child(0, || created_id);
                    assert_eq!(children.child_index(id), 0);
                });
            }
        });
    }
}

#[test]
fn test_child_array_explored_len_and_heap_bytes_estimate() {
    use crate::strategies::mcts::index::Id;
    use crate::strategies::mcts::node::ChildArray;
    use crate::strategies::mcts::node::PlayerStats;

    let children = ChildArray::<u32>::new(vec![10, 11, 12, 13], 2);
    assert_eq!(children.explored_len(), 0, "nothing resolved yet");

    children.get_or_create_child(1, Id::invalid_id);
    children.get_or_create_child(3, Id::invalid_id);
    assert_eq!(
        children.explored_len(),
        2,
        "explored_len should count only resolved slots, not len()"
    );

    let n = 4usize;
    let explored = 2usize;
    let expected = n * std::mem::size_of::<u32>()
        + n * std::mem::size_of::<std::sync::OnceLock<Id>>()
        + explored * (std::mem::size_of::<Id>() + std::mem::size_of::<usize>())
        + n * std::mem::size_of::<std::sync::atomic::AtomicU32>()
        + n * std::mem::size_of::<u32>()
        + n * 2 * std::mem::size_of::<PlayerStats>();
    assert_eq!(
        children.heap_bytes_estimate(),
        expected,
        "heap_bytes_estimate should be exactly the sum of each parallel array's element count * element size"
    );
}

#[test]
fn test_transposition_table_compact_drops_unmapped_and_remaps_survivors() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::TranspositionTable;
    use rustc_hash::FxHashMap;

    let old_index = TreeIndex::<u32>::new();
    let old_ids: Vec<_> = (0..4).map(|i| old_index.insert(Node::new(0, i))).collect();
    let new_index = TreeIndex::<u32>::new();
    let new_ids: Vec<_> = (0..2).map(|i| new_index.insert(Node::new(0, i))).collect();

    let mut table = TranspositionTable::default();
    table.insert(100, old_ids[0]);
    table.insert(200, old_ids[1]);
    table.insert(300, old_ids[2]);
    table.insert(400, old_ids[3]);

    let mut old_to_new = FxHashMap::default();
    old_to_new.insert(old_ids[0], new_ids[0]);
    old_to_new.insert(old_ids[1], new_ids[1]);

    table.compact(&old_to_new);

    assert_eq!(table.get_const(100).unwrap(), new_ids[0]);
    assert_eq!(table.get_const(200).unwrap(), new_ids[1]);
    assert!(table.get_const(300).is_none());
    assert!(table.get_const(400).is_none());
    assert_eq!(table.len(), 2);
}

#[test]
fn test_transposition_table_trusts_the_hash_first_write_wins() {
    use crate::strategies::mcts::node::Node;
    use crate::strategies::mcts::search::TreeIndex;
    use crate::strategies::mcts::table::TranspositionTable;

    let index = TreeIndex::<u32>::new();
    let ids: Vec<_> = (0..2).map(|i| index.insert(Node::new(0, i))).collect();

    let table = TranspositionTable::default();
    let first = table.get_or_insert(42, || ids[0]);
    let second = table.get_or_insert(42, || ids[1]);

    assert_eq!(first, ids[0]);
    assert_eq!(second, ids[0]);
    assert_eq!(table.len(), 1);
}

#[test]
fn test_child_array_remap_child_ids_rewrites_resolved_slots_only() {
    use crate::strategies::mcts::index::Id;
    use crate::strategies::mcts::node::{ChildArray, Node};
    use crate::strategies::mcts::search::TreeIndex;
    use rustc_hash::FxHashMap;

    let old_index = TreeIndex::<u32>::new();
    let old_ids: Vec<Id> = (0..3).map(|i| old_index.insert(Node::new(0, i))).collect();
    let new_index = TreeIndex::<u32>::new();
    let new_ids: Vec<Id> = (0..3).map(|i| new_index.insert(Node::new(0, i))).collect();

    let mut children = ChildArray::<u32>::new(vec![10, 11, 12], 1);
    children.get_or_create_child(0, || old_ids[0]);
    children.get_or_create_child(2, || old_ids[2]);

    let mut old_to_new = FxHashMap::default();
    old_to_new.insert(old_ids[0], new_ids[0]);
    old_to_new.insert(old_ids[2], new_ids[2]);

    children.remap_child_ids(&old_to_new);

    assert_eq!(children.node_id(0), Some(new_ids[0]));
    assert_eq!(children.node_id(1), None);
    assert_eq!(children.node_id(2), Some(new_ids[2]));
    assert_eq!(children.child_index(new_ids[0]), 0);
    assert_eq!(children.child_index(new_ids[2]), 2);
}

// PN-MCTS (Kowalski et al. 2023): `derive_pn_dpn`'s negamax recurrence,
// hand-verified on a tiny 3-child arena rather than through a real game --
// small enough to compute the expected pn/dpn by hand, which a purely
// behavioral (does-it-pick-the-right-move) test wouldn't necessarily
// exercise: `pn(root) = min` over children of the child's *dpn*, and
// `dpn(root) = sum` over children of the child's *pn* -- easy to get
// backwards, since it's the opposite of the naive "sum pn, min dpn" a
// classic (non-negamax) OR-node formula would suggest.
//
// Root has three child slots:
//   - idx 0: an unexplored slot (no tree node at all) -- PNS's "unknown
//     leaf" case, contributes (pn=1, dpn=1).
//   - idx 1: a tree node proven `Win` for its own mover -- contributes
//     (pn=0, dpn=MAX) via `Node::pn`/`dpn`'s `Proven` short-circuit.
//   - idx 2: an unvisited tree node (never `expand()`ed, `Unproven`) --
//     also (pn=1, dpn=1), but read from the *stored* atomic default this
//     time rather than the unexplored-slot fallback, exercising the other
//     code path that produces the same value.
//
// Expected: pn(root) = min(dpn_0=1, dpn_1=MAX, dpn_2=1) = 1.
//           dpn(root) = pn_0=1 + pn_1=0 + pn_2=1 = 2.
#[test]
fn test_derive_pn_dpn_negamax_recurrence_hand_verified() {
    use crate::strategies::mcts::backprop::derive_pn_dpn;
    use crate::strategies::mcts::node::{ChildArray, Node, NodeState, Proven};
    use crate::strategies::mcts::search::TreeIndex;

    let index = TreeIndex::<u32>::new();

    let proven_win_child = Node::new(1, 0);
    proven_win_child.try_prove(Proven::Win(1));
    let proven_win_id = index.insert(proven_win_child);

    let unvisited_child = Node::new(1, 0);
    let unvisited_id = index.insert(unvisited_child);

    let children = ChildArray::<u32>::new(vec![10, 11, 12], 2);
    children.get_or_create_child(1, || proven_win_id);
    children.get_or_create_child(2, || unvisited_id);
    // idx 0 deliberately left unresolved (no `get_or_create_child` call).

    assert_eq!(index.get(proven_win_id).pn(), 0);
    assert_eq!(index.get(proven_win_id).dpn(), u32::MAX);
    assert_eq!(index.get(unvisited_id).pn(), 1);
    assert_eq!(index.get(unvisited_id).dpn(), 1);

    let root = Node::<u32>::new(0, 0);
    root.expand(|| NodeState::Expanded(children));

    derive_pn_dpn(&root, &index);

    assert_eq!(root.pn(), 1, "pn(root) = min(dpn) over children");
    assert_eq!(root.dpn(), 2, "dpn(root) = sum(pn) over children");
}
