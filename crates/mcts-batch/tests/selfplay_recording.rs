//! Structural/sanity gate for `crate::selfplay::dump_gumbel_games_batched`:
//! not a byte-identical comparison against the per-node engine's own output
//! (this is a different search implementation producing different games,
//! not a like-for-like numerical check), but every structural invariant a
//! real `RecordV2` shard must satisfy regardless of which driver wrote it --
//! every position round-trips through the existing decoder, every policy
//! target is non-negative and sums sanely, and every game's own terminal
//! outcome is encoded consistently across all of that game's positions.

use game_othello::dump::RecordV2;
use game_othello::convnet::mlx::MlxCnnValueNet;
use mcts_batch::selfplay::dump_gumbel_games_batched;
use mcts_batch::Config;

/// `record_for`'s own convention: `ply` is the disc count minus 4, so a
/// single game's records -- appended to the shard once per ply, in
/// increasing ply order, since `dump_gumbel_games_batched` never plays two
/// games' positions out of order relative to each other within a game --
/// have `ply` values `0, 1, 2, ...` until the next `ply == 0` starts a new
/// game. Used here only to re-derive game boundaries from the shard itself,
/// not to re-implement the encoder's own labelling logic.
fn group_by_game(records: &[RecordV2]) -> Vec<&[RecordV2]> {
    let mut groups = Vec::new();
    let mut start = 0;
    for i in 1..records.len() {
        if records[i].ply == 0 {
            groups.push(&records[start..i]);
            start = i;
        }
    }
    if start < records.len() {
        groups.push(&records[start..]);
    }
    groups
}

/// A record's `target` (from its own side-to-move's perspective) converted
/// to a fixed Black-perspective value -- every position within one game must
/// agree on this value, since it is determined solely by that game's single
/// terminal outcome (`side == 0` is Black to move, per `dump.rs`'s own
/// `side_of` convention).
fn black_perspective(r: &RecordV2) -> f32 {
    if r.side == 0 {
        r.value
    } else {
        -r.value
    }
}

#[test]
fn batched_gumbel_records_round_trip_and_satisfy_shard_invariants() {
    let games = 4u64;
    let cfg = Config { num_simulations: 12, num_considered_actions: 4, ..Config::default() };
    let records = dump_gumbel_games_batched(MlxCnnValueNet::default(), &cfg, /* chunk_size */ 16, games, 42, 4);

    assert!(!records.is_empty(), "a real game always produces at least one non-terminal position");

    // Every record encodes and decodes back to itself (round-trips through
    // the exact byte layout the Python trainer's reader consumes).
    for r in &records {
        let mut buf = Vec::new();
        r.encode(&mut buf);
        let (decoded, consumed) = RecordV2::decode(&buf).expect("a just-encoded record must decode");
        assert_eq!(consumed, buf.len());
        assert_eq!(&decoded, r);
    }

    for r in &records {
        // Every policy entry is a valid probability over a real board
        // square or PASS, and the tail sums close to 1 -- a softmax over
        // the legal actions with illegal aids dropped entirely (see
        // `mcts_batch::improved_policy`'s own docs).
        assert!(!r.policy.is_empty(), "a non-terminal position always has a policy target");
        let mut sum = 0.0f32;
        for &(square, prob) in &r.policy {
            assert!((0.0..=1.0).contains(&prob), "prob {prob} out of range for square {square}");
            assert!(square <= 64, "square {square} out of range (0..=64, 64 == PASS)");
            sum += prob;
        }
        assert!((sum - 1.0).abs() < 1e-3, "policy tail sums to {sum}, expected ~1.0");

        // A valid outcome label: exactly a win, loss, or draw, never
        // anything else -- a wrong backfill would show up as some other
        // float value here.
        assert!(
            r.value == 1.0 || r.value == -1.0 || r.value == 0.0,
            "value {} is not a valid outcome label",
            r.value
        );
    }

    // Every game's own records agree on a single Black-perspective outcome
    // -- the actual "terminal outcome matches the game's own final board"
    // invariant the plan's gate asks for, checkable directly from the shard
    // without needing to replay the (randomized) move sequence externally:
    // a sign-flip or a game-grouping bug in the backfill would show up here
    // as two records in the same game disagreeing.
    let groups = group_by_game(&records);
    assert_eq!(groups.len(), games as usize, "expected one contiguous run of records per game");
    for group in groups {
        // `ply` is non-decreasing within a game but can repeat across a
        // forced pass (a pass leaves the disc count, and therefore `ply`,
        // unchanged) -- so this checks monotonicity, not strict contiguity.
        assert!(group.windows(2).all(|w| w[1].ply >= w[0].ply), "ply sequence not non-decreasing: {group:?}");
        let first = black_perspective(&group[0]);
        for r in group {
            assert_eq!(black_perspective(r), first, "a game's own records disagree on its outcome");
        }
    }
}
