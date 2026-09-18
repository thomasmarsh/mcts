//! Batched Gumbel self-play, recorded as `RecordV2` shards -- the real
//! (not throughput-only) counterpart to `examples/bench_othello_selfplay.rs`.
//!
//! `games/othello/src/dump.rs::dump_gumbel_games`'s `--head cnn` path (the
//! per-node engine, via `game_othello::selfplay::CnnGumbelPlayer`) is what
//! the production coordinator calls today and is what writes the
//! `RecordV2` format `az-train-othello-cnn`'s trainer and every gate leg
//! already consume. This module is a second driver over this crate's own
//! batched engine (`gumbel_explore`/`MlxOthelloOracle`) that writes the
//! *same* format, so nothing downstream of self-play needs to change --
//! only how the games are played (many simultaneously, one `gumbel_explore`
//! call per ply covering every still-live game) and recorded does.
//!
//! `game-othello` cannot depend on this crate (`mcts-batch` already depends
//! on `game-othello`, for `OthelloOracle`/`MlxOthelloOracle`), so this
//! driver -- unlike `dump_gumbel_games` -- lives here rather than in
//! `game_othello::dump`, reusing that module's public `Record`/`RecordV2`/
//! `record_for` directly.

use game_othello::convnet::mlx::MlxCnnValueNet;
use game_othello::dump::{record_for, Record, RecordV2};
use game_othello::{Move, Othello, Player, State};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::othello::MlxOthelloOracle;
use crate::{gumbel_explore, improved_policy, Config};

/// One dumped position before the final outcome is known -- the v2 record
/// head (`target` filled in once the game ends) plus its completed-Q
/// improved-policy target. Same shape as `game_othello::dump`'s private
/// `GumbelRow`, duplicated rather than shared since that type isn't public.
struct GumbelRow {
    head: Record,
    policy: Vec<(Move, f32)>,
}

/// One still-live game's accumulated rows plus its current position.
/// `game_index` is the game's fixed identity (0..`games`) -- unlike its
/// position in `live`/the batch id `gumbel_explore` assigns each round,
/// which shifts as other games finish, `game_index` never changes, so
/// [`forced_move`]'s per-game determinism holds across the whole game.
struct LiveGame {
    state: State,
    rows: Vec<GumbelRow>,
    ply: u8,
    game_index: u64,
}

/// A deterministic per-game, per-ply opening move for the first
/// `forced_plies` plies of a game, for wide, uniform coverage of Othello's
/// (small) opening tree across a self-play run -- the same mechanism and
/// hash `game_othello::dump`'s own private `forced_move` uses (duplicated
/// rather than shared, since that function isn't public); a stale copy here
/// would only ever mean this driver's forced openings drift from the
/// per-node engine's, not a hidden coupling bug. Search still runs and a
/// policy target is still recorded at every forced position -- only the
/// move actually played is overridden.
fn forced_move(state: &State, game_index: u64, ply: u32, forced_plies: u32) -> Option<Move> {
    if ply >= forced_plies {
        return None;
    }
    let mut actions = Vec::new();
    Othello::generate_actions(state, &mut actions);
    if actions.len() <= 1 {
        return None;
    }
    let h = (game_index ^ (ply as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    Some(actions[(h as usize) % actions.len()])
}

/// Sample one move from a `(aid -> probability)` distribution over all
/// `policy.len()` action ids (probabilities summing to ~1, `0.0` at every
/// illegal aid -- see [`improved_policy`]'s own docs), falling back to the
/// last positive-probability entry on a rounding shortfall. Mirrors
/// `game_othello::dump`'s private `sample_visit_distribution` (that
/// function samples over a `(Move, f32)` list; this one indexes a dense
/// per-aid array instead, since `improved_policy` already returns one).
fn sample_policy(policy: &[f32], rng: &mut SmallRng) -> usize {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0f32;
    for (aid, &p) in policy.iter().enumerate() {
        acc += p;
        if r < acc {
            return aid;
        }
    }
    policy
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &p)| p > 0.0)
        .map(|(aid, _)| aid)
        .expect("a non-terminal state always has at least one legal (positive-probability) action")
}

/// Argmax of `completed_qvalues` over the finite (legal) entries -- the
/// same post-temperature move-selection convention `examples/
/// bench_othello_selfplay.rs::play_games_batched` already established for
/// this crate's batched engine.
fn argmax_completed_q(qs: &[f32]) -> usize {
    qs.iter()
        .enumerate()
        .filter(|(_, &q)| q.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(aid, _)| aid)
        .expect("a non-terminal state always has at least one legal action")
}

/// Play `games` Gumbel self-play games with the batched GPU engine -- every
/// still-live game advances one ply per `gumbel_explore` call, so a
/// generation's whole self-play pass is a handful of large batched oracle
/// calls rather than one call per leaf per game -- and return one
/// [`RecordV2`] per non-terminal position across every game.
///
/// Move selection matches `games/othello/src/dump.rs`'s per-node convention:
/// for the first `forced_opening_plies` plies, [`forced_move`] overrides the
/// search entirely with a deterministic per-game opening choice (search
/// still runs and a policy target is still recorded); otherwise, for the
/// first `temp_moves` plies of a game, the move is sampled from the
/// completed-Q improved-policy distribution ([`improved_policy`]); after
/// that, the move is the argmax of `completed_qvalues`
/// ([`argmax_completed_q`], matching this crate's own established
/// throughput-benchmark convention). `chunk_size` is threaded straight into
/// [`MlxOthelloOracle::new`] -- see that constructor's own docs for why it
/// is a required memory-safety parameter, not an optional tuning knob, at
/// large net geometries.
pub fn dump_gumbel_games_batched(
    net: MlxCnnValueNet,
    cfg: &Config,
    chunk_size: usize,
    games: u64,
    seed: u64,
    temp_moves: u8,
    forced_opening_plies: u32,
) -> Vec<RecordV2> {
    let oracle = MlxOthelloOracle::new(net, chunk_size);
    let mut gumbel_rng = SmallRng::seed_from_u64(seed);
    let mut move_rng = SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);

    let mut live: Vec<LiveGame> = (0..games)
        .map(|game_index| LiveGame { state: State::default(), rows: Vec::new(), ply: 0, game_index })
        .collect();
    let mut out: Vec<RecordV2> = Vec::new();

    while !live.is_empty() {
        let states: Vec<State> = live.iter().map(|g| g.state).collect();
        let tree = gumbel_explore(cfg, &oracle, &states, &mut gumbel_rng);

        let mut next_live = Vec::with_capacity(live.len());
        for (bid, mut g) in live.into_iter().enumerate() {
            let policy = improved_policy(cfg, &tree, bid, tree.root());
            let tail: Vec<(Move, f32)> = (0..tree.num_actions())
                .filter(|&aid| tree.is_valid_action(bid, tree.root(), aid as u16))
                .map(|aid| (Move(aid as u8), policy[aid]))
                .collect();

            let head = record_for(&g.state, None);
            let forced = forced_move(&g.state, g.game_index, g.ply as u32, forced_opening_plies);
            let aid = if let Some(forced) = forced {
                forced.0 as usize
            } else if g.ply < temp_moves {
                sample_policy(&policy, &mut move_rng)
            } else {
                argmax_completed_q(&tree.completed_qvalues(bid, tree.root()))
            };
            g.rows.push(GumbelRow { head, policy: tail });

            let next_state = Othello::apply(g.state, &Move(aid as u8));
            g.ply += 1;
            if Othello::is_terminal(&next_state) {
                let winner = Othello::winner(&next_state);
                out.extend(g.rows.into_iter().map(|mut row| {
                    let side_player = if row.head.side == 0 { Player::Black } else { Player::White };
                    row.head.target = match winner {
                        None => 0.0,
                        Some(w) if w == side_player => 1.0,
                        Some(_) => -1.0,
                    };
                    RecordV2::from_record(row.head, &row.policy)
                }));
            } else {
                g.state = next_state;
                next_live.push(g);
            }
        }
        live = next_live;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{forced_move, Othello, State};
    use mcts::game::Game;

    #[test]
    fn forced_move_disabled_past_forced_plies() {
        let state = State::default();
        assert_eq!(forced_move(&state, 0, 3, 3), None);
        assert_eq!(forced_move(&state, 0, 10, 3), None);
    }

    #[test]
    fn forced_move_is_deterministic_per_game_and_ply() {
        let state = State::default();
        for g in 0..20u64 {
            for ply in 0..3u32 {
                let a = forced_move(&state, g, ply, 3).expect("opening has real choices");
                let mut actions = Vec::new();
                Othello::generate_actions(&state, &mut actions);
                assert!(actions.contains(&a));
                assert_eq!(forced_move(&state, g, ply, 3), Some(a));
            }
        }
    }
}
