//! Batched Gumbel self-play from the empty board. Every live game advances one ply (one
//! sub-decision of a turn) per `gumbel_explore` call, so a generation is a few thousand large GPU
//! calls rather than one call per leaf. Each position is recorded with the completed-Q improved
//! policy as its policy target, the same search's root value as `Record.q`, and, once the game
//! ends, the mover's outcome as its value target. Games that reach `max_plies` are dropped from
//! the shard (but still get a capped [`Terminal`] entry) and counted.

use std::time::Instant;

use mcts::game::Game;
use mcts_batch::{gumbel_explore, gumbel_explore_with_noise, gumbel_selected_action, improved_policy, EnvOracle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::config::{SearchSettings, SelfPlayConfig};
use super::encode::{action_id, legal_moves, move_from_id, num_actions};
use super::oracle::DruidOracle;
use super::shard::{Fields, Record, Terminal};
use crate::{DruidSplit, HashedState, Move, Player, Size};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub games_started: usize,
    pub games_finished: usize,
    pub games_capped: usize,
    pub positions: usize,
    /// Positions played without a search call (`skip_forced_search`'s one-legal-move shortcut);
    /// always `0` with the flag off.
    pub forced_positions: usize,
    /// Positions that went through `gumbel_explore_with_noise`.
    pub searched_positions: usize,
    pub mean_plies: f64,
    pub black_wins: usize,
    pub draws: usize,
    pub seconds: f64,
}

struct Row {
    fields: Fields,
    ply: u16,
    policy: Vec<f32>,
    /// The root search value at this position, from the mover's point of view -- `Record.q`.
    /// For a forced row (`forced == true`) this starts as a placeholder and is overwritten by
    /// `backfill_forced_q` once the game ends.
    q: f32,
    /// Whether this position had exactly one legal action, so no search ran for it.
    forced: bool,
}

struct Live {
    state: HashedState,
    rows: Vec<Row>,
    game: u32,
}

fn sample(policy: &[f32], rng: &mut SmallRng) -> u16 {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0;
    for (id, &p) in policy.iter().enumerate() {
        acc += p;
        if p > 0.0 && r < acc {
            return id as u16;
        }
    }
    policy
        .iter()
        .rposition(|&p| p > 0.0)
        .expect("a live position has a legal action") as u16
}

/// A finished game's per-row outcome value, from each row's own mover's point of view -- the
/// same target every row gets regardless of ply.
fn outcome_values(rows: &[Row], winner: Option<Player>) -> Vec<f32> {
    rows.iter()
        .map(|row| match winner {
            None => 0.0,
            Some(w) => {
                let black_moves = row.fields.is_black_to_move();
                if black_moves == (w == Player::Black) { 1.0 } else { -1.0 }
            }
        })
        .collect()
}

/// Back-fills `q` for a finished game's forced rows (in ply order, `q` and `movers` already
/// populated for every row): a forced row takes the next row's `q`, negated if the mover changes
/// between the two rows (within a turn it does not), or its own outcome `value` if it is the
/// game's last row. Runs of consecutive forced rows resolve from the end backwards, since each
/// one needs its successor's `q` already settled. Rows that were actually searched are untouched.
fn backfill_forced_q(movers: &[bool], values: &[f32], forced: &[bool], q: &mut [f32]) {
    for i in (0..q.len()).rev() {
        if !forced[i] {
            continue;
        }
        q[i] = if i + 1 == q.len() {
            values[i]
        } else if movers[i] == movers[i + 1] {
            q[i + 1]
        } else {
            -q[i + 1]
        };
    }
}

/// The per-generation accumulators every ply feeds -- bundled so the per-ply helpers below stay
/// under a sane argument count.
struct Accum {
    records: Vec<Record>,
    terminals: Vec<Terminal>,
    stats: Stats,
    total_plies: usize,
}

/// This generation's two RNG streams, bundled for the same reason as [`Accum`]: the Gumbel search
/// noise, and (only for the temperature-sampled opening plies) the move itself.
struct Rngs {
    gumbel: SmallRng,
    mv: SmallRng,
}

/// The read-only per-generation setup a ply needs: the oracle, its search config, and self-play
/// settings.
struct Ctx<'a, const N: usize> {
    oracle: &'a DruidOracle<N>,
    batch: &'a mcts_batch::Config,
    cfg: &'a SelfPlayConfig,
}

/// After a ply's move is applied: finishes the game into `accum` if it just ended (back-filling
/// any forced rows' `q` along the way), or moves it to `next` to keep playing. Shared by both
/// self-play paths (search every ply, or skip forced positions) so a finished game's rows are
/// recorded the same way either way.
fn finish_or_continue<const N: usize>(game: Live, ply: usize, accum: &mut Accum, next: &mut Vec<Live>) {
    if !DruidSplit::is_terminal(&game.state) {
        next.push(game);
        return;
    }
    let winner = DruidSplit::winner(&game.state);
    accum.stats.games_finished += 1;
    accum.stats.black_wins += usize::from(winner == Some(Player::Black));
    accum.stats.draws += usize::from(winner.is_none());
    accum.total_plies += ply + 1;
    let final_fields = Fields::of(&game.state);
    let gid = game.game;
    accum.terminals.push(Terminal {
        game: gid,
        winner: winner.map(|w| u8::from(w == Player::White)),
        capped: false,
        heights: final_fields.heights,
        owners: final_fields.owners,
    });

    let values = outcome_values(&game.rows, winner);
    let movers: Vec<bool> = game.rows.iter().map(|r| r.fields.is_black_to_move()).collect();
    let forced: Vec<bool> = game.rows.iter().map(|r| r.forced).collect();
    let mut qs: Vec<f32> = game.rows.iter().map(|r| r.q).collect();
    backfill_forced_q(&movers, &values, &forced, &mut qs);

    for ((row, value), q) in game.rows.into_iter().zip(values).zip(qs) {
        debug_assert_eq!(row.policy.len(), num_actions(N));
        accum.records.push(Record { fields: row.fields, value, game: gid, ply: row.ply, q, policy: row.policy });
    }
}

/// One ply of ordinary self-play: every live game goes through a fresh Gumbel search.
fn play_ply_searching_every_game<const N: usize>(
    ctx: &Ctx<N>,
    states: &[HashedState],
    live: Vec<Live>,
    ply: usize,
    rngs: &mut Rngs,
    accum: &mut Accum,
) -> Vec<Live> {
    let (tree, noise) = gumbel_explore_with_noise(ctx.batch, ctx.oracle, states, &mut rngs.gumbel);
    let mut next = Vec::with_capacity(live.len());
    for (bid, mut game) in live.into_iter().enumerate() {
        let policy = improved_policy(ctx.batch, &tree, bid, tree.root());
        game.rows.push(Row {
            fields: Fields::of(&game.state),
            ply: ply as u16,
            policy: policy.clone(),
            q: tree.value(bid, tree.root()),
            forced: false,
        });
        accum.stats.searched_positions += 1;
        let id = if ply < ctx.cfg.temp_moves {
            sample(&policy, &mut rngs.mv)
        } else {
            gumbel_selected_action(ctx.batch, &tree, bid, &noise[bid])
        };
        game.state = DruidSplit::apply(game.state, &move_from_id(id, N));
        finish_or_continue::<N>(game, ply, accum, &mut next);
    }
    next
}

/// One ply of `skip_forced_search`: live games with more than one legal action are searched as a
/// batch (mapped back to their game by `tree_bid`); a forced game (exactly one legal action)
/// records a one-hot policy on it and plays it with no search call. A ply where every live game
/// is forced never calls `gumbel_explore_with_noise` at all.
fn play_ply_skipping_forced<const N: usize>(
    ctx: &Ctx<N>,
    states: &[HashedState],
    live: Vec<Live>,
    ply: usize,
    rngs: &mut Rngs,
    accum: &mut Accum,
) -> Vec<Live> {
    let legal: Vec<Vec<Move>> = states.iter().map(|s| legal_moves(s).0).collect();
    let mut tree_bid = vec![usize::MAX; live.len()];
    let mut searched_states = Vec::new();
    for (i, mvs) in legal.iter().enumerate() {
        if mvs.len() > 1 {
            tree_bid[i] = searched_states.len();
            searched_states.push(states[i].clone());
        }
    }
    let search = (!searched_states.is_empty())
        .then(|| gumbel_explore_with_noise(ctx.batch, ctx.oracle, &searched_states, &mut rngs.gumbel));

    let mut next = Vec::with_capacity(live.len());
    for (i, mut game) in live.into_iter().enumerate() {
        let mvs = &legal[i];
        let (policy, q, forced, mv) = if mvs.len() == 1 {
            accum.stats.forced_positions += 1;
            let mut one_hot = vec![0.0f32; num_actions(N)];
            one_hot[usize::from(action_id(&mvs[0], N))] = 1.0;
            // Back-filled once the game ends; see `backfill_forced_q`.
            (one_hot, 0.0, true, mvs[0])
        } else {
            accum.stats.searched_positions += 1;
            let (tree, noise) = search.as_ref().expect("a non-forced game has a search tree");
            let bid = tree_bid[i];
            let policy = improved_policy(ctx.batch, tree, bid, tree.root());
            let q = tree.value(bid, tree.root());
            let id = if ply < ctx.cfg.temp_moves {
                sample(&policy, &mut rngs.mv)
            } else {
                gumbel_selected_action(ctx.batch, tree, bid, &noise[bid])
            };
            (policy, q, false, move_from_id(id, N))
        };
        game.rows.push(Row { fields: Fields::of(&game.state), ply: ply as u16, policy, q, forced });
        game.state = DruidSplit::apply(game.state, &mv);
        finish_or_continue::<N>(game, ply, accum, &mut next);
    }
    next
}

/// Plays every game to completion (or `max_plies`) and returns the recorded positions, one
/// [`Terminal`] per game for the shard's terminal sidecar, and the run's [`Stats`].
pub fn play_games<const N: usize>(
    oracle: &DruidOracle<N>,
    cfg: &SelfPlayConfig,
    seed: u64,
) -> (Vec<Record>, Vec<Terminal>, Stats) {
    let started = Instant::now();
    let batch = cfg.search.batch_config();
    let ctx = Ctx { oracle, batch: &batch, cfg };
    let mut rngs = Rngs {
        gumbel: SmallRng::seed_from_u64(seed),
        mv: SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
    };

    let mut live: Vec<Live> = (0..cfg.games)
        .map(|g| Live {
            state: HashedState::new(Size { w: N as u8, h: N as u8 }),
            rows: Vec::new(),
            game: g as u32,
        })
        .collect();
    let mut accum = Accum {
        records: Vec::new(),
        terminals: Vec::new(),
        stats: Stats { games_started: cfg.games, ..Stats::default() },
        total_plies: 0,
    };
    let mut ply = 0usize;

    while !live.is_empty() {
        if ply >= cfg.max_plies {
            accum.stats.games_capped += live.len();
            for game in &live {
                let fields = Fields::of(&game.state);
                accum.terminals.push(Terminal {
                    game: game.game,
                    winner: None,
                    capped: true,
                    heights: fields.heights,
                    owners: fields.owners,
                });
            }
            break;
        }
        let states: Vec<HashedState> = live.iter().map(|g| g.state.clone()).collect();
        live = if cfg.skip_forced_search {
            play_ply_skipping_forced(&ctx, &states, live, ply, &mut rngs, &mut accum)
        } else {
            play_ply_searching_every_game(&ctx, &states, live, ply, &mut rngs, &mut accum)
        };
        ply += 1;
    }

    accum.stats.positions = accum.records.len();
    accum.stats.mean_plies = accum.total_plies as f64 / accum.stats.games_finished.max(1) as f64;
    accum.stats.seconds = started.elapsed().as_secs_f64();
    (accum.records, accum.terminals, accum.stats)
}

/// The root value of a batch of independent positions, one per `states` entry, each from that
/// position's own mover's point of view -- the same convention `shard::Record::value` uses.
/// `sims == 0` skips the search: just the net's raw value (or a position's own terminal outcome),
/// batched through one `oracle.init` call, which already parallelizes over every position on its
/// own, so `workers` is unused there. Otherwise it is the root of a Gumbel search of `sims`
/// simulations per position, split across `workers` threads, each running its own batched search
/// over its share of `states` (so a thread's positions still batch together through the oracle).
pub fn root_values<const N: usize>(
    oracle: &DruidOracle<N>,
    states: &[HashedState],
    search: &SearchSettings,
    sims: usize,
    workers: usize,
    seed: u64,
) -> Vec<f32> {
    if states.is_empty() {
        return Vec::new();
    }
    if sims == 0 {
        return oracle.init(states).value_prior;
    }
    let batch = mcts_batch::Config { num_simulations: sims, ..search.batch_config() };
    let workers = workers.clamp(1, states.len());
    let chunk_len = states.len().div_ceil(workers);
    let mut values = vec![0.0f32; states.len()];
    std::thread::scope(|scope| {
        for (worker, (chunk, out)) in
            states.chunks(chunk_len).zip(values.chunks_mut(chunk_len)).enumerate()
        {
            scope.spawn(move || {
                let mut rng =
                    SmallRng::seed_from_u64(seed ^ (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let tree = gumbel_explore(&batch, oracle, chunk, &mut rng);
                for (bid, value) in out.iter_mut().enumerate() {
                    *value = tree.value(bid, tree.root());
                }
            });
        }
    });
    values
}

#[cfg(test)]
mod tests {
    use super::backfill_forced_q;

    /// A searched row is never touched, whatever `movers`/`values` say about it.
    #[test]
    fn a_searched_row_keeps_its_own_q() {
        let movers = [true, true];
        let values = [1.0, 1.0];
        let forced = [false, false];
        let mut q = [0.42, -0.7];
        backfill_forced_q(&movers, &values, &forced, &mut q);
        assert_eq!(q, [0.42, -0.7]);
    }

    /// A forced row followed by a searched row played by the same mover (a turn's own
    /// sub-decisions) takes that row's `q` unchanged.
    #[test]
    fn a_forced_row_takes_the_next_rows_q_unchanged_when_the_mover_is_the_same() {
        let movers = [true, true];
        let values = [1.0, 1.0];
        let forced = [true, false];
        let mut q = [0.0, 0.3];
        backfill_forced_q(&movers, &values, &forced, &mut q);
        assert_eq!(q, [0.3, 0.3]);
    }

    /// A forced row followed by a searched row played by the *other* mover (the turn passed) takes
    /// that row's `q` negated, since `q` is always from its own row's mover's point of view.
    #[test]
    fn a_forced_row_negates_the_next_rows_q_when_the_mover_changes() {
        let movers = [true, false];
        let values = [1.0, -1.0];
        let forced = [true, false];
        let mut q = [0.0, 0.3];
        backfill_forced_q(&movers, &values, &forced, &mut q);
        assert_eq!(q, [-0.3, 0.3]);
    }

    /// A forced row with no next row (its move ended the game) takes its own outcome value
    /// instead of borrowing from a successor.
    #[test]
    fn a_forced_final_row_takes_its_own_outcome_value() {
        let movers = [true, true];
        let values = [1.0, -1.0];
        let forced = [false, true];
        let mut q = [0.5, 0.0];
        backfill_forced_q(&movers, &values, &forced, &mut q);
        assert_eq!(q, [0.5, -1.0]);
    }

    /// A run of consecutive forced rows resolves from the end backwards, so an earlier forced row
    /// correctly chains through a later one that was itself just back-filled: row 2 negates
    /// searched row 3's `q` (the mover changes), then rows 1 and 0 each copy their successor's
    /// (now settled) `q` unchanged (the mover stays the same).
    #[test]
    fn a_run_of_forced_rows_resolves_from_the_end_backwards() {
        let movers = [true, true, true, false];
        let values = [1.0, 1.0, 1.0, -1.0];
        let forced = [true, true, true, false];
        let mut q = [0.0, 0.0, 0.0, 0.6];
        backfill_forced_q(&movers, &values, &forced, &mut q);
        assert_eq!(q, [-0.6, -0.6, -0.6, 0.6]);
    }
}
