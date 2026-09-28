//! Batched Gumbel self-play from the empty board. Every live game advances one ply (one
//! sub-decision of a turn) per `gumbel_explore` call, so a generation is a few thousand large GPU
//! calls rather than one call per leaf. Each position is recorded with the completed-Q improved
//! policy as its policy target and, once the game ends, the mover's outcome as its value target.
//! Games that reach `max_plies` are dropped and counted.

use std::time::Instant;

use mcts::game::Game;
use mcts_batch::{gumbel_explore, gumbel_explore_with_noise, gumbel_selected_action, improved_policy, EnvOracle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::config::{SearchSettings, SelfPlayConfig};
use super::encode::{move_from_id, num_actions};
use super::oracle::DruidOracle;
use super::shard::{Fields, Record};
use crate::{DruidSplit, HashedState, Player, Size};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub games_started: usize,
    pub games_finished: usize,
    pub games_capped: usize,
    pub positions: usize,
    pub mean_plies: f64,
    pub black_wins: usize,
    pub draws: usize,
    pub seconds: f64,
}

struct Row {
    fields: Fields,
    ply: u16,
    policy: Vec<f32>,
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

pub fn play_games<const N: usize>(
    oracle: &DruidOracle<N>,
    cfg: &SelfPlayConfig,
    seed: u64,
) -> (Vec<Record>, Stats) {
    let started = Instant::now();
    let batch = cfg.search.batch_config();
    let mut gumbel_rng = SmallRng::seed_from_u64(seed);
    let mut move_rng = SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let a = num_actions(N);

    let mut live: Vec<Live> = (0..cfg.games)
        .map(|g| Live {
            state: HashedState::new(Size { w: N as u8, h: N as u8 }),
            rows: Vec::new(),
            game: g as u32,
        })
        .collect();
    let mut records = Vec::new();
    let mut stats = Stats { games_started: cfg.games, ..Stats::default() };
    let mut total_plies = 0usize;
    let mut ply = 0usize;

    while !live.is_empty() {
        if ply >= cfg.max_plies {
            stats.games_capped += live.len();
            break;
        }
        let states: Vec<HashedState> = live.iter().map(|g| g.state.clone()).collect();
        let (tree, noise) = gumbel_explore_with_noise(&batch, oracle, &states, &mut gumbel_rng);

        let mut next = Vec::with_capacity(live.len());
        for (bid, mut game) in live.into_iter().enumerate() {
            let policy = improved_policy(&batch, &tree, bid, tree.root());
            game.rows.push(Row {
                fields: Fields::of(&game.state),
                ply: ply as u16,
                policy: policy.clone(),
            });
            let id = if ply < cfg.temp_moves {
                sample(&policy, &mut move_rng)
            } else {
                gumbel_selected_action(&batch, &tree, bid, &noise[bid])
            };
            game.state = DruidSplit::apply(game.state, &move_from_id(id, N));
            if DruidSplit::is_terminal(&game.state) {
                let winner = DruidSplit::winner(&game.state);
                stats.games_finished += 1;
                stats.black_wins += usize::from(winner == Some(Player::Black));
                stats.draws += usize::from(winner.is_none());
                total_plies += ply + 1;
                for row in game.rows {
                    debug_assert_eq!(row.policy.len(), a);
                    let value = match winner {
                        None => 0.0,
                        Some(w) => {
                            let black_moves = row.fields.is_black_to_move();
                            if black_moves == (w == Player::Black) { 1.0 } else { -1.0 }
                        }
                    };
                    records.push(Record {
                        fields: row.fields,
                        value,
                        game: game.game,
                        ply: row.ply,
                        policy: row.policy,
                    });
                }
            } else {
                next.push(game);
            }
        }
        live = next;
        ply += 1;
    }

    stats.positions = records.len();
    stats.mean_plies = total_plies as f64 / stats.games_finished.max(1) as f64;
    stats.seconds = started.elapsed().as_secs_f64();
    (records, stats)
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
