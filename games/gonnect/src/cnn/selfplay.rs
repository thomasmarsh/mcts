//! Batched Gumbel self-play from the empty board (swap is an ordinary action at ply 1). Every
//! live game advances one ply per `gumbel_explore` call, so a generation is a few thousand large
//! GPU calls rather than one call per leaf. Each position is recorded with the completed-Q
//! improved policy as its policy target and, once the game ends, the mover's outcome as its value
//! target. Games that reach `max_plies` are dropped and counted.

use std::time::Instant;

use mcts::game::Game;
use mcts_batch::{gumbel_explore_with_noise, gumbel_selected_action, improved_policy};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::config::SelfPlayConfig;
use super::encode::{
    action_id, analyse, move_from_id, num_actions, swap_id, Fields, FLAG_BLACK_TO_MOVE,
};
use super::oracle::GonnectOracle;
use super::shard::Record;
use crate::sized::{SizedGonnect, SizedState};
use crate::Player;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub games_started: usize,
    pub games_finished: usize,
    pub games_capped: usize,
    pub positions: usize,
    pub mean_plies: f64,
    pub black_wins: usize,
    /// Finished games whose second move (White's reply) was the swap.
    pub swaps: usize,
    pub seconds: f64,
}

struct Row {
    fields: Fields,
    ply: u8,
    policy: Vec<f32>,
}

struct Live<const N: usize> {
    state: SizedState<N>,
    rows: Vec<Row>,
    game: u32,
    swapped: bool,
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
    oracle: &GonnectOracle<N>,
    cfg: &SelfPlayConfig,
    seed: u64,
) -> (Vec<Record>, Stats) {
    let started = Instant::now();
    let batch = cfg.search.batch_config();
    let mut gumbel_rng = SmallRng::seed_from_u64(seed);
    let mut move_rng = SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let a = num_actions(N);

    let mut live: Vec<Live<N>> = (0..cfg.games)
        .map(|g| Live {
            state: SizedState::<N>::default(),
            rows: Vec::new(),
            game: g as u32,
            swapped: false,
        })
        .collect();
    let mut records = Vec::new();
    let mut stats = Stats {
        games_started: cfg.games,
        ..Stats::default()
    };
    let mut total_plies = 0usize;
    let mut ply = 0usize;

    while !live.is_empty() {
        if ply >= cfg.max_plies {
            stats.games_capped += live.len();
            break;
        }
        let states: Vec<SizedState<N>> = live.iter().map(|g| g.state.clone()).collect();
        let (tree, noise) = gumbel_explore_with_noise(&batch, oracle, &states, &mut gumbel_rng);

        let mut next = Vec::with_capacity(live.len());
        for (bid, mut game) in live.into_iter().enumerate() {
            let policy = improved_policy(&batch, &tree, bid, tree.root());
            let (fields, _) = analyse(&game.state.0);
            game.rows.push(Row {
                fields,
                ply: ply as u8,
                policy: policy.clone(),
            });
            let id = if ply < cfg.temp_moves {
                sample(&policy, &mut move_rng)
            } else {
                gumbel_selected_action(&batch, &tree, bid, &noise[bid])
            };
            if ply == 1 {
                game.swapped = id == swap_id(N);
            }
            let mv = move_from_id(&game.state.0, id);
            debug_assert_eq!(action_id(&mv, N), id);
            game.state = SizedGonnect::<N>::apply(game.state, &mv);
            if SizedGonnect::<N>::is_terminal(&game.state) {
                let winner = game.state.0.turn();
                stats.games_finished += 1;
                stats.black_wins += usize::from(winner == Player::Black);
                stats.swaps += usize::from(game.swapped);
                total_plies += ply + 1;
                for row in game.rows {
                    let mover_is_winner =
                        (row.fields.flags & FLAG_BLACK_TO_MOVE != 0) == (winner == Player::Black);
                    debug_assert_eq!(row.policy.len(), a);
                    records.push(Record {
                        fields: row.fields,
                        value: if mover_is_winner { 1.0 } else { -1.0 },
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
