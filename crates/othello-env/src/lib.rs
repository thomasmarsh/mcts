//! Batched Othello environment behind a flat C ABI, for the search-free PPO trainer.
//!
//! Rules are not reimplemented here: every transition goes through
//! `game_othello::Othello::apply`, legality through `generate_moves` (the function
//! `generate_actions` is built on), terminal/winner through the `Game` trait.
//!
//! A position is a packed `[u64; 3]`: `[black, white, flags]`, with `flags` bit 0 = White
//! to move and bit 1 = the previous ply was a pass. The pass flag is part of the state
//! because `Othello::is_terminal` needs it (two consecutive passes end the game).
//!
//! Actions are `u8`: `0..64` is a square (row-major, A1 = 0), `64` is PASS, which is legal
//! only when the mover has no square move (exactly `generate_actions`).

use game_othello::{generate_moves, Move, Othello, Player, State, BB};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

pub const STATE_WORDS: usize = 3;
pub const NUM_SQUARES: usize = 64;
pub const NUM_ACTIONS: usize = 65;
pub const PASS_ACTION: u8 = 64;
/// Floats per observation: 2 planes x 64 squares.
pub const OBS_LEN: usize = 2 * NUM_SQUARES;
pub const ABI_VERSION: u32 = 1;

const FLAG_WHITE_TO_MOVE: u64 = 1;
const FLAG_LAST_PASS: u64 = 2;

/// Envs per rayon task: a step is ~a microsecond, so tasks must be coarser than one env.
const PAR_CHUNK: usize = 64;

pub type PackedState = [u64; STATE_WORDS];

type StepChunk<'a> = ((&'a mut [PackedState], &'a [u8]), (&'a mut [f32], &'a mut [u8]));

fn bits(bb: BB) -> u64 {
    bb.words().next().unwrap()
}

pub fn pack(state: &State) -> PackedState {
    let mut flags = 0;
    if state.turn == Player::White {
        flags |= FLAG_WHITE_TO_MOVE;
    }
    if state.last_pass {
        flags |= FLAG_LAST_PASS;
    }
    [bits(state.black), bits(state.white), flags]
}

/// The Zobrist hashes are left zeroed: nothing here reads them, and `apply` only XORs into them.
pub fn unpack(packed: &PackedState) -> State {
    State {
        black: BB::from_bits(packed[0]),
        white: BB::from_bits(packed[1]),
        turn: if packed[2] & FLAG_WHITE_TO_MOVE != 0 { Player::White } else { Player::Black },
        last_pass: packed[2] & FLAG_LAST_PASS != 0,
        hashes: [0; 8],
    }
}

pub fn initial_state() -> PackedState {
    pack(&State::default())
}

/// Bitmask of the mover's legal square moves (empty means the only legal action is PASS).
fn legal_squares(state: &State) -> u64 {
    let (player, opponent) = match state.turn {
        Player::Black => (state.black, state.white),
        Player::White => (state.white, state.black),
    };
    bits(generate_moves(player, opponent))
}

/// A non-terminal position reached by up to `max_depth` uniformly random legal plies, with
/// the number of plies actually played.
///
/// The target depth is drawn uniformly from `0..=max_depth`. Play stops early, before the
/// ply that would end the game, so the returned state is never terminal. Passes count as
/// plies. The realized depth is therefore `min(target, plies before the game would end)`;
/// for `max_depth` up to ~40 the second term is essentially never binding.
pub fn random_start(rng: &mut SmallRng, max_depth: u32) -> (PackedState, u32) {
    let target = rng.gen_range(0..=max_depth);
    let mut state = State::default();
    let mut actions = Vec::with_capacity(32);
    let mut played = 0;
    while played < target {
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        let action = actions[rng.gen_range(0..actions.len())];
        let next = Othello::apply(state, &action);
        if Othello::is_terminal(&next) {
            break;
        }
        state = next;
        played += 1;
    }
    (pack(&state), played)
}

fn env_rng(seed: u64, index: usize) -> SmallRng {
    // splitmix64 finalizer over (seed, index) so neighbouring envs get unrelated streams and
    // the result does not depend on how the batch is split across threads.
    let mut z = seed ^ (index as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    SmallRng::seed_from_u64(z ^ (z >> 31))
}

pub fn reset_random(out: &mut [PackedState], max_depth: u32, seed: u64, parallel: bool) {
    let fill = |(chunk_index, chunk): (usize, &mut [PackedState])| {
        for (i, slot) in chunk.iter_mut().enumerate() {
            *slot = random_start(&mut env_rng(seed, chunk_index * PAR_CHUNK + i), max_depth).0;
        }
    };
    if parallel {
        out.par_chunks_mut(PAR_CHUNK).enumerate().for_each(fill);
    } else {
        out.chunks_mut(PAR_CHUNK).enumerate().for_each(fill);
    }
}

/// Apply `action` for the mover. Returns `(reward, done)`, reward being for the player who just
/// moved (+1 win, -1 loss, 0 draw or game not over), or `None` (state untouched) if the state is
/// already terminal or the action is illegal.
pub fn step_one(packed: &mut PackedState, action: u8) -> Option<(f32, bool)> {
    let state = unpack(packed);
    if Othello::is_terminal(&state) {
        return None;
    }
    let moves = legal_squares(&state);
    let legal = match action {
        0..=63 => moves >> action & 1 == 1,
        PASS_ACTION => moves == 0,
        _ => false,
    };
    if !legal {
        return None;
    }
    let mover = state.turn;
    let next = Othello::apply(state, &Move(action));
    let done = Othello::is_terminal(&next);
    let reward = match (done, Othello::winner(&next)) {
        (true, Some(winner)) if winner == mover => 1.0,
        (true, Some(_)) => -1.0,
        _ => 0.0,
    };
    *packed = pack(&next);
    Some((reward, done))
}

/// Steps every env in place. Returns how many were rejected (terminal state or illegal action);
/// a rejected env is left unchanged with reward 0 and done 0.
pub fn step(
    states: &mut [PackedState],
    actions: &[u8],
    rewards: &mut [f32],
    done: &mut [u8],
    parallel: bool,
) -> usize {
    let n = states.len();
    assert!(actions.len() == n && rewards.len() == n && done.len() == n);
    let run = |((s, a), (r, d)): StepChunk| {
        let mut rejected = 0;
        for (((state, &action), reward), done) in s.iter_mut().zip(a).zip(r).zip(d) {
            match step_one(state, action) {
                Some((rw, dn)) => {
                    *reward = rw;
                    *done = dn as u8;
                }
                None => {
                    *reward = 0.0;
                    *done = 0;
                    rejected += 1;
                }
            }
        }
        rejected
    };
    if parallel {
        states
            .par_chunks_mut(PAR_CHUNK)
            .zip(actions.par_chunks(PAR_CHUNK))
            .zip(rewards.par_chunks_mut(PAR_CHUNK).zip(done.par_chunks_mut(PAR_CHUNK)))
            .map(run)
            .sum()
    } else {
        states
            .chunks_mut(PAR_CHUNK)
            .zip(actions.chunks(PAR_CHUNK))
            .zip(rewards.chunks_mut(PAR_CHUNK).zip(done.chunks_mut(PAR_CHUNK)))
            .map(run)
            .sum()
    }
}

/// Mover-relative observation: plane 0 = the mover's discs, plane 1 = the opponent's, each 64
/// squares row-major. Identical to `CnnValueNet::input(state, 0)` (the literal orientation).
/// `mask[a]` is 1 for each legal action; `mask[64]` (PASS) is 1 only when no square is legal.
pub fn observe_one(packed: &PackedState, obs: &mut [f32], mask: &mut [u8]) {
    let state = unpack(packed);
    let (own, opp) = match state.turn {
        Player::Black => (packed[0], packed[1]),
        Player::White => (packed[1], packed[0]),
    };
    for sq in 0..NUM_SQUARES {
        obs[sq] = (own >> sq & 1) as f32;
        obs[NUM_SQUARES + sq] = (opp >> sq & 1) as f32;
    }
    let moves = legal_squares(&state);
    for (sq, m) in mask[..NUM_SQUARES].iter_mut().enumerate() {
        *m = (moves >> sq & 1) as u8;
    }
    mask[NUM_SQUARES] = (moves == 0) as u8;
}

pub fn observe(states: &[PackedState], obs: &mut [f32], mask: &mut [u8], parallel: bool) {
    let n = states.len();
    assert!(obs.len() == n * OBS_LEN && mask.len() == n * NUM_ACTIONS);
    let run = |((s, o), m): ((&[PackedState], &mut [f32]), &mut [u8])| {
        for ((state, o), m) in s.iter().zip(o.chunks_mut(OBS_LEN)).zip(m.chunks_mut(NUM_ACTIONS)) {
            observe_one(state, o, m);
        }
    };
    if parallel {
        states
            .par_chunks(PAR_CHUNK)
            .zip(obs.par_chunks_mut(PAR_CHUNK * OBS_LEN))
            .zip(mask.par_chunks_mut(PAR_CHUNK * NUM_ACTIONS))
            .for_each(run);
    } else {
        states
            .chunks(PAR_CHUNK)
            .zip(obs.chunks_mut(PAR_CHUNK * OBS_LEN))
            .zip(mask.chunks_mut(PAR_CHUNK * NUM_ACTIONS))
            .for_each(run);
    }
}

// ---------------------------------------------------------------------------
// C ABI. Every pointer is a C-contiguous array with `n` rows; `n == 0` accepts null.
// `parallel != 0` runs on rayon's global pool (size: RAYON_NUM_THREADS), 0 on the caller.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn othello_env_abi_version() -> u32 {
    ABI_VERSION
}

/// Fills `out_states` (`n x 3` u64) with non-terminal random-play positions; see [`random_start`].
///
/// # Safety
/// `out_states` must be valid for writing `n * 3` u64s.
#[no_mangle]
pub unsafe extern "C" fn othello_env_reset_random(
    n: usize,
    max_depth: u32,
    seed: u64,
    parallel: i32,
    out_states: *mut u64,
) {
    if n == 0 {
        return;
    }
    let out = std::slice::from_raw_parts_mut(out_states as *mut PackedState, n);
    reset_random(out, max_depth, seed, parallel != 0);
}

/// Steps `n` envs in place; see [`step`]. `actions` is `n` u8, `out_reward` `n` f32, `out_done`
/// `n` u8. Returns the number of rejected envs (0 means every action was legal).
///
/// # Safety
/// Every pointer must be valid for `n` rows of its type; `states` is read and written.
#[no_mangle]
pub unsafe extern "C" fn othello_env_step(
    n: usize,
    states: *mut u64,
    actions: *const u8,
    out_reward: *mut f32,
    out_done: *mut u8,
    parallel: i32,
) -> i64 {
    if n == 0 {
        return 0;
    }
    let states = std::slice::from_raw_parts_mut(states as *mut PackedState, n);
    let actions = std::slice::from_raw_parts(actions, n);
    let rewards = std::slice::from_raw_parts_mut(out_reward, n);
    let done = std::slice::from_raw_parts_mut(out_done, n);
    step(states, actions, rewards, done, parallel != 0) as i64
}

/// Writes `n x 2 x 8 x 8` f32 observations and an `n x 65` u8 legal-action mask; see [`observe_one`].
///
/// # Safety
/// `states` must be valid for reading `n * 3` u64s, `out_obs` for writing `n * 128` f32s and
/// `out_mask` for writing `n * 65` bytes.
#[no_mangle]
pub unsafe extern "C" fn othello_env_observe(
    n: usize,
    states: *const u64,
    out_obs: *mut f32,
    out_mask: *mut u8,
    parallel: i32,
) {
    if n == 0 {
        return;
    }
    let states = std::slice::from_raw_parts(states as *const PackedState, n);
    let obs = std::slice::from_raw_parts_mut(out_obs, n * OBS_LEN);
    let mask = std::slice::from_raw_parts_mut(out_mask, n * NUM_ACTIONS);
    observe(states, obs, mask, parallel != 0);
}

#[cfg(test)]
mod tests;
