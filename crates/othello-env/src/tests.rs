use super::*;
use game_othello::convnet::CnnValueNet;
use game_othello::{naive_apply, naive_generate_moves};

fn same_position(a: &State, b: &State) -> bool {
    bits(a.black) == bits(b.black) && bits(a.white) == bits(b.white) && a.turn == b.turn && a.last_pass == b.last_pass
}

fn actions_of(state: &State) -> Vec<Move> {
    let mut actions = Vec::new();
    Othello::generate_actions(state, &mut actions);
    actions
}

/// Uniformly random legal plies from the opening until the game ends.
fn playout(rng: &mut SmallRng) -> Vec<State> {
    let mut states = vec![State::default()];
    while !Othello::is_terminal(states.last().unwrap()) {
        let state = *states.last().unwrap();
        let actions = actions_of(&state);
        let action = actions[rng.gen_range(0..actions.len())];
        states.push(Othello::apply(state, &action));
    }
    states
}

fn disc_margin(state: &State, player: Player) -> i32 {
    let (b, w) = (state.black.count_ones() as i32, state.white.count_ones() as i32);
    if player == Player::Black { b - w } else { w - b }
}

#[test]
fn pack_roundtrips() {
    let mut rng = SmallRng::seed_from_u64(1);
    for state in playout(&mut rng) {
        assert!(same_position(&state, &unpack(&pack(&state))));
    }
    assert_eq!(initial_state(), [game_othello::INITIAL_BLACK, game_othello::INITIAL_WHITE, 0]);
}

#[test]
fn random_playouts_agree_with_apply_and_naive_apply() {
    let mut rng = SmallRng::seed_from_u64(2);
    let (mut passes, mut plies) = (0, 0);
    for _ in 0..1000 {
        for pair in playout(&mut rng).windows(2) {
            let (before, after) = (pair[0], pair[1]);
            let action = actions_of(&before).into_iter().find(|a| {
                same_position(&Othello::apply(before, a), &after)
            });
            let action = action.expect("the playout's next state must come from a legal action");
            assert!(same_position(&naive_apply(before, &action), &after), "naive_apply disagrees");

            let mut packed = pack(&before);
            let (_, done) = step_one(&mut packed, action.0).expect("legal action rejected");
            assert!(same_position(&unpack(&packed), &after), "env step disagrees with Othello::apply");
            assert_eq!(done, Othello::is_terminal(&after));

            let (player, opponent) = match before.turn {
                Player::Black => (before.black, before.white),
                Player::White => (before.white, before.black),
            };
            assert_eq!(bits(naive_generate_moves(player, opponent)), legal_squares(&before));
            passes += (action == Move::PASS) as usize;
            plies += 1;
        }
    }
    assert!(passes > 0, "no pass occurred in {plies} plies; the pass path is untested");
}

#[test]
fn terminal_reward_sign_and_mover_attribution_on_playouts() {
    let mut rng = SmallRng::seed_from_u64(3);
    let (mut wins, mut losses, mut draws, mut pass_ended, mut board_full) = (0, 0, 0, 0, 0);
    for _ in 0..1000 {
        let states = playout(&mut rng);
        for pair in states.windows(2) {
            let (before, after) = (pair[0], pair[1]);
            let action = actions_of(&before).into_iter().find(|a| same_position(&Othello::apply(before, a), &after)).unwrap();
            let mut packed = pack(&before);
            let (reward, done) = step_one(&mut packed, action.0).unwrap();
            if !done {
                assert_eq!(reward, 0.0, "non-terminal step carried a reward");
                continue;
            }
            let margin = disc_margin(&after, before.turn);
            assert_eq!(reward, margin.signum() as f32, "reward must be the mover's disc margin sign");
            match margin.signum() {
                1 => wins += 1,
                -1 => losses += 1,
                _ => draws += 1,
            }
            pass_ended += (action == Move::PASS) as usize;
            board_full += (after.black.count_ones() + after.white.count_ones() == 64) as usize;
            assert!(after.turn != before.turn);
        }
    }
    assert!(wins > 0 && losses > 0 && draws > 0, "wins {wins} losses {losses} draws {draws}");
    assert!(pass_ended > 0 && board_full > 0, "pass_ended {pass_ended} board_full {board_full}");
}

fn state_of(black: u64, white: u64, flags: u64) -> PackedState {
    [black, white, flags]
}

const ALL_BUT_H8: u64 = (1 << 63) - 1;

#[test]
fn terminal_reward_is_for_the_mover_not_for_black() {
    // One empty square (63). The mover flips 62 through the west direction only.
    // Mover wins big: everything else is the mover's colour.
    let mover_wins = |black_moves: bool| {
        let own = ALL_BUT_H8 & !(1 << 62);
        let opp = 1 << 62;
        if black_moves { state_of(own, opp, 0) } else { state_of(opp, own, FLAG_WHITE_TO_MOVE) }
    };
    // Mover loses: 0..=60 belong to the opponent, mover owns 61 and flips 62 (3 v 61).
    let mover_loses = |black_moves: bool| {
        let opp = (1 << 61) - 1;
        let own = 1 << 61;
        let opp_after = opp | 1 << 62;
        if black_moves { state_of(own, opp_after, 0) } else { state_of(opp_after, own, FLAG_WHITE_TO_MOVE) }
    };
    for black_moves in [true, false] {
        let mut s = mover_wins(black_moves);
        assert_eq!(step_one(&mut s, 63), Some((1.0, true)), "mover (black={black_moves}) wins");
        let mut s = mover_loses(black_moves);
        assert_eq!(step_one(&mut s, 63), Some((-1.0, true)), "mover (black={black_moves}) loses");
    }
}

#[test]
fn a_terminating_pass_is_credited_to_the_passer() {
    // Black has 2 discs, White 1, neither can move. Black passes, White has no move either.
    let mut s = state_of(0b11, 1 << 63, 0);
    assert_eq!(step_one(&mut s, PASS_ACTION), Some((1.0, true)));
    // Same position, White to move and White is the passer: White has fewer discs, so -1.
    let mut s = state_of(0b11, 1 << 63, FLAG_WHITE_TO_MOVE);
    assert_eq!(step_one(&mut s, PASS_ACTION), Some((-1.0, true)));
}

#[test]
fn pass_advances_the_turn_and_a_move_clears_the_pass_flag() {
    // White (to move) cannot flank anything, Black can play square 2 afterwards.
    let mut s = state_of(1, 1 << 1, FLAG_WHITE_TO_MOVE);
    let (mut obs, mut mask) = ([0.0; OBS_LEN], [0u8; NUM_ACTIONS]);
    observe_one(&s, &mut obs, &mut mask);
    assert_eq!(mask.iter().map(|&m| m as usize).sum::<usize>(), 1);
    assert_eq!(mask[64], 1);

    assert_eq!(step_one(&mut s, PASS_ACTION), Some((0.0, false)));
    assert_eq!(s, state_of(1, 1 << 1, FLAG_LAST_PASS), "turn passes to Black, discs untouched, pass recorded");

    assert_eq!(step_one(&mut s, 2), Some((0.0, false)));
    assert_eq!(s, state_of(0b111, 0, FLAG_WHITE_TO_MOVE), "a real move flips and clears the pass flag");
}

#[test]
fn illegal_actions_and_terminal_states_are_rejected_without_mutation() {
    let start = initial_state();
    for action in [PASS_ACTION, 0, 63, 65, 255] {
        let mut s = start;
        assert_eq!(step_one(&mut s, action), None, "action {action}");
        assert_eq!(s, start);
    }
    let mut full = state_of(u64::MAX, 0, 0);
    assert_eq!(step_one(&mut full, 0), None);
    assert_eq!(step_one(&mut full, PASS_ACTION), None);

    let mut states = [start, start];
    let (mut rewards, mut done) = ([9.0; 2], [9u8; 2]);
    let rejected = step(&mut states, &[19, 0], &mut rewards, &mut done, false);
    assert_eq!(rejected, 1);
    assert_eq!((rewards, done), ([0.0, 0.0], [0, 0]));
    assert_ne!(states[0], start);
    assert_eq!(states[1], start);
}

#[test]
fn reset_random_is_never_terminal_and_depth_is_uniform() {
    let mut rng = SmallRng::seed_from_u64(4);
    let max_depth = 20;
    let mut histogram = [0usize; 21];
    for _ in 0..21_000 {
        let (packed, played) = random_start(&mut rng, max_depth);
        let state = unpack(&packed);
        assert!(!Othello::is_terminal(&state));
        let discs = state.black.count_ones() + state.white.count_ones();
        assert!(discs >= 4 && discs <= 4 + played, "discs {discs} after {played} plies");
        if played == 0 {
            assert_eq!(packed, initial_state());
        }
        histogram[played as usize] += 1;
    }
    // Realized depth == target depth here (a 20-ply game end is vanishingly rare): 1000 each,
    // sigma ~31, so +-25% is ~8 sigma.
    for (depth, &count) in histogram.iter().enumerate() {
        assert!((750..=1250).contains(&count), "depth {depth} drawn {count} times: {histogram:?}");
    }

    let mut rng = SmallRng::seed_from_u64(5);
    let mut deepest = 0;
    for _ in 0..5000 {
        let (packed, played) = random_start(&mut rng, 60);
        assert!(!Othello::is_terminal(&unpack(&packed)));
        deepest = deepest.max(played);
    }
    assert!(deepest >= 55, "deepest reset {deepest} of max 60");
}

#[test]
fn reset_random_depends_only_on_seed_and_env_index() {
    let (mut serial, mut parallel, mut prefix, mut other) =
        (vec![[0; 3]; 300], vec![[0; 3]; 300], vec![[0; 3]; 100], vec![[0; 3]; 300]);
    reset_random(&mut serial, 30, 7, false);
    reset_random(&mut parallel, 30, 7, true);
    reset_random(&mut prefix, 30, 7, true);
    reset_random(&mut other, 30, 8, false);
    assert_eq!(serial, parallel);
    assert_eq!(serial[..100], prefix[..]);
    assert_ne!(serial, other);
}

#[test]
fn observation_and_mask_match_the_net_input_and_generate_actions() {
    let mut rng = SmallRng::seed_from_u64(6);
    let mut states = Vec::new();
    for _ in 0..40 {
        states.extend(playout(&mut rng));
    }
    let packed: Vec<PackedState> = states.iter().map(pack).collect();
    let n = packed.len();
    let (mut obs, mut mask) = (vec![0.0; n * OBS_LEN], vec![0u8; n * NUM_ACTIONS]);
    observe(&packed, &mut obs, &mut mask, false);
    let (mut obs_p, mut mask_p) = (vec![0.0; n * OBS_LEN], vec![0u8; n * NUM_ACTIONS]);
    observe(&packed, &mut obs_p, &mut mask_p, true);
    assert_eq!((&obs, &mask), (&obs_p, &mask_p));

    for (i, state) in states.iter().enumerate() {
        assert_eq!(obs[i * OBS_LEN..(i + 1) * OBS_LEN], CnnValueNet::input(state, 0), "obs {i}");

        let mut expected = [0u8; NUM_ACTIONS];
        for a in actions_of(state) {
            expected[a.0 as usize] = 1;
        }
        assert_eq!(mask[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS], expected, "mask {i}");
    }
}

#[test]
fn ffi_loop_plays_random_games_to_the_end() {
    let n = 64;
    let mut states = vec![0u64; n * STATE_WORDS];
    let (mut obs, mut mask) = (vec![0f32; n * OBS_LEN], vec![0u8; n * NUM_ACTIONS]);
    let (mut rewards, mut done) = (vec![0f32; n], vec![0u8; n]);
    let mut actions = vec![0u8; n];
    let mut rng = SmallRng::seed_from_u64(9);
    assert_eq!(othello_env_abi_version(), ABI_VERSION);
    unsafe {
        othello_env_reset_random(n, 10, 3, 1, states.as_mut_ptr());
        // A finished env is parked on the opening with a fixed legal action (square 19) so the
        // whole batch can keep stepping without a rejection.
        let mut finished = vec![false; n];
        for _ in 0..200 {
            othello_env_observe(n, states.as_ptr(), obs.as_mut_ptr(), mask.as_mut_ptr(), 1);
            for i in 0..n {
                if finished[i] {
                    states[i * STATE_WORDS..(i + 1) * STATE_WORDS].copy_from_slice(&initial_state());
                    actions[i] = 19;
                    continue;
                }
                let legal: Vec<u8> = (0..NUM_ACTIONS as u8).filter(|&a| mask[i * NUM_ACTIONS + a as usize] == 1).collect();
                actions[i] = legal[rng.gen_range(0..legal.len())];
            }
            let rejected = othello_env_step(n, states.as_mut_ptr(), actions.as_ptr(), rewards.as_mut_ptr(), done.as_mut_ptr(), 1);
            assert_eq!(rejected, 0);
            for i in 0..n {
                if !finished[i] && done[i] == 1 {
                    finished[i] = true;
                    assert!([-1.0, 0.0, 1.0].contains(&rewards[i]));
                }
            }
            if finished.iter().all(|&f| f) {
                break;
            }
        }
        assert!(finished.iter().all(|&f| f), "every env should finish within 200 plies");

        othello_env_reset_random(n, 0, 1, 0, states.as_mut_ptr());
        actions.fill(19);
        actions[0] = 0;
        let rejected = othello_env_step(n, states.as_mut_ptr(), actions.as_ptr(), rewards.as_mut_ptr(), done.as_mut_ptr(), 0);
        assert_eq!(rejected, 1);
    }
}
