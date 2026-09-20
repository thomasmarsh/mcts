//! Exact win/draw/loss solver for the last empties of an Othello game.
//!
//! Plain fail-soft alpha-beta over raw bitboards with the window (-1, 1), which
//! is exactly wide enough to separate the three outcomes. Passes are ordinary
//! moves; two consecutive passes end the game and the side with more discs wins
//! (empties count for nobody, as in `Othello::winner`). Interior nodes with
//! enough empties try the move that leaves the opponent the fewest replies
//! first, and a transposition table (kept per thread across calls, since a
//! position's value never depends on how it was reached) caches bounds.

use std::cell::RefCell;

use crate::{generate_moves, get_flips, State, Player, BB};

const TT_BITS: u32 = 19;
/// Nodes with fewer empties than this skip the table and the mobility sort:
/// they cost less than the lookup would.
const TT_MIN_EMPTIES: u32 = 6;
const SORT_MIN_EMPTIES: u32 = 7;

const EXACT: u8 = 1;
const LOWER: u8 = 2;
const UPPER: u8 = 3;

#[derive(Clone, Copy, Default)]
struct Entry {
    player: u64,
    opponent: u64,
    value: i8,
    kind: u8,
}

struct Table {
    entries: Vec<Entry>,
}

impl Table {
    fn slot(player: u64, opponent: u64) -> usize {
        let h = player.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ opponent.rotate_left(29).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        (h >> (64 - TT_BITS)) as usize
    }
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table { entries: vec![Entry::default(); 1 << TT_BITS] });
}

#[inline]
fn moves_of(player: u64, opponent: u64) -> u64 {
    generate_moves(BB::from_bits(player), BB::from_bits(opponent)).bits()
}

#[inline]
fn play(player: u64, opponent: u64, mv: u64) -> (u64, u64) {
    let flips = get_flips(BB::from_bits(player), BB::from_bits(opponent), BB::from_bits(mv)).bits();
    (player | mv | flips, opponent & !flips)
}

#[inline]
fn outcome(player: u64, opponent: u64) -> i8 {
    (player.count_ones() as i32 - opponent.count_ones() as i32).signum() as i8
}

fn search(table: &mut Table, player: u64, opponent: u64, mut alpha: i8, beta: i8) -> i8 {
    let moves = moves_of(player, opponent);
    if moves == 0 {
        if moves_of(opponent, player) == 0 {
            return outcome(player, opponent);
        }
        return -search(table, opponent, player, -beta, -alpha);
    }
    let empties = 64 - (player | opponent).count_ones();
    let alpha0 = alpha;
    let slot = Table::slot(player, opponent);
    if empties >= TT_MIN_EMPTIES {
        let e = table.entries[slot];
        if e.kind != 0 && e.player == player && e.opponent == opponent {
            match e.kind {
                EXACT => return e.value,
                LOWER if e.value >= beta => return e.value,
                UPPER if e.value <= alpha => return e.value,
                _ => {}
            }
        }
    }

    let mut best = i8::MIN;
    if empties >= SORT_MIN_EMPTIES {
        let mut order = [(0u32, 0u64); 32];
        let mut n = 0;
        let mut rest = moves;
        while rest != 0 {
            let mv = rest & rest.wrapping_neg();
            rest &= rest - 1;
            let (p, o) = play(player, opponent, mv);
            order[n] = (moves_of(o, p).count_ones(), mv);
            n += 1;
        }
        order[..n].sort_unstable_by_key(|&(mobility, _)| mobility);
        for &(_, mv) in &order[..n] {
            let (p, o) = play(player, opponent, mv);
            let v = -search(table, o, p, -beta, -alpha);
            if v > best {
                best = v;
            }
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }
    } else {
        let mut rest = moves;
        while rest != 0 {
            let mv = rest & rest.wrapping_neg();
            rest &= rest - 1;
            let (p, o) = play(player, opponent, mv);
            let v = -search(table, o, p, -beta, -alpha);
            if v > best {
                best = v;
            }
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }
    }

    if empties >= TT_MIN_EMPTIES {
        let kind = if best <= alpha0 {
            UPPER
        } else if best >= beta {
            LOWER
        } else {
            EXACT
        };
        table.entries[slot] = Entry { player, opponent, value: best, kind };
    }
    best
}

/// The exact result of `state` for the player to move: 1 win, 0 draw, -1 loss.
pub fn solve_wld(state: &State) -> i8 {
    let (player, opponent) = match state.turn {
        Player::Black => (state.black, state.white),
        Player::White => (state.white, state.black),
    };
    TABLE.with(|t| search(&mut t.borrow_mut(), player.bits(), opponent.bits(), -1, 1))
}

/// The same result without the table or the move ordering: the reference the
/// optimised search is tested against.
#[cfg(test)]
pub(crate) fn solve_wld_plain(player: u64, opponent: u64) -> i8 {
    let moves = moves_of(player, opponent);
    if moves == 0 {
        if moves_of(opponent, player) == 0 {
            return outcome(player, opponent);
        }
        return -solve_wld_plain(opponent, player);
    }
    let mut best = i8::MIN;
    let mut rest = moves;
    while rest != 0 {
        let mv = rest & rest.wrapping_neg();
        rest &= rest - 1;
        let (p, o) = play(player, opponent, mv);
        best = best.max(-solve_wld_plain(o, p));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Othello;
    use mcts::game::Game;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// A random unfinished position with exactly `empties` empty cells (playing
    /// out with passes as ordinary moves, and retrying games that end early).
    fn random_position(rng: &mut SmallRng, empties: u32) -> State {
        loop {
            let mut s = State::default();
            let mut actions = Vec::new();
            while !Othello::is_terminal(&s) && 64 - s.occupied().count_ones() > empties {
                actions.clear();
                Othello::generate_actions(&s, &mut actions);
                s = Othello::apply(s, &actions[rng.gen_range(0..actions.len())]);
            }
            if !Othello::is_terminal(&s) && 64 - s.occupied().count_ones() == empties {
                return s;
            }
        }
    }

    fn sides(s: &State) -> (u64, u64) {
        match s.turn {
            Player::Black => (s.black.bits(), s.white.bits()),
            Player::White => (s.white.bits(), s.black.bits()),
        }
    }

    #[test]
    fn the_optimised_solver_matches_the_plain_one_on_random_positions_including_pass_lines() {
        let mut rng = SmallRng::seed_from_u64(21);
        let mut seen = [0u32; 3];
        for i in 0..60 {
            let empties = if i < 40 { 7 + (i % 3) } else { 4 + (i % 5) };
            let s = random_position(&mut rng, empties);
            let (p, o) = sides(&s);
            let got = solve_wld(&s);
            assert_eq!(got, solve_wld_plain(p, o), "{empties} empties:\n{s}");
            seen[(got + 1) as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0), "wins, draws and losses all occur: {seen:?}");
    }

    #[test]
    fn positions_where_the_side_to_move_must_pass_are_solved_correctly() {
        let mut rng = SmallRng::seed_from_u64(23);
        let mut found = 0;
        for _ in 0..20_000 {
            let s = random_position(&mut rng, 6 + (found % 4));
            let (p, o) = sides(&s);
            if moves_of(p, o) == 0 {
                assert_eq!(solve_wld(&s), solve_wld_plain(p, o), "{s}");
                found += 1;
                if found == 12 {
                    return;
                }
            }
        }
        panic!("only {found} forced-pass positions found");
    }

    #[test]
    fn the_shared_table_never_changes_an_answer_when_positions_are_solved_repeatedly() {
        let mut rng = SmallRng::seed_from_u64(22);
        let states: Vec<State> = (0..12).map(|_| random_position(&mut rng, 9)).collect();
        let first: Vec<i8> = states.iter().map(solve_wld).collect();
        let second: Vec<i8> = states.iter().rev().map(solve_wld).collect();
        let expected: Vec<i8> = states
            .iter()
            .map(|s| {
                let (p, o) = sides(s);
                solve_wld_plain(p, o)
            })
            .collect();
        assert_eq!(first, expected);
        assert_eq!(second.into_iter().rev().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn a_position_where_neither_side_can_move_is_scored_by_disc_count() {
        // Full board, 40 black to 24 white: black to move wins, white to move loses.
        let black = State {
            black: BB::from_bits(u64::MAX >> 24),
            white: BB::from_bits(!(u64::MAX >> 24)),
            turn: Player::Black,
            ..State::default()
        };
        assert_eq!(solve_wld(&black), 1);
        assert_eq!(solve_wld(&State { turn: Player::White, ..black }), -1);
        let even = State {
            black: BB::from_bits(u64::MAX >> 32),
            white: BB::from_bits(!(u64::MAX >> 32)),
            ..black
        };
        assert_eq!(solve_wld(&even), 0);
        assert!(Othello::is_terminal(&even));
    }
}
