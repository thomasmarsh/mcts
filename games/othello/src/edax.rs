//! Conversions between this crate's [`State`] and the board / move strings
//! spoken by the Edax engine's native line protocol.
//!
//! Edax is an open-source alpha-beta Othello engine, superhuman on 8x8. The
//! match harness (`games/othello/examples/edax_match.rs`) drives a built
//! Edax binary as an external strength reference; these functions are the
//! wire mapping it depends on, split out here so they get a fast,
//! deterministic `cargo test --lib` check -- a transposed square mapping
//! would make every match number meaningless without failing anything else.
//!
//! ## Edax board string
//!
//! 64 characters, one per square in row-major order from A1: string offset
//! `row * 8 + col`, where `row 0` is rank 1 and `col 0` is file A. This is
//! exactly this crate's bit index (`BB::to_coord(i) == (i / 8, i % 8)`,
//! `Othello::notation` = file `a + col`, rank `row + 1`), so square `i`
//! maps to string offset `i` with no transposition.
//!
//! Edax's `board_set` reads `X`/`*`/`B` as **black** discs and `O`/`W` as
//! **white** discs regardless of side to move, `-`/`.` as empty, then a
//! trailing non-square token gives the side to move (`X`/`*`/`B` = black,
//! `O`/`W` = white). We emit `X` / `O` / `-` and a trailing `X` or `O`.
//!
//! ## Edax move string
//!
//! Edax replies `Edax plays <MOVE>` where `<MOVE>` is `<file><rank>`
//! upper-cased (e.g. `D3`), or `PA` for a pass.

use crate::{Move, Player, State};

/// Encode `state` as an Edax `setboard` argument: 64 square characters
/// followed by a space and the side-to-move token.
pub fn state_to_edax_board(state: &State) -> String {
    let black = state.black.bits();
    let white = state.white.bits();
    let mut s = String::with_capacity(66);
    for i in 0..64u32 {
        let bit = 1u64 << i;
        s.push(if black & bit != 0 {
            'X'
        } else if white & bit != 0 {
            'O'
        } else {
            '-'
        });
    }
    s.push(' ');
    s.push(match state.turn {
        Player::Black => 'X',
        Player::White => 'O',
    });
    s
}

/// Parse an Edax move token (`"D3"`, `"d3"`, `"PA"`, `"pa"`, or the tail of
/// a `"Edax plays D3"` line already split off) into a [`Move`]. Returns
/// `None` if the token is not a legal square or pass.
pub fn edax_move_to_index(token: &str) -> Option<Move> {
    let t = token.trim().to_ascii_lowercase();
    let b = t.as_bytes();
    if b == b"pa" || b == b"pass" {
        return Some(Move::PASS);
    }
    if b.len() != 2 {
        return None;
    }
    let col = b[0].checked_sub(b'a')?;
    let row = b[1].checked_sub(b'1')?;
    if col > 7 || row > 7 {
        return None;
    }
    Some(Move(row * 8 + col))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Othello, BB};
    use mcts::game::Game;

    /// The known Edax string for the standard opening: white on d4/e5,
    /// black on e4/d5, black to move. d4 = index 27, e4 = 28, d5 = 35,
    /// e5 = 36.
    const OPENING: &str =
        "---------------------------OX------XO--------------------------- X";
    // (27 dashes + "OX" + 6 dashes + "XO" + 27 dashes, then " X")

    #[test]
    fn default_state_round_trips_to_the_known_opening_string() {
        assert_eq!(state_to_edax_board(&State::default()), OPENING);
    }

    #[test]
    fn opening_string_places_the_four_center_discs() {
        let bytes = OPENING.as_bytes();
        assert_eq!(bytes[27], b'O');
        assert_eq!(bytes[28], b'X');
        assert_eq!(bytes[35], b'X');
        assert_eq!(bytes[36], b'O');
        assert_eq!(bytes[64], b' ');
        assert_eq!(bytes[65], b'X');
        assert_eq!(bytes.iter().filter(|&&c| c == b'-').count(), 60);
    }

    #[test]
    fn a_hand_placed_midgame_position_maps_square_for_square() {
        // Black plays d3 (index 19) from the opening: d3 placed, d4 flips to
        // black, white to move.
        let after_d3 = Othello::apply(State::default(), &Move(19));
        let s = state_to_edax_board(&after_d3);
        let bytes = s.as_bytes();
        assert_eq!(bytes[19], b'X'); // d3, just played
        assert_eq!(bytes[27], b'X'); // d4, flipped
        assert_eq!(bytes[28], b'X'); // e4, unchanged black
        assert_eq!(bytes[35], b'X'); // d5, unchanged black
        assert_eq!(bytes[36], b'O'); // e5, unchanged white
        assert_eq!(bytes[65], b'O'); // white to move
    }

    #[test]
    fn corners_and_edges_land_at_the_expected_offsets() {
        let st = State {
            black: BB::from_bits((1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)),
            white: BB::from_bits(1 << 8),
            turn: Player::White,
            ..State::default()
        };
        let s = state_to_edax_board(&st);
        let b = s.as_bytes();
        assert_eq!(b[0], b'X'); // a1
        assert_eq!(b[7], b'X'); // h1
        assert_eq!(b[56], b'X'); // a8
        assert_eq!(b[63], b'X'); // h8
        assert_eq!(b[8], b'O'); // a2
        assert_eq!(b[65], b'O');
    }

    #[test]
    fn move_tokens_parse_both_cases_and_pass() {
        assert_eq!(edax_move_to_index("D3"), Some(Move(19)));
        assert_eq!(edax_move_to_index("d3"), Some(Move(19)));
        assert_eq!(edax_move_to_index("A1"), Some(Move(0)));
        assert_eq!(edax_move_to_index("H8"), Some(Move(63)));
        assert_eq!(edax_move_to_index(" F5 "), Some(Move(37)));
        assert_eq!(edax_move_to_index("PA"), Some(Move::PASS));
        assert_eq!(edax_move_to_index("pass"), Some(Move::PASS));
        assert_eq!(edax_move_to_index("z9"), None);
        assert_eq!(edax_move_to_index("d"), None);
    }

    #[test]
    fn move_notation_agrees_with_the_game_impl() {
        // Every square: our own notation string, upper-cased, must parse
        // back to the same index through the Edax parser.
        for i in 0..64u8 {
            let n = Othello::notation(&State::default(), &Move(i)).to_ascii_uppercase();
            assert_eq!(edax_move_to_index(&n), Some(Move(i)), "square {i}");
        }
    }
}
