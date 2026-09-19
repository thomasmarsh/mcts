//! Balanced-opening ("XOT"-style) support for the Edax yardstick.
//!
//! Edax and our seeded engines are deterministic, so a fixed opening gives the
//! same game every time. Random openings fix that but add opening luck. The
//! papers' answer (OLIVAW, section on Edax) is XOT: the first 8 plies are
//! random, but only sequences whose end position Edax judges within a couple
//! of discs of even are used. This module holds the pure parts of that
//! protocol - sequence text format, D4 canonical position key, the balance
//! filter, the seeded sampler, and the seeded opening selection - so they get
//! fast `cargo test --lib` coverage. The Edax-driven generation is
//! `examples/gen_xot_openings.rs`; the consumers are the match examples.
//!
//! ## File format
//!
//! One opening per line: the moves concatenated as two-character lowercase
//! squares (`file` then `rank`, the same as `Othello::notation`), e.g.
//! `f5d6c3d3c4f4f6g5`. This is the format the public XOT lists use. Blank
//! lines and lines starting with `#` are ignored.

use std::path::Path;

use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use crate::edax::edax_move_to_index;
use crate::{Move, Othello, State};

/// Random plies of an XOT opening.
pub const XOT_PLIES: usize = 8;

/// A position's identity under the 8 board symmetries: the lexicographically
/// smallest `(black, white)` image, with the side to move. Two positions with
/// equal keys are the same game position up to a board rotation or reflection.
pub type CanonicalKey = (u64, u64, u8);

/// Canonical D4 key of `state`, keeping the side to move distinct.
pub fn canonical_key(state: &State) -> CanonicalKey {
    let (black, white) = crate::board_symmetries(state.black, state.white)
        [crate::canonical_symmetry(state.black, state.white)];
    (black, white, state.turn as u8)
}

/// `key` as a fixed-width 33-character string, for logs.
pub fn key_string(key: CanonicalKey) -> String {
    format!("{:016x}{:016x}{}", key.0, key.1, key.2)
}

/// Format `moves` as one opening line. Panics on a pass, which an opening
/// never contains.
pub fn format_sequence(moves: &[Move]) -> String {
    let empty = State::default();
    moves
        .iter()
        .map(|m| {
            assert!(*m != Move::PASS, "openings never contain a pass");
            Othello::notation(&empty, m)
        })
        .collect()
}

/// Parse one opening line and replay it from the standard start. Fails on a
/// malformed token, an illegal move, a forced pass, or a finished game, so a
/// corrupt line can never silently become a different opening.
pub fn parse_sequence(line: &str) -> Result<State, String> {
    let line = line.trim();
    if line.is_empty() || !line.len().is_multiple_of(2) || !line.is_ascii() {
        return Err(format!("bad opening line {line:?}"));
    }
    let mut state = State::default();
    let mut actions = Vec::new();
    for i in (0..line.len()).step_by(2) {
        let tok = &line[i..i + 2];
        let mv = edax_move_to_index(tok).ok_or_else(|| format!("bad square {tok:?} in {line:?}"))?;
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if mv == Move::PASS || !actions.contains(&mv) {
            return Err(format!("illegal move {tok:?} at ply {} of {line:?}", i / 2 + 1));
        }
        state = Othello::apply(state, &mv);
    }
    Ok(state)
}

/// Load an opening file (see the module doc for the format) as `(line, state)`
/// pairs. Every line must parse; the error names the file and line number.
pub fn load_openings(path: &Path) -> Result<Vec<(String, State)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read openings file {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let state =
            parse_sequence(t).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        out.push((t.to_string(), state));
    }
    if out.is_empty() {
        return Err(format!("openings file {} has no openings", path.display()));
    }
    Ok(out)
}

/// The `i`-th seeded random opening of `plies` real moves, or `None` when the
/// walk hits a forced pass (including one for the side to move at the end) or ends the game (such samples are rejected, not
/// retried, so the sample stream stays a pure function of `(seed, i)`).
pub fn sample_sequence(seed: u64, i: u64, plies: usize) -> Option<Vec<Move>> {
    let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut state = State::default();
    let mut actions = Vec::new();
    let mut moves = Vec::with_capacity(plies);
    for _ in 0..plies {
        if Othello::is_terminal(&state) {
            return None;
        }
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if actions == [Move::PASS] {
            return None;
        }
        let mv = actions[rng.gen_range(0..actions.len())];
        state = Othello::apply(state, &mv);
        moves.push(mv);
    }
    // The side to move must have a real move too: a pass on ply 9 is still a
    // pass, and Edax answers it with no score line.
    actions.clear();
    Othello::generate_actions(&state, &mut actions);
    if Othello::is_terminal(&state) || actions == [Move::PASS] {
        return None;
    }
    Some(moves)
}

/// The balance filter: an Edax score (side-to-move perspective, discs) is
/// balanced when it is within `tolerance` of even.
pub fn is_balanced(score: f32, tolerance: f32) -> bool {
    score.abs() <= tolerance
}

/// Pick `pairs` distinct openings from `lines` for a seeded run: the lines
/// ordered by `sha256("{seed}:{line}")`, first `pairs` taken. A different seed
/// samples a different subset of the same file, and because the rule is plain
/// hashing the Python harness (`research/ppo-train/slice0b/claim_check.py`)
/// reproduces it exactly, so every agent on the yardstick plays the same
/// openings. Errors when the file is too small: silently reusing openings
/// would shrink the effective sample size behind the reported interval.
pub fn select_openings(lines: &[&str], pairs: usize, seed: u64) -> Result<Vec<usize>, String> {
    if pairs > lines.len() {
        return Err(format!(
            "asked for {pairs} openings but the file has only {}",
            lines.len()
        ));
    }
    let mut keyed: Vec<(String, usize)> = lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let digest = Sha256::digest(format!("{seed}:{line}").as_bytes());
            (digest.iter().map(|b| format!("{b:02x}")).collect(), i)
        })
        .collect();
    keyed.sort();
    Ok(keyed.into_iter().take(pairs).map(|(_, i)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BB;
    use game_core::symmetry::board_symmetries_8x8;

    /// A legal 8-ply line, from the seeded sampler.
    fn fixture_line() -> String {
        let moves = (0..).find_map(|i| sample_sequence(3, i, XOT_PLIES)).unwrap();
        format_sequence(&moves)
    }

    fn play(moves: &[u8]) -> State {
        moves
            .iter()
            .fold(State::default(), |s, m| Othello::apply(s, &Move(*m)))
    }

    #[test]
    fn sequence_text_round_trips() {
        let line = fixture_line();
        let moves: Vec<Move> = (0..line.len() / 2)
            .map(|i| edax_move_to_index(&line[2 * i..2 * i + 2]).unwrap())
            .collect();
        assert_eq!(moves.len(), XOT_PLIES);
        assert_eq!(format_sequence(&moves), line);
        let replayed = moves.iter().fold(State::default(), Othello::apply);
        assert_eq!(canonical_key(&parse_sequence(&line).unwrap()), canonical_key(&replayed));
    }

    #[test]
    fn parse_rejects_corrupt_lines() {
        assert!(parse_sequence("").is_err());
        assert!(parse_sequence("d3c").is_err(), "odd length");
        assert!(parse_sequence("z9").is_err(), "bad square");
        assert!(parse_sequence("a1").is_err(), "illegal first move");
        assert!(parse_sequence("d3d3").is_err(), "occupied square");
    }

    #[test]
    fn canonical_key_is_invariant_under_all_eight_symmetries() {
        let state = parse_sequence(&fixture_line()).unwrap();
        let key = canonical_key(&state);
        let b = board_symmetries_8x8(state.black);
        let w = board_symmetries_8x8(state.white);
        for s in 0..8 {
            let image = State {
                black: BB::from_bits(b[s].bits()),
                white: BB::from_bits(w[s].bits()),
                ..State::default()
            };
            assert_eq!(canonical_key(&image), key, "symmetry {s}");
        }
    }

    #[test]
    fn symmetric_first_moves_share_a_key_and_side_to_move_separates() {
        // d3, c4, f5, e6 are the four equivalent first moves.
        let keys: Vec<_> = [19u8, 26, 37, 44]
            .iter()
            .map(|&m| canonical_key(&play(&[m])))
            .collect();
        assert!(keys.windows(2).all(|w| w[0] == w[1]), "{keys:?}");
        let mut flipped = play(&[19]);
        flipped.turn = crate::Player::Black;
        assert_ne!(canonical_key(&flipped), keys[0]);
    }

    #[test]
    fn distinct_positions_get_distinct_keys() {
        assert_ne!(canonical_key(&play(&[19])), canonical_key(&play(&[19, 18])));
    }

    #[test]
    fn sampler_is_deterministic_and_yields_legal_eight_ply_lines() {
        let mut got = 0;
        for i in 0..200 {
            let a = sample_sequence(7, i, XOT_PLIES);
            assert_eq!(a, sample_sequence(7, i, XOT_PLIES));
            if let Some(moves) = a {
                assert_eq!(moves.len(), XOT_PLIES);
                let end = parse_sequence(&format_sequence(&moves)).expect("sampled line replays");
                let mut actions = Vec::new();
                Othello::generate_actions(&end, &mut actions);
                assert_ne!(actions, [Move::PASS], "the side to move must have a move");
                got += 1;
            }
        }
        assert!(got > 150, "8-ply passes/game-overs are rare, got {got} of 200");
        assert_ne!(sample_sequence(7, 0, 8), sample_sequence(8, 0, 8));
    }

    #[test]
    fn balance_filter_is_inclusive_and_symmetric() {
        assert!(is_balanced(0.0, 2.0));
        assert!(is_balanced(2.0, 2.0));
        assert!(is_balanced(-2.0, 2.0));
        assert!(!is_balanced(3.0, 2.0));
        assert!(!is_balanced(-4.0, 2.0));
    }

    #[test]
    fn select_openings_is_seeded_distinct_and_bounded() {
        let lines: Vec<String> = (0..1000).map(|i| format!("line{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let a = select_openings(&refs, 100, 1).unwrap();
        assert_eq!(a, select_openings(&refs, 100, 1).unwrap());
        assert_ne!(a, select_openings(&refs, 100, 2).unwrap());
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 100);
        assert!(a.iter().all(|&i| i < 1000));
        assert!(select_openings(&refs[..10], 11, 1).is_err());
    }

    #[test]
    fn select_openings_order_is_sha256_of_seed_colon_line() {
        // Pinned so the Python harness's `hashlib.sha256(f"{seed}:{line}")` rule
        // cannot drift from this one. sha256("1:a") < sha256("1:b") < sha256("1:c")
        // is checked against the digests directly.
        let digest = |l: &str| -> Vec<u8> { Sha256::digest(format!("1:{l}").as_bytes()).to_vec() };
        let mut want = vec!["a", "b", "c"];
        want.sort_by_key(|l| digest(l));
        let picked = select_openings(&["a", "b", "c"], 3, 1).unwrap();
        let got: Vec<&str> = picked.iter().map(|&i| ["a", "b", "c"][i]).collect();
        assert_eq!(got, want);
        assert_eq!(
            digest("a")[..4],
            [0x41, 0x62, 0xfd, 0xdd][..],
            "sha256(\"1:a\") starts 4162fddd"
        );
    }

    #[test]
    fn load_openings_reads_comments_and_reports_the_bad_line() {
        let dir = std::env::temp_dir().join(format!("openings-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("ok.txt");
        std::fs::write(&ok, format!("# header\n{}\n\nf5d6\n", fixture_line())).unwrap();
        assert_eq!(load_openings(&ok).unwrap().len(), 2);
        let bad = dir.join("bad.txt");
        std::fs::write(&bad, "d3c5\na1\n").unwrap();
        let err = load_openings(&bad).unwrap_err();
        assert!(err.contains("bad.txt:2"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
