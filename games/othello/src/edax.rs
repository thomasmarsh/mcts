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

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

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

// ---------------------------------------------------------------------------
// Edax as a scoring oracle
// ---------------------------------------------------------------------------

/// One Edax evaluation of a position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdaxScore {
    /// Signed disc-difference score in Edax's native `[-64, 64]` units, from
    /// the **side-to-move** perspective (positive == the player to move is
    /// ahead).
    pub score: f32,
    /// `true` when Edax searched the position to the end of the game (an
    /// exact result), `false` for a heuristic / selectively-pruned score.
    pub exact: bool,
    /// Edax's self-reported node count for this search (`0` if the search
    /// line carried no parseable node column). Summed across calls for the
    /// bake-off's CPU cross-check.
    pub nodes: u64,
}

/// A persistent Edax subprocess driven as a *scoring* oracle -- as opposed
/// to the match harness's `EdaxPlayer`, which drives Edax for *moves*.
///
/// Edax aborts an in-progress `go` search the instant another line is queued
/// on its stdin, so [`EdaxEval::eval`] writes `setboard` + `go` and reads
/// the search result back before writing anything else. One long-lived
/// process per instance -- spawning Edax per position would dominate the CPU
/// bill and pollute the bake-off accounting.
///
/// Pinned to a single thread (`-n 1`) so its wall-time is its CPU-time.
///
/// ## Timeout
///
/// A `go` read is bounded by a wall-clock `timeout`: Edax's alpha-beta can
/// spend minutes-to-hours on a hard mid-game position at a deep level, and a
/// blocking read there would wedge a whole label run indefinitely. On
/// timeout [`EdaxEval::eval`] aborts the search (any queued stdin line does
/// that), resynchronises the line protocol against a full-board `*** Game
/// Over ***` sentinel, and falls back the same way it does for an
/// unparseable result -- one shallow retry, then a neutral score.
///
/// ## Score-line parse (pinned against `edax-reversi` v4.6, `mEdax-native`)
///
/// Run without `-q`, Edax prints one search-result line per `go`, after a
/// `depth|score|...` header and a `---+---+...` rule:
///
/// ```text
///  depth|score|       time   |  nodes (N)  |   N/s    | principal variation
/// ------+-----+--------------+-------------+----------+----------------------
///    10   +00        0:00.021        145934    6949238 d3 C5 e6 D2 c3 E3 ...
/// ```
///
/// Whitespace-split: field 0 is the depth token (`10`, or `30@73%` for a
/// selective search), field 1 the signed disc score (`+00`, `-38`), field 3
/// the node count. A position with no legal continuation prints
/// `*** Game Over ***` instead; that is handled by reading the exact
/// disc-difference straight off the board.
pub struct EdaxEval {
    child: Child,
    stdin: ChildStdin,
    /// Lines from Edax's stdout, pumped by a reader thread so `go` reads can
    /// be bounded with [`Receiver::recv_timeout`].
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    binary: String,
    eval_file: std::path::PathBuf,
    timeout: Duration,
    current_level: u32,
    total_nodes: u64,
    calls: u64,
    timeouts: u64,
}

impl EdaxEval {
    /// Spawn Edax pinned to `level` and to a single thread. Mirrors
    /// `EdaxPlayer::spawn`'s binary / weight-file guards. `timeout` bounds
    /// each `go` read (see the type-level "Timeout" note).
    pub fn spawn(binary: &str, data_dir: &str, level: u32, timeout: Duration) -> EdaxEval {
        let eval_file = Path::new(data_dir).join("eval.dat");
        assert!(
            Path::new(binary).exists(),
            "edax binary not found at {binary} -- run games/othello/edax/build-edax.sh"
        );
        assert!(
            eval_file.exists(),
            "eval weights not found at {} -- run games/othello/edax/build-edax.sh",
            eval_file.display()
        );
        let (child, stdin, lines, reader) = Self::spawn_process(binary, &eval_file, level);
        let mut e = EdaxEval {
            child,
            stdin,
            lines,
            reader: Some(reader),
            binary: binary.to_string(),
            eval_file,
            timeout,
            current_level: level,
            total_nodes: 0,
            calls: 0,
            timeouts: 0,
        };
        // `mode 3`: manual mode, no auto-play, no pondering. Running without
        // `-q` means each `go` also dumps the board; `eval` skips those lines.
        writeln!(e.stdin, "mode 3").unwrap();
        e.stdin.flush().unwrap();
        e
    }

    /// Launch one Edax subprocess plus the stdout reader thread that feeds
    /// its lines onto a channel.
    fn spawn_process(
        binary: &str,
        eval_file: &Path,
        level: u32,
    ) -> (Child, ChildStdin, Receiver<String>, JoinHandle<()>) {
        let mut child = Command::new(binary)
            // No `-q`: it silences the search-result line this oracle parses.
            .args(["-n", "1", "-book-usage", "off", "-eval-file"])
            .arg(eval_file)
            .args(["-level", &level.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn edax");
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let (tx, lines) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match stdout.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {
                        if tx.send(line.trim_end().to_string()).is_err() {
                            return;
                        }
                    }
                }
            }
        });
        (child, stdin, lines, reader)
    }

    /// Evaluate `state` at `level`, switching Edax's level first if it
    /// differs from the last call. `state` is assumed non-terminal for the
    /// heuristic path; a terminal board short-circuits to the exact margin.
    ///
    /// A position whose side to move has no legal continuation (a forced
    /// pass) makes Edax print `*** Game Over ***`; that is read off the
    /// board. If Edax somehow returns a move with no parseable score line at
    /// all, `eval` retries once at a shallow level and then falls back to a
    /// neutral score rather than aborting a multi-hour label run.
    pub fn eval(&mut self, state: &State, level: u32) -> EdaxScore {
        match self.eval_once(state, level) {
            Some(s) => s,
            None => match self.eval_once(state, 4) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "warn: Edax returned no score for {} -- using 0.0",
                        state_to_edax_board(state)
                    );
                    self.calls += 1;
                    EdaxScore { score: 0.0, exact: false, nodes: 0 }
                }
            },
        }
    }

    fn eval_once(&mut self, state: &State, level: u32) -> Option<EdaxScore> {
        if level != self.current_level {
            writeln!(self.stdin, "level {level}").unwrap();
            self.stdin.flush().unwrap();
            self.current_level = level;
        }
        let board = state_to_edax_board(state);
        writeln!(self.stdin, "setboard {board}").unwrap();
        writeln!(self.stdin, "go").unwrap();
        self.stdin.flush().unwrap();

        let empties = 64 - (state.black.bits() | state.white.bits()).count_ones();
        let mut last: Option<(f32, u32, bool, u64)> = None;
        let mut played = false;
        loop {
            let line = match self.lines.recv_timeout(self.timeout) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    self.timeouts += 1;
                    eprintln!(
                        "warn: Edax search exceeded {:?} for {board} -- killing and restarting",
                        self.timeout
                    );
                    self.restart();
                    return None;
                }
                Err(RecvTimeoutError::Disconnected) => panic!("edax closed its output mid-search"),
            };
            let t = line.trim();
            if t.contains("*** Game Over ***") {
                self.calls += 1;
                return Some(EdaxScore {
                    score: terminal_disc_diff(state) as f32,
                    exact: true,
                    nodes: 0,
                });
            }
            if let Some(p) = parse_score_line(t) {
                last = Some(p);
            }
            if t.starts_with("Edax plays") {
                played = true;
            }
            // Stop once we have the score, or -- for an "obvious move" Edax
            // resolves without a search table -- once it returns to its `>`
            // prompt after playing. The retry at a shallow level then forces
            // a table.
            if played && (last.is_some() || t == ">") {
                break;
            }
        }
        let (score, depth, selective, nodes) = last?;
        self.total_nodes += nodes;
        self.calls += 1;
        Some(EdaxScore {
            score,
            exact: !selective && depth >= empties,
            nodes,
        })
    }

    /// Kill the current Edax process and start a fresh one. Edax does not
    /// reliably abort an in-progress `go` when a line is queued on its stdin
    /// -- it runs the search to completion first -- so a runaway search can
    /// only be stopped by killing the process. Respawn re-reads `eval.dat`
    /// (tens of ms) and is only reached on a timeout, so the cost is
    /// negligible against the search that was abandoned.
    fn restart(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        let level = self.current_level;
        let (child, stdin, lines, reader) =
            Self::spawn_process(&self.binary, &self.eval_file, level);
        self.child = child;
        self.stdin = stdin;
        self.lines = lines;
        self.reader = Some(reader);
        writeln!(self.stdin, "mode 3").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Number of `eval` calls so far.
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// Number of searches aborted by the wall-clock timeout (each mapped to a
    /// shallow retry, then a neutral score).
    pub fn timeouts(&self) -> u64 {
        self.timeouts
    }

    /// Total Edax nodes searched across all `eval` calls.
    pub fn total_nodes(&self) -> u64 {
        self.total_nodes
    }
}

impl Drop for EdaxEval {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// Disc difference from the side-to-move perspective, counting empties as
/// lost by the trailing side (Edax's own final-score convention).
fn terminal_disc_diff(state: &State) -> i32 {
    let b = state.black.bits().count_ones() as i32;
    let w = state.white.bits().count_ones() as i32;
    let empty = 64 - b - w;
    let (mut mine, mut theirs) = match state.turn {
        Player::Black => (b, w),
        Player::White => (w, b),
    };
    if mine > theirs {
        mine += empty;
    } else if theirs > mine {
        theirs += empty;
    }
    mine - theirs
}

/// Parse one Edax search-result line into `(score, depth, selective, nodes)`.
/// Returns `None` for the header, the rule, and every non-result line.
fn parse_score_line(line: &str) -> Option<(f32, u32, bool, u64)> {
    let mut it = line.split_whitespace();
    let depth_tok = it.next()?;
    let digits: String = depth_tok.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let depth: u32 = digits.parse().ok()?;
    let selective = depth_tok.contains('@');
    let score_tok = it.next()?;
    if !(score_tok.starts_with('+') || score_tok.starts_with('-')) {
        return None;
    }
    let score: i32 = score_tok.parse().ok()?;
    let _time = it.next();
    let nodes: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    Some((score as f32, depth, selective, nodes))
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
    fn score_line_parse_pins_the_v46_format() {
        // Header and rule are not result lines.
        assert_eq!(
            parse_score_line(" depth|score|       time   |  nodes (N)  |   N/s    | pv"),
            None
        );
        assert_eq!(parse_score_line("------+-----+--------------+----+----+---"), None);
        assert_eq!(parse_score_line(">"), None);

        // Full-depth heuristic line: balanced opening.
        let (s, d, sel, n) =
            parse_score_line("   10   +00        0:00.021        145934    6949238 d3 C5 e6").unwrap();
        assert_eq!((s, d, sel, n), (0.0, 10, false, 145934));

        // Selective search: the `@73%` marks it non-exact.
        let (s, d, sel, _) =
            parse_score_line("30@73%  -01        0:01.415      32733547   23133249 d3 C5").unwrap();
        assert_eq!((s, d, sel), (-1.0, 30, true));

        // Short line: no n/s or pv columns.
        let (s, d, sel, n) = parse_score_line("    8   -32        0:00.000           798").unwrap();
        assert_eq!((s, d, sel, n), (-32.0, 8, false, 798));
    }

    #[test]
    fn terminal_disc_diff_awards_empties_to_the_leader() {
        // Side to move (black) has 40, white 20, 4 empty -> black leads, gets
        // the empties: 44 - 20 = +24.
        let st = State {
            black: BB::from_bits((1u64 << 40) - 1),
            white: BB::from_bits(((1u64 << 60) - 1) & !((1u64 << 40) - 1)),
            turn: Player::Black,
            ..State::default()
        };
        assert_eq!(terminal_disc_diff(&st), 24);
        let flipped = State {
            turn: Player::White,
            ..st
        };
        assert_eq!(terminal_disc_diff(&flipped), -24);
    }

    /// Live Edax check -- shells out, so kept off the `cargo test --lib` hot
    /// path (`cargo test --lib -p game-othello -- --ignored edax_eval`).
    #[test]
    #[ignore = "shells out to the vendored Edax binary"]
    fn edax_eval_opening_won_and_solved() {
        const BIN: &str = "edax/vendor/bin/mEdax-native";
        const DATA: &str = "edax/vendor/data";
        if !Path::new(BIN).exists() {
            eprintln!("skip: no Edax binary");
            return;
        }
        let mut e = EdaxEval::spawn(BIN, DATA, 12, Duration::from_secs(120));

        // The opening is near-balanced.
        let opening = e.eval(&State::default(), 12);
        assert!(opening.score.abs() <= 2.0, "opening score {}", opening.score);

        // Near-full board: black owns all but a1 (empty) and a2 (a lone
        // white disc). Black to move has the legal move a1 (a1-a2-a3 flips
        // a2); the score is an exact near-max win. Perspective sign test.
        let won_black = State {
            black: BB::from_bits(!((1u64 << 0) | (1u64 << 8))),
            white: BB::from_bits(1u64 << 8),
            turn: Player::Black,
            ..State::default()
        };
        let from_black = e.eval(&won_black, 12);
        assert!(from_black.exact, "few-empties position should solve exactly");
        assert!(from_black.score > 40.0, "won-for-black score {}", from_black.score);
        // Same board, white to move: white has no legal move, so this is
        // terminal for the scoring purpose -- strongly negative for white.
        let from_white = e.eval(
            &State {
                turn: Player::White,
                ..won_black
            },
            12,
        );
        assert!(from_white.score < -40.0, "same board, white to move: {}", from_white.score);
    }

    /// A sub-millisecond ceiling forces every search to time out; `eval`
    /// must then fall back to a neutral score, resync, and stay usable for
    /// the next position (proving the drain worked).
    #[test]
    #[ignore = "shells out to the vendored Edax binary"]
    fn a_timed_out_search_falls_back_and_the_process_stays_usable() {
        const BIN: &str = "edax/vendor/bin/mEdax-native";
        const DATA: &str = "edax/vendor/data";
        if !Path::new(BIN).exists() {
            eprintln!("skip: no Edax binary");
            return;
        }
        let mut e = EdaxEval::spawn(BIN, DATA, 18, Duration::from_millis(1));
        let a = e.eval(&State::default(), 18);
        assert_eq!(a.score, 0.0);
        assert!(!a.exact);
        let after_d3 = Othello::apply(State::default(), &Move(19));
        let b = e.eval(&after_d3, 18);
        assert_eq!(b.score, 0.0);
        assert!(e.timeouts() >= 2, "both searches should have timed out");
        assert_eq!(e.calls(), 2);
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
