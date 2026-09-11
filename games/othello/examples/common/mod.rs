//! Shared match-play scaffolding for the Othello strength-measurement
//! examples (`edax_match`, `ntuple_match`): random openings, an alternating
//! -colour series driver, Wilson-interval reporting, an Edax subprocess
//! player, and a `Box<dyn Search>` newtype.
//!
//! Both examples place an Othello engine on an external, non-self-referential
//! strength scale; everything that isn't specific to *which* comparison is
//! being run lives here.

#![allow(dead_code)] // each example uses a subset.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use game_othello::edax::{edax_move_to_index, state_to_edax_board};
use game_othello::{Move, Othello, State};
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub use mcts_bench::tournament::Result as Tally;

/// Number of uniform-random plies played from the opening before the two
/// engines take over. Both Edax and (seeded) MCTS are deterministic, so
/// without this a 40-game match is one line played 40 times; randomising
/// the first few plies makes 40 distinct games while staying near book
/// theory.
pub const OPENING_PLIES: usize = 4;

// ---------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------

/// Delegating newtype so a `Box<dyn Search>` can be handed to a driver that
/// needs a concrete `Search` type.
pub struct Boxed(pub Box<dyn Search<G = Othello>>);

impl Search for Boxed {
    type G = Othello;
    fn friendly_name(&self) -> String {
        self.0.friendly_name()
    }
    fn set_friendly_name(&mut self, name: &str) {
        self.0.set_friendly_name(name)
    }
    fn choose_action(&mut self, state: &State) -> Move {
        self.0.choose_action(state)
    }
    fn estimated_depth(&self) -> usize {
        self.0.estimated_depth()
    }
}

/// Build one of `presets.json`'s named engines, wrapped in [`Boxed`].
pub fn build_preset_engine(presets_path: &Path, preset: &str, seed: u64) -> Boxed {
    let table = PresetTable::load_from_path(presets_path)
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", presets_path.display()));
    Boxed(
        table
            .build::<Othello>(preset, seed)
            .unwrap_or_else(|e| panic!("preset {preset:?} did not resolve: {e}")),
    )
}

// ---------------------------------------------------------------------------
// Edax subprocess
// ---------------------------------------------------------------------------

/// A running Edax engine pinned to one search level.
///
/// Edax aborts an in-progress `go` search the instant another line is
/// queued on its stdin, so [`EdaxPlayer::ask`] writes `setboard`+`go` and
/// reads the `Edax plays` reply back before anything else is written --
/// never queue a command (least of all `quit`) while a search is running.
pub struct EdaxPlayer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    level: u32,
    name: String,
}

impl EdaxPlayer {
    pub fn spawn(binary: &str, data_dir: &str, level: u32) -> EdaxPlayer {
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
        let mut child = Command::new(binary)
            .args(["-q", "-book-usage", "off", "-eval-file"])
            .arg(&eval_file)
            .args(["-level", &level.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn edax");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut p = EdaxPlayer {
            child,
            stdin,
            stdout,
            level,
            name: format!("edax-L{level}"),
        };
        // `mode 3` keeps Edax in manual mode: no auto-play, no pondering.
        writeln!(p.stdin, "mode 3").unwrap();
        p.stdin.flush().unwrap();
        p
    }

    /// Ask Edax for its move in `state`. Assumes `state` has at least one
    /// real legal move (callers short-circuit forced passes).
    fn ask(&mut self, state: &State) -> Move {
        let board = state_to_edax_board(state);
        writeln!(self.stdin, "setboard {board}").unwrap();
        writeln!(self.stdin, "go").unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line).unwrap();
            assert!(n != 0, "edax closed its output mid-search");
            if let Some(rest) = line.trim().strip_prefix("Edax plays ") {
                let token = rest.trim();
                return edax_move_to_index(token)
                    .unwrap_or_else(|| panic!("unparseable edax move {token:?}"));
            }
        }
    }
}

impl Drop for EdaxPlayer {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

impl Search for EdaxPlayer {
    type G = Othello;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State) -> Move {
        let mut actions = Vec::new();
        Othello::generate_actions(state, &mut actions);
        if actions == [Move::PASS] {
            return Move::PASS;
        }
        let mv = self.ask(state);
        debug_assert!(
            actions.contains(&mv),
            "edax-L{} returned {mv:?}, not in legal {actions:?}",
            self.level
        );
        mv
    }
}

// ---------------------------------------------------------------------------
// Match play + aggregation
// ---------------------------------------------------------------------------

/// A uniform-random legal opening: `OPENING_PLIES` real (non-pass) moves
/// from the standard start, retried if a line ends early.
pub fn random_opening(rng: &mut SmallRng) -> State {
    'outer: loop {
        let mut state = State::default();
        let mut actions = Vec::new();
        for _ in 0..OPENING_PLIES {
            if Othello::is_terminal(&state) {
                continue 'outer;
            }
            actions.clear();
            Othello::generate_actions(&state, &mut actions);
            if actions == [Move::PASS] {
                continue 'outer;
            }
            let a = actions[rng.gen_range(0..actions.len())];
            state = Othello::apply(state, &a);
        }
        return state;
    }
}

/// `battle_royale`, but starting from `state` (whose side to move is played
/// by `first`). `None` draw, `Some(0)` `first` won, `Some(1)` `second` won.
pub fn play_from<A, B>(mut state: State, first: &mut A, second: &mut B) -> Option<usize>
where
    A: Search<G = Othello>,
    B: Search<G = Othello>,
{
    let players: [&mut dyn Search<G = Othello>; 2] = [first, second];
    let mut s = 0usize;
    loop {
        if Othello::is_terminal(&state) {
            let current = Othello::player_to_move(&state);
            return Othello::winner(&state).map(|p| {
                if current.to_index() == p.to_index() {
                    s
                } else {
                    1 - s
                }
            });
        }
        let m = players[s].choose_action(&state);
        state = Othello::apply(state, &m);
        s = 1 - s;
    }
}

/// Fold one `play_from` outcome into `tally`, kept from the "hero" side.
/// `hero_is_s1` says whether the hero moved first that game (colours are
/// swapped every other game).
pub fn record_game(tally: &mut Tally, hero_is_s1: bool, br: Option<usize>) {
    match br {
        None => tally.draws += 1,
        Some(w) => {
            let hero_won = (w == 0) == hero_is_s1;
            if hero_won {
                tally.wins += 1;
            } else {
                tally.losses += 1;
            }
        }
    }
}

/// Play `games` games between `hero` and `foe`, alternating who moves first.
/// Returns the tally from `hero`'s side. Per-game progress goes to stderr.
pub fn play_series<H, F>(hero: &mut H, foe: &mut F, games: u32, seed: u64, label: &str) -> Tally
where
    H: Search<G = Othello>,
    F: Search<G = Othello>,
{
    let mut tally = Tally::default();
    for g in 0..games {
        let hero_is_s1 = g % 2 == 0;
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(g as u64));
        let opening = random_opening(&mut rng);
        let br = if hero_is_s1 {
            play_from(opening, hero, foe)
        } else {
            play_from(opening, foe, hero)
        };
        record_game(&mut tally, hero_is_s1, br);
        eprintln!(
            "  {label} game {}/{}: W-D-L {}-{}-{}",
            g + 1,
            games,
            tally.wins,
            tally.draws,
            tally.losses
        );
    }
    tally
}

/// Print a one-line summary: W-D-L, win rate, and its Wilson 95% interval.
pub fn report_row(label: &str, t: &Tally) {
    let (p, (lo, hi)) = t.win_rate_ci(1.96);
    println!(
        "{label:<20} games={:>3}  W-D-L {}-{}-{}  win_rate={:.3}  ci=[{:.3}, {:.3}]",
        t.total(),
        t.wins,
        t.draws,
        t.losses,
        p,
        lo,
        hi
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_othello::{Player, BB};

    /// A player that plays a fixed, pre-scripted sequence of moves,
    /// ignoring the board it is handed.
    struct Scripted {
        moves: Vec<Move>,
        next: usize,
    }

    impl Scripted {
        fn new(moves: Vec<Move>) -> Self {
            Self { moves, next: 0 }
        }
    }

    impl Search for Scripted {
        type G = Othello;

        fn friendly_name(&self) -> String {
            "scripted".to_string()
        }

        fn set_friendly_name(&mut self, _name: &str) {}

        fn choose_action(&mut self, _state: &State) -> Move {
            let m = self.moves[self.next];
            self.next += 1;
            m
        }
    }

    /// Union of single-square bitboards at `indices`.
    fn bb(indices: impl IntoIterator<Item = usize>) -> BB {
        indices
            .into_iter()
            .fold(BB::EMPTY, |b, i| b | BB::from_index(i))
    }

    /// A state one square short of a full board (index 63 empty), split so
    /// that whichever colour plays there ends up with a strict, known
    /// count-based lead once the disc lands, with no flips: 63's only
    /// neighbours (54, 55, 62) are pre-set to `mover`'s colour, so
    /// `get_flips` finds an own-coloured disc immediately adjacent in every
    /// direction and flips nothing.
    fn one_square_from_full(mover: Player) -> (State, /* mover count after move */ u32) {
        let corner_neighbours = [54usize, 55, 62];
        let empty = 63usize;
        let mut mover_idx: Vec<usize> = corner_neighbours.to_vec();
        let mut other_idx: Vec<usize> = Vec::new();
        for i in 0..64 {
            if i == empty || corner_neighbours.contains(&i) {
                continue;
            }
            // Split the remaining 60 squares 37/23 so the mover, after
            // placing at 63, leads 41-23 (or trails into a 23-41 loss,
            // depending on which colour is `mover`) -- either way a clean,
            // unambiguous count winner.
            if mover_idx.len() < 40 {
                mover_idx.push(i);
            } else {
                other_idx.push(i);
            }
        }
        let (black_idx, white_idx) = match mover {
            Player::Black => (mover_idx, other_idx),
            Player::White => (other_idx, mover_idx),
        };
        let state = State {
            black: bb(black_idx),
            white: bb(white_idx),
            turn: mover,
            last_pass: false,
            hashes: [0u64; 8],
        };
        (state, 41)
    }

    /// One ply from a full board (`play_from`'s only ply) must attribute a
    /// decisive win to the seat whose colour actually has more discs, when
    /// that seat's colour also made the winning, board-filling move.
    #[test]
    fn play_from_attributes_a_win_made_on_the_final_ply() {
        let (state, mover_discs) = one_square_from_full(Player::Black);
        assert_eq!(state.black.count_ones() + 1, mover_discs); // sanity on the setup above.

        let mut first = Scripted::new(vec![Move(63)]);
        let mut second = Scripted::new(vec![]);
        let result = play_from(state, &mut first, &mut second);

        // Black plays the winning move and Black ends up with more discs:
        // seat 0 (`first`) wins.
        assert_eq!(result, Some(0));
    }

    /// The decisive case: the seat that makes the literal last move (fills
    /// the last empty square) is the *loser* by disc count. `play_from`
    /// must still attribute the win to the seat whose colour has more
    /// discs, not to whoever moved last -- this is exactly the shape of the
    /// Connect Four `battle_royale` bug (see `crates/mcts/src/util.rs`),
    /// which used `player_to_move` at the terminal as if it named the last
    /// mover. It is safe here because `State::apply` always advances
    /// `turn`, on every move including a pass, so `player_to_move` at the
    /// terminal names the *next* mover's colour, which stays in lockstep
    /// with the alternating seat index `s` for the whole game.
    #[test]
    fn play_from_does_not_attribute_the_win_to_the_last_mover() {
        // Two empty squares: 0 and 63, opposite corners so neither's
        // neighbourhood overlaps the other's.
        let black_neighbours = [54usize, 55, 62]; // around 63, black plays there first.
        let white_neighbours = [1usize, 8, 9]; // around 0, white plays there second.
        let empties = [0usize, 63usize];

        let mut black_idx: Vec<usize> = black_neighbours.to_vec();
        let mut white_idx: Vec<usize> = white_neighbours.to_vec();
        for i in 0..64 {
            if empties.contains(&i)
                || black_neighbours.contains(&i)
                || white_neighbours.contains(&i)
            {
                continue;
            }
            // 37 more black, 19 more white: black 40, white 22 before the
            // last two discs land: 41-23 after, an unambiguous black lead
            // even though white plays the literal final move.
            if black_idx.len() < 40 {
                black_idx.push(i);
            } else {
                white_idx.push(i);
            }
        }
        assert_eq!(black_idx.len(), 40);
        assert_eq!(white_idx.len(), 22);

        let state = State {
            black: bb(black_idx),
            white: bb(white_idx),
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };

        let mut first = Scripted::new(vec![Move(63)]); // plays black, first.
        let mut second = Scripted::new(vec![Move(0)]); // plays white, second and last.
        let result = play_from(state, &mut first, &mut second);

        let final_black = 41u32;
        let final_white = 23u32;
        assert!(final_black > final_white);
        // Black (seat 0, `first`) has the disc-count lead, even though
        // white (seat 1, `second`) made the game-ending move.
        assert_eq!(result, Some(0));
    }

    #[test]
    fn record_game_folds_colour_swaps_correctly() {
        let mut t = Tally::default();
        record_game(&mut t, true, Some(0)); // hero s1, hero wins
        record_game(&mut t, false, Some(1)); // hero s2, hero wins
        record_game(&mut t, true, Some(1)); // hero s1, hero loses
        record_game(&mut t, false, Some(0)); // hero s2, hero loses
        record_game(&mut t, true, None);
        record_game(&mut t, false, None);
        assert_eq!((t.wins, t.losses, t.draws), (2, 2, 2));
        assert_eq!(t.score(), 3.0);
    }

    #[test]
    fn wilson_bounds_match_hand_computed_values() {
        let t = Tally {
            wins: 30,
            losses: 10,
            draws: 0,
        };
        let (p, (lo, hi)) = t.win_rate_ci(1.96);
        assert!((p - 0.75).abs() < 1e-9);
        assert!((lo - 0.599).abs() < 0.01, "lo = {lo}");
        assert!((hi - 0.862).abs() < 0.01, "hi = {hi}");
    }

    #[test]
    fn draws_count_as_half_in_the_series_tally() {
        let mut t = Tally::default();
        for g in 0..10 {
            record_game(&mut t, g % 2 == 0, None);
        }
        let (p, _) = t.win_rate_ci(1.96);
        assert_eq!(t.draws, 10);
        assert!((p - 0.5).abs() < 1e-9);
    }
}
