//! Place our Othello MCTS engine on an external, non-self-referential
//! strength scale by playing it against Edax at fixed search levels.
//!
//! This gives learned-evaluator work an external reference point: strength
//! is reported as "Edax level N" rather than as a delta between two of our
//! own configs. This binary produces that integer.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example edax_match -p game-othello -- [CONFIG] [MODE]
//! ```
//!
//! `CONFIG` defaults to `games/othello/edax/match.toml`. `MODE` is one of:
//!
//! - `ladder` -- Run A: validate the ladder is monotone (Edax L vs L+2).
//! - `place`  -- Run B: place `our_preset` on the ladder (default).
//! - `both`   -- ladder then place.
//!
//! Build Edax first with `games/othello/edax/build-edax.sh`. This is a
//! background job: per-game progress goes to stderr.
//!
//! ## Edax protocol (verified against the pinned v4.6 build)
//!
//! One long-lived child process per Edax level. Native line protocol:
//!
//! - `setboard <64-char-board> <side>` -- `X`/`O`/`-` squares in row-major
//!   order from A1 (see `game_othello::edax`), trailing `X`/`O` = side to
//!   move.
//! - `go` -- search at the configured level; replies `Edax plays <MOVE>`
//!   (upper-case `<file><rank>`, or `PA` for a pass).
//! - launched with `-q -book-usage off -eval-file <dir>/eval.dat -level N`.
//!
//! **The one gotcha:** Edax aborts an in-progress `go` search the instant
//! another line is queued on its stdin. The driver must therefore write
//! `setboard`+`go` and read the `Edax plays` reply back *before* writing
//! anything else -- never queue a command (least of all `quit`) while a
//! search is running.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use game_othello::edax::{edax_move_to_index, state_to_edax_board};
use game_othello::{Move, Othello, State};
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use mcts_bench::tournament::Result as Tally;
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Number of uniform-random plies played from the opening before the two
/// engines take over. Both Edax and (seeded) MCTS are deterministic, so
/// without this a "40-game" match is really one line played 40 times;
/// randomising the first few plies turns it into 40 distinct games while
/// staying near book theory.
const OPENING_PLIES: usize = 4;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Config {
    edax_binary: String,
    edax_data_dir: String,
    our_preset: String,
    levels: Vec<u32>,
    games_per_level: u32,
    seed: u64,
}

// ---------------------------------------------------------------------------
// Edax subprocess
// ---------------------------------------------------------------------------

/// A running Edax engine pinned to one search level.
struct EdaxPlayer {
    #[allow(dead_code)]
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    level: u32,
    name: String,
}

impl EdaxPlayer {
    fn spawn(binary: &str, data_dir: &str, level: u32) -> EdaxPlayer {
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
// Our engine
// ---------------------------------------------------------------------------

/// Delegating newtype so a `Box<dyn Search>` (what `PresetTable::build`
/// hands back) can be passed to `battle_royale`, which needs a concrete
/// `Search` type.
struct Boxed(Box<dyn Search<G = Othello>>);

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

fn build_our_engine(preset: &str, seed: u64) -> Boxed {
    let table = PresetTable::load_from_path(Path::new("games/othello/presets.json"))
        .expect("games/othello/presets.json must parse");
    Boxed(
        table
            .build::<Othello>(preset, seed)
            .expect("preset must resolve"),
    )
}

// ---------------------------------------------------------------------------
// Match play + aggregation
// ---------------------------------------------------------------------------

/// A uniform-random legal opening: `OPENING_PLIES` real (non-pass) moves
/// from the standard start, retried if a line ends early.
fn random_opening(rng: &mut SmallRng) -> State {
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

/// `mcts::util::battle_royale`, but starting from `state` (whose side to
/// move is played by `first`) instead of the default opening. Same return
/// convention: `None` draw, `Some(0)` `first` won, `Some(1)` `second` won.
fn play_from<A, B>(mut state: State, first: &mut A, second: &mut B) -> Option<usize>
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

/// Fold one `battle_royale` outcome into `tally`, where `tally` is kept from
/// the perspective of the player we care about (the "hero"). `hero_is_s1`
/// says whether the hero played as `s1` in that game (colours are swapped
/// every other game). `br` is `battle_royale`'s return: `None` draw,
/// `Some(0)` s1 won, `Some(1)` s2 won.
fn record_game(tally: &mut Tally, hero_is_s1: bool, br: Option<usize>) {
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

/// Play `games` games between `hero` and `foe`, alternating who moves first
/// (equivalently, who plays Black). Returns the tally from `hero`'s side.
fn play_series<H, F>(hero: &mut H, foe: &mut F, games: u32, seed: u64, label: &str) -> Tally
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

fn report_row(label: &str, t: &Tally) {
    let (p, (lo, hi)) = t.win_rate_ci(1.96);
    println!(
        "{label:<16} games={:>3}  W-D-L {}-{}-{}  win_rate={:.3}  ci=[{:.3}, {:.3}]",
        t.total(),
        t.wins,
        t.draws,
        t.losses,
        p,
        lo,
        hi
    );
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/// Run A: deeper Edax must beat shallower with a CI excluding 0.5.
fn run_ladder(cfg: &Config) {
    println!("== Run A: ladder monotonicity (Edax L vs L+2) ==");
    let n = cfg.games_per_level.max(30);
    let mut ok = true;
    for l in [1u32, 3, 5, 7] {
        let mut deep = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l + 2);
        let mut shallow = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l);
        let label = format!("L{}v L{}", l + 2, l);
        let seed = 0xA000 ^ ((l as u64) << 8);
        let t = play_series(&mut deep, &mut shallow, n, seed, &label);
        report_row(&label, &t);
        let (_, (lo, _)) = t.win_rate_ci(1.96);
        if lo <= 0.5 {
            println!("  !! L{} did not clearly beat L{} (ci lower bound {lo:.3})", l + 2, l);
            ok = false;
        }
    }
    println!(
        "ladder monotonicity: {}",
        if ok { "PASS" } else { "FAIL -- fix the driver before trusting Run B" }
    );
}

/// Run B: place `our_preset` on the ladder.
fn run_place(cfg: &Config) {
    println!("== Run B: {} vs Edax ==", cfg.our_preset);
    let n = cfg.games_per_level;
    let mut rows: Vec<(u32, Tally)> = Vec::new();
    for &l in &cfg.levels {
        let mut ours = build_our_engine(&cfg.our_preset, cfg.seed.wrapping_add(l as u64));
        let mut edax = EdaxPlayer::spawn(&cfg.edax_binary, &cfg.edax_data_dir, l);
        let label = format!("{} v L{l}", cfg.our_preset);
        let seed = cfg.seed.wrapping_add((l as u64) << 8);
        let t = play_series(&mut ours, &mut edax, n, seed, &label);
        report_row(&format!("vs edax-L{l}"), &t);
        rows.push((l, t));
    }

    // N = highest level where our CI lower bound is still >= 0.5.
    let mut n_clear = None;
    let mut cross = None;
    for (l, t) in &rows {
        let (p, (lo, _)) = t.win_rate_ci(1.96);
        if lo >= 0.5 {
            n_clear = Some(*l);
        }
        if p < 0.5 && cross.is_none() {
            cross = Some(*l);
        }
    }
    println!();
    match n_clear {
        Some(l) => println!("N = {l}  (our '{}' preset clearly beats Edax up to level {l})", cfg.our_preset),
        None => println!("N = 0  (our '{}' preset does not clearly beat even the lowest level tested)", cfg.our_preset),
    }
    match cross {
        Some(l) => println!("point estimate crosses 0.5 at level {l}"),
        None => println!(
            "point estimate never crossed 0.5 -- extend `levels` upward in match.toml and rerun"
        ),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cfg_path = args
        .next()
        .filter(|a| !a.starts_with("--") && a != "ladder" && a != "place" && a != "both")
        .unwrap_or_else(|| "games/othello/edax/match.toml".to_string());
    let mode = args.next().unwrap_or_else(|| "place".to_string());

    let cfg: Config = toml::from_str(
        &std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}")),
    )
    .expect("config must parse");

    println!(
        "edax={}  preset={}  levels={:?}  games/level={}  seed={}",
        cfg.edax_binary, cfg.our_preset, cfg.levels, cfg.games_per_level, cfg.seed
    );

    match mode.as_str() {
        "ladder" => run_ladder(&cfg),
        "place" => run_place(&cfg),
        "both" => {
            run_ladder(&cfg);
            run_place(&cfg);
        }
        other => panic!("unknown mode {other:?} (want ladder | place | both)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_game_folds_colour_swaps_correctly() {
        let mut t = Tally::default();
        // Hero as s1, hero wins (br = Some(0)).
        record_game(&mut t, true, Some(0));
        // Hero as s2, hero wins (br = Some(1)).
        record_game(&mut t, false, Some(1));
        // Hero as s1, hero loses (br = Some(1)).
        record_game(&mut t, true, Some(1));
        // Hero as s2, hero loses (br = Some(0)).
        record_game(&mut t, false, Some(0));
        // Two draws.
        record_game(&mut t, true, None);
        record_game(&mut t, false, None);
        assert_eq!((t.wins, t.losses, t.draws), (2, 2, 2));
        assert_eq!(t.score(), 3.0);
    }

    #[test]
    fn wilson_bounds_match_hand_computed_values() {
        // 30 wins, 10 losses, 0 draws: p_hat = 0.75, n = 40.
        let t = Tally {
            wins: 30,
            losses: 10,
            draws: 0,
        };
        let (p, (lo, hi)) = t.win_rate_ci(1.96);
        assert!((p - 0.75).abs() < 1e-9);
        // Wilson 95% interval for 30/40 is ~[0.599, 0.862].
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
