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
use std::time::Instant;

use game_othello::edax::{edax_move_to_index, parse_score_line, state_to_edax_board};
use game_othello::openings::{load_openings, select_openings};
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
///
/// Runs without `-q` so each `go` prints its search-result line, which is
/// where the per-move node count comes from (see [`EdaxPlayer::nodes`]).
/// Pinned to a fixed thread count (`-n`, default 1): Edax otherwise takes
/// every core, which makes its timing depend on what else is running and
/// steals CPU from the engine it is playing.
pub struct EdaxPlayer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    level: u32,
    name: String,
    nodes: u64,
    searches: u64,
}

impl EdaxPlayer {
    /// A single-threaded Edax at `level`.
    pub fn spawn(binary: &str, data_dir: &str, level: u32) -> EdaxPlayer {
        Self::spawn_with_threads(binary, data_dir, level, 1)
    }

    /// Total nodes Edax reported over all its searches so far.
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// Number of searches that reported a node count.
    pub fn searches(&self) -> u64 {
        self.searches
    }

    pub fn spawn_with_threads(
        binary: &str,
        data_dir: &str,
        level: u32,
        threads: u32,
    ) -> EdaxPlayer {
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
            .args(["-n", &threads.to_string(), "-book-usage", "off", "-eval-file"])
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
            nodes: 0,
            searches: 0,
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
        let mut last_nodes = None;
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line).unwrap();
            assert!(n != 0, "edax closed its output mid-search");
            if let Some((_, _, _, nodes)) = parse_score_line(line.trim()) {
                last_nodes = Some(nodes);
            }
            if let Some(rest) = line.trim().strip_prefix("Edax plays ") {
                if let Some(nodes) = last_nodes {
                    self.nodes += nodes;
                    self.searches += 1;
                }
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
    random_opening_plies(rng, OPENING_PLIES)
}

/// [`random_opening`] with a caller-chosen ply count.
pub fn random_opening_plies(rng: &mut SmallRng, plies: usize) -> State {
    'outer: loop {
        let mut state = State::default();
        let mut actions = Vec::new();
        for _ in 0..plies {
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

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Wraps an engine to measure wall-clock seconds per real move (forced
/// passes are free and not counted).
pub struct Timed<S> {
    pub inner: S,
    pub moves: u64,
    pub secs: f64,
}

impl<S> Timed<S> {
    pub fn new(inner: S) -> Self {
        Timed {
            inner,
            moves: 0,
            secs: 0.0,
        }
    }

    pub fn secs_per_move(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.secs / self.moves as f64
        }
    }
}

impl<S: Search<G = Othello>> Search for Timed<S> {
    type G = Othello;
    fn friendly_name(&self) -> String {
        self.inner.friendly_name()
    }
    fn set_friendly_name(&mut self, name: &str) {
        self.inner.set_friendly_name(name)
    }
    fn choose_action(&mut self, state: &State) -> Move {
        let started = Instant::now();
        let mv = self.inner.choose_action(state);
        if mv != Move::PASS {
            self.moves += 1;
            self.secs += started.elapsed().as_secs_f64();
        }
        mv
    }
    fn estimated_depth(&self) -> usize {
        self.inner.estimated_depth()
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Read a TOML file into `T`, first applying any `--set key=value` overrides
/// found in `args` (top-level keys only; the value is parsed as a TOML value,
/// so `--set levels=[1,2]` and `--set out="x.jsonl"` work, and a bare word
/// falls back to a string).
pub fn load_toml_config<T: serde::de::DeserializeOwned>(path: &str, args: &[String]) -> T {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut table: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("{path} is not valid TOML: {e}"));
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--set" {
            let kv = args
                .get(i + 1)
                .unwrap_or_else(|| panic!("--set needs key=value"));
            let (k, v) = kv
                .split_once('=')
                .unwrap_or_else(|| panic!("--set expects key=value, got {kv:?}"));
            let value = format!("v = {v}")
                .parse::<toml::Table>()
                .ok()
                .and_then(|mut t| t.remove("v"))
                .unwrap_or_else(|| toml::Value::String(v.to_string()));
            table.insert(k.trim().to_string(), value);
            i += 1;
        }
        i += 1;
    }
    toml::Value::Table(table)
        .try_into()
        .unwrap_or_else(|e| panic!("{path} (with --set overrides) does not fit the config: {e}"))
}

fn one() -> u32 {
    1
}
fn six() -> usize {
    6
}

/// Everything the Edax gate needs that is not the agent under test. Shared by
/// `edax_match` (TOML) and `gumbel_gate --edax-only` (`--gate-config`).
#[derive(serde::Deserialize, Clone, Debug)]
pub struct GateConfig {
    pub edax_binary: String,
    pub edax_data_dir: String,
    /// Edax search threads (`-n`). 1 keeps Edax reproducible and off our CPUs.
    #[serde(default = "one")]
    pub edax_threads: u32,
    pub levels: Vec<u32>,
    /// Games per level; each opening is played from both seats, so this is
    /// twice the number of openings and must be even.
    pub games_per_level: u32,
    /// Seeds the opening subset (file mode) or the random openings.
    pub seed: u64,
    /// Balanced-opening file (`games/othello/openings/xot8.txt`). Absent or
    /// empty (`--set openings=\"\"`) means random `opening_plies`-ply openings,
    /// the pre-XOT protocol.
    #[serde(default)]
    pub openings: Option<String>,
    #[serde(default = "six")]
    pub opening_plies: usize,
    /// JSONL file appended per game and per level as the gate runs.
    #[serde(default)]
    pub out: Option<String>,
}

// ---------------------------------------------------------------------------
// Openings
// ---------------------------------------------------------------------------

/// Where a gate's opening positions come from. Pair `p` is the same opening
/// for every level and every agent, so results are comparable across them.
pub enum Openings {
    /// `picks` indexes into `states`, a seeded subset of the balanced file.
    File {
        path: String,
        states: Vec<State>,
        picks: Vec<usize>,
    },
    /// Fresh seeded random `plies`-ply openings.
    Random { plies: usize, seed: u64 },
}

impl Openings {
    pub fn from_config(cfg: &GateConfig) -> Openings {
        let pairs = (cfg.games_per_level / 2) as usize;
        match cfg.openings.as_ref().filter(|p| !p.is_empty()) {
            Some(path) => {
                let loaded = load_openings(Path::new(path)).unwrap_or_else(|e| panic!("{e}"));
                let lines: Vec<&str> = loaded.iter().map(|(l, _)| l.as_str()).collect();
                let picks = select_openings(&lines, pairs, cfg.seed)
                    .unwrap_or_else(|e| panic!("{path}: {e}"));
                let states = loaded.into_iter().map(|(_, s)| s).collect();
                Openings::File {
                    path: path.clone(),
                    states,
                    picks,
                }
            }
            None => Openings::Random {
                plies: cfg.opening_plies,
                seed: cfg.seed,
            },
        }
    }

    /// The opening for pair `p`, and its index in the file when there is one.
    pub fn pair(&self, p: usize) -> (Option<usize>, State) {
        match self {
            Openings::File { states, picks, .. } => (Some(picks[p]), states[picks[p]]),
            Openings::Random { plies, seed } => {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(p as u64));
                (None, random_opening_plies(&mut rng, *plies))
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Openings::File { path, states, .. } => format!("{path} ({} openings)", states.len()),
            Openings::Random { plies, .. } => format!("random {plies}-ply"),
        }
    }
}

// ---------------------------------------------------------------------------
// The Edax gate
// ---------------------------------------------------------------------------

/// One level's result for one agent.
pub struct GateRow {
    pub level: u32,
    pub tally: Tally,
    pub agent_secs_per_move: f64,
    pub edax_secs_per_move: f64,
    pub edax_nodes_per_move: f64,
    pub secs: f64,
}

fn append_jsonl(path: &Option<String>, row: &serde_json::Value) {
    if let Some(path) = path {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("cannot open --out {path}: {e}"));
        writeln!(f, "{row}").unwrap();
    }
}

/// Place one agent on the Edax ladder under the balanced-opening protocol:
/// per level, `games_per_level / 2` shared openings each played from both
/// seats, a Wilson 95% interval on the score (draw = half), and the seconds
/// per move both sides took plus Edax's measured nodes per move.
///
/// This is the engine-builder hook: `make(seed)` returns a fresh agent, and
/// `agent_nodes_per_move` is that agent's nominal search budget per move
/// (MCTS iterations, Gumbel sims, 1 for a raw net forward pass), or `None`
/// when it has no such number. It is a budget, not a measurement - only
/// Edax reports the nodes it actually searched. Any agent (search preset,
/// released net raw, the TD agent later) is gated by giving it a `make`.
pub fn run_edax_gate<S: Search<G = Othello>>(
    cfg: &GateConfig,
    label: &str,
    agent_nodes_per_move: Option<u64>,
    make: impl Fn(u64) -> S,
) -> Vec<GateRow> {
    assert!(
        cfg.games_per_level.is_multiple_of(2) && cfg.games_per_level > 0,
        "games_per_level must be a positive even number (each opening is played from both seats)"
    );
    let openings = Openings::from_config(cfg);
    let pairs = (cfg.games_per_level / 2) as usize;
    println!(
        "gate {label}: levels {:?}, {} games/level over {pairs} openings ({}), edax -n {}, seed {}",
        cfg.levels,
        cfg.games_per_level,
        openings.describe(),
        cfg.edax_threads,
        cfg.seed
    );
    let mut rows = Vec::new();
    for &level in &cfg.levels {
        let started = Instant::now();
        let mut hero = Timed::new(make(cfg.seed.wrapping_add(level as u64)));
        let mut edax = Timed::new(EdaxPlayer::spawn_with_threads(
            &cfg.edax_binary,
            &cfg.edax_data_dir,
            level,
            cfg.edax_threads,
        ));
        let mut tally = Tally::default();
        for p in 0..pairs {
            let (opening_idx, opening) = openings.pair(p);
            for hero_first in [true, false] {
                let (hero_secs0, edax_secs0) = (hero.secs, edax.secs);
                let br = if hero_first {
                    play_from(opening, &mut hero, &mut edax)
                } else {
                    play_from(opening, &mut edax, &mut hero)
                };
                let before = tally;
                record_game(&mut tally, hero_first, br);
                let result = if tally.wins > before.wins {
                    "win"
                } else if tally.losses > before.losses {
                    "loss"
                } else {
                    "draw"
                };
                append_jsonl(
                    &cfg.out,
                    &serde_json::json!({
                        "type": "game", "agent": label, "level": level, "pair": p,
                        "opening": opening_idx, "hero_first": hero_first, "result": result,
                        "agent_secs": hero.secs - hero_secs0, "edax_secs": edax.secs - edax_secs0,
                    }),
                );
            }
            if (p + 1) % 10 == 0 || p + 1 == pairs {
                eprintln!(
                    "  {label} v edax-L{level} {} games: W-D-L {}-{}-{}",
                    2 * (p + 1),
                    tally.wins,
                    tally.draws,
                    tally.losses
                );
            }
        }
        let row = GateRow {
            level,
            tally,
            agent_secs_per_move: hero.secs_per_move(),
            edax_secs_per_move: edax.secs_per_move(),
            edax_nodes_per_move: edax.inner.nodes() as f64 / edax.inner.searches().max(1) as f64,
            secs: started.elapsed().as_secs_f64(),
        };
        print_gate_row(label, &row, agent_nodes_per_move);
        let (score, (lo, hi)) = tally.win_rate_ci(1.96);
        append_jsonl(
            &cfg.out,
            &serde_json::json!({
                "type": "level", "agent": label, "level": level, "games": tally.total(),
                "w": tally.wins, "d": tally.draws, "l": tally.losses,
                "score": score, "lo": lo, "hi": hi,
                "openings": openings.describe(), "seed": cfg.seed, "edax_threads": cfg.edax_threads,
                "agent_secs_per_move": row.agent_secs_per_move,
                "agent_nodes_per_move_budget": agent_nodes_per_move,
                "edax_secs_per_move": row.edax_secs_per_move,
                "edax_nodes_per_move": row.edax_nodes_per_move, "secs": row.secs,
            }),
        );
        rows.push(row);
    }
    rows
}

fn print_gate_row(label: &str, r: &GateRow, agent_nodes: Option<u64>) {
    let (p, (lo, hi)) = r.tally.win_rate_ci(1.96);
    let budget = agent_nodes.map_or("-".to_string(), |n| n.to_string());
    println!(
        "{label} L{:<2} games={:>3} W-D-L {}-{}-{}  score={p:.3} wilson95=[{lo:.3}, {hi:.3}]  \
         agent {:.1} ms/move (budget {budget} nodes)  edax {:.1} ms/move {:.0} nodes/move",
        r.level,
        r.tally.total(),
        r.tally.wins,
        r.tally.draws,
        r.tally.losses,
        r.agent_secs_per_move * 1e3,
        r.edax_secs_per_move * 1e3,
        r.edax_nodes_per_move,
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

    #[test]
    fn timed_counts_real_moves_but_not_passes() {
        struct Fixed(Move);
        impl Search for Fixed {
            type G = Othello;
            fn friendly_name(&self) -> String {
                "fixed".into()
            }
            fn set_friendly_name(&mut self, _: &str) {}
            fn choose_action(&mut self, _: &State) -> Move {
                self.0
            }
        }
        let mut t = Timed::new(Fixed(Move(19)));
        t.choose_action(&State::default());
        t.choose_action(&State::default());
        assert_eq!(t.moves, 2);
        t.inner.0 = Move::PASS;
        t.choose_action(&State::default());
        assert_eq!(t.moves, 2, "a pass is not a move");
    }

    #[test]
    fn set_overrides_replace_toml_keys_and_parse_values() {
        #[derive(serde::Deserialize)]
        struct C {
            a: u32,
            levels: Vec<u32>,
            name: String,
        }
        let dir = std::env::temp_dir().join(format!("gate-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.toml");
        std::fs::write(&path, "a = 1\nlevels = [1]\nname = \"x\"\n").unwrap();
        let args: Vec<String> = ["--set", "a=7", "--set", "levels=[2,3]", "--set", "name=bare"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let c: C = load_toml_config(path.to_str().unwrap(), &args);
        assert_eq!((c.a, c.levels, c.name.as_str()), (7, vec![2, 3], "bare"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn random_openings_are_pure_in_the_pair_index() {
        let o = Openings::Random { plies: 6, seed: 1 };
        let (_, a) = o.pair(3);
        let (_, b) = o.pair(3);
        assert_eq!((a.black, a.white), (b.black, b.white));
        let (_, c) = o.pair(4);
        assert_ne!((a.black, a.white), (c.black, c.white));
    }
}
