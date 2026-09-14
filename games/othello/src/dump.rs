//! `game-othello dump` -- write self-play positions as fixed-width
//! little-endian records for the offline learned-evaluation pipeline.
//!
//! This is Othello-specific scaffolding, deliberately kept out of the
//! generic `game-host` CLI: `main.rs` intercepts `dump` as the first
//! argument and calls [`run`] before `game_host::run_cli` ever sees the
//! args.
//!
//! ## Record format
//!
//! 22 bytes, packed (no padding), little-endian:
//!
//! | field  | type   | bytes |
//! |--------|--------|-------|
//! | black  | u64 LE | 8     |
//! | white  | u64 LE | 8     |
//! | side   | u8     | 1     |
//! | ply    | u8     | 1     |
//! | target | f32 LE | 4     |
//!
//! `side` is 0 for black to move, 1 for white to move. `ply` is the number
//! of discs on the board minus 4 (0 at the opening, up to 60 at a full
//! board). `target` is the final game result **from the side-to-move
//! player's perspective**: `+1.0` that player won, `-1.0` they lost, `0.0`
//! a draw.
//!
//! ## Label modes
//!
//! - `--label outcome` (default): seeded self-play, every non-terminal
//!   position labelled with the final game outcome.
//! - `--label harvest --out <dir>`: one search-per-move self-play pass
//!   (config from `games/othello/ntuple/harvest.toml`, or
//!   `--harvest-config`) writing four label streams from the *same*
//!   searches -- `arm_a` (played root <- outcome), `arm_b` (played root <-
//!   its searched value), `arm_c` (every internal node <- its searched
//!   value, TreeStrap; see `harvest.rs`), `arm_d` (played root <- the
//!   TD(lambda) return along the game). Each `arm_*.bin` has a matching
//!   `arm_*.json` manifest; `harvest.json` records the counts and the
//!   harvest ratio.
//! - `--label gumbel --out <path>`: Gumbel self-play with either the n-tuple
//!   value head + policy sidecar (`GumbelPlayer`, `--head ntuple`, the
//!   default) or the joint CNN value+policy container (`CnnGumbelPlayer`,
//!   `--head cnn`), both in `crate::selfplay`, writing [`RecordV2`]s whose
//!   policy tail is the completed-Q improved-policy target either way.
//!   `--head ntuple`'s `--weights-dir <dir>` points at a trained checkpoint
//!   (`model.toml` + `weights.bin` + `weights.meta.json` + `policy.bin` +
//!   `policy.meta.json`, `research/az-train`'s layout); absent, self-play
//!   uses the all-zero generation-0 net over `--model`'s geometry (default
//!   `games/othello/ntuple/model.toml`). `--head cnn`'s `--cnn-weights
//!   <path>` points at a single `OTCNN001`-layout checkpoint file
//!   (`crate::convnet::CnnValueNet::load`); absent, self-play uses the
//!   all-zero CNN. `--sims` / `--max-considered` set the Gumbel budget;
//!   `--temp-moves` samples the Sequential-Halving visit distribution for
//!   the first N plies before switching to the argmax;
//!   `--forced-opening-plies` forces a deterministic per-game opening
//!   choice (see [`forced_move`]) for wide, uniform opening coverage.
//!
//! ## Position source
//!
//! Without `--engine`, moves are uniform-random. With `--engine <preset>`,
//! moves come from that `games/othello/presets.json` engine, except that
//! with probability `--epsilon` (default 0.1) a uniform-random legal move
//! is played instead -- diversity so a deterministic seeded engine doesn't
//! emit the same game repeatedly. The label is unchanged either way.
//! (`--label gumbel`'s position source is Gumbel self-play itself, not this
//! `--engine` mechanism.)
//!
//! ## Record v2 (Gumbel self-play)
//!
//! [`RecordV2`] adds a completed-Q improved-policy tail to the 22-byte head
//! above (23-byte head + `n_policy` `(square, prob)` pairs), following
//! `games/connect4/src/dump.rs`'s v2-connect4 record shape exactly:
//! [`RecordV2::from_record`] converts a completed-Q distribution
//! (`mcts::algorithms::mcts::gumbel::GumbelOutcome::improved_policy`) into
//! the tail. `--label gumbel` is the only mode that produces these records.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mcts::algorithms::mcts::gumbel::{GumbelConfig, GumbelOutcome};
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::convnet::CnnValueNet;
use crate::ntuple::{NTupleEval, NTupleModel, NTupleModelEval};
use crate::policy::NTuplePolicyNet;
use crate::selfplay::{CnnGumbelPlayer, GumbelPlayer};
use crate::{Move, Othello, Player, State};

// ---------------------------------------------------------------------------
// Record v2: fixed head + a variable-length improved-policy tail
// ---------------------------------------------------------------------------

/// One dumped position with a completed-Q improved-policy target, following
/// `games/connect4/src/dump.rs`'s v2-connect4 record shape. A 22-byte head
/// identical in meaning to [`Record`] (see the module docs), followed by
/// `n_policy` `(square: u8, prob: f32 LE)` pairs -- `square` is a [`Move`]'s
/// raw `u8` (0..=63 a board square, 64 == [`Move::PASS`]). `policy` is empty
/// for a record with no search-derived policy (e.g. `--label outcome`).
///
/// v2 records are variable-width, so a reader must walk them sequentially
/// (the Python counterpart, `research/az-train/src/az_train/records_othello.py`,
/// is added alongside the Gumbel self-play driver that will produce these
/// records).
#[derive(Debug, Clone, PartialEq)]
pub struct RecordV2 {
    pub black: u64,
    pub white: u64,
    pub side: u8,
    pub ply: u8,
    pub value: f32,
    /// `(square, probability)` pairs -- the completed-Q improved-policy
    /// target. Empty when no search produced a policy for this position.
    pub policy: Vec<(u8, f32)>,
}

/// Size of a [`RecordV2`]'s fixed head, in bytes (everything up to and
/// including the `n_policy` count byte, before the policy tail).
pub const RECORD_V2_HEAD_BYTES: usize = 23;

/// Size of one policy tail entry, in bytes: `(square: u8, prob: f32 LE)`.
pub const POLICY_ENTRY_BYTES: usize = 5;

impl RecordV2 {
    /// Build a [`RecordV2`] from an existing [`Record`] head plus a
    /// completed-Q improved-policy distribution
    /// (`mcts::algorithms::mcts::gumbel::GumbelOutcome::improved_policy`).
    pub fn from_record(head: Record, policy: &[(Move, f32)]) -> RecordV2 {
        RecordV2 {
            black: head.black,
            white: head.white,
            side: head.side,
            ply: head.ply,
            value: head.target,
            policy: policy.iter().map(|(m, p)| (m.0, *p)).collect(),
        }
    }

    /// Append this record's little-endian bytes to `buf`. Fields are
    /// written one at a time -- never a `#[repr(C)]` struct cast, which
    /// could introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.black.to_le_bytes());
        buf.extend_from_slice(&self.white.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.value.to_le_bytes());
        let n: u8 = self
            .policy
            .len()
            .try_into()
            .expect("policy tail longer than 255 entries");
        buf.push(n);
        for (square, prob) in &self.policy {
            buf.push(*square);
            buf.extend_from_slice(&prob.to_le_bytes());
        }
    }

    /// Decode one record from the front of `bytes`, returning it and the
    /// number of bytes it consumed. Returns `None` if `bytes` is too short
    /// to hold a complete record.
    pub fn decode(bytes: &[u8]) -> Option<(RecordV2, usize)> {
        if bytes.len() < RECORD_V2_HEAD_BYTES {
            return None;
        }
        let black = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let white = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let side = bytes[16];
        let ply = bytes[17];
        let value = f32::from_le_bytes(bytes[18..22].try_into().unwrap());
        let n_policy = bytes[22] as usize;
        let total = RECORD_V2_HEAD_BYTES + n_policy * POLICY_ENTRY_BYTES;
        if bytes.len() < total {
            return None;
        }
        let mut policy = Vec::with_capacity(n_policy);
        for i in 0..n_policy {
            let off = RECORD_V2_HEAD_BYTES + i * POLICY_ENTRY_BYTES;
            let square = bytes[off];
            let prob = f32::from_le_bytes(bytes[off + 1..off + 5].try_into().unwrap());
            policy.push((square, prob));
        }
        Some((
            RecordV2 {
                black,
                white,
                side,
                ply,
                value,
                policy,
            },
            total,
        ))
    }
}

/// One dumped position. See the module docs for field semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Record {
    pub black: u64,
    pub white: u64,
    pub side: u8,
    pub ply: u8,
    pub target: f32,
}

/// Serialized size of one [`Record`], in bytes.
pub const RECORD_BYTES: usize = 22;

impl Record {
    /// Append this record's 22 little-endian bytes to `buf`. Fields are
    /// written one at a time -- never a `#[repr(C)]` struct cast, which
    /// could introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.black.to_le_bytes());
        buf.extend_from_slice(&self.white.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.target.to_le_bytes());
    }

    /// Decode one record from exactly [`RECORD_BYTES`] bytes.
    pub fn decode(b: &[u8; RECORD_BYTES]) -> Record {
        Record {
            black: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            white: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            side: b[16],
            ply: b[17],
            target: f32::from_le_bytes(b[18..22].try_into().unwrap()),
        }
    }
}

fn side_of(p: Player) -> u8 {
    match p {
        Player::Black => 0,
        Player::White => 1,
    }
}

/// Build a [`Record`] for a mid-game `state` given the final `winner`
/// (`None` == draw). `target` is from `state.turn`'s perspective.
pub fn record_for(state: &State, winner: Option<Player>) -> Record {
    let side = side_of(state.turn);
    let ply = (state.occupied().count_ones() as u8).saturating_sub(4);
    let target = match winner {
        None => 0.0,
        Some(w) if side_of(w) == side => 1.0,
        Some(_) => -1.0,
    };
    Record {
        black: state.black.bits(),
        white: state.white.bits(),
        side,
        ply,
        target,
    }
}

struct Config {
    out: PathBuf,
    manifest: Option<PathBuf>,
    games: u64,
    seed: u64,
    engine: Option<String>,
    epsilon: f64,
    presets_path: PathBuf,
    label: String,
    harvest_config: PathBuf,
    games_overridden: bool,
    seed_overridden: bool,
    /// `--label harvest` target oracle: `mcts` (the searched value read from
    /// the tree) or `edax` (an independent Edax evaluation of each position).
    /// Arm A (outcome) is unaffected either way.
    oracle: String,
    edax_config: PathBuf,
    /// `--edax-level N` overrides the config's `edax_level` (used by the
    /// `edax_level_sweep.sh` diagnostic and the higher-level held-out relabel).
    edax_level_override: Option<u32>,
    /// `max_iterations` / `max_playout_depth` for `--engine ntuple` (an
    /// n-tuple-evaluator-guided UCB1 search, reading `$OTHELLO_NTUPLE_WEIGHTS`
    /// -- unlike other `--engine` values, this bypasses `presets.json`
    /// because the evaluator is a game-specific `mcts::Evaluator`, not a
    /// generic `config_ir` axis).
    ntuple_iters: usize,
    ntuple_depth: usize,
    /// `--label gumbel` only: `model.toml` geometry to use when
    /// `--weights-dir` is absent (the generation-0 net: zero value, zero
    /// policy). Ignored when `--weights-dir` is given, since
    /// `NTupleModel::from_dir` reads its own `model.toml`.
    model_toml: PathBuf,
    /// `--label gumbel` only: a trained checkpoint directory holding
    /// `model.toml` + `weights.bin` + `weights.meta.json` + `policy.bin` +
    /// `policy.meta.json` (`research/az-train`'s per-generation output
    /// layout). Absent == the all-zero generation-0 net. Ignored when
    /// `head == "cnn"`.
    weights_dir: Option<PathBuf>,
    /// `--label gumbel` only: which value/policy model class self-play uses
    /// -- `ntuple` (default, `GumbelPlayer`) or `cnn` (`CnnGumbelPlayer`).
    head: String,
    /// `--label gumbel --head cnn` only: a single `OTCNN001`-layout
    /// checkpoint file (`CnnValueNet::load`). Absent == the all-zero CNN.
    cnn_weights: Option<PathBuf>,
    /// `--label gumbel` only: Gumbel simulation budget and root candidate cap.
    gumbel_sims: u32,
    gumbel_max_considered: usize,
    /// `--label gumbel` only: number of opening plies whose move is *sampled*
    /// from the Sequential-Halving visit distribution rather than taken as
    /// the argmax. Keeps self-play trajectories diverse so the value head
    /// trains on a distribution that does not collapse onto its own current
    /// best line each generation, even before any generational feedback
    /// loop exists to amplify a narrowing self-play distribution.
    temp_moves: u8,
    /// `--label gumbel` only: number of opening plies whose move is *forced*
    /// to a deterministic per-game choice (see [`forced_move`]) rather than
    /// chosen by search, for wide, uniform opening coverage across a run.
    /// `0` disables forcing. Search still runs and a policy target is still
    /// recorded at every forced position.
    forced_opening_plies: u32,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Config {
    let mut out = None;
    let mut manifest = None;
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut label = "outcome".to_string();
    let mut engine = None;
    let mut epsilon = 0.1f64;
    let mut presets_path = PathBuf::from("games/othello/presets.json");
    let mut harvest_config = PathBuf::from("games/othello/ntuple/harvest.toml");
    let mut oracle = "mcts".to_string();
    let mut edax_config = PathBuf::from("games/othello/ntuple/harvest_edax.toml");
    let mut edax_level_override = None;
    let mut games_overridden = false;
    let mut seed_overridden = false;
    let mut ntuple_iters = 200usize;
    let mut ntuple_depth = 0usize;
    let mut model_toml = PathBuf::from("games/othello/ntuple/model.toml");
    let mut weights_dir = None;
    let mut head = "ntuple".to_string();
    let mut cnn_weights = None;
    let mut gumbel_sims = 32u32;
    let mut gumbel_max_considered = 8usize;
    let mut temp_moves = 6u8;
    let mut forced_opening_plies = 0u32;
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--manifest" => manifest = Some(PathBuf::from(val())),
            "--games" => {
                games = val().parse().expect("--games must be an integer");
                games_overridden = true;
            }
            "--seed" => {
                seed = val().parse().expect("--seed must be an integer");
                seed_overridden = true;
            }
            "--label" => label = val(),
            "--engine" => engine = Some(val()),
            "--epsilon" => epsilon = val().parse().expect("--epsilon must be a float"),
            "--presets" => presets_path = PathBuf::from(val()),
            "--harvest-config" => harvest_config = PathBuf::from(val()),
            "--oracle" => oracle = val(),
            "--edax-config" => edax_config = PathBuf::from(val()),
            "--edax-level" => {
                edax_level_override = Some(val().parse().expect("--edax-level must be an integer"))
            }
            "--ntuple-iters" => {
                ntuple_iters = val().parse().expect("--ntuple-iters must be an integer")
            }
            "--ntuple-depth" => {
                ntuple_depth = val().parse().expect("--ntuple-depth must be an integer")
            }
            "--model" => model_toml = PathBuf::from(val()),
            "--weights-dir" => weights_dir = Some(PathBuf::from(val())),
            "--head" => head = val(),
            "--cnn-weights" => cnn_weights = Some(PathBuf::from(val())),
            "--sims" => gumbel_sims = val().parse().expect("--sims must be an integer"),
            "--max-considered" => {
                gumbel_max_considered = val().parse().expect("--max-considered must be an integer")
            }
            "--temp-moves" => temp_moves = val().parse().expect("--temp-moves must be an integer"),
            "--forced-opening-plies" => {
                forced_opening_plies = val()
                    .parse()
                    .expect("--forced-opening-plies must be an integer")
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-othello dump --out <path> [--games N] [--seed N] \
                     [--label outcome|harvest|gumbel] [--engine <preset>|ntuple] [--epsilon P] \
                     [--presets <path>] [--manifest <path>] [--harvest-config <path>] \
                     [--ntuple-iters N] [--ntuple-depth N] \
                     [--model <model.toml>] [--weights-dir <dir>] [--head ntuple|cnn] \
                     [--cnn-weights <path>] [--sims N] \
                     [--max-considered N] [--temp-moves N] [--forced-opening-plies N]\n\
                     \n\
                     --engine ntuple: self-play guided by $OTHELLO_NTUPLE_WEIGHTS instead of \
                     a presets.json entry.\n\
                     --label harvest: --out is a directory; writes arm_{{a,b,c,d}}.bin \
                     + manifests + harvest.json\n\
                     --label gumbel: Gumbel self-play with either the n-tuple value/policy \
                     heads (--head ntuple, default) or the joint CNN value+policy container \
                     (--head cnn), writing v2 records with a completed-Q policy tail. \
                     --head ntuple's --weights-dir points at a research/az-train checkpoint \
                     (model.toml + weights.bin + weights.meta.json + policy.bin + \
                     policy.meta.json); absent, self-play uses the all-zero generation-0 net \
                     over --model's geometry. --head cnn's --cnn-weights points at a single \
                     OTCNN001-layout checkpoint file; absent, self-play uses the all-zero CNN."
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    match oracle.as_str() {
        "mcts" | "edax" => {}
        other => panic!("unknown --oracle {other:?} (want mcts | edax)"),
    }
    match label.as_str() {
        "outcome" | "harvest" | "gumbel" => {}
        "treestrap" | "root_value" => panic!(
            "--label {label} is superseded by --label harvest, which emits arms a/b/c/d \
             (outcome, root searched value, TreeStrap, TD(lambda)) from one self-play pass"
        ),
        other => panic!("unknown --label mode: {other}"),
    }
    assert!(
        (0.0..=1.0).contains(&epsilon),
        "--epsilon must be in [0, 1], got {epsilon}"
    );
    assert!(
        matches!(head.as_str(), "ntuple" | "cnn"),
        "unknown --head mode: {head}"
    );
    assert!(gumbel_sims >= 1, "--sims must be positive");
    assert!(
        gumbel_max_considered >= 1,
        "--max-considered must be positive"
    );
    Config {
        out: out.expect("--out is required"),
        manifest,
        games,
        seed,
        engine,
        epsilon,
        presets_path,
        label,
        harvest_config,
        games_overridden,
        seed_overridden,
        oracle,
        edax_config,
        edax_level_override,
        ntuple_iters,
        ntuple_depth,
        model_toml,
        weights_dir,
        head,
        cnn_weights,
        gumbel_sims,
        gumbel_max_considered,
        temp_moves,
        forced_opening_plies,
    }
}

/// Backfill the final-outcome target into every record pushed for this
/// game (those from index `first` onward), now that `winner` is known.
fn finish_game(records: &mut [Record], first: usize, winner: Option<Player>) {
    for rec in &mut records[first..] {
        let side_player = if rec.side == 0 {
            Player::Black
        } else {
            Player::White
        };
        rec.target = match winner {
            None => 0.0,
            Some(w) if w == side_player => 1.0,
            Some(_) => -1.0,
        };
    }
}

/// Play one uniform-random game, pushing a [`Record`] for every
/// non-terminal position onto `records` once the outcome is known.
fn dump_one_game(rng: &mut SmallRng, records: &mut Vec<Record>) {
    let mut state = State::default();
    let mut actions = Vec::new();
    let first = records.len();
    while !Othello::is_terminal(&state) {
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        // Placeholder target, backfilled below once the winner is known.
        records.push(record_for(&state, None));
        let action = actions[rng.gen_range(0..actions.len())];
        state = Othello::apply(state, &action);
    }
    finish_game(records, first, Othello::winner(&state));
}

/// Play one game driven by `engine`, except that with probability `epsilon`
/// a uniform-random legal move is substituted. Same labelling as
/// [`dump_one_game`].
fn dump_one_game_engine(
    rng: &mut SmallRng,
    records: &mut Vec<Record>,
    engine: &mut dyn Search<G = Othello>,
    epsilon: f64,
) {
    let mut state = State::default();
    let mut actions = Vec::new();
    let first = records.len();
    while !Othello::is_terminal(&state) {
        actions.clear();
        Othello::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        records.push(record_for(&state, None));
        let action = if actions == [Move::PASS] {
            Move::PASS
        } else if rng.gen_bool(epsilon) {
            actions[rng.gen_range(0..actions.len())]
        } else {
            engine.choose_action(&state)
        };
        state = Othello::apply(state, &action);
    }
    finish_game(records, first, Othello::winner(&state));
}

/// `EvaluatedCutoff` + the n-tuple evaluator: UCB1 to `ntuple_iters`, playout
/// cut off at `ntuple_depth` and replaced by the n-tuple value -- the same
/// recipe `examples/ntuple_match.rs`'s `ContenderProfile` uses to gate the
/// evaluator, reused here as a self-play *source*: a modest node budget
/// guided by a real trained value net approximates what early-generation
/// self-play in a self-play training loop looks like (sharper than uniform
/// rollouts, without full-strength search cost).
type NtupleSelfPlayProfile =
    profile::Mcts<select::Ucb1, simulate::EvaluatedCutoff<Othello, NTupleEval, simulate::Uniform>>;

/// Build one `--engine ntuple` game's search, reading
/// `$OTHELLO_NTUPLE_WEIGHTS` (via [`NTupleEval`]'s lazy process-wide load).
fn ntuple_engine(k: usize, d: usize, seed: u64) -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, NtupleSelfPlayProfile>::new().config(
            SearchConfig::new()
                .name("dump/ntuple-selfplay")
                .expand_threshold(1)
                .q_init(QInit::Loss)
                .max_iterations(k)
                .max_playout_depth(d)
                .simulate(simulate::EvaluatedCutoff::new())
                .seed(seed),
        ),
    )
}

// ---------------------------------------------------------------------------
// `--label gumbel`: Gumbel self-play with the n-tuple value/policy heads
// ---------------------------------------------------------------------------

/// Load the value/policy nets for `--label gumbel` self-play: a trained
/// checkpoint from `--weights-dir` (`model.toml` + `weights.bin` +
/// `weights.meta.json` + `policy.bin` + `policy.meta.json`, `research/
/// az-train`'s per-generation output layout), or the all-zero generation-0
/// net over `--model`'s geometry when no checkpoint is given yet.
fn load_gumbel_nets(cfg: &Config) -> (NTupleModelEval, NTuplePolicyNet) {
    match &cfg.weights_dir {
        Some(dir) => {
            let model = NTupleModel::from_dir(dir);
            let geom = model.geometry().clone();
            let policy = NTuplePolicyNet::from_dir(geom, dir);
            (NTupleModelEval::new(model), policy)
        }
        None => {
            let bytes = std::fs::read(&cfg.model_toml)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.model_toml.display()));
            let geom = crate::ntuple::ModelGeometry::parse(&bytes);
            (NTupleModelEval::default(), NTuplePolicyNet::zeros(geom))
        }
    }
}

/// Draw one move from a policy distribution (probabilities summing to 1),
/// falling back to the first entry on a rounding shortfall.
fn sample_visit_distribution(dist: &[(Move, f32)], rng: &mut SmallRng) -> Move {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0f32;
    for (m, p) in dist {
        acc += *p;
        if r < acc {
            return *m;
        }
    }
    dist[0].0
}

/// Pick the move for a Gumbel self-play position: `forced` when one applies,
/// else the Sequential-Halving visit distribution for the first `temp_moves`
/// plies, else the argmax.
fn choose_selfplay_move(
    outcome: &GumbelOutcome<Move>,
    forced: Option<Move>,
    ply: u8,
    temp_moves: u8,
    rng: &mut SmallRng,
) -> Move {
    if let Some(forced) = forced {
        forced
    } else if ply < temp_moves {
        sample_visit_distribution(&outcome.improved_policy, rng)
    } else {
        outcome.action
    }
}

/// The forced opening move for `game_index` at `ply`, or `None` past the
/// forced prefix or when the position has no real choice to force (a single
/// legal action, including a forced pass). Unlike Connect Four's
/// `forced_opening_column` (digits of `game_index` in a fixed base 7, one
/// per column), Othello's branching factor is not fixed -- it varies move to
/// move and can be as low as 1 -- so this hashes `(game_index, ply)` into an
/// index over whatever the *actual* legal-move list is at this position,
/// rather than reading digits of a fixed base.
fn forced_move(state: &State, game_index: u64, ply: u32, forced_plies: u32) -> Option<Move> {
    if ply >= forced_plies {
        return None;
    }
    let mut actions = Vec::new();
    Othello::generate_actions(state, &mut actions);
    if actions.len() <= 1 {
        return None;
    }
    let h = (game_index ^ (ply as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    Some(actions[(h as usize) % actions.len()])
}

/// One dumped Gumbel self-play position before the final outcome is known:
/// the v2 record head (`target` filled in below) plus its completed-Q
/// improved-policy target.
struct GumbelRow {
    head: Record,
    policy: Vec<(Move, f32)>,
}

/// Play one Gumbel self-play game via `choose` (either `GumbelPlayer::choose`
/// or `CnnGumbelPlayer::choose` -- the two players share no trait beyond
/// `Search`, so a closure over the already-constructed player is the
/// smallest way to share this loop between `--head ntuple` and `--head cnn`
/// rather than duplicating it), returning one [`RecordV2`] with a
/// completed-Q improved-policy tail per non-terminal position.
fn play_one_gumbel_game(
    mut choose: impl FnMut(&State) -> GumbelOutcome<Move>,
    g: u64,
    game_seed: u64,
    cfg: &Config,
) -> Vec<RecordV2> {
    let mut move_rng = SmallRng::seed_from_u64(game_seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut state = State::default();
    let mut rows: Vec<GumbelRow> = Vec::new();
    let mut ply = 0u8;
    while !Othello::is_terminal(&state) {
        let outcome = choose(&state);
        rows.push(GumbelRow {
            head: record_for(&state, None),
            policy: outcome.improved_policy.clone(),
        });
        let forced = forced_move(&state, g, ply as u32, cfg.forced_opening_plies);
        let action = choose_selfplay_move(&outcome, forced, ply, cfg.temp_moves, &mut move_rng);
        state = Othello::apply(state, &action);
        ply += 1;
    }

    let winner = Othello::winner(&state);
    for row in &mut rows {
        let side_player = if row.head.side == 0 {
            Player::Black
        } else {
            Player::White
        };
        row.head.target = match winner {
            None => 0.0,
            Some(w) if w == side_player => 1.0,
            Some(_) => -1.0,
        };
    }
    rows.into_iter()
        .map(|row| RecordV2::from_record(row.head, &row.policy))
        .collect()
}

/// Play `cfg.games` Gumbel self-play games, pushing a [`RecordV2`] with the
/// completed-Q improved policy as its policy tail for every non-terminal
/// position, using either the n-tuple heads (`--head ntuple`, default) or
/// the joint CNN value+policy container (`--head cnn`).
fn dump_gumbel_games(cfg: &Config, out: &mut Vec<RecordV2>) {
    let gcfg = GumbelConfig {
        sims: cfg.gumbel_sims,
        max_considered: cfg.gumbel_max_considered,
        ..GumbelConfig::default()
    };

    if cfg.head == "cnn" {
        let net = match &cfg.cnn_weights {
            Some(p) => CnnValueNet::load(p)
                .unwrap_or_else(|e| panic!("cannot load CNN weights {}: {e}", p.display())),
            None => CnnValueNet::default(),
        };
        for g in 0..cfg.games {
            let game_seed = cfg.seed.wrapping_add(g).wrapping_add(1);
            let mut player = CnnGumbelPlayer::new(net.clone(), gcfg, game_seed);
            out.extend(play_one_gumbel_game(
                |s| player.choose(s),
                g,
                game_seed,
                cfg,
            ));
            if (g + 1) % 25 == 0 || g + 1 == cfg.games {
                eprintln!(
                    "  played {}/{} gumbel games ({} records) [cnn]",
                    g + 1,
                    cfg.games,
                    out.len()
                );
            }
        }
        return;
    }

    let (value_net, policy_net) = load_gumbel_nets(cfg);
    for g in 0..cfg.games {
        let game_seed = cfg.seed.wrapping_add(g).wrapping_add(1);
        let mut player =
            GumbelPlayer::with_policy(value_net.clone(), policy_net.clone(), gcfg, game_seed);
        out.extend(play_one_gumbel_game(
            |s| player.choose(s),
            g,
            game_seed,
            cfg,
        ));
        if (g + 1) % 25 == 0 || g + 1 == cfg.games {
            eprintln!(
                "  played {}/{} gumbel games ({} records)",
                g + 1,
                cfg.games,
                out.len()
            );
        }
    }
}

/// Entry point for `game-othello dump ...`. `args` is the argument iterator
/// positioned just past the `dump` token.
pub fn run(args: impl Iterator<Item = String>) {
    let cfg = parse_args(args);
    if cfg.label == "harvest" {
        return run_harvest(&cfg);
    }
    if cfg.label == "gumbel" {
        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);

        let mut buf = Vec::with_capacity(records.len() * RECORD_V2_HEAD_BYTES);
        for r in &records {
            r.encode(&mut buf);
        }
        if let Some(parent) = cfg.out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).expect("cannot create --out parent directory");
        }
        let mut w = BufWriter::new(File::create(&cfg.out).expect("cannot create --out file"));
        w.write_all(&buf).expect("write failed");
        w.flush().expect("flush failed");

        eprintln!(
            "wrote {} v2 records ({} bytes) from {} gumbel games to {}",
            records.len(),
            buf.len(),
            cfg.games,
            cfg.out.display()
        );
        return;
    }
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut records = Vec::new();

    let preset_table = cfg
        .engine
        .as_ref()
        .filter(|e| e.as_str() != "ntuple")
        .map(|_| {
            PresetTable::load_from_path(&cfg.presets_path)
                .unwrap_or_else(|e| panic!("cannot load {}: {e}", cfg.presets_path.display()))
        });

    for g in 0..cfg.games {
        match cfg.engine.as_deref() {
            Some("ntuple") => {
                let mut engine = ntuple_engine(
                    cfg.ntuple_iters,
                    cfg.ntuple_depth,
                    cfg.seed.wrapping_add(g).wrapping_add(1),
                );
                dump_one_game_engine(&mut rng, &mut records, &mut *engine, cfg.epsilon);
            }
            Some(preset) => {
                // Rebuild per game with a game-specific seed so a persistent
                // search tree can't carry across games.
                let table = preset_table.as_ref().expect("preset table not loaded");
                let mut engine = table
                    .build::<Othello>(preset, cfg.seed.wrapping_add(g).wrapping_add(1))
                    .unwrap_or_else(|e| panic!("preset {preset:?} did not resolve: {e}"));
                dump_one_game_engine(&mut rng, &mut records, &mut *engine, cfg.epsilon);
            }
            None => dump_one_game(&mut rng, &mut records),
        }
        if (g + 1) % 100 == 0 || g + 1 == cfg.games {
            eprintln!(
                "  dumped {}/{} games ({} records)",
                g + 1,
                cfg.games,
                records.len()
            );
        }
    }

    let mut buf = Vec::with_capacity(records.len() * RECORD_BYTES);
    for r in &records {
        r.encode(&mut buf);
    }
    let mut w = BufWriter::new(File::create(&cfg.out).expect("cannot create --out file"));
    w.write_all(&buf).expect("write failed");
    w.flush().expect("flush failed");

    if let Some(path) = &cfg.manifest {
        let mut json = String::from("[\n");
        for (i, r) in records.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"black\": \"{:016x}\", \"white\": \"{:016x}\", \"side\": {}, \"ply\": {}, \"target\": {}}}",
                r.black, r.white, r.side, r.ply, r.target
            ));
            json.push_str(if i + 1 == records.len() { "\n" } else { ",\n" });
        }
        json.push_str("]\n");
        std::fs::write(path, json).expect("cannot write --manifest file");
    }

    eprintln!(
        "wrote {} records ({} bytes) from {} games to {}",
        records.len(),
        buf.len(),
        cfg.games,
        cfg.out.display()
    );
}

// ---------------------------------------------------------------------------
// `--label harvest`: one search-per-move self-play pass, four label streams
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct HarvestParams {
    #[allow(dead_code)]
    engine: String,
    label_iters: usize,
    epsilon: f64,
    opening_plies: usize,
    games: u64,
    seed: u64,
    min_visits: u32,
    max_per_search: usize,
    dedup: bool,
    td_lambda: f64,
}

/// Parameters for the `--oracle edax` target teacher
/// (`games/othello/ntuple/harvest_edax.toml`). The self-play corpus params
/// still come from `harvest.toml`; only the label oracle changes.
#[derive(serde::Deserialize)]
struct EdaxParams {
    edax_level: u32,
    edax_exact_ply: u32,
    target_mode: String,
    squash_t: f32,
    /// Wall-clock ceiling on a single Edax search; a slower one is aborted
    /// and relabelled neutral rather than wedging the run.
    eval_timeout_s: f64,
    edax_binary: String,
    edax_data_dir: String,
}

/// A [`TargetOracle`] that labels each position with an independent Edax
/// evaluation, mapped into `[-1, 1]`. Near the endgame it raises the search
/// to a full solve.
struct EdaxLabel {
    edax: crate::edax::EdaxEval,
    level: u32,
    exact_ply: u32,
    mode: EdaxMode,
    squash_t: f32,
    /// Positions that came back as exact solves (label-quality cross-check).
    exact_hits: u64,
}

#[derive(Clone, Copy)]
enum EdaxMode {
    Sign,
    Squash,
}

impl EdaxLabel {
    fn new(p: &EdaxParams, level_override: Option<u32>) -> Self {
        let mode = match p.target_mode.as_str() {
            "sign" => EdaxMode::Sign,
            "squash" => EdaxMode::Squash,
            other => panic!("target_mode must be \"sign\" or \"squash\", got {other:?}"),
        };
        let level = level_override.unwrap_or(p.edax_level);
        EdaxLabel {
            edax: crate::edax::EdaxEval::spawn(
                &p.edax_binary,
                &p.edax_data_dir,
                level,
                std::time::Duration::from_secs_f64(p.eval_timeout_s),
            ),
            level,
            exact_ply: p.edax_exact_ply,
            mode,
            squash_t: p.squash_t,
            exact_hits: 0,
        }
    }

    fn map_score(&self, score: f32) -> f32 {
        match self.mode {
            EdaxMode::Sign => {
                if score > 0.0 {
                    1.0
                } else if score < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            EdaxMode::Squash => (score / self.squash_t).tanh(),
        }
    }
}

/// Repeatedly applies a forced pass -- a position whose *only* legal action
/// is `Move::PASS` -- until the side to move has a real action or the game
/// is over, folding the perspective flip each pass causes. A pass changes
/// no discs, so `value(pre-pass state) == -value(post-pass state)`.
///
/// Edax auto-plays a lone forced pass on its own when it has *some* other
/// legal continuation to search into, but a `go` on a position whose only
/// legal action is that pass does not reliably return at all -- a harvested
/// MCTS tree node can be exactly such a position, even though it never
/// arises in ordinary alternating self-play. Passing ourselves first
/// sidesteps asking Edax to search a lone-pass position at all.
fn skip_forced_passes(mut state: State) -> (State, f32) {
    let mut sign = 1.0f32;
    loop {
        if Othello::is_terminal(&state) {
            return (state, sign);
        }
        let mut acts = Vec::new();
        Othello::generate_actions(&state, &mut acts);
        if acts.len() == 1 && acts[0] == Move::PASS {
            state = Othello::apply(state, &Move::PASS);
            sign = -sign;
        } else {
            return (state, sign);
        }
    }
}

impl crate::harvest::TargetOracle for EdaxLabel {
    fn target(&mut self, state: &State) -> f32 {
        let (state, sign) = skip_forced_passes(*state);

        // A harvested tree node can be a genuinely terminal position (no
        // legal move for either side). Handing Edax a `setboard` on an
        // already-over game does not reliably reach its `*** Game Over
        // ***` report either; the final margin is already exact and known
        // without a search, so skip Edax entirely.
        if Othello::is_terminal(&state) {
            self.exact_hits += 1;
            return sign * self.map_score(crate::edax::terminal_disc_diff(&state) as f32);
        }

        let empties = 64 - state.occupied().count_ones();
        let level = if empties <= self.exact_ply {
            60
        } else {
            self.level
        };
        let s = self.edax.eval(&state, level);
        if s.exact {
            self.exact_hits += 1;
        }
        sign * self.map_score(s.score)
    }
}

/// Encode `records` to `<dir>/<name>.bin` and a matching JSON manifest at
/// `<dir>/<name>.json`.
fn write_arm(dir: &std::path::Path, name: &str, records: &[Record]) {
    let mut buf = Vec::with_capacity(records.len() * RECORD_BYTES);
    for r in records {
        r.encode(&mut buf);
    }
    std::fs::write(dir.join(format!("{name}.bin")), &buf).expect("cannot write arm .bin");

    let mut json = String::from("[\n");
    for (i, r) in records.iter().enumerate() {
        json.push_str(&format!(
            "  {{\"black\": \"{:016x}\", \"white\": \"{:016x}\", \"side\": {}, \"ply\": {}, \"target\": {}}}",
            r.black, r.white, r.side, r.ply, r.target
        ));
        json.push_str(if i + 1 == records.len() { "\n" } else { ",\n" });
    }
    json.push_str("]\n");
    std::fs::write(dir.join(format!("{name}.json")), json).expect("cannot write arm .json");
}

/// Play `opening_plies` uniform-random real (non-pass) plies from the start,
/// retrying if a line ends early.
fn random_opening(rng: &mut SmallRng, opening_plies: usize) -> State {
    'outer: loop {
        let mut state = State::default();
        let mut actions = Vec::new();
        for _ in 0..opening_plies {
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

fn run_harvest(cfg: &Config) {
    use crate::harvest::{
        harvest_tree_scored, harvest_tree_scored_oracle, label_search, root_value, HarvestFilter,
        TargetOracle,
    };
    use mcts::algorithms::Search;
    use mcts::game::PlayerIndex;
    use std::time::{Duration, Instant};

    let use_edax = cfg.oracle == "edax";
    let mut edax: Option<EdaxLabel> = if use_edax {
        let etext = std::fs::read_to_string(&cfg.edax_config)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.edax_config.display()));
        let ep: EdaxParams = toml::from_str(&etext).expect("edax config must parse");
        eprintln!(
            "oracle=edax  level={}  exact_ply={}  mode={}",
            cfg.edax_level_override.unwrap_or(ep.edax_level),
            ep.edax_exact_ply,
            ep.target_mode
        );
        Some(EdaxLabel::new(&ep, cfg.edax_level_override))
    } else {
        None
    };

    // CPU accounting: the self-play search is the controlled variable and is
    // paid by every arm; the Edax label cost is broken out per arm (root
    // evals feed B and D, the per-node harvest is arm C's alone).
    let mut t_selfplay = Duration::ZERO;
    let mut t_edax_root = Duration::ZERO;
    let mut t_edax_harvest = Duration::ZERO;

    let text = std::fs::read_to_string(&cfg.harvest_config)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cfg.harvest_config.display()));
    let mut params: HarvestParams = toml::from_str(&text).expect("harvest config must parse");
    if cfg.games_overridden {
        params.games = cfg.games;
    }
    if cfg.seed_overridden {
        params.seed = cfg.seed;
    }
    assert!(
        (0.0..=1.0).contains(&params.epsilon),
        "harvest epsilon must be in [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&params.td_lambda),
        "td_lambda must be in [0, 1]"
    );

    let dir = &cfg.out;
    std::fs::create_dir_all(dir).expect("cannot create --out directory");

    let filter = HarvestFilter {
        min_visits: params.min_visits,
        max_per_search: params.max_per_search,
        max_depth: None,
    };

    let mut arm_a: Vec<Record> = Vec::new();
    let mut arm_b: Vec<Record> = Vec::new();
    let mut arm_c: Vec<(u32, Record)> = Vec::new();
    let mut arm_d: Vec<Record> = Vec::new();

    for g in 0..params.games {
        let mut walk_rng = SmallRng::seed_from_u64(params.seed.wrapping_add(g).wrapping_add(1));
        let mut engine = label_search(
            params.label_iters,
            params.seed.wrapping_add(g).wrapping_add(1),
        );
        let mut state = random_opening(&mut walk_rng, params.opening_plies);
        let a_first = arm_a.len();

        // Per label-search ply: (arm_a record index, side, root value in the
        // fixed Black-to-move perspective).
        let mut traj: Vec<(usize, u8, f64)> = Vec::new();
        let mut actions = Vec::new();

        while !Othello::is_terminal(&state) {
            actions.clear();
            Othello::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }
            if actions == [Move::PASS] {
                state = Othello::apply(state, &Move::PASS);
                continue;
            }
            if walk_rng.gen_bool(params.epsilon) {
                let a = actions[walk_rng.gen_range(0..actions.len())];
                state = Othello::apply(state, &a);
                continue;
            }

            let t0 = Instant::now();
            let action = engine.choose_action(&state);
            t_selfplay += t0.elapsed();

            let pidx = state.turn.to_index();
            let rv: f64 = if let Some(o) = edax.as_mut() {
                let t = Instant::now();
                let v = o.target(&state) as f64;
                t_edax_root += t.elapsed();
                v
            } else {
                root_value(&engine, &state, pidx)
            };
            let rv_black = if state.turn == Player::Black { rv } else { -rv };

            let mut rec = record_for(&state, None);
            arm_a.push(rec); // outcome target backfilled below
            rec.target = rv as f32;
            arm_b.push(rec);
            if let Some(o) = edax.as_mut() {
                let t = Instant::now();
                arm_c.extend(harvest_tree_scored_oracle(&engine, &state, &filter, o));
                t_edax_harvest += t.elapsed();
            } else {
                arm_c.extend(harvest_tree_scored(&engine, &state, &filter));
            }
            traj.push((arm_a.len() - 1, rec.side, rv_black));

            state = Othello::apply(state, &action);
        }

        let winner = Othello::winner(&state);
        finish_game(&mut arm_a, a_first, winner);
        let z_black = match winner {
            None => 0.0,
            Some(Player::Black) => 1.0,
            Some(Player::White) => -1.0,
        };

        // Forward-view TD(lambda) return, computed backward along the played
        // line: G_t = (1-lambda) V(s_{t+1}) + lambda G_{t+1}, with the value
        // past the last recorded ply pinned to the terminal outcome.
        let mut g_next = z_black;
        for t in (0..traj.len()).rev() {
            let (a_idx, side, _) = traj[t];
            let v_next = if t + 1 == traj.len() {
                z_black
            } else {
                traj[t + 1].2
            };
            let g = (1.0 - params.td_lambda) * v_next + params.td_lambda * g_next;
            let mut rec = arm_a[a_idx];
            rec.target = (if side == 0 { g } else { -g }) as f32;
            arm_d.push(rec);
            g_next = g;
        }

        if (g + 1) % 100 == 0 || g + 1 == params.games {
            eprintln!(
                "  harvest {}/{} games  arm_a={} arm_c={} (pre-dedup)",
                g + 1,
                params.games,
                arm_a.len(),
                arm_c.len()
            );
        }
    }

    // arm C dedup: collapse (black, white, side) across the whole file,
    // keeping the highest-visit target.
    let arm_c_raw_count = arm_c.len();
    let harvest_ratio_raw = arm_c_raw_count as f64 / arm_a.len().max(1) as f64;
    if params.dedup {
        use std::collections::HashMap;
        let mut best: HashMap<(u64, u64, u8), (u32, f32)> = HashMap::new();
        for (v, r) in arm_c.drain(..) {
            let key = (r.black, r.white, r.side);
            let e = best.entry(key).or_insert((0, 0.0));
            if v >= e.0 {
                *e = (v, r.target);
            }
        }
        arm_c = best
            .into_iter()
            .map(|((black, white, side), (v, target))| {
                let ply = ((black | white).count_ones() as u8).saturating_sub(4);
                (
                    v,
                    Record {
                        black,
                        white,
                        side,
                        ply,
                        target,
                    },
                )
            })
            .collect();
    }
    let arm_c_records: Vec<Record> = arm_c.iter().map(|(_, r)| *r).collect();

    write_arm(dir, "arm_a", &arm_a);
    write_arm(dir, "arm_b", &arm_b);
    write_arm(dir, "arm_c", &arm_c_records);
    write_arm(dir, "arm_d", &arm_d);

    // CPU accounting. For `--oracle mcts` the arms differ only in training
    // cost (bakeoff.sh times that); for `--oracle edax` the plot needs the
    // Edax label bill broken out -- root evals feed arms B/D, the per-node
    // harvest is arm C's alone, and arm A pays no Edax at all.
    let cpu_block = if use_edax {
        let o = edax.as_ref().unwrap();
        format!(
            ",\n  \"oracle\": \"edax\",\n  \"cpu\": {{\n    \
             \"selfplay_s\": {:.2},\n    \"edax_root_s\": {:.2},\n    \
             \"edax_harvest_s\": {:.2},\n    \"edax_calls\": {},\n    \
             \"edax_nodes\": {},\n    \"edax_exact_hits\": {},\n    \"edax_timeouts\": {}\n  }}",
            t_selfplay.as_secs_f64(),
            t_edax_root.as_secs_f64(),
            t_edax_harvest.as_secs_f64(),
            o.edax.calls(),
            o.edax.total_nodes(),
            o.exact_hits,
            o.edax.timeouts(),
        )
    } else {
        format!(
            ",\n  \"oracle\": \"mcts\",\n  \"selfplay_s\": {:.2}",
            t_selfplay.as_secs_f64()
        )
    };
    let summary = format!(
        "{{\n  \"games\": {},\n  \"label_iters\": {},\n  \"epsilon\": {},\n  \"td_lambda\": {},\n  \
         \"min_visits\": {},\n  \"max_per_search\": {},\n  \"dedup\": {},\n  \
         \"arm_a\": {},\n  \"arm_b\": {},\n  \"arm_c\": {},\n  \"arm_c_raw\": {},\n  \"arm_d\": {},\n  \
         \"harvest_ratio\": {:.2},\n  \"harvest_ratio_raw\": {:.2}{}\n}}\n",
        params.games,
        params.label_iters,
        params.epsilon,
        params.td_lambda,
        params.min_visits,
        params.max_per_search,
        params.dedup,
        arm_a.len(),
        arm_b.len(),
        arm_c_records.len(),
        arm_c_raw_count,
        arm_d.len(),
        arm_c_records.len() as f64 / arm_a.len().max(1) as f64,
        harvest_ratio_raw,
        cpu_block,
    );
    std::fs::write(dir.join("harvest.json"), &summary).expect("cannot write harvest.json");

    eprintln!(
        "wrote arm_a={} arm_b={} arm_c={} arm_d={} to {}  (harvest ratio {:.2}x)",
        arm_a.len(),
        arm_b.len(),
        arm_c_records.len(),
        arm_d.len(),
        dir.display(),
        arm_c_records.len() as f64 / arm_a.len().max(1) as f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gumbel_config() -> Config {
        Config {
            out: PathBuf::new(),
            manifest: None,
            games: 2,
            seed: 3,
            engine: None,
            epsilon: 0.1,
            presets_path: PathBuf::from("games/othello/presets.json"),
            label: "gumbel".to_string(),
            harvest_config: PathBuf::from("games/othello/ntuple/harvest.toml"),
            games_overridden: false,
            seed_overridden: false,
            oracle: "mcts".to_string(),
            edax_config: PathBuf::from("games/othello/ntuple/harvest_edax.toml"),
            edax_level_override: None,
            ntuple_iters: 200,
            ntuple_depth: 0,
            model_toml: PathBuf::from("ntuple/tests/tiny.toml"),
            weights_dir: None,
            head: "ntuple".to_string(),
            cnn_weights: None,
            gumbel_sims: 8,
            gumbel_max_considered: 4,
            temp_moves: 6,
            forced_opening_plies: 0,
        }
    }

    /// `model_toml` above is relative to the crate root (`CARGO_MANIFEST_DIR`),
    /// matching how `--model` is resolved from a shell invocation; tests run
    /// from the crate directory too, so resolve it the same way rather than
    /// hand-building an absolute path per call site.
    fn with_crate_relative_model(mut cfg: Config) -> Config {
        cfg.model_toml =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&cfg.model_toml);
        cfg
    }

    /// Builds a throwaway `research/az-train`-shaped checkpoint directory
    /// (`model.toml` + `weights.bin` + `weights.meta.json` + `policy.bin` +
    /// `policy.meta.json`, all zero-weight) from the committed `tiny.toml`
    /// fixture, so `--weights-dir`'s load path -- distinct from the
    /// `--model`-only generation-0 path the other tests exercise -- gets a
    /// real end-to-end check. Removed by the caller.
    fn write_tiny_checkpoint_dir() -> std::path::PathBuf {
        let tiny_toml = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/tests/tiny.toml"),
        )
        .unwrap();
        let geom = crate::ntuple::ModelGeometry::parse(&tiny_toml);

        let dir = std::env::temp_dir().join(format!(
            "othello_gumbel_dump_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.toml"), &tiny_toml).unwrap();

        let value_weights = vec![0.0f32; geom.n_weights()];
        let value_bytes: Vec<u8> = value_weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(dir.join("weights.bin"), &value_bytes).unwrap();
        std::fs::write(
            dir.join("weights.meta.json"),
            format!(
                r#"{{"model_toml_sha256": "{}", "n_weights": {}}}"#,
                geom.sha256_hex(),
                geom.n_weights()
            ),
        )
        .unwrap();

        let policy_weights = vec![0.0f32; geom.n_weights() * 64];
        let policy_bytes: Vec<u8> = policy_weights
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        std::fs::write(dir.join("policy.bin"), &policy_bytes).unwrap();
        std::fs::write(
            dir.join("policy.meta.json"),
            format!(
                r#"{{"model_toml_sha256": "{}", "n_weights": {}}}"#,
                geom.sha256_hex(),
                geom.n_weights()
            ),
        )
        .unwrap();

        dir
    }

    #[test]
    fn a_gumbel_selfplay_run_loads_a_trained_checkpoint_via_weights_dir() {
        let dir = write_tiny_checkpoint_dir();
        let mut cfg = gumbel_config();
        cfg.weights_dir = Some(dir.clone());
        cfg.games = 1;

        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);
        assert!(!records.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Ignored in the default `cargo test --lib` run: `dump_gumbel_games`
    /// always plays to a real terminal (Othello games run ~60 plies, unlike
    /// a hand-built short fixture), and `CnnValueNet`'s direct-loop
    /// numpy-style convolution -- fast under `--release` -- costs ~1s/ply
    /// under an unoptimized debug build even at the smallest possible
    /// search budget (1 game, 2 sims, 1 max-considered), landing this
    /// single test at ~60s. That is exactly the class of check
    /// `AGENTS.md`'s "keep `cargo test --lib` fast" rule exists for. Run
    /// explicitly (`cargo test --release -p game-othello
    /// dump::tests::a_cnn_gumbel_selfplay_run_produces_completed_q_records`)
    /// after touching `--head cnn` wiring; `selfplay::tests::cnn_gumbel_*`
    /// above cover the same seam on hand-built near-terminal fixtures at
    /// negligible cost and do run by default.
    #[test]
    #[ignore = "plays a full ~60-ply game through CnnValueNet's unoptimized-debug-build conv; ~60s, run with --release"]
    fn a_cnn_gumbel_selfplay_run_produces_completed_q_records() {
        let mut cfg = gumbel_config();
        cfg.head = "cnn".to_string();
        cfg.games = 1;
        cfg.gumbel_sims = 2;
        cfg.gumbel_max_considered = 1;

        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);
        assert!(!records.is_empty());
        for r in &records {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(
                !r.policy.is_empty(),
                "cnn gumbel positions carry a policy target"
            );
            let sum: f32 = r.policy.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy tail sums to {sum}");
        }
    }

    #[test]
    fn a_gumbel_selfplay_game_records_a_completed_q_policy_tail_per_position() {
        let cfg = with_crate_relative_model(gumbel_config());
        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);
        assert!(!records.is_empty());
        for r in &records {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(
                !r.policy.is_empty(),
                "gumbel positions carry a policy target"
            );
            let sum: f32 = r.policy.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy tail sums to {sum}");
            for (square, _) in &r.policy {
                assert!(*square as usize <= Move::PASS.0 as usize);
            }
        }
    }

    /// A forced-opening-plies run must still start every game at ply 0 and
    /// increment by one, so the shard stays compatible with a whole-game
    /// replay splitter -- mirrors Connect Four's
    /// `a_forced_opening_game_follows_the_prefix_then_stays_a_valid_shard`.
    #[test]
    fn a_forced_opening_gumbel_run_still_labels_plies_consistently() {
        let mut cfg = with_crate_relative_model(gumbel_config());
        cfg.games = 3;
        cfg.forced_opening_plies = 4;
        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);
        assert_eq!(records[0].ply, 0);
        for pair in records.windows(2) {
            // A pass leaves the disc count (and so `ply`) unchanged, unlike
            // Connect Four where every move places a disc -- non-decreasing
            // is the real invariant here, matching
            // `a_dumped_game_labels_every_position_consistently` above.
            assert!(pair[1].ply == 0 || pair[1].ply >= pair[0].ply);
        }
    }

    /// `forced_move` must never fabricate a choice at a genuinely
    /// single-legal-action position (a forced pass, or Othello's rare
    /// single-real-move positions) -- it returns `None` and lets the search
    /// decide, rather than looping the trivial one-element action list.
    #[test]
    fn forced_move_defers_to_search_when_there_is_no_real_choice() {
        let lone_pass = State {
            black: crate::BB::from_bits(1 << 0),
            white: crate::BB::from_bits(1 << 63),
            turn: Player::Black,
            last_pass: false,
            ..State::default()
        };
        let mut actions = Vec::new();
        Othello::generate_actions(&lone_pass, &mut actions);
        assert_eq!(actions, vec![Move::PASS]);
        assert_eq!(forced_move(&lone_pass, 0, 0, 10), None);
    }

    #[test]
    fn forced_move_picks_a_legal_action_deterministically_per_game_and_ply() {
        let state = State::default();
        for g in 0..20u64 {
            for ply in 0..3u32 {
                let a = forced_move(&state, g, ply, 3).expect("opening has real choices");
                let mut actions = Vec::new();
                Othello::generate_actions(&state, &mut actions);
                assert!(actions.contains(&a));
                // Deterministic: same inputs, same output.
                assert_eq!(forced_move(&state, g, ply, 3), Some(a));
            }
        }
    }

    fn sample_states() -> Vec<State> {
        let d3 = Othello::apply(State::default(), &Move(19));
        let d3c5 = Othello::apply(d3, &Move(34));
        let hand = State {
            black: crate::BB::from_bits((1 << 0) | (1 << 63)),
            white: crate::BB::from_bits((1 << 7) | (1 << 56)),
            turn: Player::White,
            last_pass: true,
            ..State::default()
        };
        vec![State::default(), d3, d3c5, hand]
    }

    #[test]
    fn record_v2_round_trips_through_encode_decode_with_an_empty_policy() {
        for st in sample_states() {
            let head = record_for(&st, Some(Player::Black));
            let rec = RecordV2::from_record(head, &[]);
            let mut buf = Vec::new();
            rec.encode(&mut buf);
            assert_eq!(buf.len(), RECORD_V2_HEAD_BYTES);
            let (back, consumed) = RecordV2::decode(&buf).expect("decode");
            assert_eq!(consumed, buf.len());
            assert_eq!(back, rec);
        }
    }

    #[test]
    fn record_v2_round_trips_a_completed_q_shaped_policy_tail() {
        let st = sample_states()[1];
        let head = record_for(&st, Some(Player::White));
        let policy = vec![(Move(19), 0.6_f32), (Move(26), 0.25), (Move::PASS, 0.15)];
        let rec = RecordV2::from_record(head, &policy);
        assert_eq!(rec.policy, vec![(19, 0.6), (26, 0.25), (64, 0.15)]);

        let mut buf = Vec::new();
        rec.encode(&mut buf);
        assert_eq!(
            buf.len(),
            RECORD_V2_HEAD_BYTES + policy.len() * POLICY_ENTRY_BYTES
        );
        let (back, consumed) = RecordV2::decode(&buf).expect("decode");
        assert_eq!(consumed, buf.len());
        assert_eq!(back, rec);
    }

    #[test]
    fn record_v2_decode_returns_none_on_a_truncated_buffer() {
        let policy = vec![(Move(0), 1.0_f32)];
        let rec = RecordV2::from_record(record_for(&State::default(), None), &policy);
        let mut buf = Vec::new();
        rec.encode(&mut buf);
        // Missing the last policy entry byte.
        assert!(RecordV2::decode(&buf[..buf.len() - 1]).is_none());
        // Missing the whole tail (head only, but n_policy byte says 1).
        assert!(RecordV2::decode(&buf[..RECORD_V2_HEAD_BYTES]).is_none());
        // Truncated before even the head is complete.
        assert!(RecordV2::decode(&buf[..RECORD_V2_HEAD_BYTES - 1]).is_none());
    }

    #[test]
    fn records_round_trip_through_encode_decode() {
        for (i, st) in sample_states().into_iter().enumerate() {
            for winner in [None, Some(Player::Black), Some(Player::White)] {
                let rec = record_for(&st, winner);
                let mut buf = Vec::new();
                rec.encode(&mut buf);
                assert_eq!(buf.len(), RECORD_BYTES, "state {i}");
                let back = Record::decode(&buf.as_slice().try_into().unwrap());
                assert_eq!(back, rec, "state {i} winner {winner:?}");
            }
        }
    }

    #[test]
    fn target_is_from_side_to_move_perspective() {
        // Opening: black to move. Black wins -> +1; white wins -> -1.
        let s = State::default();
        assert_eq!(record_for(&s, Some(Player::Black)).target, 1.0);
        assert_eq!(record_for(&s, Some(Player::White)).target, -1.0);
        assert_eq!(record_for(&s, None).target, 0.0);
        assert_eq!(record_for(&s, None).side, 0);
        assert_eq!(record_for(&s, None).ply, 0);
    }

    #[test]
    fn a_dumped_game_labels_every_position_consistently() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut recs = Vec::new();
        dump_one_game(&mut rng, &mut recs);
        assert!(!recs.is_empty());
        // Every target is one of the three legal values, and ply is
        // non-decreasing across the game.
        let mut last_ply = 0u8;
        for r in &recs {
            assert!([1.0f32, -1.0, 0.0].contains(&r.target));
            assert!(r.ply >= last_ply);
            last_ply = r.ply;
        }
    }

    /// Decode an Edax `setboard` string (64 square chars + side-to-move
    /// token) into a `State`. `hashes` are left at the default zeroed
    /// value -- fine for tests that don't touch Zobrist lookups.
    fn state_from_edax_board(board: &str) -> State {
        let (squares, turn_tok) = board.split_once(' ').unwrap();
        let bytes = squares.as_bytes();
        assert_eq!(bytes.len(), 64);
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, &c) in bytes.iter().enumerate() {
            match c {
                b'X' => black |= 1 << i,
                b'O' => white |= 1 << i,
                b'-' => {}
                other => panic!("unexpected board char {other}"),
            }
        }
        State {
            black: crate::BB::from_bits(black),
            white: crate::BB::from_bits(white),
            turn: if turn_tok == "X" {
                Player::Black
            } else {
                Player::White
            },
            ..State::default()
        }
    }

    /// A position where White to move has no real action, only
    /// `Move::PASS`: Edax's `go` never returns for a lone-pass position,
    /// so `skip_forced_passes` must resolve it locally instead of ever
    /// handing it to Edax.
    #[test]
    fn skip_forced_passes_resolves_a_lone_pass_without_asking_edax() {
        let white_to_move_only_pass = state_from_edax_board(
            "OOOOOOOOXOXXXOOOXXOXXOOOXOXXOXOOXOOXXOOOXOOOOXOXXOOOOOXXXOOXO-XX O",
        );
        let mut acts = Vec::new();
        Othello::generate_actions(&white_to_move_only_pass, &mut acts);
        assert_eq!(acts, vec![Move::PASS], "fixture must be a lone-pass position");

        let (resolved, sign) = skip_forced_passes(white_to_move_only_pass);
        assert_eq!(sign, -1.0, "one pass folds one perspective flip");
        assert_eq!(resolved.turn, Player::Black);
        let mut resolved_acts = Vec::new();
        Othello::generate_actions(&resolved, &mut resolved_acts);
        assert_ne!(
            resolved_acts,
            vec![Move::PASS],
            "must stop once the side to move has a real action"
        );
        // Passing changes no discs.
        assert_eq!(resolved.black, white_to_move_only_pass.black);
        assert_eq!(resolved.white, white_to_move_only_pass.white);
    }

    #[test]
    fn skip_forced_passes_is_a_no_op_when_a_real_move_exists() {
        let (resolved, sign) = skip_forced_passes(State::default());
        assert_eq!(sign, 1.0);
        assert_eq!(resolved, State::default());
    }

    #[test]
    fn skip_forced_passes_stops_immediately_at_an_already_terminal_state() {
        let full_board = State {
            black: crate::BB::from_bits(u64::MAX),
            white: crate::BB::from_bits(0),
            turn: Player::Black,
            ..State::default()
        };
        let (resolved, sign) = skip_forced_passes(full_board);
        assert_eq!(sign, 1.0);
        assert!(Othello::is_terminal(&resolved));
    }
}
