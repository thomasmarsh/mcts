//! `game-ttt dump` -- write self-play positions as little-endian records for
//! an offline self-play training loop (Gumbel-style: a value/policy target
//! per position, refit between generations).
//!
//! This is kept out of the generic `game-host` CLI -- `main.rs` intercepts
//! `dump` as the first argument and calls [`run`] before `game_host::run_cli`
//! sees the args, the same split `games/othello/src/dump.rs` uses.
//!
//! ## Record format v2
//!
//! A fixed 11-byte head followed by a variable-length policy tail, packed
//! (no padding), little-endian:
//!
//! | field    | type          | bytes        |
//! |----------|---------------|--------------|
//! | board    | u32 LE        | 4            |
//! | side     | u8            | 1            |
//! | ply      | u8            | 1            |
//! | value    | f32 LE        | 4            |
//! | n_policy | u8            | 1            |
//! | policy   | n * (u8, f32) | n_policy * 5 |
//!
//! `board` is `game_ttt::Position`'s packed-`u32` encoding (2 bits per cell,
//! digit 1 = X, 2 = O). `side` is 0 for X to move, 1 for O. `ply` is the
//! number of pieces on the board. `value` is the final game result **from
//! the side-to-move player's perspective**: `+1.0` win, `-1.0` loss, `0.0`
//! draw. The policy tail is the improved-policy training target -- pairs of
//! `(cell_index, probability)` from the Gumbel Sequential-Halving visit
//! distribution at the root. It is empty for `--label outcome` dumps, which
//! record positions from uniform-random or preset-engine self-play with no
//! search-derived policy.
//!
//! v2 records are variable-width, so a reader must walk them sequentially --
//! `np.fromfile` with a fixed dtype does not apply (see
//! `research/az-train/`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::algorithms::Search;
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::selfplay::GumbelPlayer;
use crate::valuenet::LinearValueNet;
use crate::{HashedPosition, Move, Piece, Position, TicTacToe};

/// Draw one move from a Sequential-Halving visit distribution (probabilities
/// summing to 1), falling back to the first entry on a rounding shortfall.
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

/// One dumped position. See the module docs for field semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub board: u32,
    pub side: u8,
    pub ply: u8,
    pub value: f32,
    /// `(cell_index, probability)` pairs -- the improved-policy target.
    /// Empty when no search produced a policy for this position.
    pub policy: Vec<(u8, f32)>,
}

/// Size of a [`Record`]'s fixed head, in bytes (everything up to and
/// including the `n_policy` count byte, before the policy tail).
pub const RECORD_HEAD_BYTES: usize = 11;

/// Size of one policy tail entry, in bytes: `(action: u8, prob: f32 LE)`.
pub const POLICY_ENTRY_BYTES: usize = 5;

impl Record {
    /// Append this record's little-endian bytes to `buf`. Fields are written
    /// one at a time -- never a `#[repr(C)]` struct cast, which could
    /// introduce padding.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.board.to_le_bytes());
        buf.push(self.side);
        buf.push(self.ply);
        buf.extend_from_slice(&self.value.to_le_bytes());
        let n: u8 = self
            .policy
            .len()
            .try_into()
            .expect("policy tail longer than 255 entries");
        buf.push(n);
        for (action, prob) in &self.policy {
            buf.push(*action);
            buf.extend_from_slice(&prob.to_le_bytes());
        }
    }

    /// Decode one record from the front of `bytes`, returning it and the
    /// number of bytes it consumed. Returns `None` if `bytes` is too short
    /// to hold a complete record.
    pub fn decode(bytes: &[u8]) -> Option<(Record, usize)> {
        if bytes.len() < RECORD_HEAD_BYTES {
            return None;
        }
        let board = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let side = bytes[4];
        let ply = bytes[5];
        let value = f32::from_le_bytes(bytes[6..10].try_into().unwrap());
        let n_policy = bytes[10] as usize;
        let tail_bytes = n_policy * POLICY_ENTRY_BYTES;
        let total = RECORD_HEAD_BYTES + tail_bytes;
        if bytes.len() < total {
            return None;
        }
        let mut policy = Vec::with_capacity(n_policy);
        for i in 0..n_policy {
            let off = RECORD_HEAD_BYTES + i * POLICY_ENTRY_BYTES;
            let action = bytes[off];
            let prob = f32::from_le_bytes(bytes[off + 1..off + 5].try_into().unwrap());
            policy.push((action, prob));
        }
        Some((
            Record {
                board,
                side,
                ply,
                value,
                policy,
            },
            total,
        ))
    }
}

fn side_of(p: Piece) -> u8 {
    match p {
        Piece::X => 0,
        Piece::O => 1,
    }
}

fn piece_count(board: u32) -> u8 {
    (0..9).filter(|&i| (board >> (i << 1)) & 0b11 != 0).count() as u8
}

/// Build a [`Record`] for `state` given the final `winner` (`None` == draw)
/// and an optional policy target. `value` is from `state`'s player-to-move
/// perspective.
pub fn record_for(state: &Position, winner: Option<Piece>, policy: Vec<(u8, f32)>) -> Record {
    let side = side_of(state.turn);
    let value = match winner {
        None => 0.0,
        Some(w) if side_of(w) == side => 1.0,
        Some(_) => -1.0,
    };
    Record {
        board: state.board,
        side,
        ply: piece_count(state.board),
        value,
        policy,
    }
}

struct Config {
    out: PathBuf,
    games: u64,
    seed: u64,
    label: String,
    engine: Option<String>,
    epsilon: f64,
    presets_path: PathBuf,
    /// `--label gumbel` only: value-head weights (`az-train` output). Absent
    /// == the all-zero generation-0 net.
    weights: Option<PathBuf>,
    /// `--label gumbel` only: Gumbel simulation budget and root candidate cap.
    sims: u32,
    max_considered: usize,
    /// `--label gumbel` only: number of opening plies whose move is *sampled*
    /// from the Sequential-Halving visit distribution rather than taken as
    /// the argmax. Keeps self-play trajectories diverse so the value head
    /// trains on a distribution that does not collapse onto its own current
    /// best line each generation.
    temp_moves: u8,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Config {
    let mut out = None;
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut label = "outcome".to_string();
    let mut engine = None;
    let mut epsilon = 0.1f64;
    let mut presets_path = PathBuf::from("games/ttt/presets.json");
    let mut weights = None;
    let mut sims = 32u32;
    let mut max_considered = 8usize;
    let mut temp_moves = 3u8;
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--games" => games = val().parse().expect("--games must be an integer"),
            "--seed" => seed = val().parse().expect("--seed must be an integer"),
            "--label" => label = val(),
            "--engine" => engine = Some(val()),
            "--epsilon" => epsilon = val().parse().expect("--epsilon must be a float"),
            "--presets" => presets_path = PathBuf::from(val()),
            "--weights" => weights = Some(PathBuf::from(val())),
            "--sims" => sims = val().parse().expect("--sims must be an integer"),
            "--max-considered" => {
                max_considered = val().parse().expect("--max-considered must be an integer")
            }
            "--temp-moves" => temp_moves = val().parse().expect("--temp-moves must be an integer"),
            "-h" | "--help" => {
                eprintln!(
                    "usage: game-ttt dump --out <path> [--games N] [--seed N] \
                     [--label outcome|gumbel] [--engine <preset>] [--epsilon P] \
                     [--presets <path>] [--weights <weights.bin>] [--sims N] \
                     [--max-considered N] [--temp-moves N]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown dump argument: {other}"),
        }
    }
    match label.as_str() {
        "outcome" | "gumbel" => {}
        other => panic!("unknown --label mode: {other}"),
    }
    assert!(sims >= 1, "--sims must be positive");
    assert!(max_considered >= 1, "--max-considered must be positive");
    assert!(
        (0.0..=1.0).contains(&epsilon),
        "--epsilon must be in [0, 1], got {epsilon}"
    );
    Config {
        out: out.expect("--out is required"),
        games,
        seed,
        label,
        engine,
        epsilon,
        presets_path,
        weights,
        sims,
        max_considered,
        temp_moves,
    }
}

/// Play `cfg.games` Gumbel self-play games, pushing a [`Record`] with the
/// Sequential-Halving visit distribution as its policy tail for every
/// non-terminal position.
fn dump_gumbel_games(cfg: &Config, records: &mut Vec<Record>) {
    let net = match &cfg.weights {
        Some(p) => LinearValueNet::load(p)
            .unwrap_or_else(|e| panic!("cannot load weights {}: {e}", p.display())),
        None => LinearValueNet::default(),
    };
    let gcfg = GumbelConfig {
        sims: cfg.sims,
        max_considered: cfg.max_considered,
        ..GumbelConfig::default()
    };

    for g in 0..cfg.games {
        let game_seed = cfg.seed.wrapping_add(g).wrapping_add(1);
        let mut player = GumbelPlayer::new(net.clone(), gcfg, game_seed);
        let mut move_rng = SmallRng::seed_from_u64(game_seed ^ 0x9E37_79B9_7F4A_7C15);

        let mut state = HashedPosition::new();
        let first = records.len();
        let mut ply = 0u8;
        while !TicTacToe::is_terminal(&state) {
            let outcome = player.choose(&state);
            let policy: Vec<(u8, f32)> = outcome
                .visit_distribution
                .iter()
                .map(|(m, p)| (m.0, *p))
                .collect();
            records.push(record_for(&state.position, None, policy));
            let action = if ply < cfg.temp_moves {
                sample_visit_distribution(&outcome.visit_distribution, &mut move_rng)
            } else {
                outcome.action
            };
            state = TicTacToe::apply(state, &action);
            ply += 1;
        }
        finish_game(records, first, winner_of(&state));

        if (g + 1) % 50 == 0 || g + 1 == cfg.games {
            eprintln!(
                "  played {}/{} gumbel games ({} records)",
                g + 1,
                cfg.games,
                records.len()
            );
        }
    }
}

/// Backfill the final-outcome value into every record pushed for this game
/// (those from index `first` onward), now that `winner` is known.
fn finish_game(records: &mut [Record], first: usize, winner: Option<Piece>) {
    for rec in &mut records[first..] {
        let side_piece = if rec.side == 0 { Piece::X } else { Piece::O };
        rec.value = match winner {
            None => 0.0,
            Some(w) if w == side_piece => 1.0,
            Some(_) => -1.0,
        };
    }
}

fn winner_of(state: &HashedPosition) -> Option<Piece> {
    if TicTacToe::is_terminal(state) {
        TicTacToe::winner(state)
    } else {
        None
    }
}

/// Play one game, pushing a [`Record`] for every non-terminal position onto
/// `records`. Moves are uniform-random unless `engine` is set, in which case
/// they come from that engine except with probability `epsilon`.
fn dump_one_game(
    rng: &mut SmallRng,
    records: &mut Vec<Record>,
    mut engine: Option<&mut dyn Search<G = TicTacToe>>,
    epsilon: f64,
) {
    let mut state = HashedPosition::new();
    let mut actions = Vec::new();
    let first = records.len();
    while !TicTacToe::is_terminal(&state) {
        actions.clear();
        TicTacToe::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            break;
        }
        records.push(record_for(&state.position, None, Vec::new()));
        let action = match engine.as_deref_mut() {
            Some(e) if !rng.gen_bool(epsilon) => e.choose_action(&state),
            _ => actions[rng.gen_range(0..actions.len())],
        };
        state = TicTacToe::apply(state, &action);
    }
    finish_game(records, first, winner_of(&state));
}

/// Entry point for `game-ttt dump ...`. `args` is the argument iterator
/// positioned just past the `dump` token.
pub fn run(args: impl Iterator<Item = String>) {
    let cfg = parse_args(args);

    let mut records = Vec::new();

    if cfg.label == "gumbel" {
        dump_gumbel_games(&cfg, &mut records);
    } else {
        let mut rng = SmallRng::seed_from_u64(cfg.seed);
        let preset_table = cfg.engine.as_ref().map(|_| {
            PresetTable::load_from_path(&cfg.presets_path)
                .unwrap_or_else(|e| panic!("cannot load {}: {e}", cfg.presets_path.display()))
        });

        for g in 0..cfg.games {
            match (&cfg.engine, &preset_table) {
                (Some(preset), Some(table)) => {
                    let mut engine = table
                        .build::<TicTacToe>(preset, cfg.seed.wrapping_add(g).wrapping_add(1))
                        .unwrap_or_else(|e| panic!("preset {preset:?} did not resolve: {e}"));
                    dump_one_game(&mut rng, &mut records, Some(&mut *engine), cfg.epsilon);
                }
                _ => dump_one_game(&mut rng, &mut records, None, cfg.epsilon),
            }
            if (g + 1) % 200 == 0 || g + 1 == cfg.games {
                eprintln!(
                    "  dumped {}/{} games ({} records)",
                    g + 1,
                    cfg.games,
                    records.len()
                );
            }
        }
    }

    let mut buf = Vec::with_capacity(records.len() * RECORD_HEAD_BYTES);
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
        "wrote {} records ({} bytes) from {} games to {}",
        records.len(),
        buf.len(),
        cfg.games,
        cfg.out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Move;

    fn gumbel_player(seed: u64, cfg: GumbelConfig) -> GumbelPlayer {
        GumbelPlayer::new(LinearValueNet::default(), cfg, seed)
    }

    /// Sequential Halving spends its budget on the candidate set only:
    /// with `max_considered = 2` on a three-move root, at most two children
    /// are ever visited, and the recorded distribution is over exactly
    /// those.
    #[test]
    fn gumbel_visits_only_the_candidate_set() {
        let mut state = HashedPosition::new();
        for m in [0u8, 1, 2, 3, 4, 5] {
            state = TicTacToe::apply(state, &Move(m));
        }
        let cfg = GumbelConfig {
            sims: 12,
            max_considered: 2,
            ..GumbelConfig::default()
        };
        let out = gumbel_player(7, cfg).choose(&state);

        assert!(out.visit_distribution.len() <= 2);
        let mut legal = Vec::new();
        TicTacToe::generate_actions(&state, &mut legal);
        assert!(legal.contains(&out.action));
        let sum: f32 = out.visit_distribution.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4, "distribution sums to {sum}");
        for (m, _) in &out.visit_distribution {
            assert!(legal.contains(m));
        }
    }

    #[test]
    fn gumbel_move_is_seed_deterministic() {
        let mut state = HashedPosition::new();
        for m in [4u8, 0] {
            state = TicTacToe::apply(state, &Move(m));
        }
        let cfg = GumbelConfig {
            sims: 16,
            ..GumbelConfig::default()
        };
        let a = gumbel_player(1, cfg).choose(&state).action;
        let b = gumbel_player(1, cfg).choose(&state).action;
        let c = gumbel_player(2, cfg).choose(&state).action;
        assert_eq!(a, b);
        let _ = c; // a different seed may or may not differ; determinism per seed is the contract.
    }

    #[test]
    fn a_gumbel_selfplay_game_records_a_policy_tail_per_position() {
        let cfg = Config {
            out: PathBuf::new(),
            games: 2,
            seed: 3,
            label: "gumbel".to_string(),
            engine: None,
            epsilon: 0.0,
            presets_path: PathBuf::new(),
            weights: None,
            sims: 16,
            max_considered: 4,
            temp_moves: 3,
        };
        let mut records = Vec::new();
        dump_gumbel_games(&cfg, &mut records);
        assert!(!records.is_empty());
        for r in &records {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(!r.policy.is_empty(), "gumbel positions carry a policy target");
            let sum: f32 = r.policy.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    fn sample_positions() -> Vec<Position> {
        let mut p1 = Position::new();
        p1.apply(Move(4));
        let mut p2 = p1;
        p2.apply(Move(0));
        p2.apply(Move(8));
        vec![Position::new(), p1, p2]
    }

    #[test]
    fn records_round_trip_through_encode_decode() {
        let policies: Vec<Vec<(u8, f32)>> = vec![
            Vec::new(),
            vec![(4, 1.0)],
            vec![(0, 0.25), (1, 0.5), (8, 0.25)],
        ];
        for pos in sample_positions() {
            for winner in [None, Some(Piece::X), Some(Piece::O)] {
                for policy in &policies {
                    let rec = record_for(&pos, winner, policy.clone());
                    let mut buf = Vec::new();
                    rec.encode(&mut buf);
                    assert_eq!(buf.len(), RECORD_HEAD_BYTES + policy.len() * POLICY_ENTRY_BYTES);
                    let (back, consumed) = Record::decode(&buf).unwrap();
                    assert_eq!(consumed, buf.len());
                    assert_eq!(back, rec);
                }
            }
        }
    }

    #[test]
    fn decode_walks_a_concatenated_stream() {
        let recs: Vec<Record> = sample_positions()
            .into_iter()
            .map(|p| record_for(&p, Some(Piece::X), vec![(3, 1.0)]))
            .collect();
        let mut buf = Vec::new();
        for r in &recs {
            r.encode(&mut buf);
        }
        let mut cursor = &buf[..];
        let mut decoded = Vec::new();
        while let Some((rec, n)) = Record::decode(cursor) {
            decoded.push(rec);
            cursor = &cursor[n..];
        }
        assert!(cursor.is_empty());
        assert_eq!(decoded, recs);
    }

    #[test]
    fn value_is_from_side_to_move_perspective() {
        let p = Position::new(); // X to move
        assert_eq!(record_for(&p, Some(Piece::X), Vec::new()).value, 1.0);
        assert_eq!(record_for(&p, Some(Piece::O), Vec::new()).value, -1.0);
        assert_eq!(record_for(&p, None, Vec::new()).value, 0.0);
        assert_eq!(record_for(&p, None, Vec::new()).side, 0);
        assert_eq!(record_for(&p, None, Vec::new()).ply, 0);
    }

    #[test]
    fn a_dumped_game_labels_every_position_consistently() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut recs = Vec::new();
        dump_one_game(&mut rng, &mut recs, None, 0.0);
        assert!(!recs.is_empty());
        let mut last_ply = 0u8;
        for r in &recs {
            assert!([1.0f32, -1.0, 0.0].contains(&r.value));
            assert!(r.ply >= last_ply);
            last_ply = r.ply;
        }
    }
}
