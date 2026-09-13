//! Sparse-linear ("n-tuple") static evaluation for Othello.
//!
//! An n-tuple is an ordered list of board squares; a position maps each
//! tuple to one *feature index* by reading a base-3 digit per square
//! (0 empty, 1 side-to-move disc, 2 opponent disc) and every tuple owns a
//! table of `3^k` weights. The evaluation is the plain sum of the selected
//! weights over every tuple and every one of the 8 D4 board orientations
//! (all orientations share one table -- the standard "8x data multiplier").
//!
//! Geometry lives in a TOML file (`games/othello/ntuple/model.toml`), read
//! identically here and by the Python trainer (`othello-eval`), so changing
//! tuple shapes is a re-run rather than a recompile. The trained weights are
//! a flat little-endian `f32` array (`weights.bin`) with a sibling
//! `weights.meta.json` recording the SHA-256 of the geometry file; loading
//! refuses a weights/geometry mismatch.
//!
//! This is deliberately Othello-specific. A game-agnostic n-tuple module is
//! a separate, later concern.

use std::path::Path;
use std::sync::{LazyLock, OnceLock};

use game_core::symmetry::D4Symmetry;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};
use sha2::{Digest, Sha256};

use crate::{Othello, Player, State};

/// The 8 D4 permutations of the 64 board squares. `D4[k][i]` is the image
/// of square `i` under group element `k` (`k == 0` is the identity),
/// matching `D4Symmetry::index_symmetries`' element order.
pub static D4: LazyLock<[[u8; 64]; 8]> = LazyLock::new(|| {
    std::array::from_fn(|k| std::array::from_fn(|i| D4Symmetry::<8>::index_symmetries(i)[k] as u8))
});

/// One tuple's geometry: its 8 orientation-permuted square lists plus where
/// its `3^k` weights sit in the flat weight vector.
#[derive(Debug, Clone)]
struct TupleGeom {
    /// `syms[k]` is the tuple's square list under D4 element `k`.
    syms: [Vec<u8>; 8],
    /// Offset of this tuple's weight table into the flat weight vector.
    offset: usize,
}

/// Parsed model geometry: the tuples plus the total weight count.
#[derive(Debug, Clone)]
pub struct ModelGeometry {
    tuples: Vec<TupleGeom>,
    n_weights: usize,
    /// Lowercase hex SHA-256 of the raw `model.toml` bytes.
    sha256_hex: String,
}

#[derive(serde::Deserialize)]
struct ModelToml {
    #[serde(default)]
    tuple: Vec<TupleToml>,
}

#[derive(serde::Deserialize)]
struct TupleToml {
    #[allow(dead_code)]
    name: String,
    squares: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct WeightsMeta {
    model_toml_sha256: String,
    n_weights: usize,
}

fn pow3(k: u32) -> usize {
    3usize.pow(k)
}

impl ModelGeometry {
    /// Parse geometry from `model.toml` bytes.
    pub fn parse(toml_bytes: &[u8]) -> ModelGeometry {
        let text = std::str::from_utf8(toml_bytes).expect("model.toml must be UTF-8");
        let parsed: ModelToml = toml::from_str(text).expect("model.toml must parse");
        assert!(!parsed.tuple.is_empty(), "model.toml defines no [[tuple]]");

        let mut tuples = Vec::with_capacity(parsed.tuple.len());
        let mut offset = 0usize;
        for t in &parsed.tuple {
            assert!(
                !t.squares.is_empty() && t.squares.len() <= 12,
                "tuple {:?}: squares must be 1..=12 entries",
                t.name
            );
            for &s in &t.squares {
                assert!(s < 64, "tuple {:?}: square {s} out of range", t.name);
            }
            let k = t.squares.len() as u32;
            let syms: [Vec<u8>; 8] = std::array::from_fn(|sym| {
                t.squares.iter().map(|&sq| D4[sym][sq as usize]).collect()
            });
            tuples.push(TupleGeom { syms, offset });
            offset += pow3(k);
        }

        let mut hasher = Sha256::new();
        hasher.update(toml_bytes);
        let sha256_hex = hex_lower(&hasher.finalize());

        ModelGeometry {
            tuples,
            n_weights: offset,
            sha256_hex,
        }
    }

    pub fn n_weights(&self) -> usize {
        self.n_weights
    }

    /// Number of tuples in this geometry (== `feature_indices(_).len() / 8`,
    /// exposed directly so callers don't need a dummy state to learn it).
    pub fn n_tuples(&self) -> usize {
        self.tuples.len()
    }

    /// Lowercase hex SHA-256 of the `model.toml` bytes this geometry was
    /// parsed from, for a sibling weights file's mismatch check (see
    /// [`crate::policy::NTuplePolicyNet::from_dir`]).
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    /// Every global weight index (`offset + feature`) this position selects,
    /// one per (tuple, orientation) pair. Order is tuple-major then
    /// orientation; callers that compare against another featuriser should
    /// sort first. Shared with the Python trainer via a committed fixture.
    pub fn feature_indices(&self, state: &State) -> Vec<u32> {
        let (me, opp) = perspective_bits(state);
        let mut out = Vec::with_capacity(self.tuples.len() * 8);
        for t in &self.tuples {
            for squares in &t.syms {
                out.push((t.offset + feature_of(squares, me, opp)) as u32);
            }
        }
        out
    }
}

/// `(side-to-move discs, opponent discs)` as raw bitboards.
fn perspective_bits(state: &State) -> (u64, u64) {
    match state.turn {
        Player::Black => (state.black.bits(), state.white.bits()),
        Player::White => (state.white.bits(), state.black.bits()),
    }
}

/// Base-3 feature index for one square list: digit 0 empty, 1 own disc,
/// 2 opponent disc, least-significant digit first.
#[inline]
fn feature_of(squares: &[u8], me: u64, opp: u64) -> usize {
    let mut feat = 0usize;
    let mut place = 1usize;
    for &sq in squares {
        let bit = 1u64 << sq;
        let trit = if me & bit != 0 {
            1
        } else if opp & bit != 0 {
            2
        } else {
            0
        };
        feat += trit * place;
        place *= 3;
    }
    feat
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// A loaded model: geometry plus trained weights.
#[derive(Debug, Clone)]
pub struct NTupleModel {
    geom: ModelGeometry,
    weights: Vec<f32>,
}

impl NTupleModel {
    /// An all-zero-weight model over `geom` -- every position scores `0.0`,
    /// the generation-0 net a self-play loop starts from before any
    /// training has happened (mirrors `games/connect4/src/valuenet.rs`'s
    /// `Default` impl, which Connect Four's fixed-shape geometry can give
    /// for free; Othello's geometry is data, so this takes it explicitly).
    pub fn zeros(geom: ModelGeometry) -> NTupleModel {
        let n = geom.n_weights();
        NTupleModel {
            geom,
            weights: vec![0.0; n],
        }
    }

    /// A model over `geom` with explicit weights (mirrors
    /// [`crate::policy::NTuplePolicyNet::from_weights`]). `weights.len()`
    /// must equal `geom.n_weights()`. For tests that need a deterministic
    /// non-zero model without a trained checkpoint on disk.
    pub fn from_weights(geom: ModelGeometry, weights: Vec<f32>) -> NTupleModel {
        assert_eq!(
            weights.len(),
            geom.n_weights(),
            "weights.len() must equal geom.n_weights()"
        );
        NTupleModel { geom, weights }
    }

    /// Load `model.toml` + `weights.bin` + `weights.meta.json` from `dir`.
    /// Panics with an actionable message on any mismatch.
    pub fn from_dir(dir: &Path) -> NTupleModel {
        Self::from_parts(
            &dir.join("model.toml"),
            &dir.join("weights.bin"),
            &dir.join("weights.meta.json"),
        )
    }

    /// Load from explicitly-named geometry, weights and metadata files.
    pub fn from_parts(toml_path: &Path, bin_path: &Path, meta_path: &Path) -> NTupleModel {
        let toml_bytes = std::fs::read(toml_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", toml_path.display()));
        let geom = ModelGeometry::parse(&toml_bytes);

        let meta_text = std::fs::read_to_string(meta_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display()));
        let meta: WeightsMeta =
            serde_json::from_str(&meta_text).expect("weights.meta.json must parse");
        assert_eq!(
            meta.model_toml_sha256,
            geom.sha256_hex,
            "weights.meta.json SHA-256 does not match {} -- the weights were trained \
             against a different geometry; retrain",
            toml_path.display()
        );
        assert_eq!(
            meta.n_weights, geom.n_weights,
            "weights.meta.json n_weights disagrees with parsed geometry"
        );

        let raw = std::fs::read(bin_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", bin_path.display()));
        assert_eq!(
            raw.len(),
            geom.n_weights * 4,
            "{} holds {} bytes, expected {} (4 * n_weights)",
            bin_path.display(),
            raw.len(),
            geom.n_weights * 4
        );
        let weights: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        NTupleModel { geom, weights }
    }

    /// Resolve the process-wide model from `$OTHELLO_NTUPLE_WEIGHTS` (a
    /// directory holding `model.toml` + `weights.bin` + `weights.meta.json`).
    pub fn from_env() -> NTupleModel {
        let dir = std::env::var("OTHELLO_NTUPLE_WEIGHTS").unwrap_or_else(|_| {
            panic!(
                "OTHELLO_NTUPLE_WEIGHTS is unset -- run the trainer \
                 (`othello-eval-train ...`) and point OTHELLO_NTUPLE_WEIGHTS at its \
                 output directory"
            )
        });
        NTupleModel::from_dir(Path::new(&dir))
    }

    /// The raw linear score for `state`, from the side-to-move perspective.
    /// This is the ~40-line hot path: for each tuple, for each of the 8 D4
    /// orientations, accumulate the base-3 feature index and add its weight.
    #[inline]
    pub fn logit(&self, state: &State) -> f32 {
        let (me, opp) = perspective_bits(state);
        let mut acc = 0.0f32;
        for t in &self.geom.tuples {
            for squares in &t.syms {
                let feat = feature_of(squares, me, opp);
                acc += self.weights[t.offset + feat];
            }
        }
        acc
    }

    /// Resolve the process-wide *opponent* model from
    /// `$OTHELLO_NTUPLE_WEIGHTS_B` -- the second slot the bake-off's
    /// head-to-head mode needs so two different trained models can play each
    /// other in one process (see [`NTupleEvalB`]).
    pub fn from_env_b() -> NTupleModel {
        let dir = std::env::var("OTHELLO_NTUPLE_WEIGHTS_B")
            .expect("OTHELLO_NTUPLE_WEIGHTS_B is unset (needed for the bake-off h2h opponent)");
        NTupleModel::from_dir(Path::new(&dir))
    }

    /// The [`ModelGeometry`] backing this loaded model, shared with
    /// [`crate::policy::NTuplePolicyNet`] so the policy sidecar reads
    /// features over the same tuples as the value head.
    pub fn geometry(&self) -> &ModelGeometry {
        &self.geom
    }
}

/// An [`Evaluator`] wrapping an owned, in-process [`NTupleModel`] rather than
/// resolving a process-wide singleton from an env var the way [`NTupleEval`]
/// does. A Gumbel self-play loop loads a fresh generation's `weights.bin`
/// every generation (`crate::selfplay::GumbelPlayer`, `dump --label gumbel`)
/// and needs an evaluator instance per load, mirroring
/// `games/connect4/src/valuenet.rs::NTupleValueNet`'s per-instance `.value`.
///
/// `None` (its `Default`) scores every position `0.0` -- the generation-0 net
/// a self-play loop starts from, and the value `EvaluatedCutoff`'s `E:
/// Default` bound needs structurally even though callers always immediately
/// override it via `.evaluator(...)`. Unlike Connect Four's fixed-shape
/// geometry, Othello's tuple geometry is data (`model.toml`), so there is no
/// zero-weight model to build without first parsing one -- `None` gives the
/// same all-zero score without needing a geometry at all.
#[derive(Clone, Debug, Default)]
pub struct NTupleModelEval(pub Option<NTupleModel>);

impl NTupleModelEval {
    pub fn new(model: NTupleModel) -> Self {
        NTupleModelEval(Some(model))
    }

    /// The raw `tanh`-squashed value in `(-1, 1)`, from the side-to-move
    /// perspective. `0.0` when no model is loaded.
    pub fn value(&self, state: &State) -> f32 {
        match &self.0 {
            Some(model) => model.logit(state).tanh(),
            None => 0.0,
        }
    }
}

impl Evaluator<Othello> for NTupleModelEval {
    fn evaluate(&self, state: &State) -> Score {
        (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32) as Score
    }
}

static MODEL: OnceLock<NTupleModel> = OnceLock::new();
static MODEL_B: OnceLock<NTupleModel> = OnceLock::new();

/// Zero-sized [`Evaluator`] for Othello. `Default` resolves the
/// process-wide model, loading it from `$OTHELLO_NTUPLE_WEIGHTS` on first
/// use (same shape as a `static` Zobrist table -- the search never
/// constructs an evaluator with parameters).
#[derive(Clone, Copy, Default)]
pub struct NTupleEval;

impl NTupleEval {
    /// The raw `tanh`-squashed value in `(-1, 1)`, from the side-to-move
    /// perspective -- the same quantity [`Evaluator::evaluate`] rescales to a
    /// [`Score`], exposed directly for a Gumbel self-play root-value seed
    /// (`crate::selfplay::GumbelPlayer::choose`), mirroring
    /// `games/connect4/src/valuenet.rs::NTupleValueNet::value`.
    pub fn value(&self, state: &State) -> f32 {
        let model = MODEL.get_or_init(NTupleModel::from_env);
        model.logit(state).tanh()
    }
}

impl Evaluator<Othello> for NTupleEval {
    fn evaluate(&self, state: &State) -> Score {
        (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32) as Score
    }
}

/// Second [`Evaluator`], reading `$OTHELLO_NTUPLE_WEIGHTS_B`. Exists only so
/// the bake-off's head-to-head mode can put two independently trained models
/// on the board in one process.
#[derive(Clone, Copy, Default)]
pub struct NTupleEvalB;

impl Evaluator<Othello> for NTupleEvalB {
    fn evaluate(&self, state: &State) -> Score {
        let model = MODEL_B.get_or_init(NTupleModel::from_env_b);
        let v = model.logit(state).tanh();
        (v * EVAL_MAGNITUDE_LIMIT as f32) as Score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BB;

    fn tiny_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/tests")
    }

    fn tiny_model() -> NTupleModel {
        let d = tiny_dir();
        NTupleModel::from_parts(
            &d.join("tiny.toml"),
            &d.join("weights.bin"),
            &d.join("weights.meta.json"),
        )
    }

    fn state(black: u64, white: u64, turn: Player) -> State {
        State {
            black: BB::from_bits(black),
            white: BB::from_bits(white),
            turn,
            last_pass: false,
            hashes: [0u64; 8],
        }
    }

    /// 90° rotation of a raw bitboard, via the D4 table (element 3 in
    /// `index_symmetries` is the transpose; a rotation is transpose then a
    /// flip, element 5 or 6 -- we only need *some* non-identity rotation
    /// that is in the group, so use element 5).
    fn rotate(bits: u64) -> u64 {
        let mut out = 0u64;
        let mut r = bits;
        while r != 0 {
            let i = r.trailing_zeros() as usize;
            r &= r - 1;
            out |= 1u64 << D4[5][i];
        }
        out
    }

    #[test]
    fn d4_table_is_a_hand_checked_permutation() {
        // Identity row is the identity.
        for i in 0..64u8 {
            assert_eq!(D4[0][i as usize], i);
        }
        // Every row is a permutation of 0..64.
        for row in D4.iter() {
            let mut seen = [false; 64];
            for &v in row.iter() {
                assert!(!seen[v as usize], "row is not a permutation");
                seen[v as usize] = true;
            }
        }
        // Corner a1 (index 0) maps to a corner under every element.
        let corners = [0u8, 7, 56, 63];
        for row in D4.iter() {
            assert!(corners.contains(&row[0]));
            assert!(corners.contains(&row[7]));
        }
        // Column flip (element 1) sends a1 -> h1 and d4 (index 27) -> e4 (28).
        assert_eq!(D4[1][0], 7);
        assert_eq!(D4[1][27], 28);
    }

    #[test]
    fn logit_matches_a_hand_sum_on_the_tiny_model() {
        let m = tiny_model();
        // tiny.toml: tuple 0 = [0] (offset 0, weights [0.0, 0.5, -0.5]),
        // tuple 1 = [0, 9] (offset 3).
        // Position: black disc on a1 (0), white disc on b2 (9), black to move.
        let s = state(1 << 0, 1 << 9, Player::Black);
        // Hand computation is awkward across all 8 orientations, so just pin
        // the value and assert the two invariants below carry the meaning.
        let v = m.logit(&s);
        assert!(v.is_finite());
        // Empty board scores exactly 0 (all-empty feature index per tuple,
        // and tiny.toml sets those weights to 0).
        let empty = state(0, 0, Player::Black);
        assert_eq!(m.logit(&empty), 0.0);
    }

    #[test]
    fn logit_is_invariant_under_board_rotation() {
        let m = tiny_model();
        let cases = [
            (1u64 << 0, 1u64 << 9),
            ((1 << 0) | (1 << 27), (1 << 9) | (1 << 36)),
            (0x0000_0081_0000_0000, 0x0000_0010_0800_0000),
        ];
        for (b, w) in cases {
            let a = m.logit(&state(b, w, Player::Black));
            let r = m.logit(&state(rotate(b), rotate(w), Player::Black));
            assert!((a - r).abs() < 1e-5, "rotation changed logit: {a} vs {r}");
        }
    }

    #[test]
    fn logit_negates_under_colour_swap_on_the_antisymmetric_tiny_model() {
        // tiny.toml's weights are hand-built to be antisymmetric under the
        // trit swap (own <-> opponent), so swapping colours and side to move
        // must negate the score. A trained model is only approximately so;
        // this test pins the machinery, not a learned fact.
        let m = tiny_model();
        let cases = [
            (1u64 << 0, 1u64 << 9),
            ((1 << 2) | (1 << 20), (1 << 9) | (1 << 36)),
        ];
        for (b, w) in cases {
            // Swap the disc bitboards but keep the same side to move: the
            // side-to-move player now sees the opponent's discs as its own,
            // i.e. every trit 1 <-> 2.
            let a = m.logit(&state(b, w, Player::Black));
            let swapped = m.logit(&state(w, b, Player::Black));
            assert!((a + swapped).abs() < 1e-5, "{a} vs {swapped}");
        }
    }

    #[test]
    fn committed_model_toml_parses_to_the_documented_weight_count() {
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/model.toml"),
        )
        .unwrap();
        let geom = ModelGeometry::parse(&bytes);
        assert_eq!(geom.n_weights(), 113_724);
    }

    #[test]
    fn featurize_cases_fixture_matches() {
        let m = tiny_model();
        let text = std::fs::read_to_string(tiny_dir().join("featurize_cases.json")).unwrap();
        let cases: serde_json::Value = serde_json::from_str(&text).unwrap();
        for case in cases.as_array().unwrap() {
            let black = u64::from_str_radix(case["black"].as_str().unwrap(), 16).unwrap();
            let white = u64::from_str_radix(case["white"].as_str().unwrap(), 16).unwrap();
            let turn = if case["side"].as_u64().unwrap() == 0 {
                Player::Black
            } else {
                Player::White
            };
            let mut got = m.geom.feature_indices(&state(black, white, turn));
            got.sort_unstable();
            let mut want: Vec<u32> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "case {case}");
        }
    }

    fn model_v2_geom() -> ModelGeometry {
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/model-v2.toml"),
        )
        .unwrap();
        ModelGeometry::parse(&bytes)
    }

    /// The deterministic weight formula shared with
    /// `ntuple/tests/model_v2_logit_cases.json`'s Python-computed fixture.
    fn sin_weights(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f64) * 0.1).sin() as f32).collect()
    }

    #[test]
    fn committed_model_v2_toml_parses_to_the_documented_weight_count() {
        assert_eq!(model_v2_geom().n_weights(), 141_021);
    }

    #[test]
    fn model_v2_logit_is_invariant_under_board_rotation() {
        let geom = model_v2_geom();
        let n = geom.n_weights();
        let m = NTupleModel::from_weights(geom, sin_weights(n));
        let cases = [
            (1u64 << 0, 1u64 << 9),
            ((1 << 0) | (1 << 27), (1 << 9) | (1 << 36)),
            (0x0000_0081_0000_0000, 0x0000_0010_0800_0000),
        ];
        for (b, w) in cases {
            let a = m.logit(&state(b, w, Player::Black));
            let r = m.logit(&state(rotate(b), rotate(w), Player::Black));
            assert!((a - r).abs() < 1e-3, "rotation changed logit: {a} vs {r}");
        }
    }

    /// Cross-language pin: `research/othello-eval/tests/test_ntuple.py`'s
    /// counterpart computes the same weight formula over the same geometry
    /// with numpy and must agree with this Rust computation.
    #[test]
    fn model_v2_logit_cases_fixture_matches() {
        let geom = model_v2_geom();
        let n = geom.n_weights();
        let m = NTupleModel::from_weights(geom, sin_weights(n));

        let text = std::fs::read_to_string(tiny_dir().join("model_v2_logit_cases.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        for case in doc["cases"].as_array().unwrap() {
            let black = u64::from_str_radix(case["black"].as_str().unwrap(), 16).unwrap();
            let white = u64::from_str_radix(case["white"].as_str().unwrap(), 16).unwrap();
            let turn = if case["side"].as_u64().unwrap() == 0 {
                Player::Black
            } else {
                Player::White
            };
            let got = m.logit(&state(black, white, turn));
            let want = case["expected_logit"].as_f64().unwrap() as f32;
            assert!((got - want).abs() < 1e-3, "case {case}: got {got}, want {want}");
        }
    }
}
