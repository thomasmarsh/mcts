//! Stable, replay-incompatible Connect Four reference-diagnostic records.
//!
//! These records are intentionally not self-play records.  They carry a
//! source-game outcome next to an independently searched reference label so
//! diagnostic consumers cannot accidentally feed them to the trainer.

use crate::{BitBoard, Move, Player, Standard, State};
use mcts::algorithms::negamax::{MaterialBlind, Negamax, NegamaxOptions};
use mcts::game::Game;

pub const MAGIC: &[u8; 8] = b"C4REFD01";
pub const VERSION: u32 = 1;
/// Expanded-corpus artifact: identical record layout, distinct magic and
/// version so a fresh freeze is never confused with the original.
pub const MAGIC_V2: &[u8; 8] = b"C4REFD02";
pub const VERSION_V2: u32 = 2;
/// Shared prefix of every reference-diagnostic magic. Consumers use this to
/// tell a diagnostic artifact apart from a self-play replay shard.
pub const MAGIC_PREFIX: &[u8; 6] = b"C4REFD";
pub const HEADER_BYTES: usize = 23;
pub const RECORD_BYTES: usize = 30;

/// Increasing bounded-depth negamax schedule for positions too deep to search
/// to the end. Each entry is attempted in turn until one proves a win or loss;
/// a neutral score at the final entry stays `Unresolved`.
pub const BOUNDED_DEPTH_SCHEDULE: &[u32] = &[6, 8, 10, 12, 14];

/// Opening-band positions (see [`ply_band`]) cap their bounded search here:
/// forced tactical results that deep are vanishingly rare from near-balanced
/// play, so the extra depth only costs wall time.
pub const OPENING_BAND_DEPTH_CAP: u32 = 12;

/// Remaining-ply cap at or below which a position is searched to the end for an
/// exact label rather than run through [`BOUNDED_DEPTH_SCHEDULE`].
pub const EXACT_SEARCH_REMAINING_CAP: u32 = 12;

/// Map a disc count to an opening (`0`), middle (`1`), or late (`2`) band.
/// Edges match the Python diagnostic reader.
pub fn ply_band(ply: u8) -> u8 {
    match ply {
        0..=14 => 0,
        15..=22 => 1,
        _ => 2,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Split {
    Train = 0,
    Validation = 1,
}
impl TryFrom<u8> for Split {
    type Error = &'static str;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Train),
            1 => Ok(Self::Validation),
            _ => Err("invalid split"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReferenceLabel {
    ExactWin = 0,
    ExactLoss = 1,
    ExactDraw = 2,
    BoundedWin = 3,
    BoundedLoss = 4,
    Unresolved = 5,
}
impl ReferenceLabel {
    pub fn sign(self) -> Option<f32> {
        match self {
            Self::ExactWin | Self::BoundedWin => Some(1.0),
            Self::ExactLoss | Self::BoundedLoss => Some(-1.0),
            Self::ExactDraw => Some(0.0),
            Self::Unresolved => None,
        }
    }
    pub fn exact(self) -> bool {
        matches!(self, Self::ExactWin | Self::ExactLoss | Self::ExactDraw)
    }
}
impl TryFrom<u8> for ReferenceLabel {
    type Error = &'static str;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::ExactWin),
            1 => Ok(Self::ExactLoss),
            2 => Ok(Self::ExactDraw),
            3 => Ok(Self::BoundedWin),
            4 => Ok(Self::BoundedLoss),
            5 => Ok(Self::Unresolved),
            _ => Err("invalid reference label"),
        }
    }
}

/// Classify a negamax score without treating a depth-cutoff neutral score as a draw.
pub fn classify_score(score: i32, fully_searched: bool) -> ReferenceLabel {
    match score.cmp(&0) {
        std::cmp::Ordering::Greater if fully_searched => ReferenceLabel::ExactWin,
        std::cmp::Ordering::Less if fully_searched => ReferenceLabel::ExactLoss,
        std::cmp::Ordering::Equal if fully_searched => ReferenceLabel::ExactDraw,
        std::cmp::Ordering::Greater => ReferenceLabel::BoundedWin,
        std::cmp::Ordering::Less => ReferenceLabel::BoundedLoss,
        std::cmp::Ordering::Equal => ReferenceLabel::Unresolved,
    }
}

/// Transposition-table size for the reference solver, as a power of two.
const REFERENCE_TABLE_BITS: u32 = 20;

/// Connect Four under the strict negamax turn convention.
///
/// [`Standard::apply`] leaves `turn` on the player who completed four in a
/// row, so a won terminal state reports its `player_to_move` *as the winner*.
/// Negamax's terminal scoring reads a win for the player to move as
/// `WIN_SCORE`, which inverts the sign of every proven line and makes the
/// solver walk into losses and avoid its own wins. This newtype delegates
/// every rule to [`Standard`] but reports the player to move at a won
/// terminal as the loser, restoring the convention that a node you are "to
/// move" in, with the opponent already connected, is a loss for you.
#[derive(Clone)]
pub struct Connect4Negamax;

impl Game for Connect4Negamax {
    type S = State<6, 7>;
    type A = Move;
    type P = Player;

    fn apply(state: Self::S, action: &Self::A) -> Self::S {
        Standard::apply(state, action)
    }
    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        Standard::generate_actions(state, actions);
    }
    fn is_terminal(state: &Self::S) -> bool {
        Standard::is_terminal(state)
    }
    fn winner(state: &Self::S) -> Option<Self::P> {
        Standard::winner(state)
    }
    fn player_to_move(state: &Self::S) -> Self::P {
        let mover = Standard::player_to_move(state);
        if state.has_winner() {
            mover.next()
        } else {
            mover
        }
    }
    fn zobrist_hash(state: &Self::S) -> u64 {
        Standard::zobrist_hash(state)
    }
}

/// Single-threaded, deterministic bounded `MaterialBlind` negamax score for
/// `state`, in mate-distance units (`WIN_SCORE - ply` for a proven win).
pub fn reference_negamax_score(state: &State<6, 7>, depth: u32) -> i32 {
    let mut solver = Negamax::<Connect4Negamax, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(depth)
            .with_table_bits(REFERENCE_TABLE_BITS),
    );
    solver.bounded_negamax(state, depth.max(1)).1
}

/// Ordered `(depth, fully_searched)` negamax attempts for resolving a position
/// at disc count `ply`.
///
/// A position with few remaining plies gets a single exact search to the end.
/// Otherwise the increasing [`BOUNDED_DEPTH_SCHEDULE`] is returned -- capped at
/// [`OPENING_BAND_DEPTH_CAP`] in the opening band -- with a trailing exact
/// search substituted for any scheduled depth that would already reach the end.
/// Both [`searched_reference_label`] and [`optimal_move_set`] walk this list so
/// the depth policy lives in exactly one place.
pub fn resolution_schedule(ply: u8) -> Vec<(u32, bool)> {
    let remaining = (42 - ply as u32).max(1);
    if remaining <= EXACT_SEARCH_REMAINING_CAP {
        return vec![(remaining, true)];
    }
    let depth_cap = if ply_band(ply) == 0 {
        OPENING_BAND_DEPTH_CAP
    } else {
        u32::MAX
    };
    let mut out = Vec::new();
    for &depth in BOUNDED_DEPTH_SCHEDULE {
        if depth > depth_cap {
            break;
        }
        if depth >= remaining {
            out.push((remaining, true));
            return out;
        }
        out.push((depth, false));
    }
    out
}

/// Bounded-depth reference label for `state` at disc count `ply`, plus its
/// `(first proof depth, maximum attempted depth)`.
///
/// Walks [`resolution_schedule`]: an exact search yields its label directly, a
/// bounded search is accepted only when it proves a win or loss, and a neutral
/// score at the final attempted depth stays [`ReferenceLabel::Unresolved`]
/// rather than being read as a draw.
pub fn searched_reference_label(state: &State<6, 7>, ply: u8) -> (ReferenceLabel, u8, u8) {
    let mut max = 0u8;
    for (depth, exact) in resolution_schedule(ply) {
        let label = classify_score(reference_negamax_score(state, depth), exact);
        if exact {
            return (label, depth as u8, depth as u8);
        }
        max = depth as u8;
        if label != ReferenceLabel::Unresolved {
            return (label, depth as u8, max);
        }
    }
    (ReferenceLabel::Unresolved, 0, max)
}

/// Proven outcome class for the mover at `state`: `1` a forced win, `0` a
/// draw, `-1` a forced loss, `None` if [`resolution_schedule`] cannot resolve
/// it. Mate distance is deliberately discarded -- a slower forced win is still
/// a win.
fn proven_outcome_class(state: &State<6, 7>, ply: u8) -> Option<i32> {
    for (depth, exact) in resolution_schedule(ply) {
        match classify_score(reference_negamax_score(state, depth), exact) {
            ReferenceLabel::ExactWin | ReferenceLabel::BoundedWin => return Some(1),
            ReferenceLabel::ExactLoss | ReferenceLabel::BoundedLoss => return Some(-1),
            ReferenceLabel::ExactDraw => return Some(0),
            ReferenceLabel::Unresolved => continue,
        }
    }
    None
}

/// Why [`optimal_move_set`] declined to return a move set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleSkip {
    /// No legal child could be resolved even at the capped depth.
    NoChildResolved,
    /// Some child stayed unresolved and no resolved child forces a win, so an
    /// unresolved child could secretly beat the best resolved outcome.
    AmbiguousUnresolvedChild,
}

/// The set of columns whose child preserves the best achievable proven outcome
/// class for the mover at a non-terminal `state`.
///
/// Every legal child is scored with [`proven_outcome_class`]. Optimal = every
/// move reaching the best class among `win > draw > loss`; a lost position
/// still returns the (whole) least-bad set. Returns `Err` when the position
/// should be skipped: either nothing resolved, or an unresolved child could
/// still beat the best resolved outcome (see [`OracleSkip`]).
pub fn optimal_move_set(state: &State<6, 7>) -> Result<Vec<u8>, OracleSkip> {
    let child_ply = (state.black().count_ones() + state.white().count_ones()) as u8 + 1;
    let mut actions = Vec::new();
    crate::Standard::generate_actions(state, &mut actions);
    debug_assert!(
        !crate::Standard::is_terminal(state) && !actions.is_empty(),
        "oracle needs a non-terminal position with legal moves"
    );
    let mut classes: Vec<(u8, i32)> = Vec::new();
    let mut unresolved = false;
    for action in &actions {
        let child = crate::Standard::apply(*state, action);
        if child.has_winner() {
            classes.push((action.0, 1));
        } else if crate::Standard::is_terminal(&child) {
            classes.push((action.0, 0));
        } else {
            match proven_outcome_class(&child, child_ply) {
                // The child score is from the child mover's perspective; negate
                // it back to the perspective of the mover at `state`.
                Some(child_class) => classes.push((action.0, -child_class)),
                None => unresolved = true,
            }
        }
    }
    if classes.is_empty() {
        return Err(OracleSkip::NoChildResolved);
    }
    let best = classes.iter().map(|&(_, c)| c).max().unwrap();
    if unresolved && best < 1 {
        return Err(OracleSkip::AmbiguousUnresolvedChild);
    }
    Ok(classes
        .iter()
        .filter(|&&(_, c)| c == best)
        .map(|&(col, _)| col)
        .collect())
}

/// Soft searched-value scalar in `[-1, 1]` from the side-to-move perspective.
///
/// `MaterialBlind` contributes no positional heuristic, so an unresolved
/// cutoff carries no usable signal and maps to `0.0`; proven tactical results
/// and exact draws map to their label sign.
pub fn searched_value_scalar(label: ReferenceLabel) -> f32 {
    label.sign().unwrap_or(0.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticRecord {
    pub black: u64,
    pub white: u64,
    pub side: u8,
    pub ply: u8,
    pub group: u32,
    pub split: Split,
    pub source_outcome: f32,
    pub label: ReferenceLabel,
    pub proof_depth: u8,
    pub max_depth: u8,
}

pub fn split_for_group(group: u32, seed: u64) -> Split {
    let mut x = (group as u64) ^ seed ^ 0x9e37_79b9_7f4a_7c15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    if x.is_multiple_of(5) {
        Split::Validation
    } else {
        Split::Train
    }
}

pub fn mirror_bits(bits: u64) -> u64 {
    let mut out = 0;
    for row in 0..6 {
        for col in 0..7 {
            let i = row * 7 + col;
            if bits & (1 << i) != 0 {
                out |= 1 << (row * 7 + 6 - col);
            }
        }
    }
    out
}
pub fn canonical_key(black: u64, white: u64, side: u8) -> (u64, u64, u8) {
    let mirrored = (mirror_bits(black), mirror_bits(white), side);
    (black, white, side).min(mirrored)
}

fn has_four(bits: u64) -> bool {
    const DIRECTIONS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    for row in 0..6 {
        for col in 0..7 {
            for &(dr, dc) in &DIRECTIONS {
                let end_row = row as isize + dr * 3;
                let end_col = col as isize + dc * 3;
                if !(0..6).contains(&end_row) || !(0..7).contains(&end_col) {
                    continue;
                }
                if (0..4).all(|step| {
                    let row = row as isize + dr * step;
                    let col = col as isize + dc * step;
                    bits & (1 << (row * 7 + col)) != 0
                }) {
                    return true;
                }
            }
        }
    }
    false
}

/// Validate by replaying every occupied cell in gravity order.  This catches
/// bit-range, overlap, parity, gravity, terminal, and turn mismatches.
pub fn state_from_record(record: &DiagnosticRecord) -> Result<State<6, 7>, String> {
    if (record.black | record.white) >> 42 != 0 {
        return Err("board has bits outside 6x7".into());
    }
    if record.black & record.white != 0 {
        return Err("board colors overlap".into());
    }
    let count = (record.black | record.white).count_ones() as u8;
    if count != record.ply {
        return Err("ply does not match board count".into());
    }
    if record.side > 1
        || !record.source_outcome.is_finite()
        || ![-1.0, 0.0, 1.0].contains(&record.source_outcome)
    {
        return Err("invalid scalar field".into());
    }
    let black_count = record.black.count_ones();
    let white_count = record.white.count_ones();
    if black_count != white_count && black_count != white_count + 1 {
        return Err("illegal color counts".into());
    }
    let expected_side = if black_count == white_count { 0 } else { 1 };
    if record.side != expected_side {
        return Err("side does not match color counts".into());
    }
    for col in 0..7 {
        let mut gap = false;
        for row in 0..6 {
            let occupied = (record.black | record.white) & (1 << (row * 7 + col)) != 0;
            if !occupied {
                gap = true;
            } else if gap {
                return Err("gravity-invalid board".into());
            }
        }
    }
    if count == 42 || has_four(record.black) || has_four(record.white) {
        return Err("diagnostic records must contain non-terminal positions".into());
    }
    Ok(State::from_parts(
        BitBoard::from_bits(record.black),
        BitBoard::from_bits(record.white),
        if record.side == 0 {
            Player::Black
        } else {
            Player::White
        },
        false,
    ))
}

pub fn encode(records: &[DiagnosticRecord]) -> Vec<u8> {
    encode_with(records, MAGIC, VERSION)
}

/// Encode as the expanded-corpus `C4REFD02` artifact.
pub fn encode_v2(records: &[DiagnosticRecord]) -> Vec<u8> {
    encode_with(records, MAGIC_V2, VERSION_V2)
}

fn encode_with(records: &[DiagnosticRecord], magic: &[u8; 8], version: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + records.len() * RECORD_BYTES);
    out.extend_from_slice(magic);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    out.push(0xff);
    out.extend_from_slice(&[0; 6]);
    for r in records {
        out.extend_from_slice(&r.black.to_le_bytes());
        out.extend_from_slice(&r.white.to_le_bytes());
        out.push(r.side);
        out.push(r.ply);
        out.extend_from_slice(&r.group.to_le_bytes());
        out.push(r.split as u8);
        out.push(r.label as u8);
        out.push(r.proof_depth);
        out.push(r.max_depth);
        out.extend_from_slice(&r.source_outcome.to_le_bytes());
    }
    out
}
pub fn decode(bytes: &[u8]) -> Result<Vec<DiagnosticRecord>, String> {
    if bytes.len() < HEADER_BYTES {
        return Err("reference diagnostic magic mismatch".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let magic_ok = match &bytes[..8] {
        m if m == MAGIC => version == VERSION,
        m if m == MAGIC_V2 => version == VERSION_V2,
        _ => return Err("reference diagnostic magic mismatch".into()),
    };
    if !magic_ok {
        return Err("unsupported reference diagnostic version".into());
    }
    if bytes[16] != 0xff {
        return Err("invalid diagnostic header sentinel".into());
    }
    let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if bytes.len() != HEADER_BYTES + n * RECORD_BYTES {
        return Err("diagnostic length does not match header count".into());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bytes[HEADER_BYTES + i * RECORD_BYTES..HEADER_BYTES + (i + 1) * RECORD_BYTES];
        let r = DiagnosticRecord {
            black: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            white: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            side: b[16],
            ply: b[17],
            group: u32::from_le_bytes(b[18..22].try_into().unwrap()),
            split: Split::try_from(b[22])?,
            label: ReferenceLabel::try_from(b[23])?,
            proof_depth: b[24],
            max_depth: b[25],
            source_outcome: f32::from_le_bytes(b[26..30].try_into().unwrap()),
        };
        state_from_record(&r)?;
        out.push(r);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub count: usize,
    pub mse: f64,
    pub pearson: f64,
    pub sign_agreement: f64,
    pub balanced_sign_accuracy: f64,
    pub mean_abs_prediction: f64,
}
pub fn metrics(prediction: &[f64], target: &[f64]) -> Metrics {
    assert_eq!(prediction.len(), target.len());
    let n = prediction.len();
    if n == 0 {
        return Metrics {
            count: 0,
            mse: 0.0,
            pearson: 0.0,
            sign_agreement: 0.0,
            balanced_sign_accuracy: 0.0,
            mean_abs_prediction: 0.0,
        };
    }
    let mse = prediction
        .iter()
        .zip(target)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64;
    let abs = prediction.iter().map(|p| p.abs()).sum::<f64>() / n as f64;
    let (mp, mt) = (
        prediction.iter().sum::<f64>() / n as f64,
        target.iter().sum::<f64>() / n as f64,
    );
    let (mut pp, mut tt, mut pt) = (0., 0., 0.);
    for (&p, &t) in prediction.iter().zip(target) {
        pp += (p - mp).powi(2);
        tt += (t - mt).powi(2);
        pt += (p - mp) * (t - mt);
    }
    let pearson = if n < 2 || pp == 0. || tt == 0. {
        0.
    } else {
        pt / (pp * tt).sqrt()
    };
    let mut classes = [Vec::new(), Vec::new()];
    for (&p, &t) in prediction.iter().zip(target) {
        if t > 0. {
            classes[1].push(p.signum() == 1.);
        } else if t < 0. {
            classes[0].push(p.signum() == -1.);
        }
    }
    let signs: Vec<_> = classes.iter().flatten().copied().collect();
    let sign_agreement = if signs.is_empty() {
        0.
    } else {
        signs.iter().filter(|&&x| x).count() as f64 / signs.len() as f64
    };
    let present: Vec<f64> = classes
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().filter(|&&x| x).count() as f64 / v.len() as f64)
        .collect();
    let balanced_sign_accuracy = if present.is_empty() {
        0.
    } else {
        present.iter().sum::<f64>() / present.len() as f64
    };
    Metrics {
        count: n,
        mse,
        pearson,
        sign_agreement,
        balanced_sign_accuracy,
        mean_abs_prediction: abs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Move, Standard};
    use mcts::game::Game;
    fn rec() -> DiagnosticRecord {
        DiagnosticRecord {
            black: 1,
            white: 1 << 1,
            side: 0,
            ply: 2,
            group: 7,
            split: Split::Train,
            source_outcome: 1.,
            label: ReferenceLabel::BoundedWin,
            proof_depth: 3,
            max_depth: 6,
        }
    }
    #[test]
    fn codec_and_magic() {
        let b = encode(&[rec()]);
        assert_eq!(decode(&b).unwrap(), vec![rec()]);
        let mut b = b;
        b[0] = 0;
        assert!(decode(&b).is_err());
        let mut b = encode(&[rec()]);
        b[8] = 2;
        assert!(decode(&b).is_err());
    }
    #[test]
    fn codec_v2_round_trip_and_header_rules() {
        let b = encode_v2(&[rec()]);
        assert_eq!(&b[..8], MAGIC_V2);
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), VERSION_V2);
        assert_eq!(decode(&b).unwrap(), vec![rec()]);
        // both magics remain decodable so old artifacts keep working
        assert_eq!(decode(&encode(&[rec()])).unwrap(), vec![rec()]);
        // v2 magic with a v1 version is rejected
        let mut wrong = encode_v2(&[rec()]);
        wrong[8] = VERSION as u8;
        assert!(decode(&wrong).is_err());
        assert!(b.starts_with(MAGIC_PREFIX));
    }
    #[test]
    fn ply_bands_and_depth_schedule() {
        assert_eq!(ply_band(0), 0);
        assert_eq!(ply_band(14), 0);
        assert_eq!(ply_band(15), 1);
        assert_eq!(ply_band(22), 1);
        assert_eq!(ply_band(23), 2);
        assert_eq!(ply_band(41), 2);
        assert!(BOUNDED_DEPTH_SCHEDULE.windows(2).all(|w| w[0] < w[1]));
        assert!(BOUNDED_DEPTH_SCHEDULE[0] > 0);
        assert!(*BOUNDED_DEPTH_SCHEDULE.last().unwrap() >= EXACT_SEARCH_REMAINING_CAP);
    }
    #[test]
    fn split_and_mirror() {
        assert_eq!(split_for_group(4, 9), split_for_group(4, 9));
        assert_eq!(
            canonical_key(1, 2, 0),
            canonical_key(mirror_bits(1), mirror_bits(2), 0)
        );
        let mut train = std::collections::HashSet::new();
        let mut validation = std::collections::HashSet::new();
        for group in 0..100 {
            match split_for_group(group, 9) {
                Split::Train => assert!(train.insert(group)),
                Split::Validation => assert!(validation.insert(group)),
            }
        }
        assert!(train.is_disjoint(&validation));
    }
    #[test]
    fn invalid_boards_rejected() {
        let mut r = rec();
        r.white = 1;
        assert!(state_from_record(&r).is_err());
        r = rec();
        r.black = 1 << 7;
        assert!(state_from_record(&r).is_err());
        r = rec();
        r.black = 0b1111;
        r.white = (1 << 7) | (1 << 8) | (1 << 9);
        r.ply = 7;
        r.side = 1;
        assert!(state_from_record(&r).is_err());
    }
    #[test]
    fn finite_edge_metrics() {
        for (p, t) in [
            (&[][..], &[][..]),
            (&[0.][..], &[1.][..]),
            (&[1., 1.][..], &[1., 1.][..]),
            (&[1., -1.][..], &[1., 1.][..]),
        ] {
            let m = metrics(p, t);
            assert!(
                m.mse.is_finite() && m.pearson.is_finite() && m.balanced_sign_accuracy.is_finite()
            );
        }
    }
    #[test]
    fn labels_keep_sign_and_exactness_separate() {
        assert_eq!(ReferenceLabel::ExactDraw.sign(), Some(0.));
        assert!(ReferenceLabel::ExactDraw.exact());
        assert_eq!(ReferenceLabel::BoundedLoss.sign(), Some(-1.));
        assert!(!ReferenceLabel::BoundedLoss.exact());
        assert_eq!(ReferenceLabel::Unresolved.sign(), None);
    }
    #[test]
    fn score_classification_distinguishes_draws_from_cutoffs() {
        assert_eq!(classify_score(0, true), ReferenceLabel::ExactDraw);
        assert_eq!(classify_score(0, false), ReferenceLabel::Unresolved);
        assert_eq!(classify_score(7, true), ReferenceLabel::ExactWin);
        assert_eq!(classify_score(-7, true), ReferenceLabel::ExactLoss);
        assert_eq!(classify_score(7, false), ReferenceLabel::BoundedWin);
        assert_eq!(classify_score(-7, false), ReferenceLabel::BoundedLoss);
    }
    #[test]
    fn source_outcome_is_from_the_recorded_mover() {
        let black_to_move = rec();
        assert_eq!(black_to_move.source_outcome, 1.);
        let mut white_to_move = black_to_move.clone();
        white_to_move.side = 1;
        white_to_move.black = (1 << 0) | (1 << 1);
        white_to_move.white = 1 << 7;
        white_to_move.ply = 3;
        white_to_move.source_outcome = -1.;
        assert!(state_from_record(&white_to_move).is_ok());
        assert_eq!(-black_to_move.source_outcome, white_to_move.source_outcome);
    }
    #[test]
    fn searched_label_proves_a_shallow_forced_win_and_maps_to_plus_one() {
        // Black holds the bottom row at columns 0, 1, 2 with White scattered;
        // Black to move has an immediate column-3 win.
        let mut state = State::<6, 7>::default();
        for col in [0u8, 4, 1, 5, 2, 6] {
            state = Standard::apply(state, &Move(col));
        }
        assert_eq!(state.turn(), Player::Black);
        let ply = (state.black().count_ones() + state.white().count_ones()) as u8;
        let (label, proof_depth, _max) = searched_reference_label(&state, ply);
        assert_eq!(label.sign(), Some(1.0));
        assert!(proof_depth >= 1);
        assert_eq!(searched_value_scalar(label), 1.0);
    }

    #[test]
    fn searched_label_proves_a_forced_loss_for_the_side_facing_a_double_threat() {
        // Black holds bottom-row columns 2, 3, 4; White to move cannot block
        // both the column-1 and column-5 completions.
        let mut state = State::<6, 7>::default();
        for col in [2u8, 0, 3, 6, 4] {
            state = Standard::apply(state, &Move(col));
        }
        assert_eq!(state.turn(), Player::White);
        let ply = (state.black().count_ones() + state.white().count_ones()) as u8;
        assert!(reference_negamax_score(&state, 6) < 0);
        let (label, _proof, _max) = searched_reference_label(&state, ply);
        assert_eq!(label.sign(), Some(-1.0));
    }

    #[test]
    fn raw_connect4_inverts_a_terminal_win_but_the_negamax_newtype_does_not() {
        use mcts::algorithms::negamax::{Negamax, NegamaxOptions};
        let mut state = State::<6, 7>::default();
        for col in [0u8, 4, 1, 5, 2, 6] {
            state = Standard::apply(state, &Move(col));
        }
        let options = || NegamaxOptions::default().with_max_depth(4).with_table_bits(0);
        // `Standard`'s won terminal reports the winner as the player to move,
        // so negamax scores the immediate win as a loss and never plays it.
        let raw = Negamax::<Standard, MaterialBlind>::new_with_options(MaterialBlind, options())
            .bounded_negamax(&state, 4)
            .1;
        assert!(raw <= 0, "raw Standard inverts the win, got {raw}");
        // The newtype restores the convention.
        assert!(reference_negamax_score(&state, 4) > 0);
    }

    #[test]
    fn searched_value_scalar_maps_every_label_to_its_sign() {
        assert_eq!(searched_value_scalar(ReferenceLabel::ExactWin), 1.0);
        assert_eq!(searched_value_scalar(ReferenceLabel::BoundedWin), 1.0);
        assert_eq!(searched_value_scalar(ReferenceLabel::ExactLoss), -1.0);
        assert_eq!(searched_value_scalar(ReferenceLabel::BoundedLoss), -1.0);
        assert_eq!(searched_value_scalar(ReferenceLabel::ExactDraw), 0.0);
        assert_eq!(searched_value_scalar(ReferenceLabel::Unresolved), 0.0);
    }

    #[test]
    fn optimal_move_set_is_a_singleton_for_a_forced_win() {
        // Black holds bottom-row columns 0,1,2; White holds 4,5,6. Black to
        // move: column 3 wins immediately, and every other move lets White
        // complete columns 3-6, so column 3 is the only optimal move.
        let mut state = State::<6, 7>::default();
        for col in [0u8, 4, 1, 5, 2, 6] {
            state = Standard::apply(state, &Move(col));
        }
        assert_eq!(state.turn(), Player::Black);
        assert_eq!(optimal_move_set(&state), Ok(vec![3]));
    }

    #[test]
    fn optimal_move_set_returns_both_drawing_moves() {
        // A full 40-disc board with no four in a row (colour = `(row + 2*col)
        // mod 4 < 2`), minus the two top cells of columns 0 and 6. Black to
        // move; only columns 0 and 6 are legal, each leaves White a single
        // forced reply, and neither ordering completes a four -- so both moves
        // hold the draw.
        let (mut black, mut white) = (0u64, 0u64);
        for row in 0..6u64 {
            for col in 0..7u64 {
                if (row, col) == (5, 0) || (row, col) == (5, 6) {
                    continue;
                }
                let bit = 1u64 << (row * 7 + col);
                if (row + 2 * col) % 4 < 2 {
                    black |= bit;
                } else {
                    white |= bit;
                }
            }
        }
        let state = State::<6, 7>::from_parts(
            BitBoard::from_bits(black),
            BitBoard::from_bits(white),
            Player::Black,
            false,
        );
        assert!(!Standard::is_terminal(&state));
        let mut set = optimal_move_set(&state).expect("2-cell endgame resolves");
        set.sort_unstable();
        assert_eq!(set, vec![0, 6]);
    }

    #[test]
    fn optimal_move_set_returns_every_move_for_a_lost_position() {
        // Black holds bottom-row columns 2,3,4; White to move cannot block both
        // the column-1 and column-5 completions. No White move wins or draws,
        // so the least-bad set is every legal column.
        let mut state = State::<6, 7>::default();
        for col in [2u8, 0, 3, 6, 4] {
            state = Standard::apply(state, &Move(col));
        }
        assert_eq!(state.turn(), Player::White);
        assert!(reference_negamax_score(&state, 6) < 0);
        assert_eq!(optimal_move_set(&state), Ok(vec![0, 1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn forced_child_win_has_the_opposite_parent_perspective() {
        let mut state = State::default();
        for col in [1, 0, 1, 0, 2, 0, 3] {
            state = Standard::apply(state, &Move(col));
        }
        assert_eq!(state.turn(), Player::White);
        let child = Standard::apply(state, &Move(0));
        assert!(Standard::is_terminal(&child));
        assert_eq!(Standard::winner(&child), Some(Player::White));
        let child_value = classify_score(1, false).sign().unwrap();
        assert!(child_value > 0., "White has the immediate column-0 win");
        assert!(
            -child_value < 0.,
            "the same child is bad for its Black parent"
        );
    }
}
