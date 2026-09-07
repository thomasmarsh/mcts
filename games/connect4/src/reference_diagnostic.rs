//! Stable, replay-incompatible Connect Four reference-diagnostic records.
//!
//! These records are intentionally not self-play records.  They carry a
//! source-game outcome next to an independently searched reference label so
//! diagnostic consumers cannot accidentally feed them to the trainer.

use crate::{BitBoard, Move, Player, Standard, State};
use mcts::game::Game;

pub const MAGIC: &[u8; 8] = b"C4REFD01";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 23;
pub const RECORD_BYTES: usize = 30;

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
    let mut out = Vec::with_capacity(HEADER_BYTES + records.len() * RECORD_BYTES);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
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
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err("reference diagnostic magic mismatch".into());
    }
    if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != VERSION {
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
