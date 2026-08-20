use super::config::McgsCorrection;

/// One residual check's result, per `McgsCorrection::Residual`: how far the
/// edge and node estimates disagreed, and the corrected value a trial should
/// use in place of the raw node estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correction {
    pub residual: f64,
    pub corrected: f64,
}

/// Pure algebra for the paper's residual information-leak check (arXiv
/// 2012.11045v1, Section III.C): compare an edge's local expected score for
/// the parent's mover against the shared target node's expected score for
/// that same player -- both already player-relative, exactly what
/// `ChildSnapshot::expected_score`/`NodeStats::expected_score` return.
///
/// Returns `None` when correction is off (`McgsCorrection::Disabled`),
/// either side has zero visits (nothing yet to compare), or the two
/// estimates already agree within `epsilon`. Otherwise returns `Some` with
/// the signed residual (`node_expected - edge_expected`) and a corrected
/// value clamped to the engine's utility range, `[-1.0, 1.0]` (see
/// `TerminalStatus::utilities`) -- the node's own estimate can't leave that
/// range on its own, but clamping keeps this helper's output safe to use
/// even if a caller feeds it a value from a different scale.
pub fn residual_correction(
    config: McgsCorrection,
    edge_expected: f64,
    edge_visits: u32,
    node_expected: f64,
    node_visits: u32,
) -> Option<Correction> {
    let McgsCorrection::Residual { epsilon } = config else {
        return None;
    };
    if edge_visits == 0 || node_visits == 0 {
        return None;
    }
    let residual = node_expected - edge_expected;
    if residual.abs() <= epsilon {
        return None;
    }
    Some(Correction {
        residual,
        corrected: node_expected.clamp(-1.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESIDUAL: McgsCorrection = McgsCorrection::Residual { epsilon: 0.1 };

    #[test]
    fn disabled_never_corrects() {
        assert_eq!(
            residual_correction(McgsCorrection::Disabled, -1.0, 10, 1.0, 10),
            None
        );
    }

    #[test]
    fn zero_edge_visits_skips() {
        assert_eq!(residual_correction(RESIDUAL, 0.0, 0, 1.0, 10), None);
    }

    #[test]
    fn zero_node_visits_skips() {
        assert_eq!(residual_correction(RESIDUAL, 0.0, 10, 1.0, 0), None);
    }

    #[test]
    fn residual_exactly_at_tolerance_boundary_skips() {
        // |0.5 - 0.4| == epsilon (0.1) exactly -- the boundary is inclusive
        // of "no correction", i.e. correction only fires on a strict excess.
        assert_eq!(residual_correction(RESIDUAL, 0.4, 5, 0.5, 5), None);
    }

    #[test]
    fn residual_just_past_tolerance_corrects() {
        let got = residual_correction(RESIDUAL, 0.4, 5, 0.5001, 5).unwrap();
        assert!((got.residual - 0.1001).abs() < 1e-9);
        assert!((got.corrected - 0.5001).abs() < 1e-9);
    }

    #[test]
    fn opposite_perspective_produces_large_signed_residual() {
        // The edge thinks this line is winning for the mover (1.0) but the
        // shared node -- informed by other parents -- thinks it's losing
        // (-1.0). The correction should point at the node's value, not
        // split the difference.
        let got = residual_correction(RESIDUAL, 1.0, 20, -1.0, 20).unwrap();
        assert_eq!(got.residual, -2.0);
        assert_eq!(got.corrected, -1.0);
    }

    #[test]
    fn corrected_value_is_bounded_to_utility_range() {
        let got = residual_correction(RESIDUAL, 0.0, 5, 5.0, 5).unwrap();
        assert_eq!(got.corrected, 1.0);
        let got = residual_correction(RESIDUAL, 0.0, 5, -5.0, 5).unwrap();
        assert_eq!(got.corrected, -1.0);
    }
}
