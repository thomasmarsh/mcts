use super::config::McgsCorrection;
use super::select::RaveSchedule;

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

/// Pure algebra for the RAVE-blended alternative to `residual_correction`
/// (`McgsCorrection::RaveBlend`): blend a DAG-merged target node's pooled
/// `expected_score` into an edge's own selection score using a
/// `RaveSchedule`-style decay on the edge's own visit count, exactly the
/// shape `select::Rave::score_child` already uses to blend AMAF/GRAVE's
/// pooled estimate into a direct one -- see `RaveSchedule::beta`. Unlike
/// `residual_correction`, this is unconditional: no `epsilon` threshold, no
/// `Option`, always blends. That's the point -- there's no "fire" event to
/// gate descent or backprop on, so the edge this feeds into is never
/// skipped and always keeps accumulating its own direct samples, whatever
/// the blend says this iteration.
pub fn rave_blend_correction(
    schedule: RaveSchedule,
    edge_expected: f64,
    edge_visits: u32,
    node_expected: f64,
    node_visits: u32,
) -> f64 {
    let beta = schedule.beta(edge_visits, node_visits);
    beta * node_expected + (1.0 - beta) * edge_expected
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

    #[test]
    fn rave_blend_at_zero_edge_visits_trusts_the_pooled_node_value() {
        // `HandSelected`'s beta is 1.0 at n == 0 regardless of k, same as
        // RAVE's own "no direct evidence yet, trust the pooled estimate
        // fully" starting point.
        let schedule = RaveSchedule::HandSelected { k: 1000 };
        let blended = rave_blend_correction(schedule, 1.0, 0, -1.0, 50);
        assert_eq!(blended, -1.0);
    }

    #[test]
    fn rave_blend_decays_toward_the_edges_own_estimate_as_its_visits_grow() {
        let schedule = RaveSchedule::HandSelected { k: 1000 };
        let near_zero_visits = rave_blend_correction(schedule, 1.0, 1, -1.0, 10_000);
        let many_visits = rave_blend_correction(schedule, 1.0, 100_000, -1.0, 10_000);
        // Both start off pulled toward the pooled value (-1.0) but the
        // edge with far more of its own visits should sit closer to its own
        // estimate (1.0) than the nearly-unvisited one does.
        assert!(many_visits > near_zero_visits);
        assert!(many_visits > 0.0, "many_visits = {many_visits}");
    }

    #[test]
    fn rave_blend_never_gates_unlike_residual_correction() {
        // Unlike `residual_correction`, which returns `None` (no correction
        // signal) once the two estimates agree, `rave_blend_correction`
        // always blends -- there's no threshold to fall under.
        let schedule = RaveSchedule::Threshold { rave: 700 };
        let blended = rave_blend_correction(schedule, 0.5, 5, 0.5, 5);
        assert_eq!(blended, 0.5);
    }
}
