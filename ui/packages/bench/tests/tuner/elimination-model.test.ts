import { describe, expect, it } from "vitest";
import { deriveElimination } from "../../src/tuner/models/elimination-model.js";

describe("deriveElimination", () => {
  it("is absent when neither elimination section is present", () => {
    expect(deriveElimination({}).present).toBe(false);
    expect(deriveElimination({}).mode).toBe("none");
  });

  it("reads shadow calibration and summary KPIs", () => {
    const view = deriveElimination({
      shadow_elimination: {
        summary: {
          counterfactual_eliminations: 2,
          top_set_false_eliminations: 1,
          top_set_false_elimination_rate: 0.25,
          trash_precision: null,
          true_trash_eliminations: 0,
          brier_score: 0.13,
        },
        scope: { completed_cohorts: 3, recorded_looks: 8, active_path_looks: 6, held_out_validation_used: false },
        policy: { policy_kind: "paired_bootstrap", policy_version: "v1" },
        strata: [
          { reversals: 1, elimination_reversals: 0 },
          { reversals: 2, elimination_reversals: 1 },
        ],
        calibration_bins: [
          { lower: 0.8, upper: 1.0, mean_prediction: 0.95, observed_success_rate: 0.7, count: 6 },
        ],
      },
    });
    expect(view.mode).toBe("shadow");
    expect(view.enforced).toBe(false);
    expect(view.policyKind).toBe("paired_bootstrap");
    expect(view.calibrationBins).toHaveLength(1);
    expect(view.calibrationBins[0]!.observedRate).toBe(0.7);
    const reversals = view.kpis.find((k) => k.label === "Stratum reversals");
    expect(reversals?.value).toBe("3");
  });

  it("reads active elimination and surfaces suspension", () => {
    const view = deriveElimination({
      active_elimination: {
        suspended: true,
        suspension: { reason: "audited boundary reversal" },
        active_interval: { first_cohort_index: 0, last_cohort_index: null },
        policy: { policy_kind: "successive_halving", audit_probability: 0.25, policy_version: "sh-v1" },
        summary: {
          pruned: 1,
          nominal_eliminations: 1,
          elimination_decisions: 1,
          audited_continuations: 0,
          audit_continued: 0,
          audited_boundary_reversals: 1,
          estimated_boundary_reversals: 0,
          estimated_reversal_rate: 0.0,
          observed_audit_reversal_rate: null,
          planned_unique_pair_savings: 2,
        },
      },
    });
    expect(view.mode).toBe("active");
    expect(view.enforced).toBe(true);
    expect(view.suspended).toBe(true);
    expect(view.suspensionReason).toBe("audited boundary reversal");
    expect(view.calibrationBins).toHaveLength(0);
    expect(view.kpis.find((k) => k.label === "Pruned")?.value).toBe("1");
  });

  it("prefers active over shadow when both are present", () => {
    expect(
      deriveElimination({ active_elimination: { summary: {} }, shadow_elimination: { summary: {} } }).mode,
    ).toBe("active");
  });
});
