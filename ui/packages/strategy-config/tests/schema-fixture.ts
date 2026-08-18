// schema-fixture.ts — a hand-built `AxisSchema` fixture, small but the real
// shape (a wrapper variant that `wraps` a `_base` axis, a nested non-axis
// enum field, ordinary scalar fields) -- enough to exercise
// `StrategyConfigEditor`'s recursion without pulling in the full ~20-variant
// schema `axis_schema()` produces.

import type { AxisSchema } from "@mcts/game";

export const fixtureSchema: AxisSchema = {
  select: {
    variants: [
      { kind: "ucb1", fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }] },
      {
        kind: "epsilon_greedy",
        fields: [{ name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 }],
        wraps: "select_base",
      },
    ],
  },
  select_base: {
    variants: [{ kind: "ucb1", fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }] }],
  },
  simulate: {
    variants: [
      { kind: "uniform", fields: [] },
      {
        kind: "decisive_move_nst",
        fields: [
          {
            name: "mode",
            type: "enum",
            default: "win",
            bare: true,
            variants: [
              { kind: "win", fields: [] },
              { kind: "win_loss", fields: [] },
              { kind: "win_loss_draw", fields: [] },
            ],
          },
          { name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 },
          { name: "nst_backoff_threshold", type: "int", bounds: [0, 100], default: 5 },
        ],
      },
    ],
  },
  simulate_base: {
    variants: [{ kind: "uniform", fields: [] }],
  },
  backprop: {
    variants: [{ kind: "classic", fields: [] }],
  },
  final_action: {
    variants: [
      { kind: "robust_child", fields: [] },
      { kind: "secure_child", fields: [{ name: "a", type: "float", bounds: [0, 10], default: 4.0 }] },
    ],
  },
};

export function fixtureDefaultConfig() {
  return {
    search: {
      select: { kind: "ucb1", c: 1.4142135623730951 },
      simulate: { kind: "uniform" },
      backprop: { kind: "classic" },
      final_action: { kind: "robust_child" },
    },
    max_iterations: 10_000,
  };
}
