// schema-fixture.ts — a hand-built `AxisSchema` fixture, small but the real
// shape (a wrapper variant that `wraps` a `_base` axis, a nested non-axis
// enum field, ordinary scalar fields) -- enough to exercise
// `StrategyConfigEditor`'s recursion without pulling in the full ~20-variant
// schema `axis_schema()` produces.

import type { AxisSchema, TunerInfo } from "@mcts/game";

export const fixtureSchema: AxisSchema = {
  select: {
    variants: [
      {
        kind: "ucb1",
        fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }],
      },
      {
        kind: "epsilon_greedy",
        fields: [{ name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 }],
        wraps: "select_base",
      },
    ],
  },
  select_base: {
    variants: [
      {
        kind: "ucb1",
        fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }],
      },
    ],
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
      {
        kind: "secure_child",
        fields: [{ name: "a", type: "float", bounds: [0, 10], default: 4.0 }],
      },
    ],
  },
};

/** A hand-built `TunerInfo`, small but the real shape `strategy_tuner_info`
 * emits (`mcts-tune/src/tuner_info.rs`): `algorithm` is the always-active
 * root categorical; picking `mcts` activates the axis categoricals
 * (`select`) and the orthogonal `contempt` switch. It exercises every shape
 * by-algorithm mode's activation walk needs: `c` is gated by two separate
 * conditions (`select: ucb1` and `rave_ucb: [ucb1, tuned]`), so it's active
 * if either is satisfied; `rave_ucb` is itself gated on `select`, so
 * activating `c` via `rave_ucb` requires a further pass (`algorithm ->
 * select -> rave_ucb -> c`); `contempt`/`contempt_factor` is a second,
 * independent parent/child pair rooted at the same `algorithm: mcts`. */
export const fixtureTunerInfo: TunerInfo = {
  id: "fixture",
  baselines: ["random"],
  eval_rounds: 10,
  parameters: [
    {
      name: "algorithm",
      type: "categorical",
      choices: ["mcts", "random", "negamax"],
      default: "mcts",
    },
    { name: "select", type: "categorical", choices: ["ucb1", "rave"], default: "ucb1" },
    { name: "rave_ucb", type: "categorical", choices: ["ucb1", "tuned"], default: "ucb1" },
    { name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 },
    { name: "contempt", type: "categorical", choices: ["on", "off"], default: "off" },
    { name: "contempt_factor", type: "float", bounds: [-1, 1], default: 0 },
  ],
  conditions: [
    { if: { algorithm: "mcts" }, then: ["select", "contempt"] },
    { if: { select: "rave" }, then: ["rave_ucb"] },
    { if: { select: "ucb1" }, then: ["c"] },
    { if: { rave_ucb: ["ucb1", "tuned"] }, then: ["c"] },
    { if: { contempt: "on" }, then: ["contempt_factor"] },
  ],
  game_config: {},
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
