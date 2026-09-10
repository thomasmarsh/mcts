#!/usr/bin/env bash
# Gumbel AlphaZero graded coordinator for Connect Four with the compact
# C4CNN001 value+policy head and the b=0.75 outcome/searched-value mixture
# value target. Sibling of coordinator_c4.sh, which runs the n-tuple head.
#
#   research/az-train/coordinator_c4_cnn.sh
#
# Per generation: diverse forced-opening self-play -> offline bounded-negamax
# searched-value annotation -> mixture fit (az_train.mixture_selfplay_c4) ->
# gen-vs-zero and gen-vs-gen0 equal-budget Gumbel gates with the fixed
# battle_royale harness -> one merged JSON metrics line appended to
# coordinator-metrics.jsonl. Every generation checkpoints its shard, searched
# values, weights, result.json and metrics line before the next starts, so an
# interrupt resumes with START set to the first unfinished generation.
#
# Each generation also gates gen(N) vs gen(N-1). Two kill-gates stop the loop
# early: gen1-vs-gen0 Wilson lower bound < KILL_GEN1_LB after gen1, and any
# gen(N)-vs-gen(N-1) score share < KILL_PREV_SHARE (catastrophic regression).
# On a kill-gate the loop writes KILL-GATE.txt and stops with the finished
# generations checkpointed.
#
# Env knobs: RUN_DIR, GAMES (self-play games/gen), GENS, EPOCHS, SIMS,
# FORCED (forced opening plies), GATE_GAMES, B (mixture weight), L2,
# GEN0_RESERVOIR (expected gen0-shard share of the resampled training rows;
# 0.0, the default, disables it -- a graded run showed a 0.5 reservoir reverses
# the value-head reference-Pearson decay but degrades the policy head and
# weakens end play), START, KILL_GEN1_LB, KILL_PREV_SHARE.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export LIBRARY_PATH=${LIBRARY_PATH:-/opt/homebrew/lib}

RUN_DIR=${RUN_DIR:-local/output/az/connect4/recovery/graded-smoke-cnn}
REF=${REF:-local/output/az/connect4/recovery/target-mixture/corpus-v2.c4ref}
ZERO=${ZERO:-local/output/az/connect4/recovery/compact-conv/zero.c4cnn}
GAMES=${GAMES:-200}
GENS=${GENS:-3}
EPOCHS=${EPOCHS:-25}
SIMS=${SIMS:-32}
FORCED=${FORCED:-4}
GATE_GAMES=${GATE_GAMES:-120}
B=${B:-0.75}
L2=${L2:-1e-4}
GEN0_RESERVOIR=${GEN0_RESERVOIR:-0.0}
START=${START:-0}
KILL_GEN1_LB=${KILL_GEN1_LB:-0.45}
KILL_PREV_SHARE=${KILL_PREV_SHARE:-0.40}

mkdir -p "$RUN_DIR"

cargo build --release -p game-connect4
cargo build --release -p mcts-tests \
  --example connect4_replay_searched_value --example connect4_cnn_smoke_gate

BIN="$ROOT/target/release/game-connect4"
ANNOT="$ROOT/target/release/examples/connect4_replay_searched_value"
GATE="$ROOT/target/release/examples/connect4_cnn_smoke_gate"

for g in $(seq "$START" $((GENS - 1))); do
  gen_start=$(date +%s)
  seed=$((1000 + g * 100000))
  if [ "$g" -eq 0 ]; then weights="$ZERO"; else weights="$RUN_DIR/gen$((g - 1)).c4cnn"; fi

  echo "=== generation $g: self-play ($GAMES games, $SIMS sims) @ $(date) ==="
  "$BIN" dump --label gumbel --head cnn --value-weights "$weights" \
    --out "$RUN_DIR/gen$g.bin" --games "$GAMES" --seed "$seed" --sims "$SIMS" \
    --max-considered 7 --temp-moves 6 --forced-opening-plies "$FORCED"

  echo "=== generation $g: searched-value annotation @ $(date) ==="
  "$ANNOT" --positions "$RUN_DIR/gen$g.bin" --out "$RUN_DIR/gen$g.sv.f32"

  # Retain the full replay every generation, always including the diverse
  # zero-net generation-0 data.
  pos=""; sv=""
  for h in $(seq 0 "$g"); do
    pos="$pos${pos:+,}$RUN_DIR/gen$h.bin"
    sv="$sv${sv:+,}$RUN_DIR/gen$h.sv.f32"
  done

  echo "=== generation $g -> $((g + 1)): mixture fit @ $(date) ==="
  uv run --project research/az-train python -m az_train.mixture_selfplay_c4 \
    --reference-corpus "$REF" --positions "$pos" --searched-values "$sv" \
    --out-dir "$RUN_DIR" --generation "$g" --b "$B" --l2 "$L2" --epochs "$EPOCHS" \
    --gen0-reservoir-fraction "$GEN0_RESERVOIR"

  echo "=== generation $g: gates (fixed harness, $GATE_GAMES games) @ $(date) ==="
  "$GATE" "$RUN_DIR/gen$g.c4cnn" "$GATE_GAMES" "$SIMS" \
    | tee "$RUN_DIR/gen$g.gate-vs-zero.txt"
  "$GATE" "$RUN_DIR/gen$g.c4cnn" "$GATE_GAMES" "$SIMS" --opponent "$RUN_DIR/gen0.c4cnn" \
    | tee "$RUN_DIR/gen$g.gate-vs-gen0.txt"
  prev_metric_args=()
  if [ "$g" -ge 1 ]; then
    "$GATE" "$RUN_DIR/gen$g.c4cnn" "$GATE_GAMES" "$SIMS" \
      --opponent "$RUN_DIR/gen$((g - 1)).c4cnn" \
      | tee "$RUN_DIR/gen$g.gate-vs-prev.txt"
    prev_metric_args=(--gate-vs-prev "$RUN_DIR/gen$g.gate-vs-prev.txt")
  fi

  gen_wall=$(($(date +%s) - gen_start))
  uv run --project research/az-train python -m az_train.coordinator_metrics_c4 \
    --result "$RUN_DIR/gen$g.result.json" \
    --gate-vs-zero "$RUN_DIR/gen$g.gate-vs-zero.txt" \
    --gate-vs-gen0 "$RUN_DIR/gen$g.gate-vs-gen0.txt" \
    ${prev_metric_args[@]+"${prev_metric_args[@]}"} \
    --generation "$g" --wall-seconds "$gen_wall" \
    | tee -a "$RUN_DIR/coordinator-metrics.jsonl"

  verdict=$(uv run --project research/az-train python - \
    "$RUN_DIR/coordinator-metrics.jsonl" "$g" "$KILL_GEN1_LB" "$KILL_PREV_SHARE" <<'PY'
import json, sys
path, gen, gen1_lb, prev_share = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
line = json.loads(open(path).read().splitlines()[-1])
msgs = []
if gen == 1:
    lb = line["gate_vs_gen0"]["wilson_lower_bound"]
    if lb < gen1_lb:
        msgs.append(f"gen1-vs-gen0 Wilson LB {lb} < {gen1_lb}")
prev = line.get("gate_vs_prev")
if prev is not None and prev["score_share"] < prev_share:
    msgs.append(f"gen{gen}-vs-gen{gen-1} share {prev['score_share']} < {prev_share}")
print("STOP: " + "; ".join(msgs) if msgs else "CONTINUE")
PY
)
  echo "kill-gate check (gen $g): $verdict"
  case "$verdict" in
    STOP*)
      printf '%s\nfired after generation %s\nresume: START=%s bash research/az-train/coordinator_c4_cnn.sh\n' \
        "$verdict" "$g" "$((g + 1))" > "$RUN_DIR/KILL-GATE.txt"
      break
      ;;
  esac
done

echo "=== done @ $(date) ==="
echo "curve: $RUN_DIR/coordinator-metrics.jsonl"
