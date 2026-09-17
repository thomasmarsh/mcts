#!/usr/bin/env bash
# Gumbel AlphaZero coordinator for Othello with the OTCNN001 CNN value+policy
# head. Sibling of coordinator_othello.sh, which drives the n-tuple head;
# only the model class and checkpoint format differ -- self-play, the
# self-play-outcome value target plus completed-Q policy target, and the
# cumulative replay-window shape are unchanged from that script.
#
#   RUN_DIR=local/output/az/othello-cnn/run0 bash research/az-train/coordinator_othello_cnn.sh
#
# Per generation: Gumbel self-play (gen k weights, `--head cnn`) -> az-train-
# othello-cnn fit (PyTorch Adam, value MSE + masked policy cross-entropy
# against the self-play outcome/completed-Q label, cosine LR decay, best-
# validation-checkpoint selection -- `az_train.convnet_othello_torch.
# fit_torch`) -> gumbel_gate --head cnn vs gen0 and vs gen(k-1), with an
# Edax yardstick line folded into the same gate run -> one merged JSON
# metrics line appended to log.jsonl, plus a per-epoch JSONL sidecar
# (gen<N>.epochs.jsonl) written incrementally during that generation's fit.
# Every generation checkpoints its shard, weights, gate output, and metrics
# line before the next starts, so an interrupt resumes with START set to
# the first unfinished generation.
#
# Checkpoint format: unlike the n-tuple head's checkpoint *directory*
# (`model.toml` geometry-as-data + `weights.bin` + `policy.bin`), the CNN's
# architecture is fixed code (`OTCNN001`), so each generation is a single
# `gen<N>.cnn.bin` file (`games/othello/src/convnet.rs::CnnValueNet::load`'s
# exact byte layout) plus a `gen<N>.cnn.bin.meta.json` sidecar. `az-train-
# othello-cnn` always writes the fit's best-validation-checkpoint weights
# (not the final epoch's) to this file.
#
# The gate call and its stdout->JSON merge mirror coordinator_othello.sh's
# own n-tuple gate wiring exactly (same `gumbel_gate` binary, same
# `az_train.coordinator_metrics_othello.parse_gate_output` stdout parser --
# `run_checks`'s output text is identical across `--head ntuple`/`--head
# cnn`), just with `--head cnn` added to the gate invocation and the CNN's
# single `.meta.json` sidecar merged in place of the n-tuple's separate
# weights/policy meta files.
#
# Progress while this is running: `tail -f "$RUN_DIR/log.jsonl"` for one
# line per completed generation (wall clock, positions, value/policy
# metrics, both gate results), or `tail -f "$RUN_DIR/gen<N>.epochs.jsonl"`
# for that generation's own live per-epoch validation trace while it fits.
#
# Env knobs: RUN_DIR, GAMES (self-play games/gen), GENS, SIMS,
# MAX_CONSIDERED, TEMP_MOVES, FORCED_OPENING_PLIES, EPOCHS, BATCH_SIZE, LR,
# L2, VALIDATION_FRACTION, DEVICE, GATE_GAMES, EDAX_BINARY, EDAX_DATA_DIR,
# EDAX_LEVEL, REPLAY_WINDOW, START, EVALUATOR.
#
# EVALUATOR: which `CnnValueNet` forward-pass backend self-play uses --
# `cpu` (default, always available) or `mlx` (GPU-backed via
# `games/othello/src/convnet/mlx.rs::MlxCnnValueNet`, ~4.5-5.4x faster
# wall-clock on this machine's real self-play call pattern, byte-identical
# output to the CPU path on the same seed). Requires building
# `game-othello` with `--features mlx`, which this script only turns on
# when `EVALUATOR=mlx` -- Homebrew's `mlx`/`mlx-c` must be installed. Only
# self-play uses this knob; training and gating are untouched and always
# run on the CPU path.
#
# REPLAY_WINDOW: number of most recent generations' shards to train on each
# generation (default 0 = unlimited/cumulative, every shard from gen0 on,
# the original behaviour). When set to N > 0, generation g trains only on
# shards from generations max(0, g - N + 1)..=g -- a real sliding window,
# which (unlike the default) does *not* force gen0's diverse shard to stay
# in the window once it ages out, since the point of this knob is to
# measure the staleness/RAM tradeoff a true bounded window has, not to
# preserve the default's own gen0-anchoring choice.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export LIBRARY_PATH=${LIBRARY_PATH:-/opt/homebrew/lib}

RUN_DIR=${RUN_DIR:-local/output/az/othello-cnn/run0}
GAMES=${GAMES:-800}
GENS=${GENS:-5}
SIMS=${SIMS:-32}
MAX_CONSIDERED=${MAX_CONSIDERED:-8}
TEMP_MOVES=${TEMP_MOVES:-12}
FORCED_OPENING_PLIES=${FORCED_OPENING_PLIES:-6}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-4096}
LR=${LR:-2e-3}
L2=${L2:-1e-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
DEVICE=${DEVICE:-}
GATE_GAMES=${GATE_GAMES:-100}
EDAX_BINARY=${EDAX_BINARY:-games/othello/edax/vendor/bin/mEdax-native}
EDAX_DATA_DIR=${EDAX_DATA_DIR:-games/othello/edax/vendor/data}
EDAX_LEVEL=${EDAX_LEVEL:-3}
REPLAY_WINDOW=${REPLAY_WINDOW:-0}
START=${START:-0}
EVALUATOR=${EVALUATOR:-cpu}

mkdir -p "$RUN_DIR/shards"

feature_args=()
if [ "$EVALUATOR" = "mlx" ]; then feature_args=(--features mlx); fi
cargo build --release -p game-othello --bin game-othello --example gumbel_gate "${feature_args[@]}"

BIN="$ROOT/target/release/game-othello"
GATE="$ROOT/target/release/examples/gumbel_gate"

# Generation-0 checkpoint: the all-zero OTCNN001 net, written as a real,
# loadable file (matching coordinator_othello.sh's own "gen0 is a real
# checkpoint, not a special-cased no-weights self-play path" convention) so
# every generation's self-play call takes the same `--cnn-weights` flag.
if [ ! -f "$RUN_DIR/gen0.cnn.bin" ]; then
  echo "=== generation 0: writing the all-zero OTCNN001 checkpoint @ $(date) ==="
  uv run --project research/az-train python - "$RUN_DIR/gen0.cnn.bin" <<'PY'
import sys
import numpy as np
from othello_eval import convnet

out = sys.argv[1]
convnet.write_weights(out, np.zeros(convnet.N_WEIGHTS, dtype=np.float32))
print(f"wrote {out} ({convnet.N_WEIGHTS} weights)")
PY
fi

for g in $(seq "$START" $((GENS - 1))); do
  gen_start=$(date +%s)
  seed=$((1000 + g * 100000))

  echo "=== generation $g: self-play ($GAMES games, $SIMS sims) @ $(date) ==="
  "$BIN" dump --label gumbel --head cnn --cnn-weights "$RUN_DIR/gen$g.cnn.bin" \
    --evaluator "$EVALUATOR" \
    --out "$RUN_DIR/shards/gen$g.bin" --games "$GAMES" --seed "$seed" --sims "$SIMS" \
    --max-considered "$MAX_CONSIDERED" --temp-moves "$TEMP_MOVES" \
    --forced-opening-plies "$FORCED_OPENING_PLIES"

  # Replay window: default (REPLAY_WINDOW=0) is every generation's shards,
  # always including the diverse generation-0 (zero-net) data -- same
  # rationale as coordinator_othello.sh. REPLAY_WINDOW=N > 0 switches to a
  # true sliding window of the last N generations only (see the env-knob
  # doc comment above).
  parts=""
  if [ "$REPLAY_WINDOW" -gt 0 ]; then
    window_start=$((g - REPLAY_WINDOW + 1))
    if [ "$window_start" -lt 0 ]; then window_start=0; fi
  else
    window_start=0
  fi
  for s in $(seq "$window_start" "$g"); do parts="$parts${parts:+,}$RUN_DIR/shards/gen$s.bin"; done

  echo "=== generation $g -> $((g + 1)): train @ $(date) ==="
  device_arg=()
  if [ -n "$DEVICE" ]; then device_arg=(--device "$DEVICE"); fi
  uv run --project research/az-train az-train-othello-cnn \
    --positions "$parts" --out "$RUN_DIR/gen$((g + 1)).cnn.bin" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LR" --l2 "$L2" \
    --validation-fraction "$VALIDATION_FRACTION" --split-seed "$g" \
    --epoch-log "$RUN_DIR/gen$((g + 1)).epochs.jsonl" "${device_arg[@]}"

  echo "=== generation $((g + 1)): gates ($GATE_GAMES games) @ $(date) ==="
  # Gates are diagnostic, not a hard stop -- a FAIL must not abort the loop,
  # same rationale as coordinator_othello.sh.
  set +e
  "$GATE" "$RUN_DIR/gen0.cnn.bin" "$RUN_DIR/gen$((g + 1)).cnn.bin" "$GATE_GAMES" "$SIMS" \
    --head cnn --edax-binary "$EDAX_BINARY" --edax-data-dir "$EDAX_DATA_DIR" --edax-level "$EDAX_LEVEL" \
    | tee "$RUN_DIR/gen$((g + 1)).gate-vs-gen0.txt"
  prev_gate_arg=""
  if [ "$g" -ge 1 ]; then
    "$GATE" "$RUN_DIR/gen$g.cnn.bin" "$RUN_DIR/gen$((g + 1)).cnn.bin" "$GATE_GAMES" "$SIMS" --head cnn \
      | tee "$RUN_DIR/gen$((g + 1)).gate-vs-prev.txt"
    prev_gate_arg="$RUN_DIR/gen$((g + 1)).gate-vs-prev.txt"
  fi
  set -e

  gen_wall=$(($(date +%s) - gen_start))
  uv run --project research/az-train python - \
    "$RUN_DIR/gen$((g + 1)).cnn.bin.meta.json" "$g" "$gen_wall" \
    "$RUN_DIR/gen$((g + 1)).gate-vs-gen0.txt" "$prev_gate_arg" <<'PY' | tee -a "$RUN_DIR/log.jsonl"
import json
import sys

from az_train.coordinator_metrics_othello import parse_gate_output

meta_path, generation, wall_seconds, gate_vs_gen0_path, gate_vs_prev_path = sys.argv[1:6]
meta = json.loads(open(meta_path).read())
line = {
    "generation": int(generation) + 1,
    "wall_seconds": round(float(wall_seconds), 1),
    "positions": meta["train"]["positions"],
    "train_games": meta["train"]["train_games"],
    "validation_games": meta["train"]["validation_games"],
    "final_validation_metrics": meta["metrics"]["final_validation_metrics"],
    "gate_vs_gen0": parse_gate_output(open(gate_vs_gen0_path).read()),
    "gate_vs_prev": parse_gate_output(open(gate_vs_prev_path).read()) if gate_vs_prev_path else None,
}
print(json.dumps(line, sort_keys=True))
PY
done

echo "=== done @ $(date) ==="
echo "curve: $RUN_DIR/log.jsonl"
