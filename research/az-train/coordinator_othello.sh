#!/usr/bin/env bash
# Gumbel AlphaZero coordinator for Othello: self-play -> train -> gate ->
# repeat, with the n-tuple value head + D4 policy sidecar. Sibling of
# coordinator_c4.sh, which does the same for Connect Four; unlike Connect
# Four this also reports an Edax fixed-depth yardstick every generation, an
# external, non-self-referential strength reference Othello has and Connect
# Four does not.
#
#   RUN_DIR=local/output/az/othello/run0 bash research/az-train/coordinator_othello.sh
#
# Per generation: Gumbel self-play (gen k weights) -> az-train-othello fit
# (value + policy, both against the self-play outcome label) -> gumbel_gate
# vs gen0 and vs gen(k-1), with an Edax yardstick line folded into the same
# gate run -> one merged JSON metrics line appended to log.jsonl. Every
# generation checkpoints its shard, weights, and metrics line before the
# next starts, so an interrupt resumes with START set to the first
# unfinished generation.
#
# Env knobs: RUN_DIR, GAMES (self-play games/gen), GENS, SIMS,
# MAX_CONSIDERED, TEMP_MOVES, FORCED_OPENING_PLIES, GATE_GAMES,
# POLICY_L2, POLICY_EPOCHS, EDAX_LEVEL, START.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export LIBRARY_PATH=${LIBRARY_PATH:-/opt/homebrew/lib}

RUN_DIR=${RUN_DIR:-local/output/az/othello/run0}
MODEL_TOML=${MODEL_TOML:-games/othello/ntuple/model.toml}
TRAIN_CONFIG=${TRAIN_CONFIG:-games/othello/ntuple/train.toml}
GAMES=${GAMES:-800}
GENS=${GENS:-5}
SIMS=${SIMS:-32}
MAX_CONSIDERED=${MAX_CONSIDERED:-8}
TEMP_MOVES=${TEMP_MOVES:-12}
FORCED_OPENING_PLIES=${FORCED_OPENING_PLIES:-6}
GATE_GAMES=${GATE_GAMES:-100}
POLICY_L2=${POLICY_L2:-1e-4}
POLICY_EPOCHS=${POLICY_EPOCHS:-60}
EDAX_BINARY=${EDAX_BINARY:-games/othello/edax/vendor/bin/mEdax-native}
EDAX_DATA_DIR=${EDAX_DATA_DIR:-games/othello/edax/vendor/data}
EDAX_LEVEL=${EDAX_LEVEL:-3}
START=${START:-0}

mkdir -p "$RUN_DIR/shards"

cargo build --release -p game-othello --bin game-othello --example gumbel_gate

BIN="$ROOT/target/release/game-othello"
GATE="$ROOT/target/release/examples/gumbel_gate"

# Generation-0 checkpoint: the all-zero net over $MODEL_TOML's geometry, so
# gen0 is a real, loadable, self-contained checkpoint directory rather than
# a special-cased "no --weights-dir" self-play path -- the gate and every
# later generation's fit treat every generation uniformly.
if [ ! -f "$RUN_DIR/gen0/weights.bin" ]; then
  echo "=== generation 0: writing the all-zero checkpoint @ $(date) ==="
  uv run --project research/az-train python - "$MODEL_TOML" "$RUN_DIR/gen0" <<'PY'
import hashlib
import json
import shutil
import struct
import sys
import tomllib
from pathlib import Path

model_toml, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
raw = model_toml.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
doc = tomllib.loads(raw.decode("utf-8"))
n_weights = sum(3 ** len(t["squares"]) for t in doc["tuple"])

shutil.copy(model_toml, out_dir / "model.toml")
(out_dir / "weights.bin").write_bytes(struct.pack(f"<{n_weights}f", *([0.0] * n_weights)))
(out_dir / "weights.meta.json").write_text(
    json.dumps({"model_toml_sha256": sha, "n_weights": n_weights}, indent=2) + "\n"
)
(out_dir / "policy.bin").write_bytes(
    struct.pack(f"<{n_weights * 64}f", *([0.0] * (n_weights * 64)))
)
(out_dir / "policy.meta.json").write_text(
    json.dumps({"model_toml_sha256": sha, "n_weights": n_weights}, indent=2) + "\n"
)
print(f"wrote {out_dir} ({n_weights} weights)")
PY
fi

for g in $(seq "$START" $((GENS - 1))); do
  gen_start=$(date +%s)
  seed=$((1000 + g * 100000))

  echo "=== generation $g: self-play ($GAMES games, $SIMS sims) @ $(date) ==="
  "$BIN" dump --label gumbel --weights-dir "$RUN_DIR/gen$g" \
    --out "$RUN_DIR/shards/gen$g.bin" --games "$GAMES" --seed "$seed" --sims "$SIMS" \
    --max-considered "$MAX_CONSIDERED" --temp-moves "$TEMP_MOVES" \
    --forced-opening-plies "$FORCED_OPENING_PLIES"

  # Replay window: every generation's shards, always including the diverse
  # generation-0 (zero-net) data, so later fits keep board coverage the
  # sharpening self-play policy stops visiting on its own.
  parts=""
  for s in $(seq 0 "$g"); do parts="$parts${parts:+,}$RUN_DIR/shards/gen$s.bin"; done

  echo "=== generation $g -> $((g + 1)): train @ $(date) ==="
  uv run --project research/az-train az-train-othello \
    --positions "$parts" --model "$MODEL_TOML" --train-config "$TRAIN_CONFIG" \
    --out "$RUN_DIR/gen$((g + 1))" --policy-l2 "$POLICY_L2" --policy-epochs "$POLICY_EPOCHS"

  echo "=== generation $((g + 1)): gates ($GATE_GAMES games) @ $(date) ==="
  # Gates are diagnostic, not a hard stop -- a FAIL must not abort the loop.
  # Deciding what a FAIL means for the run as a whole is a judgment call on
  # the finished curve, not something this script should short-circuit on.
  set +e
  "$GATE" "$RUN_DIR/gen0" "$RUN_DIR/gen$((g + 1))" "$GATE_GAMES" "$SIMS" \
    --edax-binary "$EDAX_BINARY" --edax-data-dir "$EDAX_DATA_DIR" --edax-level "$EDAX_LEVEL" \
    | tee "$RUN_DIR/gen$((g + 1)).gate-vs-gen0.txt"
  prev_metric_args=()
  if [ "$g" -ge 1 ]; then
    "$GATE" "$RUN_DIR/gen$g" "$RUN_DIR/gen$((g + 1))" "$GATE_GAMES" "$SIMS" \
      | tee "$RUN_DIR/gen$((g + 1)).gate-vs-prev.txt"
    prev_metric_args=(--gate-vs-prev "$RUN_DIR/gen$((g + 1)).gate-vs-prev.txt")
  fi
  set -e

  gen_wall=$(($(date +%s) - gen_start))
  uv run --project research/az-train python -m az_train.coordinator_metrics_othello \
    --weights-meta "$RUN_DIR/gen$((g + 1))/weights.meta.json" \
    --policy-meta "$RUN_DIR/gen$((g + 1))/policy.meta.json" \
    --gate-vs-gen0 "$RUN_DIR/gen$((g + 1)).gate-vs-gen0.txt" \
    ${prev_metric_args[@]+"${prev_metric_args[@]}"} \
    --generation $((g + 1)) --wall-seconds "$gen_wall" \
    | tee -a "$RUN_DIR/log.jsonl"
done

echo "=== done @ $(date) ==="
echo "curve: $RUN_DIR/log.jsonl"
