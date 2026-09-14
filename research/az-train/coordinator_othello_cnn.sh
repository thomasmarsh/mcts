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
# othello-cnn fit (value + policy, both against the self-play outcome/
# completed-Q label) -> one merged JSON metrics line appended to log.jsonl.
# Every generation checkpoints its shard, weights, and metrics line before
# the next starts, so an interrupt resumes with START set to the first
# unfinished generation.
#
# Checkpoint format: unlike the n-tuple head's checkpoint *directory*
# (`model.toml` geometry-as-data + `weights.bin` + `policy.bin`), the CNN's
# architecture is fixed code (`OTCNN001`), so each generation is a single
# `gen<N>.cnn.bin` file (`games/othello/src/convnet.rs::CnnValueNet::load`'s
# exact byte layout) plus a `gen<N>.cnn.bin.meta.json` sidecar.
#
# KNOWN GAP, not fixed by this script: there is no CNN-aware gate yet.
# `games/othello/examples/gumbel_gate.rs` is written
# directly against `NTupleModel`/`NTuplePolicyNet`/`GumbelPlayer`, not
# `CnnValueNet`/`CnnGumbelPlayer`, so it cannot score a CNN checkpoint as-is.
# This coordinator therefore logs self-play and training metrics only --
# gen-vs-gen0/gen-vs-prev/Edax head-to-head columns are absent from
# `log.jsonl` until `gumbel_gate.rs` gets a `--head cnn` path (a real but
# modest change: both `GumbelPlayer` and `CnnGumbelPlayer` already implement
# the same `mcts::algorithms::Search` trait `battle_royale` is generic over,
# so `score_share`/`load_dir` need to become head-dispatching, not a
# redesign). Do not treat a run driven by this script as gated until that
# lands and is re-run against it.
#
# Env knobs: RUN_DIR, GAMES (self-play games/gen), GENS, SIMS,
# MAX_CONSIDERED, TEMP_MOVES, FORCED_OPENING_PLIES, EPOCHS, BATCH_SIZE, LR,
# L2, VALIDATION_FRACTION, START.
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
EPOCHS=${EPOCHS:-24}
BATCH_SIZE=${BATCH_SIZE:-4096}
LR=${LR:-2e-3}
L2=${L2:-1e-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
START=${START:-0}

mkdir -p "$RUN_DIR/shards"

cargo build --release -p game-othello --bin game-othello

BIN="$ROOT/target/release/game-othello"

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
    --out "$RUN_DIR/shards/gen$g.bin" --games "$GAMES" --seed "$seed" --sims "$SIMS" \
    --max-considered "$MAX_CONSIDERED" --temp-moves "$TEMP_MOVES" \
    --forced-opening-plies "$FORCED_OPENING_PLIES"

  # Replay window: every generation's shards, always including the diverse
  # generation-0 (zero-net) data -- same rationale as coordinator_othello.sh.
  parts=""
  for s in $(seq 0 "$g"); do parts="$parts${parts:+,}$RUN_DIR/shards/gen$s.bin"; done

  echo "=== generation $g -> $((g + 1)): train @ $(date) ==="
  uv run --project research/az-train az-train-othello-cnn \
    --positions "$parts" --out "$RUN_DIR/gen$((g + 1)).cnn.bin" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LR" --l2 "$L2" \
    --validation-fraction "$VALIDATION_FRACTION" --split-seed "$g"

  gen_wall=$(($(date +%s) - gen_start))
  uv run --project research/az-train python - \
    "$RUN_DIR/gen$((g + 1)).cnn.bin.meta.json" "$g" "$gen_wall" <<'PY' | tee -a "$RUN_DIR/log.jsonl"
import json
import sys

meta_path, generation, wall_seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
meta = json.loads(open(meta_path).read())
line = {
    "generation": generation + 1,
    "wall_seconds": round(wall_seconds, 1),
    "positions": meta["train"]["positions"],
    "train_games": meta["train"]["train_games"],
    "validation_games": meta["train"]["validation_games"],
    "final_validation_metrics": meta["metrics"]["final_validation_metrics"],
}
print(json.dumps(line, sort_keys=True))
PY
done

echo "=== done @ $(date) ==="
echo "curve: $RUN_DIR/log.jsonl (self-play/training metrics only -- no gate, see this script's header)"
