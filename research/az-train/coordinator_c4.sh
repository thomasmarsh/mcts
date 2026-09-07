#!/usr/bin/env bash
# Gumbel AlphaZero Phase-1 coordinator: self-play -> train -> repeat, on
# Connect Four, with n-tuple value and policy sidecars.
# The tic-tac-toe counterpart is coordinator.sh.
#
#   research/az-train/coordinator_c4.sh [run_dir]
#
# Env knobs: GAMES (self-play games per generation), SIMS (Gumbel budget per
# move), GENS (generations), GATE_GAMES (head-to-head games in the gate),
# TEMP_MOVES (opening plies sampled from the improved policy in self-play).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

RUN_DIR=${1:-local/output/az/connect4/run0}
GAMES=${GAMES:-400}
SIMS=${SIMS:-32}
GENS=${GENS:-5}
GATE_GAMES=${GATE_GAMES:-200}
TEMP_MOVES=${TEMP_MOVES:-6}

mkdir -p "$RUN_DIR/weights" "$RUN_DIR/shards"

export LIBRARY_PATH=${LIBRARY_PATH:-/opt/homebrew/lib}
cargo build --release -p game-connect4
cargo build --release -p mcts-tests --example gumbel_connect4_gate
BIN="$ROOT/target/release/game-connect4"

# Generation 0 weights: the all-zero n-tuple net (5590 f32; every position
# scores as a draw). Layout: az_train.ntuple_c4 /
# game_connect4::valuenet::NTupleValueNet.
NT_WEIGHTS=5590
python3 -c "import struct,sys; n=int(sys.argv[1]); open('$RUN_DIR/weights/gen_0.bin','wb').write(struct.pack('<%df'%n, *([0.0]*n)))" "$NT_WEIGHTS"
POLICY_WEIGHTS=39130
python3 -c "import struct,sys; n=int(sys.argv[1]); open('$RUN_DIR/weights/gen_0.policy.bin','wb').write(struct.pack('<%df'%n, *([0.0]*n)))" "$POLICY_WEIGHTS"

for g in $(seq 0 $((GENS - 1))); do
  echo "=== generation $g: self-play ==="
  "$BIN" dump --label gumbel \
    --out "$RUN_DIR/shards/gen_$g.bin" \
    --games "$GAMES" --seed "$g" --sims "$SIMS" --temp-moves "$TEMP_MOVES" \
    --value-weights "$RUN_DIR/weights/gen_$g.bin" \
    --policy-weights "$RUN_DIR/weights/gen_$g.policy.bin"

  # Replay window: every generation's shards, always including the diverse
  # generation-0 (zero-net) data so later fits keep board coverage the
  # sharpening self-play policy stops visiting on its own.
  parts=""
  for s in $(seq 0 "$g"); do parts="$parts$ROOT/$RUN_DIR/shards/gen_$s.bin,"; done

  echo "=== generation $g -> $((g + 1)): train ==="
  ( cd research/az-train && uv run az-train --game connect4 --head ntuple \
      --positions "${parts%,}" \
      --out "$ROOT/$RUN_DIR/weights/gen_$((g + 1)).bin" \
      --policy-out "$ROOT/$RUN_DIR/weights/gen_$((g + 1)).policy.bin" )
done

# Gates are diagnostic; a FAIL must not abort the report of the other gate.
set +e
for lo in 0 1; do
  echo "=== gate: gen $GENS vs gen $lo ==="
  cargo run --release -p mcts-tests --example gumbel_connect4_gate -- \
    "$RUN_DIR/weights/gen_$lo.bin" "$RUN_DIR/weights/gen_$GENS.bin" "$GATE_GAMES" "$SIMS" \
    --baseline-policy "$RUN_DIR/weights/gen_$lo.policy.bin" --candidate-policy "$RUN_DIR/weights/gen_$GENS.policy.bin"
done

echo "=== per-generation gate ladder (gen k vs gen k-1) ==="
for g in $(seq 1 "$GENS"); do
  echo "--- gen $g vs gen $((g - 1)) ---"
  cargo run --release -p mcts-tests --example gumbel_connect4_gate -- \
    "$RUN_DIR/weights/gen_$((g - 1)).bin" "$RUN_DIR/weights/gen_$g.bin" "$GATE_GAMES" "$SIMS" \
    --baseline-policy "$RUN_DIR/weights/gen_$((g - 1)).policy.bin" --candidate-policy "$RUN_DIR/weights/gen_$g.policy.bin"
done
