#!/usr/bin/env bash
# Gumbel AlphaZero Phase-0 coordinator: self-play -> train -> repeat, on
# tic-tac-toe, with the linear value head and a uniform policy stub.
#
#   research/az-train/coordinator.sh [run_dir]
#
# Env knobs: GAMES (self-play games per generation), SIMS (Gumbel budget per
# move), GENS (generations), GATE_GAMES (head-to-head games in the gate),
# TEMP_MOVES (opening plies sampled from the visit distribution in self-play).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

RUN_DIR=${1:-local/output/az/ttt/run0}
GAMES=${GAMES:-400}
SIMS=${SIMS:-32}
GENS=${GENS:-3}
GATE_GAMES=${GATE_GAMES:-200}
TEMP_MOVES=${TEMP_MOVES:-3}

mkdir -p "$RUN_DIR/weights" "$RUN_DIR/shards"

export LIBRARY_PATH=${LIBRARY_PATH:-/opt/homebrew/lib}
cargo build --release -p game-ttt
BIN="$ROOT/target/release/game-ttt"

# Generation 0 weights: the all-zero net (every position scores as a draw).
python3 -c "import struct; open('$RUN_DIR/weights/gen_0.bin','wb').write(struct.pack('<19f', *([0.0]*19)))"

for g in $(seq 0 $((GENS - 1))); do
  echo "=== generation $g: self-play ==="
  "$BIN" dump --label gumbel \
    --out "$RUN_DIR/shards/gen_$g.bin" \
    --games "$GAMES" --seed "$g" --sims "$SIMS" --temp-moves "$TEMP_MOVES" \
    --weights "$RUN_DIR/weights/gen_$g.bin"

  # Replay window: every generation's shards, always including the diverse
  # generation-0 (zero-net) data so later fits keep board coverage the
  # sharpening self-play policy stops visiting on its own.
  parts=""
  for s in $(seq 0 "$g"); do parts="$parts$ROOT/$RUN_DIR/shards/gen_$s.bin,"; done

  echo "=== generation $g -> $((g + 1)): train ==="
  ( cd research/az-train && uv run az-train \
      --positions "${parts%,}" \
      --out "$ROOT/$RUN_DIR/weights/gen_$((g + 1)).bin" )
done

# Gates are diagnostic; a FAIL must not abort the report of the other gate.
set +e
echo "=== gate: gen $GENS vs gen 0 (did the loop improve on its starting point?) ==="
cargo run --release -p mcts-tests --example gumbel_ttt_gate -- \
  "$RUN_DIR/weights/gen_0.bin" "$RUN_DIR/weights/gen_$GENS.bin" "$GATE_GAMES" "$SIMS"

echo "=== gate: gen $GENS vs gen 1 (did it keep improving after the first fit?) ==="
cargo run --release -p mcts-tests --example gumbel_ttt_gate -- \
  "$RUN_DIR/weights/gen_1.bin" "$RUN_DIR/weights/gen_$GENS.bin" "$GATE_GAMES" "$SIMS"
