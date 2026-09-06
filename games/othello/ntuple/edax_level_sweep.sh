#!/usr/bin/env bash
# Choose `edax_level` for harvest_edax.toml.
#
# Label a small, fixed set of independent self-play positions with Edax at
# each of several levels, fit a quick arm-B n-tuple model per level, and
# read held-out MSE (relabelled by Edax at a strictly higher level) against
# the training level. Pick the lowest level where MSE has plateaued --
# deeper is just spent CPU. Set this before running bakeoff.sh, from this
# diagnostic alone, not from any downstream strength number.
#
# Usage:  games/othello/ntuple/edax_level_sweep.sh <work-dir>
#
# Writes <work-dir>/level_sweep.md and .csv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LIBRARY_PATH="${LIBRARY_PATH:-/opt/homebrew/lib}"

WORK="${1:?usage: edax_level_sweep.sh <work-dir>}"
LEVELS=(${LEVELS:-6 8 10 12 16})
GAMES="${GAMES:-60}"               # ~60 games -> ~1.5-2k arm_b positions
HELDOUT_LEVEL="${HELDOUT_LEVEL:-20}"
HARVEST_CFG="games/othello/ntuple/harvest.toml"
EDAX_CFG="games/othello/ntuple/harvest_edax.toml"
MODEL_TOML="games/othello/ntuple/model.toml"
TRAIN_CFG="games/othello/ntuple/train.toml"

mkdir -p "$WORK"
exec > >(tee -a "$WORK/log") 2>&1
echo "=== edax level sweep $(date -u +%FT%TZ)  levels=${LEVELS[*]}  games=$GAMES ==="

dump() { cargo run --release -q -p game-othello -- dump --label harvest \
  --harvest-config "$HARVEST_CFG" --edax-config "$EDAX_CFG" --oracle edax "$@"; }

# One held-out set, relabelled above every training level under test.
if [[ ! -f "$WORK/heldout/arm_b.bin" ]]; then
  dump --out "$WORK/heldout" --seed 424243 --games "$GAMES" --edax-level "$HELDOUT_LEVEL"
fi

CSV="$WORK/level_sweep.csv"
echo "level,arm_b_positions,held_out_mse,sign_acc" > "$CSV"
for L in "${LEVELS[@]}"; do
  d="$WORK/lvl_$L"
  [[ -f "$d/arm_b.bin" ]] || dump --out "$d" --seed 4242 --games "$GAMES" --edax-level "$L"
  [[ -f "$WORK/weights_$L/weights.bin" ]] || \
    uv run --project othello-eval othello-eval-train \
      --positions "$d/arm_b.bin" --model "$MODEL_TOML" --config "$TRAIN_CFG" \
      --out "$WORK/weights_$L"
  uv run --project othello-eval othello-eval-mse --held "$WORK/heldout/arm_b.bin" \
    --weights "L$L=$WORK/weights_$L" --json-out "$WORK/mse_$L.json"
  python - "$WORK/mse_$L.json" "$L" "$(wc -c < "$d/arm_b.bin")" "$CSV" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1])); lvl = sys.argv[2]
n = int(sys.argv[3]) // 22
m = next(iter(rep["models"].values()))
open(sys.argv[4], "a").write(f"{lvl},{n},{m['mse']:.6f},{m['sign_accuracy']:.4f}\n")
PY
done

echo "=== sweep done ==="
column -s, -t "$CSV" | tee "$WORK/level_sweep.md"
echo "Set edax_level in harvest_edax.toml to the lowest level whose MSE is within noise of the best."
