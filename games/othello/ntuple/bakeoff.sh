#!/usr/bin/env bash
# The signal bake-off, end to end.
#
# One shared search-per-move self-play corpus feeds four label arms:
#   A  played root <- final outcome            (TD / AZ-style)
#   B  played root <- its own searched value
#   C  every internal node <- its searched value  (TreeStrap)
#   D  played root <- the TD(lambda) return
# Every arm pays the identical search cost; they differ only in what they
# keep. We train one n-tuple model per arm with identical hyperparameters,
# then measure strength (Edax-ladder placement + A/B/C/D round-robin) and
# held-out regression MSE, and plot strength against cumulative Rust
# CPU-seconds.
#
# Usage:  games/othello/ntuple/bakeoff.sh <work-dir> [harvest.toml] [train.toml]
#
# Environment:
#   ORACLE=mcts|edax        target teacher (default mcts). With edax, arms
#                           B/C/D read their target from an independent Edax
#                           evaluation instead of the MCTS tree's own value.
#   EDAX_CFG=<path>          Edax oracle config (default harvest_edax.toml).
#   HELDOUT_EDAX_LEVEL=<N>   held-out relabel level, strictly above the
#                           training level (default 18); ORACLE=edax only.
#   EDAX_GAMES=<N>           corpus size for ORACLE=edax (default 300).
#                           Arm C pays one Edax call per harvested node
#                           (~10^2-10^3 per move), so the MCTS default of
#                           3000 games is intractable here -- keep it small.
#
# Runs for hours -- launch it as a background job and tail <work-dir>/log.
set -euo pipefail

ORACLE="${ORACLE:-mcts}"
EDAX_CFG="${EDAX_CFG:-games/othello/ntuple/harvest_edax.toml}"
HELDOUT_EDAX_LEVEL="${HELDOUT_EDAX_LEVEL:-18}"
EDAX_GAMES="${EDAX_GAMES:-300}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

WORK="${1:?usage: bakeoff.sh <work-dir> [harvest.toml] [train.toml]}"
HARVEST_CFG="${2:-games/othello/ntuple/harvest.toml}"
TRAIN_CFG="${3:-games/othello/ntuple/train.toml}"
MODEL_TOML="games/othello/ntuple/model.toml"
MATCH_CFG="games/othello/ntuple/match.toml"

export LIBRARY_PATH="${LIBRARY_PATH:-/opt/homebrew/lib}"
mkdir -p "$WORK"
LOG="$WORK/log"
exec > >(tee -a "$LOG") 2>&1
echo "=== bake-off start $(date -u +%FT%TZ)  work=$WORK  oracle=$ORACLE ==="
ORACLE_ARGS=(--oracle "$ORACLE")
[[ "$ORACLE" == edax ]] && ORACLE_ARGS+=(--edax-config "$EDAX_CFG" --games "$EDAX_GAMES")

ACCT="$WORK/cpu_seconds.tsv"
: > "$ACCT"
stage() {  # stage <name> -- run "$@" (rest), record wall-seconds
  local name="$1"; shift
  local t0=$SECONDS
  "$@"
  printf '%s\t%d\n' "$name" "$((SECONDS - t0))" >> "$ACCT"
}

# ---------------------------------------------------------------------------
# 1. Harvest: one self-play pass, four arms, provably the same searches.
# ---------------------------------------------------------------------------
if [[ ! -f "$WORK/harvest/harvest.json" ]]; then
  stage harvest cargo run --release -q -p game-othello -- \
    dump --label harvest --out "$WORK/harvest" --harvest-config "$HARVEST_CFG" \
    "${ORACLE_ARGS[@]}"
fi
cat "$WORK/harvest/harvest.json"

# ---------------------------------------------------------------------------
# 2. Held-out set: an independent harvest, arm_b (root positions) relabelled
#    by a stronger teacher than training used --
#    ORACLE=mcts: a deeper MCTS label search (via HARVEST_CFG);
#    ORACLE=edax: Edax at HELDOUT_EDAX_LEVEL, strictly above the training
#    level -- so the held-out MSE is a near-ground-truth reference rather
#    than a re-run of the (possibly weak) training teacher.
# ---------------------------------------------------------------------------
HELDOUT_ARGS=(--oracle "$ORACLE" --seed 999983 --games 400)
if [[ "$ORACLE" == edax ]]; then
  # arm_b only (one Edax call per ply, no per-node harvest), but at the
  # higher held-out level -- keep the count modest.
  HELDOUT_ARGS=(--oracle edax --seed 999983 --games 200 \
    --edax-config "$EDAX_CFG" --edax-level "$HELDOUT_EDAX_LEVEL")
fi
if [[ ! -f "$WORK/heldout/arm_b.bin" ]]; then
  stage heldout cargo run --release -q -p game-othello -- \
    dump --label harvest --out "$WORK/heldout" \
    --harvest-config "$HARVEST_CFG" "${HELDOUT_ARGS[@]}"
fi
HELD="$WORK/heldout/arm_b.bin"

# ---------------------------------------------------------------------------
# 3. Train one model per arm, identical hyperparameters.
# ---------------------------------------------------------------------------
for arm in a b c d; do
  out="$WORK/weights_$arm"
  if [[ ! -f "$out/weights.bin" ]]; then
    stage "train_$arm" uv run --project othello-eval othello-eval-train \
      --positions "$WORK/harvest/arm_$arm.bin" \
      --model "$MODEL_TOML" --config "$TRAIN_CFG" --out "$out"
  fi
done

# ---------------------------------------------------------------------------
# 4. Held-out MSE + pairwise bootstrap CI (A vs C is the gate tie-breaker).
# ---------------------------------------------------------------------------
stage mse uv run --project othello-eval othello-eval-mse --held "$HELD" \
  --weights "A=$WORK/weights_a" --weights "B=$WORK/weights_b" \
  --weights "C=$WORK/weights_c" --weights "D=$WORK/weights_d" \
  --json-out "$WORK/mse.json"

# ---------------------------------------------------------------------------
# 5. Edax-ladder placement, one process per arm.
# ---------------------------------------------------------------------------
for arm in a b c d; do
  echo "--- Edax ladder: arm $arm ---"
  stage "edax_$arm" env OTHELLO_NTUPLE_WEIGHTS="$WORK/weights_$arm" \
    cargo run --release -q --example ntuple_match -p game-othello -- "$MATCH_CFG" edax \
    | tee "$WORK/edax_$arm.txt"
done

# ---------------------------------------------------------------------------
# 6. Head-to-head round-robin (ordered pairs; colours alternate inside each).
# ---------------------------------------------------------------------------
for pair in "a c" "a b" "b c" "a d" "b d" "c d"; do
  set -- $pair
  echo "--- h2h: $1 vs $2 ---"
  stage "h2h_${1}_${2}" env \
    OTHELLO_NTUPLE_WEIGHTS="$WORK/weights_$1" \
    OTHELLO_NTUPLE_WEIGHTS_B="$WORK/weights_$2" \
    cargo run --release -q --example ntuple_match -p game-othello -- "$MATCH_CFG" h2h \
    | tee "$WORK/h2h_${1}_${2}.txt"
done

# ---------------------------------------------------------------------------
# 7. Roll up: strength vs cumulative CPU-seconds.
# ---------------------------------------------------------------------------
uv run --project othello-eval python games/othello/ntuple/plot_bakeoff.py "$WORK"

echo "=== bake-off done $(date -u +%FT%TZ) ==="
echo "CPU-second accounting: $ACCT"
echo "Record the verdict and the numbers above where this experiment is tracked."
