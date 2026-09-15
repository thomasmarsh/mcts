#!/usr/bin/env bash
# Cost-knob sweep driver for the Othello CNN Gumbel loop
# (coordinator_othello_cnn.sh). Runs a list of named configs sequentially,
# never concurrently -- two coordinator runs sharing one RUN_DIR/machine can
# leave an orphaned training process racing a fresh run for CPU and roughly
# doubling apparent wall-clock cost, so this driver never launches the next
# config until the previous one's coordinator process has exited -- each
# config gets its own RUN_DIR under $SWEEP_ROOT/<name>/, and this script
# appends one line to
# $SWEEP_ROOT/manifest.jsonl per finished config so a partial sweep (killed,
# crashed, or still running) can be read without waiting for the rest.
#
# Configs are read from a file, one per line: `<name> <ENV=val> <ENV=val> ...`
# (blank lines and lines starting with # are skipped). Every ENV=val pair is
# exported into coordinator_othello_cnn.sh's own environment for that one
# run only; anything not overridden keeps the coordinator's own default, so
# this script adds no new hyperparameters of its own -- config-as-data lives
# entirely in the config file, per AGENTS.md.
#
#   SWEEP_ROOT=local/output/az/othello-cnn/sweep \
#     bash research/az-train/sweep_othello_cnn.sh \
#     research/az-train/sweep-configs/othello-cnn-cost-knobs.txt
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

CONFIG_FILE=${1:?usage: sweep_othello_cnn.sh <config-file>}
SWEEP_ROOT=${SWEEP_ROOT:-local/output/az/othello-cnn/sweep}
mkdir -p "$SWEEP_ROOT"
MANIFEST="$SWEEP_ROOT/manifest.jsonl"

while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac

  name=$(echo "$line" | awk '{print $1}')
  overrides=$(echo "$line" | cut -d' ' -f2-)

  run_dir="$SWEEP_ROOT/$name"
  if [ -f "$run_dir/log.jsonl" ] && [ "$(wc -l < "$run_dir/log.jsonl")" -ge 1 ]; then
    gens_done=$(wc -l < "$run_dir/log.jsonl")
    echo "=== sweep config '$name': skipping, $run_dir/log.jsonl already has $gens_done generation(s) @ $(date) ==="
    continue
  fi

  echo "=== sweep config '$name': starting ($overrides) @ $(date) ==="
  start=$(date +%s)

  env_vars="RUN_DIR=$run_dir"
  for kv in $overrides; do env_vars="$env_vars $kv"; done

  # shellcheck disable=SC2086
  env $env_vars bash research/az-train/coordinator_othello_cnn.sh

  wall=$(($(date +%s) - start))
  echo "=== sweep config '$name': done, ${wall}s wall @ $(date) ==="
  python3 -c "
import json, sys
name, run_dir, wall = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({'name': name, 'run_dir': run_dir, 'wall_seconds': wall}))
" "$name" "$run_dir" "$wall" >> "$MANIFEST"
done < "$CONFIG_FILE"

echo "=== sweep done @ $(date) ==="
echo "manifest: $MANIFEST"
