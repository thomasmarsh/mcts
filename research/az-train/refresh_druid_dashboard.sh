#!/bin/sh
# Re-render a Druid CNN run's dashboard every INTERVAL seconds until its trainer exits.
# Run from the repo root:
#   nohup research/az-train/refresh_druid_dashboard.sh RUN_DIR TOTAL CONFIG [INTERVAL] &
run=$1 total=$2 config=$3 interval=${4:-300}
render() {
  uv run --project research/az-train python -m az_train.druid_dashboard \
    "$run" "$run/dashboard.html" "$total" "$config"
}
while pgrep -f "az-train-druid-cnn.*$run" >/dev/null; do
  render
  sleep "$interval"
done
render
