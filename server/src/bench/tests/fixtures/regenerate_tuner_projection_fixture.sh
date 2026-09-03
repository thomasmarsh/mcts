#!/usr/bin/env bash
# Regenerate the checked-in read-only projection fixture used by the tuner_api
# endpoint tests.
#
# The fixture is a full SQLite projection built by `tuner-project`. The two
# complete runs come from the tuner crate's checked-in version-4 run fixtures
# (the same source its own `version4.dump.sql` golden is built from), so a diff
# in those rows reflects a real change to what the projection materializes. A
# third `broken` run with a garbage manifest is projected alongside them so the
# endpoint tests can exercise the `ingest_error` path.
#
# Run from the repository root:
#
#     server/src/bench/tests/fixtures/regenerate_tuner_projection_fixture.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
fixtures="$repo_root/tuner/tests/fixtures/projection-root"
out="$repo_root/server/src/bench/tests/fixtures/tuner-projection.sqlite"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp -RL "$fixtures"/version4 "$staging/version4"
cp -RL "$fixtures"/version4-active-halving "$staging/version4-active-halving"
mkdir -p "$staging/broken"
printf '{ this is not valid json' > "$staging/broken/manifest.json"

# A still-running run: version4's evidence truncated just after its first
# cohort completes, so `terminal_status` is "open" and no report.json exists,
# yet the proposals / observations / shadow_decisions / compute_phases rows
# are populated. This is the fixture the live-science endpoint tests read.
cp -RL "$fixtures"/version4 "$staging/version4-partial"
rm -f "$staging/version4-partial/report.json" \
      "$staging/version4-partial/scientific_projection.json"
head -160 "$fixtures/version4/evidence.jsonl" > "$staging/version4-partial/evidence.jsonl"

rm -f "$out"
uv run --project "$repo_root/tuner" tuner-project \
    --runs-root "$staging" --db "$out" --rebuild
echo "wrote $out"
