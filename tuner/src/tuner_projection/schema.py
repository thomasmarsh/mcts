"""The SQLite read-model DDL as one versioned string.

Every table is ``WITHOUT ROWID`` with an explicit primary key so a dump walks
each b-tree in primary-key order, making the canonical dump independent of the
order in which runs were projected.
"""

from __future__ import annotations

PROJECTION_SCHEMA_VERSION = 2

# Tables carrying projected content, in the order the canonical dump emits them.
# ``ingest_state`` is deliberately excluded here: it holds the per-run change
# fingerprint (evidence size / mtime), which is machine-specific and must not
# reach the checked-in golden dump.
CONTENT_TABLES: tuple[str, ...] = (
    "projection_meta",
    "runs",
    "run_manifest",
    "run_report",
    "cohorts",
    "candidates",
    "proposals",
    "pairs",
    "games",
    "observations",
    "shadow_decisions",
    "active_elimination_decisions",
    "validation_rows",
    "compute_phases",
)

DDL = """
CREATE TABLE projection_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE ingest_state (
    run_id               TEXT PRIMARY KEY,
    evidence_size        INTEGER NOT NULL,
    evidence_mtime_ns    INTEGER NOT NULL,
    manifest_fingerprint TEXT NOT NULL
) WITHOUT ROWID;

-- A pickled `tuner_cli.replay.ReplayCheckpoint`, keyed by run, that lets a
-- live run's next pass fold only the events past `last_sequence` instead of
-- replaying the whole evidence log from scratch. Bookkeeping like
-- `ingest_state`, not projected content: excluded from `CONTENT_TABLES` and
-- the canonical dump, safe to drop and never affects what a rebuild produces.
CREATE TABLE run_checkpoints (
    run_id             TEXT PRIMARY KEY,
    checkpoint_version INTEGER NOT NULL,
    last_sequence      INTEGER NOT NULL,
    manifest_fingerprint TEXT NOT NULL,
    state              BLOB NOT NULL
) WITHOUT ROWID;

CREATE TABLE runs (
    run_id               TEXT PRIMARY KEY,
    manifest_run_id      TEXT,
    manifest_fingerprint TEXT,
    terminal_status      TEXT,
    report_available     INTEGER NOT NULL,
    ingest_error         TEXT
) WITHOUT ROWID;

CREATE TABLE run_manifest (
    run_id             TEXT PRIMARY KEY REFERENCES runs(run_id),
    manifest_json      TEXT NOT NULL,
    game_kind          TEXT NOT NULL,
    objective_id       TEXT NOT NULL,
    cohort_size        INTEGER NOT NULL,
    finalists          INTEGER NOT NULL,
    seed               INTEGER NOT NULL,
    task_seed          INTEGER NOT NULL,
    shadow_policy_kind TEXT NOT NULL,
    active_elimination INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE run_report (
    run_id           TEXT PRIMARY KEY REFERENCES runs(run_id),
    report_json      TEXT NOT NULL,
    schema_version   INTEGER NOT NULL,
    status           TEXT NOT NULL,
    validation_claim TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE cohorts (
    run_id                 TEXT NOT NULL REFERENCES runs(run_id),
    cohort_index           INTEGER NOT NULL,
    candidate_ids          TEXT NOT NULL,
    retained_candidate_ids TEXT NOT NULL,
    PRIMARY KEY (run_id, cohort_index)
) WITHOUT ROWID;

CREATE TABLE candidates (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    candidate_id        TEXT NOT NULL,
    fingerprint         TEXT NOT NULL,
    canonical_config    TEXT NOT NULL,
    cohort_index        INTEGER NOT NULL,
    cohort_slot         INTEGER NOT NULL,
    source              TEXT NOT NULL,
    parent_candidate_id TEXT,
    PRIMARY KEY (run_id, candidate_id)
) WITHOUT ROWID;

CREATE TABLE proposals (
    run_id               TEXT NOT NULL REFERENCES runs(run_id),
    proposal_index       INTEGER NOT NULL,
    cohort_index         INTEGER NOT NULL,
    cohort_slot          INTEGER NOT NULL,
    candidate_id         TEXT NOT NULL,
    source               TEXT NOT NULL,
    source_attempt       INTEGER NOT NULL,
    disposition          TEXT,
    frontier_id          TEXT NOT NULL,
    origin               TEXT,
    acquisition          REAL,
    prediction           REAL,
    uncertainty          REAL,
    parent_candidate_id  TEXT,
    refill_of_candidate_id TEXT,
    PRIMARY KEY (run_id, proposal_index)
) WITHOUT ROWID;

CREATE TABLE pairs (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    pair_id      TEXT NOT NULL,
    phase        TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    opponent_id  TEXT NOT NULL,
    pair_utility REAL NOT NULL,
    PRIMARY KEY (run_id, pair_id)
) WITHOUT ROWID;

CREATE TABLE games (
    run_id                    TEXT NOT NULL REFERENCES runs(run_id),
    game_id                   TEXT NOT NULL,
    pair_id                   TEXT NOT NULL,
    candidate_side            TEXT NOT NULL,
    outcome                   TEXT NOT NULL,
    plies                     INTEGER NOT NULL,
    elapsed_ms                INTEGER NOT NULL,
    candidate_iterations_total INTEGER NOT NULL,
    opponent_iterations_total  INTEGER NOT NULL,
    PRIMARY KEY (run_id, game_id)
) WITHOUT ROWID;

CREATE TABLE observations (
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    observation_id TEXT NOT NULL,
    candidate_id   TEXT NOT NULL,
    phase          TEXT NOT NULL,
    prefix_id      TEXT NOT NULL,
    mean           REAL NOT NULL,
    lower          REAL NOT NULL,
    upper          REAL NOT NULL,
    PRIMARY KEY (run_id, observation_id)
) WITHOUT ROWID;

CREATE TABLE shadow_decisions (
    run_id                TEXT NOT NULL REFERENCES runs(run_id),
    race_index            INTEGER NOT NULL,
    cohort_index          INTEGER NOT NULL,
    prefix_id             TEXT NOT NULL,
    candidate_id          TEXT NOT NULL,
    boundary_candidate_id TEXT NOT NULL,
    disposition           TEXT NOT NULL,
    policy_kind           TEXT NOT NULL,
    policy_version        TEXT NOT NULL,
    PRIMARY KEY (run_id, race_index, candidate_id)
) WITHOUT ROWID;

CREATE TABLE active_elimination_decisions (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    batch_index  INTEGER NOT NULL,
    cohort_index INTEGER NOT NULL,
    prefix_id    TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    action       TEXT NOT NULL,
    margin_kind  TEXT NOT NULL,
    PRIMARY KEY (run_id, batch_index, candidate_id)
) WITHOUT ROWID;

CREATE TABLE validation_rows (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    candidate_id TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    estimate     REAL NOT NULL,
    lower        REAL NOT NULL,
    upper        REAL NOT NULL,
    wins         INTEGER NOT NULL,
    draws        INTEGER NOT NULL,
    losses       INTEGER NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
) WITHOUT ROWID;

CREATE TABLE compute_phases (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    phase            TEXT NOT NULL,
    pair_attempts    INTEGER NOT NULL,
    completed_pairs  INTEGER NOT NULL,
    failed_attempts  INTEGER NOT NULL,
    censored_attempts INTEGER NOT NULL,
    physical_games   INTEGER NOT NULL,
    search_iterations INTEGER NOT NULL,
    wall_time_ms     INTEGER NOT NULL,
    PRIMARY KEY (run_id, phase)
) WITHOUT ROWID;

CREATE INDEX idx_candidates_run ON candidates(run_id);
CREATE INDEX idx_proposals_run ON proposals(run_id);
CREATE INDEX idx_pairs_run_candidate ON pairs(run_id, candidate_id);
CREATE INDEX idx_games_run_pair ON games(run_id, pair_id);
CREATE INDEX idx_observations_run_candidate ON observations(run_id, candidate_id);
"""
