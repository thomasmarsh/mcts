module Identity where

import Candidate (Candidate, ObservationFrontier)
import Deployment (ObjectiveEpoch, Opponent, OpponentPanel)
import Effort (SearchEffort)
import Evaluation
  ( DiagnosticPairTask
  , PairTask
  , Phase
  , TaskCase
  , TaskCorpus
  , TaskPrefix
  )
import Evidence (Observation)
import Json (JsonValue)
import Proposal (ObservationReference)
import Statistics (Estimate)

-- | SHA-256 hex digest of a canonical JSON value.
type Fingerprint = String

-- | Canonical JSON encoding: sorted keys, compact separators, no non-finite floats.
canonicalJson :: JsonValue -> String
canonicalJson = undefined

-- | SHA-256 hex digest of a canonical JSON value.
fingerprint :: JsonValue -> Fingerprint
fingerprint = undefined

-- | SHA-256 hex digest of a file's contents.
sha256File :: String -> Fingerprint
sha256File = undefined

-- | A stable, deterministic ID built from a kind tag and an identity payload.
stableId :: String -> JsonValue -> String
stableId = undefined

-- | Derive a deterministic task seed from a root seed, phase, and ordinal.
deriveTaskSeed :: Int -> String -> Int -> Int
deriveTaskSeed = undefined

-- | Build an immutable candidate from a config value.
candidateFromConfig :: JsonValue -> Candidate
candidateFromConfig = undefined

-- | Build an immutable candidate from an already canonical config string.
candidateFromCanonicalConfig :: String -> Candidate
candidateFromCanonicalConfig = undefined

-- | Freeze a set of opponents into a weighted panel.
opponentPanel :: [Opponent] -> OpponentPanel
opponentPanel = undefined

-- | Build one task case from its deterministically derived identity.
taskCase :: Phase -> Int -> Int -> Opponent -> OpponentPanel -> String -> TaskCase
taskCase = undefined

-- | Freeze ordered task cases into a task corpus.
taskCorpus :: Phase -> [TaskCase] -> OpponentPanel -> TaskCorpus
taskCorpus = undefined

-- | Select a prefix from a corpus.
taskPrefix :: TaskCorpus -> Int -> TaskPrefix
taskPrefix = undefined

-- | Freeze an objective epoch from its identity payload.
objectiveEpoch :: JsonValue -> ObjectiveEpoch
objectiveEpoch = undefined

-- | Project an observation into its model-consumable reference.
observationReference :: Observation -> ObservationReference
observationReference = undefined

-- | Deterministic observation identity.
observationId :: String -> JsonValue -> [Double] -> Estimate -> String
observationId = undefined

-- | Freeze a comparable observation frontier.
observationFrontier :: [ObservationReference] -> ObservationFrontier
observationFrontier = undefined

-- | Build a seat-swapped pair task for a candidate and task case.
pairTask :: Candidate -> TaskCase -> SearchEffort -> PairTask
pairTask = undefined

-- | Deterministic game identity for one seat of a pair task.
gameId :: Either PairTask DiagnosticPairTask -> String -> String
gameId = undefined

-- | Deterministic diagnostic matchup edge identity.
diagnosticEdgeId :: String -> Candidate -> Candidate -> String
diagnosticEdgeId = undefined

-- | Build a diagnostic candidate-vs-candidate pair task.
diagnosticPairTask
  :: String -> Int -> Int -> Int -> Candidate -> Candidate -> SearchEffort -> DiagnosticPairTask
diagnosticPairTask = undefined
