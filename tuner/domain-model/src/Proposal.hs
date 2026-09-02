module Proposal where

import Candidate (Candidate, ObservationFrontier, ProposalSource)
import Effort (SearchEffort)
import Evidence (Observation, ObservationContext)

-- | A model observation: a candidate with its reference and cost.
data ModelObservation = ModelObservation
  { moCandidate   :: Candidate
  , moReference   :: ObservationReference
  , moCost        :: Double
  }
  deriving (Eq, Show)

-- | Reference to an observation for model consumption.
data ObservationReference = ObservationReference
  { orObservationId    :: String
  , orCandidateId      :: String
  , orObjectiveEpochId  :: String
  , orPrefixId         :: String
  , orTaskIds          :: [String]
  , orSearchEffort      :: SearchEffort
  }
  deriving (Eq, Show)

-- | An attempt ordinal and seed for guided proposers.
data ModelAttempt = ModelAttempt
  { maSourceAttempt :: Int
  , maSeed          :: Int
  }
  deriving (Eq, Show)

-- | The complete, immutable proposal request visible to a guided proposer.
data ProposalRequest = ProposalRequest
  { prqObservations               :: [ModelObservation]
  , prqFrontier                   :: ObservationFrontier
  , prqExcludedFingerprints       :: [String]
  , prqAttempt                    :: ModelAttempt
  , prqGenerationIndex            :: Int
  , prqRankedParents              :: [Candidate]
  , prqGuidedCandidatesPerGeneration :: Int
  }
  deriving (Eq, Show)

-- | A proposed configuration returned by a model proposer.
data ProposedConfiguration = ProposedConfiguration
  { pcCandidate          :: Candidate
  , pcOrigin             :: Maybe String
  , pcAcquisition        :: Maybe Double
  , pcPrediction         :: Maybe Double
  , pcUncertainty        :: Maybe Double
  , pcParentCandidateId   :: Maybe String
  }
  deriving (Eq, Show)

-- | Proposer policy: which guided source to use.
data ProposerPolicy = SmacMixed | Random | Qmc | IRaceGenerational
  deriving (Eq, Show)

-- | The model proposer interface. Given observations at a common frontier,
-- proposes one new configuration.
data ModelProposer = ModelProposer
  { mpAdapterVersion :: String
  , mpAsk            :: ProposalRequest -> ProposedConfiguration
  }

-- | Source schedule for the initial cohort: schema_default, bootstrap_random entries,
-- then a weighted mix of guided and random_reserve slots.
sourceSchedule :: Int -> Int -> Int -> ProposerPolicy -> [ProposalSource]
sourceSchedule = undefined

-- | Source schedule for challenger cohorts: elites retained, then guided + reserve.
challengerSourceSchedule :: Int -> Int -> Int -> ProposerPolicy -> [ProposalSource]
challengerSourceSchedule = undefined

-- | Derive a deterministic seed for a proposal source namespace.
derivedSeed :: Int -> String -> Int -> Int
derivedSeed = undefined

-- | Build an empty observation frontier for a context.
emptyFrontier :: ObservationContext -> ObservationFrontier
emptyFrontier = undefined

-- | Build a tuning frontier from completed comparable observations.
tuningFrontier :: [Observation] -> ObservationFrontier
tuningFrontier = undefined

-- | Extract model observations from tuning observations at a frontier.
modelObservations :: [Observation] -> [(String, Candidate)] -> ObservationFrontier -> [ModelObservation]
modelObservations = undefined

-- | Cost for a tuning observation: 1.0 - mean estimate, clamped to [0, 1].
costFromObservation :: Observation -> Double
costFromObservation = undefined