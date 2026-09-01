module Racing where

import Candidate (Candidate)
import Evidence (Observation)
import Evaluation (TaskPrefix)

-- | A cohort is a group of candidates evaluated together on common task blocks.
-- All active candidates in a race see the same ordered task blocks.
data Cohort = Cohort
  { chIndex              :: Int
  , chCandidates         :: [Candidate]
  , chRetainedEliteIds   :: [String]  -- elites carried forward from prior cohort
  }
  deriving (Eq, Show)

-- | A completed cohort with its records.
data CohortRecord = CohortRecord
  { crCohort   :: Cohort
  , crObservations :: [Observation]
  }
  deriving (Eq, Show)

-- | The iterated racing state: a sequence of cohorts, each evaluated on
-- progressively deeper task blocks. Elites are retained between cohorts.
data RacingState = RacingState
  { rsCohorts            :: [CohortRecord]
  , rsActiveCandidates   :: [Candidate]
  , rsActiveElites       :: [Candidate]
  , rsCurrentBlockIndex  :: Int
  , rsFinalists          :: Maybe [Candidate]
  , rsTerminalStatus     :: RacingStatus
  }
  deriving (Eq, Show)

data RacingStatus
  = RacingOpen
  | RacingConfigurationFailed
  | RacingComplete
  deriving (Eq, Show)

-- | Depth of evaluation for a cohort: which task block prefix.
data EvaluationDepth = EvaluationDepth
  { edBlockIndex :: Int
  , edPrefix     :: TaskPrefix
  }
  deriving (Eq, Show)

-- | Deepen: advance the active cohort to the next cumulative task block.
-- All survivors complete the same block before any decision.
deepen :: RacingState -> TaskPrefix -> RacingState
deepen = undefined

-- | Eliminate candidates that have negligible probability of reaching
-- the current promotion boundary, given the evidence so far and a
-- practical-effect margin delta.
eliminate
  :: RacingState
  -> Double        -- ^ practical effect margin (delta)
  -> Double        -- ^ false-elimination risk threshold (alpha)
  -> TaskPrefix    -- ^ the prefix at which elimination is tested
  -> ([Candidate], [Candidate])  -- ^ (survivors, eliminated)
eliminate = undefined

-- | Complete the current cohort: close it and record observations.
completeCohort :: RacingState -> CohortRecord
completeCohort = undefined

-- | Start a new challenger cohort, retaining the top-N elites from the
-- prior cohort for continuity.
startNextCohort :: RacingState -> Int -> RacingState
startNextCohort = undefined

-- | Select finalists from accumulated cohort evidence.
selectFinalists :: RacingState -> Int -> [Candidate]
selectFinalists = undefined

-- | Select the top-N candidates from a cohort by mean observed utility.
selectTop :: [Observation] -> [Candidate] -> Int -> [Candidate]
selectTop = undefined

-- | The promotion boundary: the current N-th-best candidate that others
-- must beat to avoid elimination. N is typically the number of finalists.
promotionBoundary :: [Observation] -> Int -> Maybe Candidate
promotionBoundary = undefined