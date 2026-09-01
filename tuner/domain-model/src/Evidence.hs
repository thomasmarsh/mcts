module Evidence where

import Deployment (SearchEffort)
import Evaluation (Phase, TaskPrefix, PairResult)

-- | An observation is the aggregated evidence from completed task prefixes
-- for one candidate at one fidelity context. It is comparable only with other
-- observations that share the same epoch, phase, prefix, and search effort.
data Observation = Observation
  { obsId            :: String
  , obsCandidateId   :: String
  , obsContext       :: ObservationContext
  , obsPairUtilities :: [Double]  -- one per pair in the common prefix
  , obsEstimate      :: Estimate
  }
  deriving (Eq, Show)

-- | The context under which an observation was collected.
-- Observations in different contexts are never directly compared.
data ObservationContext = ObservationContext
  { ocEpochId      :: String
  , ocPhase        :: Phase
  , ocPrefix       :: TaskPrefix
  , ocSearchEffort :: SearchEffort
  }
  deriving (Eq, Show)

-- | A point estimate with a confidence interval.
data Estimate = Estimate
  { estMean  :: Double
  , estLower :: Double
  , estUpper :: Double
  }
  deriving (Eq, Show)

-- | Build an observation from a candidate and a completed common task prefix.
-- All pairs must share the same context.
observe :: ObservationContext -> [PairResult] -> Maybe Observation
observe = undefined

-- | Check whether two observations are comparable (same epoch, phase, prefix, effort).
comparable :: Observation -> Observation -> Bool
comparable = undefined

-- | Compute the paired difference estimate between two comparable observations.
pairedDifference :: Observation -> Observation -> Maybe Estimate
pairedDifference = undefined

-- | A marginal interval from a series of pair utilities using Hoeffding bounds.
marginalInterval :: [Double] -> Estimate
marginalInterval = undefined

-- | The reference to an observation for model consumption.
data ObservationReference = ObservationReference
  { refId          :: String
  , refCandidateId :: String
  , refEpochId     :: String
  , refPrefixId    :: String
  , refTaskIds     :: [String]
  , refEffort      :: SearchEffort
  }
  deriving (Eq, Show)

-- | The frontier of observations visible to the proposer.
data ObservationFrontier = ObservationFrontier
  { frontierId             :: String
  , frontierEpochId        :: String
  , frontierPrefixId       :: String
  , frontierTaskIds        :: [String]
  , frontierEffort         :: SearchEffort
  , frontierObservationIds :: [String]
  }
  deriving (Eq, Show)