module Evidence where

import Effort (SearchEffort)
import Evaluation (Phase, TaskPrefix)
import Statistics (Estimate)

-- | Context under which an observation was collected.
data ObservationContext = ObservationContext
  { ocObjectiveEpochId :: String
  , ocPhase            :: Phase
  , ocTaskPrefix       :: TaskPrefix
  , ocSearchEffort     :: SearchEffort
  }
  deriving (Eq, Show)

-- | An observation: aggregated evidence from completed task prefixes for one
-- candidate at one fidelity context.
data Observation = Observation
  { obsId            :: String
  , obsCandidateId   :: String
  , obsContext       :: ObservationContext
  , obsPairUtilities :: [Double]
  , obsEstimate      :: Estimate
  }
  deriving (Eq, Show)

-- | A task count fidelity: an observation's prefix as a fidelity level.
data TaskCountFidelity = TaskCountFidelity
  { tcfTaskPrefix :: TaskPrefix
  }
  deriving (Eq, Show)