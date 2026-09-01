module Allocation where

import Evaluation (PairTask, TaskPrefix)
import Racing (RacingState)

-- | The three uses of compute controlled by the single allocator:
-- introduce a new configuration, deepen an existing candidate, or
-- spend evidence on validation.
data AllocationDecision
  = IntroduceNewConfig
  | DeepenExisting
  | RefineRanking
  | NoDecisionYet
  deriving (Eq, Show)

-- | The concrete allocation of a resource: which candidate gets what.
data ResourceAllocation
  = IntroduceCandidate
      { raCohortSlot :: Int
      , raSource     :: String  -- proposal source
      }
  | DeepenCohort
      { raBlockIndex :: Int
      , raPrefixId   :: String
      }
  | BeginValidation
      { raTuningPrefixId :: String
      }
  | RetainElites
      { raCohortIndex  :: Int
      , raCandidateIds :: [String]
      , raPrefixId     :: String
      }
  | RefillFailedSlot
      { raCohortSlot      :: Int
      , raFailedCandidate :: String
      }
  deriving (Eq, Show)

-- | Compute budget in pair attempts, separately for tuning and validation.
data ComputeBudget = ComputeBudget
  { cbTuningPairAttempts     :: Int
  , cbValidationPairAttempts :: Int
  }
  deriving (Eq, Show)

-- | Accumulated compute usage.
data ComputeLedger = ComputeLedger
  { clTuning     :: PhaseCompute
  , clValidation :: PhaseCompute
  }
  deriving (Eq, Show)

data PhaseCompute = PhaseCompute
  { pcPairAttempts     :: Int
  , pcCompletedPairs   :: Int
  , pcFailedAttempts   :: Int
  , pcCensoredAttempts :: Int
  , pcPhysicalGames    :: Int
  , pcSearchIterations :: Int
  , pcWallTimeMs       :: Int
  }
  deriving (Eq, Show)

-- | The authoritative allocator: given the current state, decide which of the
-- three uses of compute to pursue next.
allocate :: RacingState -> ComputeBudget -> ComputeLedger -> AllocationDecision
allocate = undefined

-- | Translate an allocation decision into a concrete resource allocation.
resource :: AllocationDecision -> RacingState -> Maybe ResourceAllocation
resource = undefined

-- | Check whether the tuning budget can fund another challenger cohort
-- given the used budget and per-cohort costs.
canFundChallenger :: ComputeBudget -> ComputeLedger -> Int -> Int -> Bool
canFundChallenger = undefined

-- | The next pair task that is ready to execute, given the active prefix
-- and candidates. Returns the first task whose pair hasn't been completed.
nextReadyPair :: RacingState -> TaskPrefix -> Maybe PairTask
nextReadyPair = undefined