module OpponentInteractions where

import Deployment (OpponentPanel)
import Evaluation (PairResult)
import Evidence (Observation)
import Racing (CohortRecord)
import Statistics (Estimate, TieRelation)

-- | A candidate's stationary tuning response against one opponent.
data OpponentResponse = OpponentResponse
  { orCandidateId :: String
  , orOpponentId  :: String
  , orEstimate    :: Estimate
  , orPairCount   :: Int
  , orPairs       :: [PairResult]
  }
  deriving (Eq, Show)

-- | A paired candidate difference within one opponent stratum.
data OpponentContrast = OpponentContrast
  { ocOpponentId       :: String
  , ocPairedDifference :: Estimate
  , ocRelation         :: TieRelation
  }
  deriving (Eq, Show)

-- | An opponent ranking reversal between two candidate contrasts.
data OpponentRankingReversal = OpponentRankingReversal
  { orrLeftOpponentId  :: String
  , orrRightOpponentId :: String
  }
  deriving (Eq, Show)

-- | One candidate pair's per-opponent contrasts and detected reversals.
data CandidateOpponentInteraction = CandidateOpponentInteraction
  { coiLeftCandidateId  :: String
  , coiRightCandidateId :: String
  , coiContrasts        :: [OpponentContrast]
  , coiRankingReversals :: [OpponentRankingReversal]
  }
  deriving (Eq, Show)

-- | The candidate-by-opponent response matrix, plus pairwise interactions.
data OpponentResponseAnalysis = OpponentResponseAnalysis
  { oraResponses    :: [OpponentResponse]
  , oraInteractions :: [CandidateOpponentInteraction]
  }
  deriving (Eq, Show)

-- | Project complete maximum-prefix tuning evidence in frozen roster order.
buildOpponentResponseAnalysis
  :: OpponentPanel -> CohortRecord -> [Observation] -> [PairResult]
  -> OpponentResponseAnalysis
buildOpponentResponseAnalysis = undefined
