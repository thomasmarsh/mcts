module Ranking where

import Candidate (Candidate)
import Statistics (Estimate)

-- | One entry in the ranked set.
data RankedEntry = RankedEntry
  { reCandidate         :: Candidate
  , reDeploymentScore   :: Estimate
  , reTopKProbability   :: Double
  , rePairCount         :: Int
  , reOpponentCount     :: Int
  , reIsDistinguishable :: Bool
  , reTiedWith          :: [String]
  }
  deriving (Eq, Show)

-- | A pairwise matchup record between two candidates.
data PairwiseMatchup = PairwiseMatchup
  { pmLeftId   :: String
  , pmRightId  :: String
  , pmWins     :: Int
  , pmLosses   :: Int
  , pmDraws    :: Int
  , pmNetScore :: Double
  }
  deriving (Eq, Show)

-- | A complete matchup matrix for non-transitivity detection.
data MatchupMatrix = MatchupMatrix
  { mmCandidates :: [String]
  , mmMatchups   :: [(String, String, PairwiseMatchup)]
  }
  deriving (Eq, Show)

-- | The ranked set output.
data RankedSet = RankedSet
  { rsEntries          :: [RankedEntry]
  , rsEpochFingerprint :: String
  , rsDeploymentVersion :: String
  }
  deriving (Eq, Show)

-- | Compute deployment scores from validation observations.
deploymentScores :: [Candidate] -> [(Candidate, Estimate)] -> [RankedEntry]
deploymentScores = undefined

-- | Produce the ranked set, grouping candidates that are practically tied.
rank :: [RankedEntry] -> Double -> RankedSet
rank = undefined

-- | Estimate the probability each candidate belongs to the top k.
topKMembership :: [RankedEntry] -> Int -> [(String, Double)]
topKMembership = undefined

-- | A tuning result: ranked set with full evidence and reproducibility info.
data TuningResult = TuningResult
  { trRankedSet        :: RankedSet
  , trMatchupMatrix    :: MatchupMatrix
  , trDetectedCycles   :: [[String]]
  , trObjectiveEpochId :: String
  , trProductionBudget :: String
  , trTotalCompute     :: String
  , trWallTime         :: String
  }
  deriving (Eq, Show)