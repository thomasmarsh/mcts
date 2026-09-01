module Ranking where

import Candidate (Candidate)
import Evidence (Estimate)

-- | A ranked set of configurations evaluated at production fidelity.
-- The output may declare ties and always carries uncertainty.
data RankedSet = RankedSet
  { rsEntries            :: [RankedEntry]
  , rsEpochFingerprint   :: String
  , rsDeploymentVersion  :: String
  }
  deriving (Eq, Show)

-- | One entry in the ranked set: a configuration with its deployment score,
-- uncertainty, and evidence trail.
data RankedEntry = RankedEntry
  { reCandidate        :: Candidate
  , reDeploymentScore  :: Estimate
  , reTopKProbability  :: Double          -- probability of belonging to top k
  , rePairCount        :: Int
  , reOpponentCount    :: Int
  , reIsDistinguishable :: Bool           -- practically distinguishable from neighbors
  , reTiedWith         :: [String]        -- candidate IDs this is tied with
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

-- | Detect non-transitive cycles in the matchup matrix.
-- Returns list of cycles (A > B > C > A).
detectCycles :: MatchupMatrix -> [[String]]
detectCycles = undefined

-- | Compute the deployment score for each candidate in the finalist set,
-- using held-out production-validation evidence.
deploymentScores
  :: [Candidate]             -- ^ finalists
  -> [(Candidate, Estimate)] -- ^ production-validation observations
  -> [RankedEntry]
deploymentScores = undefined

-- | Produce the ranked set, grouping candidates that are practically tied.
rank :: [RankedEntry] -> Double -> RankedSet
  -- ^ practical-difference threshold for declaring ties
rank = undefined

-- | Estimate the probability each candidate belongs to the top k.
topKMembership :: [RankedEntry] -> Int -> [(String, Double)]
topKMembership = undefined

-- | A production claim: the ranked set with full evidence and reproducibility info.
data TuningResult = TuningResult
  { trRankedSet         :: RankedSet
  , trMatchupMatrix     :: MatchupMatrix
  , trDetectedCycles    :: [[String]]
  , trObjectiveEpochId  :: String
  , trProductionBudget  :: String       -- e.g. "10000 iterations"
  , trTotalCompute       :: String      -- human-readable compute summary
  , trWallTime           :: String
  }
  deriving (Eq, Show)