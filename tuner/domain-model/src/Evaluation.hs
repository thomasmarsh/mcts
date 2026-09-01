module Evaluation where

import Deployment (DeploymentCase, SearchEffort)

-- | A task case is one concrete game to be played: which candidate, which
-- opponent, which deployment case, at what phase and budget.
data TaskCase = TaskCase
  { tcTaskId    :: String
  , tcPhase     :: Phase
  , tcOrdinal   :: Int
  , tcSeed      :: Int
  , tcStratumId :: String
  , tcCase      :: DeploymentCase
  }
  deriving (Eq, Show)

-- | A corpus of ordered task cases for one phase, stratified to be representative
-- of the target distribution.
data TaskCorpus = TaskCorpus
  { corpusId          :: String
  , corpusFingerprint :: String
  , corpusPhase       :: Phase
  , corpusCases       :: [TaskCase]
  }
  deriving (Eq, Show)

-- | A prefix of a task corpus: the first N cases, forming a common task block.
-- All candidates in a race see these same cases in the same order.
data TaskPrefix = TaskPrefix
  { prefixId      :: String
  , prefixCorpusId :: String
  , prefixLength  :: Int
  , prefixTaskIds :: [String]
  }
  deriving (Eq, Show)

-- | Make a prefix from a corpus.
takePrefix :: TaskCorpus -> Int -> TaskPrefix
takePrefix = undefined

-- | A pair task: a seat-swapped pair of games for one candidate against one opponent.
-- The smallest atomic unit of evidence.
data PairTask = PairTask
  { ptPairId      :: String
  , ptCandidateId :: String
  , ptTaskCase    :: TaskCase
  , ptBudget      :: SearchEffort
  }
  deriving (Eq, Show)

-- | Phase of evaluation.
data Phase = Tuning | Validation | ProductionValidation
  deriving (Eq, Show)

-- | Result of a single game.
data GameResult = GameResult
  { grGameId        :: String
  , grCandidateSide :: Side
  , grOutcome       :: Outcome
  , grSeed          :: Int
  , grPlies         :: Int
  , grElapsedMs     :: Int
  }
  deriving (Eq, Show)

data Side = First | Second
  deriving (Eq, Show)

data Outcome = CandidateWin | BaselineWin | Draw
  deriving (Eq, Show)

-- | A paired result: two games, candidate plays first then second.
-- Seat-swapping controls first-player advantage.
data PairResult = PairResult
  { prTask  :: PairTask
  , prGames :: (GameResult, GameResult)
  }
  deriving (Eq, Show)

-- | Seat-balanced utility for a pair: (candidate_win_count + draw_count * 0.5) / 2.
-- Range: [0, 1]. win=1, draw=0.5, loss=0 per seat, averaged.
pairUtility :: PairResult -> Double
pairUtility = undefined