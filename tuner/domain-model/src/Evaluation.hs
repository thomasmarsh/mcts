module Evaluation where

import Deployment (Opponent, OpponentPanel)
import Effort (SearchEffort)

-- | Phase of evaluation.
data Phase = Tuning | Validation
  deriving (Eq, Show)

-- | A task case: one concrete paired comparison to be played.
data TaskCase = TaskCase
  { tcTaskId               :: String
  , tcPhase                :: Phase
  , tcOrdinal              :: Int
  , tcSeed                 :: Int
  , tcStratumId            :: String
  , tcOpponentId           :: String
  , tcOpponentFingerprint  :: String
  , tcPanelFingerprint     :: String
  , tcGameConfigFingerprint :: String
  , tcStart                :: String  -- "default"
  }
  deriving (Eq, Show)

-- | A corpus of ordered task cases for one phase.
data TaskCorpus = TaskCorpus
  { corpusId           :: String
  , corpusFingerprint  :: String
  , corpusPhase        :: Phase
  , corpusTaskPolicyVersion :: String
  , corpusCases        :: [TaskCase]
  }
  deriving (Eq, Show)

-- | A prefix of a task corpus: the first N cases forming a common task block.
data TaskPrefix = TaskPrefix
  { prefixId      :: String
  , prefixCorpusId :: String
  , prefixLength  :: Int
  , prefixTaskIds :: [String]
  }
  deriving (Eq, Show)

-- | A pair task: a seat-swapped pair of games for one candidate against one opponent.
data PairTask = PairTask
  { ptPairId      :: String
  , ptCandidateId :: String
  , ptTaskCase    :: TaskCase
  , ptBudget      :: SearchEffort
  }
  deriving (Eq, Show)

-- | Strategy metrics for one side of a game.
data StrategyMetrics = StrategyMetrics
  { smIterationsTotal      :: Int
  , smIterationsFirstHalf  :: Int
  , smMoveTimeMs           :: Int
  }
  deriving (Eq, Show)

-- | Result of a single game.
data GameResult = GameResult
  { grGameId           :: String
  , grCandidateSide    :: String  -- "first" | "second"
  , grOutcome          :: String  -- "candidate_win" | "baseline_win" | "draw"
  , grDerivedSeed      :: Int
  , grRound            :: Int
  , grSeq              :: Int
  , grTraceGameSeq     :: Maybe Int
  , grPlies            :: Int
  , grElapsedMs        :: Int
  , grCandidateMetrics :: StrategyMetrics
  , grOpponentMetrics  :: StrategyMetrics
  , grRawRecord        :: String
  }
  deriving (Eq, Show)

-- | A paired result: two games, candidate plays first then second.
data PairResult = PairResult
  { prTask  :: PairTask
  , prGames :: (GameResult, GameResult)
  }
  deriving (Eq, Show)

-- | A diagnostic pair task: direct candidate-vs-candidate matchup.
data DiagnosticPairTask = DiagnosticPairTask
  { dptPairId           :: String
  , dptEdgeId           :: String
  , dptOrdinal          :: Int
  , dptLeftCandidateId  :: String
  , dptRightCandidateId :: String
  , dptSeed             :: Int
  , dptSearchEffort     :: SearchEffort
  }
  deriving (Eq, Show)

-- | A diagnostic pair result.
data DiagnosticPairResult = DiagnosticPairResult
  { dprTask  :: DiagnosticPairTask
  , dprGames :: (GameResult, GameResult)
  }
  deriving (Eq, Show)

-- | Build a task case.
mkTaskCase :: Phase -> Int -> Int -> Opponent -> OpponentPanel -> String -> TaskCase
mkTaskCase = undefined

-- | Build a task corpus.
mkTaskCorpus :: Phase -> Int -> Int -> OpponentPanel -> String -> TaskCorpus
mkTaskCorpus = undefined

-- | Select a prefix from a corpus.
takePrefix :: TaskCorpus -> Int -> TaskPrefix
takePrefix = undefined