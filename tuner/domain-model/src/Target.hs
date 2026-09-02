module Target where

import Candidate (Candidate, ValidationResult)
import Deployment (Opponent)
import Evaluation
  ( DiagnosticPairResult
  , DiagnosticPairTask
  , PairResult
  , PairTask
  )
import Json (JsonValue)

-- | A subprocess failure before a complete pair was produced.
data PairExecutionError = PairExecutionError
  { peeKind       :: String
  , peeMessage    :: String
  , peeCommand    :: [String]
  , peeReturncode :: Maybe Int
  , peeStderr     :: String
  , peeStdout     :: String
  }
  deriving (Eq, Show)

-- | The subject a candidate is compared against: a panel opponent for objective
-- pairs, or another candidate for diagnostic pairs.
data EvaluationSubject
  = OpponentSubject Opponent
  | CandidateSubject Candidate
  deriving (Eq, Show)

-- | The result of a completed pair evaluation.
data EvaluationResult
  = ObjectiveResult PairResult
  | DiagnosticResult DiagnosticPairResult
  deriving (Eq, Show)

-- | The game-binary boundary. Function fields are not inspected for equality.
data Target = Target
  { tDescribe :: () -> JsonValue
  , tValidate :: [Candidate] -> Opponent -> String -> ValidationResult
  , tEvaluate
      :: Either PairTask DiagnosticPairTask
      -> Candidate
      -> EvaluationSubject
      -> String
      -> Int
      -> EvaluationResult
  , tCancel :: () -> ()
  }

-- | One scheduled pair evaluation.
data PairJob = PairJob
  { pjTask           :: PairTask
  , pjCandidate      :: Candidate
  , pjOpponent       :: Opponent
  , pjGameConfig     :: String
  , pjTimeoutSeconds :: Int
  }
  deriving (Eq, Show)

-- | A successful pair evaluation.
data PairSucceeded = PairSucceeded
  { psJob    :: PairJob
  , psResult :: PairResult
  }
  deriving (Eq, Show)

-- | A failed pair evaluation.
data PairFailed = PairFailed
  { pfJob   :: PairJob
  , pfError :: PairExecutionError
  }
  deriving (Eq, Show)

-- | An interrupted pair evaluation.
data PairInterrupted = PairInterrupted
  { piJob :: PairJob
  }
  deriving (Eq, Show)

-- | One bounded pair execution outcome.
data PairOutcome
  = PairSucceededOutcome PairSucceeded
  | PairFailedOutcome PairFailed
  | PairInterruptedOutcome PairInterrupted
  deriving (Eq, Show)

-- | Bounded, ordered execution of allocator-provided pair jobs.
data PairExecutor = PairExecutor
  { peCapacity :: Int
  , peEvaluate :: Target -> [PairJob] -> [PairOutcome]
  , peCancel   :: Target -> ()
  }
