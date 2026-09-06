module Racing where

import Candidate (Candidate, CandidateFailure, PairAttemptFacts, Proposal, ProposalDisposition)
import Diagnostic (EvaluateDiagnosticPair)
import Elimination (ApplyElimination, SuspendActiveElimination)
import Evaluation (PairResult, DiagnosticPairResult, PairTask, Phase)
import Evidence (Observation)
import Shadow (ShadowRaceDecision)

-- | Terminal status of a tuning run.
data TerminalStatus = Open | ConfigurationFailed | Complete
  deriving (Eq, Show)

-- | A cohort record: candidates that completed a tuning cycle.
data CohortRecord = CohortRecord
  { crCohortIndex          :: Int
  , crCandidates           :: [Candidate]
  , crRetainedCandidateIds :: [String]
  }
  deriving (Eq, Show)

-- | A compute budget for pair attempts.
data ComputeBudget = ComputeBudget
  { cbTuningPairAttempts     :: Int
  , cbValidationPairAttempts :: Int
  , cbDiagnosticPairAttempts :: Int
  }
  deriving (Eq, Show)

-- | Compute usage for one phase.
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

-- | Full compute ledger: tuning, validation, and diagnostic phases.
data ComputeLedger = ComputeLedger
  { clTuning     :: PhaseCompute
  , clValidation :: PhaseCompute
  , clDiagnostic :: PhaseCompute
  }
  deriving (Eq, Show)

-- | A concrete resource allocation: what to record in the evidence log.
data ResourceAllocation
  = IntroduceCandidate     { raCohortSlot :: Int, raSource :: String }
  | RefillCandidate       { raCohortSlot :: Int, raSource :: String, raFailedCandidateId :: String }
  | DeepenCohortAllocation { raBlockIndex :: Int, raPrefixId :: String }
  | BeginValidation        { raTuningPrefixId :: String }
  | RetainElites          { raCohortIndex :: Int, raCandidateIds :: [String], raPrefixId :: String }
  | ApplyElimAction       { raElimination :: ApplyElimination }
  | SuspendActiveElim     { raSuspension :: SuspendActiveElimination }
  | EvaluateDiagnostic    { raDiagnostic :: EvaluateDiagnosticPair }
  deriving (Eq, Show)

-- | The immutable replay state: a checkpoint of every decision, observation,
-- and compute usage accumulated since the tuner started.
data ReplayState = ReplayState
  { rsProposals                  :: [Proposal]
  , rsDispositions               :: [(Int, ProposalDisposition)]
  , rsCompletedCohorts           :: [CohortRecord]
  , rsActiveElites               :: [Candidate]
  , rsCompletedPairs             :: [PairResult]
  , rsObservations               :: [Observation]
  , rsFinalists                  :: Maybe [Candidate]
  , rsTerminalStatus             :: TerminalStatus
  , rsTuningBlockIndex           :: Int
  , rsPendingResourceAllocation  :: Maybe ResourceAllocation
  , rsCompute                    :: ComputeLedger
  , rsShadowRaces                :: [ShadowRaceDecision]
  , rsCandidateFailures          :: [CandidateFailure]
  , rsPairAttempts               :: [(String, PairAttemptFacts)]
  , rsRefillAttempts             :: [(Int, String)]
  , rsEliminationAllocations     :: [ApplyElimination]
  , rsActiveEliminationSuspension :: Maybe SuspendActiveElimination
  , rsDiagnosticPairs            :: [DiagnosticPairResult]
  , rsDiagnosticAttempts         :: [(String, PairAttemptFacts)]
  , rsEffectiveBudget            :: ComputeBudget
  , rsSupersededFinalists        :: [[Candidate]]
  }
  deriving (Eq, Show)

-- | An allocation decision: what the allocator should do next.
data AllocationDecision
  = ResolveProposal      { adProposalIndex :: Int }
  | ExecutePair          { adTask :: PairTask }
  | ChooseDiagnosticPair { adCohortIndex :: Int }
  | EmitObservation      { adCandidateId :: String, adPhase :: Phase }
  | EmitShadowRace       { adCohortIndex :: Int, adPrefixId :: String }
  | EnforceElimination   { adCohortIndex :: Int, adPrefixId :: String }
  | SuspendElimination   { adAfterCohortIndex :: Int }
  | CompleteCohort
  | StartNextCohort
  | DeepenCohort         { adBlockIndex :: Int, adPrefixId :: String }
  | SelectFinalists
  | IntroduceProposal
  | FailCandidate        { adFailure :: CandidateFailure }
  | CompleteRun
  | NoDecision
  deriving (Eq, Show)