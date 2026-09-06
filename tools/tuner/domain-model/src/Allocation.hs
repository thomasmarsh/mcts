module Allocation where

import Artifacts (Manifest)
import Candidate (Candidate, CandidateFailure, Proposal)
import Evaluation (PairResult, PairTask, Phase, TaskPrefix)
import Racing (AllocationDecision, CohortRecord, ReplayState, ResourceAllocation)

-- | The allocation policy version for a manifest.
allocationPolicyVersion :: Manifest -> String
allocationPolicyVersion = undefined

-- | Decide the next allocation action from the current state.
decideAllocation :: Manifest -> ReplayState -> AllocationDecision
decideAllocation = undefined

-- | Translate an allocation decision into a concrete resource allocation.
resourceAllocation
  :: AllocationDecision -> Manifest -> ReplayState -> Maybe ResourceAllocation
resourceAllocation = undefined

-- | Return a proposal by index.
proposalAt :: ReplayState -> Int -> Proposal
proposalAt = undefined

-- | Candidates currently in contest (not failed, not pruned).
currentActiveCandidates :: ReplayState -> [Candidate]
currentActiveCandidates = undefined

-- | Candidates admitted (active elites + accepted proposals, minus failures).
currentAdmittedCandidates :: ReplayState -> [Candidate]
currentAdmittedCandidates = undefined

-- | The candidates that pair execution should schedule.
pairCandidates :: ReplayState -> [Candidate]
pairCandidates = undefined

-- | The phase of the pair work currently scheduled.
pairPhase :: ReplayState -> Phase
pairPhase = undefined

-- | The active prefix for the current phase (tuning or validation).
activePrefix :: Manifest -> ReplayState -> TaskPrefix
activePrefix = undefined

-- | Return the incomplete tasks in the active prefix's canonical order.
readyPairs :: Manifest -> ReplayState -> Maybe Int -> [PairTask]
readyPairs = undefined

-- | Check whether a candidate failure is due.
candidateFailureDue :: Manifest -> ReplayState -> Maybe CandidateFailure
candidateFailureDue = undefined

-- | The first unrefilled terminal candidate failure, if any.
pendingRefillFailure :: ReplayState -> Maybe CandidateFailure
pendingRefillFailure = undefined

-- | The first currently ready pair.
pendingPair :: Manifest -> ReplayState -> Maybe PairTask
pendingPair = undefined

-- | Completed pairs for one candidate in one phase.
matchingPairs :: ReplayState -> Candidate -> Phase -> [PairResult]
matchingPairs = undefined

-- | Check whether a candidate is observation-due.
observationDue :: Manifest -> ReplayState -> Maybe Candidate
observationDue = undefined

-- | The latest completed cohort, if any.
latestCompletedCohort :: ReplayState -> Maybe CohortRecord
latestCompletedCohort = undefined

-- | Check whether the validation phase is complete.
validationComplete :: Manifest -> ReplayState -> Bool
validationComplete = undefined
