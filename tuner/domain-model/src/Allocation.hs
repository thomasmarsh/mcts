module Allocation where

import Candidate (Candidate, CandidateFailure)
import Evaluation (PairTask, TaskPrefix)
import Racing (AllocationDecision (..), CohortRecord, ComputeBudget, ReplayState, ResourceAllocation (..))

-- | The allocation policy version.
allocationPolicyVersion :: String
allocationPolicyVersion = "budgeted-multi-cohort-diagnostic-v2"

-- | Decide the next allocation action from the current state.
decideAllocation :: ComputeBudget -> ReplayState -> AllocationDecision
decideAllocation = undefined

-- | Translate an allocation decision into a concrete resource allocation.
resourceAllocation :: AllocationDecision -> ReplayState -> Maybe ResourceAllocation
resourceAllocation = undefined

-- | Find the next ready pair task.
readyPairs :: [TaskPrefix] -> ReplayState -> Int -> [PairTask]
readyPairs = undefined

-- | The active prefix for the current phase (tuning or validation).
activePrefix :: [TaskPrefix] -> TaskPrefix -> ReplayState -> TaskPrefix
activePrefix = undefined

-- | Candidates currently in contest (not failed, not pruned).
currentActiveCandidates :: ReplayState -> [Candidate]
currentActiveCandidates = undefined

-- | Candidates admitted (active elites + accepted proposals, minus failures).
currentAdmittedCandidates :: ReplayState -> [Candidate]
currentAdmittedCandidates = undefined

-- | Check whether a candidate is observation-due.
observationDue :: [TaskPrefix] -> ReplayState -> Maybe Candidate
observationDue = undefined

-- | Check whether a candidate failure is due.
candidateFailureDue :: ComputeBudget -> ReplayState -> Maybe CandidateFailure
candidateFailureDue = undefined

-- | The latest completed cohort, if any.
latestCompletedCohort :: ReplayState -> Maybe CohortRecord
latestCompletedCohort = undefined

-- | Check whether the validation phase is complete.
validationComplete :: [TaskPrefix] -> ReplayState -> Bool
validationComplete = undefined