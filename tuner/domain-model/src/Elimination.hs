module Elimination where

import Shadow (ShadowRaceDecision)

-- | An active elimination action: prune a candidate or audit-continue it.
data EliminationAction = Prune | AuditContinue
  deriving (Eq, Show)

-- | A single candidate elimination action with decision margin.
data CandidateEliminationAction = CandidateEliminationAction
  { ceaCandidateId   :: String
  , ceaAction        :: EliminationAction
  , ceaDecisionMargin :: Double
  }
  deriving (Eq, Show)

-- | An apply-elimination allocation: actions for one cohort at one prefix.
data ApplyElimination = ApplyElimination
  { aeCohortIndex :: Int
  , aePrefixId    :: String
  , aeActions     :: [CandidateEliminationAction]
  }
  deriving (Eq, Show)

-- | A suspension of active elimination after a boundary reversal.
data SuspendActiveElimination = SuspendActiveElimination
  { saeAfterCohortIndex       :: Int
  , saeTriggeringCandidateIds  :: [String]
  , saeTriggeringPrefixIds    :: [String]
  , saeSafeyRuleVersion       :: String
  }
  deriving (Eq, Show)

-- | An audited boundary reversal: a pruned candidate that reached the
-- boundary after audit continuation.
data AuditedBoundaryReversal = AuditedBoundaryReversal
  { abrCohortIndex                     :: Int
  , abrCandidateId                     :: String
  , abrPrefixId                        :: String
  , abrBoundaryCandidateId             :: String
  , abrMaximumPrefixPairedMeanDifference :: Double
  }
  deriving (Eq, Show)

-- | Determine active elimination from a shadow race with audit sampling.
activeEliminationAllocation :: ShadowRaceDecision -> Double -> ApplyElimination
activeEliminationAllocation = undefined

-- | Find completed-cohort audit continuations that reach the boundary.
auditedBoundaryReversals :: [ApplyElimination] -> [ShadowRaceDecision] -> [AuditedBoundaryReversal]
auditedBoundaryReversals = undefined