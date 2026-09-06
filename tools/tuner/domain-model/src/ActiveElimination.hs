module ActiveElimination where

import Artifacts (Manifest)
import Elimination
  ( ApplyElimination
  , AuditedBoundaryReversal
  )
import Racing (CohortRecord, ReplayState)
import Shadow (ShadowRaceDecision)

-- | Determine enforced active elimination from a shadow race with the frozen
-- prospective audit sampling.
activeEliminationAllocation
  :: Manifest -> ReplayState -> ShadowRaceDecision -> ApplyElimination
activeEliminationAllocation = undefined

-- | Find completed-cohort audit continuations that reach their recorded
-- boundary at maximum tuning fidelity.
auditedBoundaryReversals
  :: Manifest -> ReplayState -> CohortRecord -> [AuditedBoundaryReversal]
auditedBoundaryReversals = undefined
