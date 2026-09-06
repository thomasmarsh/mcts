module DiagnosticMatchmaking where

import Artifacts (Manifest)
import Diagnostic (EvaluateDiagnosticPair)
import Racing (ReplayState)

-- | Find the next diagnostic pair to allocate, if one is due and the
-- diagnostic budget remains.
nextDiagnosticAllocation :: Manifest -> ReplayState -> Maybe EvaluateDiagnosticPair
nextDiagnosticAllocation = undefined
