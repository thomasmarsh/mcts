module Selection where

import Candidate (Candidate)
import Diagnostic (DiagnosticGraph)
import Evidence (Observation)

-- | Pick the top candidates by mean observed utility.
selectTopCandidates :: [Candidate] -> [Observation] -> Int -> [Candidate]
selectTopCandidates = undefined

-- | Select the validation shortlist accounting for non-transitive cycles.
selectValidationShortlist
  :: [Candidate] -> [Observation] -> Int -> DiagnosticGraph
  -> ([Candidate], Maybe String, Maybe String)
selectValidationShortlist = undefined
