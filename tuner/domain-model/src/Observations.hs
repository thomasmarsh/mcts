module Observations where

import Candidate (Candidate)
import Evidence (Observation, ObservationContext)
import Evaluation (TaskPrefix)
import Statistics (Estimate)

-- | Check that two observations share the same context.
comparable :: Observation -> Observation -> Bool
comparable = undefined

-- | Compute the paired difference estimate between two comparable observations.
pairedDifferenceEstimate :: Observation -> Observation -> Estimate
pairedDifferenceEstimate = undefined

-- | Select exactly one tuning observation per candidate at one common prefix.
comparablePrefixObservations
  :: [Observation] -> [Candidate] -> TaskPrefix -> [Observation]
comparablePrefixObservations = undefined

-- | Build an observation from a candidate, context, and completed pair utilities.
contextualObservation
  :: Candidate -> ObservationContext -> [Double] -> Observation
contextualObservation = undefined