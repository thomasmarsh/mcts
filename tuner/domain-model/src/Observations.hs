module Observations where

import Candidate (Candidate)
import Evaluation (PairResult, TaskPrefix)
import Evidence (Observation, ObservationContext)
import Statistics (Estimate)

-- | Build an observation from a candidate id, context, and completed pair utilities.
observation :: String -> ObservationContext -> [Double] -> Observation
observation = undefined

-- | Build an observation from a complete, ordered common task prefix.
contextualObservation
  :: Candidate -> ObservationContext -> [PairResult] -> Observation
contextualObservation = undefined

-- | Check that two observations share the same context.
comparable :: Observation -> Observation -> Either String ()
comparable = undefined

-- | Compute the paired difference estimate between two comparable observations.
pairedDifference :: Observation -> Observation -> Estimate
pairedDifference = undefined

-- | Select exactly one tuning observation per candidate at one common prefix.
comparablePrefixObservations
  :: [Observation] -> [Candidate] -> TaskPrefix -> [Observation]
comparablePrefixObservations = undefined
