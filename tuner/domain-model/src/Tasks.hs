module Tasks where

import Deployment (OpponentPanel)
import Evaluation (Phase (..), TaskCorpus, TaskPrefix)

-- | Build a weighted-fair schedule of opponent indices for `count` tasks.
weightedSchedule :: OpponentPanel -> Int -> [Int]
weightedSchedule = undefined

-- | Validate that a task count is a complete cycle endpoint.
validateCycleEndpoint :: OpponentPanel -> Int -> Bool
validateCycleEndpoint = undefined

-- | Build a task crpus for a phase.
buildCorpus :: Phase -> Int -> Int -> OpponentPanel -> String -> TaskCorpus
buildCorpus = undefined

-- | Select a prefix from a corpus.
selectedPrefix :: TaskCorpus -> Int -> TaskPrefix
selectedPrefix = undefined

-- | Build tuning blocks: cumulative prefixes at each complete weighted cycle.
tuningBlocks :: TaskCorpus -> OpponentPanel -> [TaskPrefix]
tuningBlocks = undefined

-- | Verify that a corpus matches the weighted-fair schedule.
verifyWeightedCorpus :: TaskCorpus -> OpponentPanel -> Bool
verifyWeightedCorpus = undefined