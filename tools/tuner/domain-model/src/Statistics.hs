module Statistics where

-- | The fixed Hoeffding confidence level used across the tuner.
alpha :: Double
alpha = 0.05

-- | Game outcome utility: win=1, draw=0.5, loss=0.
data Utility = Utility Double
  deriving (Eq, Show)

-- | A point estimate with Hoeffding confidence bounds.
data Estimate = Estimate
  { estMean  :: Double
  , estLower :: Double
  , estUpper :: Double
  }
  deriving (Eq, Show)

-- | Game utility from an outcome string.
gameUtility :: String -> Double
gameUtility = undefined

-- | Pair utility: average of the two games (seat-balanced, [0,1]).
pairUtility :: Double -> Double -> Double
pairUtility first second = (first + second) / 2.0

-- | Marginal Hoeffding interval (alpha = 0.05) for a sequence of pair utilities.
marginalInterval :: [Double] -> Estimate
marginalInterval = undefined

-- | Numeric primitive for paired difference intervals between two
-- equal-length sequences; contextual callers use 'Observations.pairedDifference'.
pairedDifferenceValues :: [Double] -> [Double] -> Estimate
pairedDifferenceValues = undefined

-- | Deterministic percentile bootstrap interval for independent complete runs.
bootstrapMeanInterval :: [Double] -> Int -> Int -> Estimate
bootstrapMeanInterval = undefined

-- | Tie relation from a difference estimate.
data TieRelation = Better | Worse | Tie
  deriving (Eq, Show)

tieRelation :: Estimate -> TieRelation
tieRelation = undefined