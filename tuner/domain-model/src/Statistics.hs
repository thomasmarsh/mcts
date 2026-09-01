module Statistics where

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

-- | Paired difference interval between two equal-length sequences.
pairedDifference :: [Double] -> [Double] -> Estimate
pairedDifference = undefined

-- | Tie relation from a difference estimate.
data TieRelation = Better | Worse | Tie
  deriving (Eq, Show)

tieRelation :: Estimate -> TieRelation
tieRelation = undefined