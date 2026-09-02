module Mechanism where

import Artifacts (Manifest)
import Candidate (Candidate)
import Evaluation (TaskPrefix)
import Evidence (Observation)
import Shadow (ShadowRaceDecision)

-- | A propensity bin for drawing calibrated pair utilities.
data UtilityBin = UtilityBin
  { ubLo             :: Double
  , ubHi             :: Double
  , ubCount          :: Int
  , ubMeanPropensity :: Double
  , ubCdf            :: [(Double, Double)]
  }
  deriving (Eq, Show)

-- | A frozen Druid-realism calibration extracted from recorded runs.
data Calibration = Calibration
  { calGenerated            :: String
  , calPairUtilityBins      :: [UtilityBin]
  , calStrengthMean         :: Double
  , calStrengthStd          :: Double
  , calBoundaryGapMean      :: Double
  , calBoundaryGapStd       :: Double
  , calPairsPerStratum      :: [Int]
  , calDeviationCorrelation :: Double
  , calDeviationStd         :: Double
  }
  deriving (Eq, Show)

-- | A synthetic cohort with latent per-stratum propensities.
data SyntheticCohort = SyntheticCohort
  { scCandidates    :: [Candidate]
  , scPropensities  :: [[(String, Double)]]  -- per-candidate per-stratum propensities
  , scLatentStrength :: [Double]
  }
  deriving (Eq, Show)

-- | One labeled shadow-race mechanism trial.
data TrialClassification = TrialClassification
  { tcPolicy                   :: String
  , tcEliminated               :: Int
  , tcTopSetFalseEvictions     :: Int
  , tcBoundaryReversals        :: Int
  , tcRuleTieEvictions         :: Int
  , tcPerStratumDangerousFlips :: Int
  , tcUniquePairsSaved         :: Int
  }
  deriving (Eq, Show)

-- | Aggregated eviction metrics for one policy across many trials.
data PolicyRates = PolicyRates
  { prTrials                        :: Int
  , prEliminated                    :: Int
  , prMeanEliminatedPerTrial        :: Double
  , prMeanUniquePairsSaved          :: Double
  , prMeanBoundaryReversalsPerTrial :: Double
  , prMeanTopSetFalsePerTrial       :: Double
  , prTopSetFalseEvictionRate       :: Double
  , prTopSetFalseEvictionUpper      :: Double
  , prBoundaryReversalRate          :: Double
  , prBoundaryReversalUpper         :: Double
  , prRuleTieEvictionRate           :: Double
  , prPerStratumDangerousFlipRate   :: Double
  }
  deriving (Eq, Show)

-- | One sweep cell over a boundary-gap/spread-scale pair.
data CellResult = CellResult
  { crBoundaryGap :: Double
  , crSpreadScale :: Double
  , crPolicies    :: [(String, PolicyRates)]
  }
  deriving (Eq, Show)

-- | A preregistered gate result for one policy.
data GateResult = GateResult
  { grPolicy                       :: String
  , grClauses                      :: [(String, Bool)]
  , grWorstCellBoundaryReversalUpper :: Double
  , grPassed                       :: Bool
  }
  deriving (Eq, Show)

-- | The complete mechanism sweep result.
data SweepResult = SweepResult
  { swBoundaryGaps    :: [Double]
  , swSpreadScales    :: [Double]
  , swTrialsPerCell   :: Int
  , swSeed            :: Int
  , swPairedResamples :: Int
  , swCells           :: [CellResult]
  , swOverall         :: [(String, PolicyRates)]
  , swGates           :: [(String, GateResult)]
  }
  deriving (Eq, Show)

-- | Wilson score interval for a success count.
wilsonInterval :: Int -> Int -> Double -> (Double, Double)
wilsonInterval = undefined

-- | Extract a calibration from recorded run directories.
buildCalibration :: [String] -> Int -> Calibration
buildCalibration = undefined

-- | Draw one synthetic cohort with an exactly controlled boundary gap.
sampleCohort
  :: Calibration -> Manifest -> Int -> Double -> Double -> SyntheticCohort
sampleCohort = undefined

-- | Draw one full latent trial, with each early prefix a true prefix of the
-- maximum prefix.
drawTrial
  :: SyntheticCohort -> Calibration -> Manifest -> Int
  -> [(Int, [Observation])]
drawTrial = undefined

-- | Label eliminated candidates against maximum-prefix evidence.
classifyTrial
  :: Manifest -> ShadowRaceDecision -> [Observation] -> [Observation]
  -> [Candidate] -> TaskPrefix -> TaskPrefix -> TrialClassification
classifyTrial = undefined

-- | Run one mechanism trial across the shipped policies and softenings.
runTrial
  :: Calibration -> Manifest -> Int -> Double -> Double -> Int
  -> [(String, TrialClassification)]
runTrial = undefined

-- | Evaluate the preregistered PASS gate for one policy.
evaluateGate :: String -> [CellResult] -> PolicyRates -> GateResult
evaluateGate = undefined

-- | Run the full grid sweep.
runSweep
  :: Calibration -> Manifest -> [Double] -> [Double] -> Int -> Int -> Int -> SweepResult
runSweep = undefined
