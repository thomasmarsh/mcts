module ShadowAudit where

import Artifacts (Manifest)
import EventPayloads (EvidenceEvent)
import Racing (PhaseCompute, ReplayState)
import Shadow (ShadowDisposition)

-- | Per-stratum counterfactual audit for one shadow look.
data StratumAudit = StratumAudit
  { saStratumId              :: String
  , saEarlyMeanDifference    :: Double
  , saMaximumMeanDifference  :: Double
  , saReversal               :: Bool
  , saFavorableResamples     :: Maybe Int
  , saFavorableProbability   :: Maybe Double
  }
  deriving (Eq, Show)

-- | A single recorded shadow decision labeled with maximum-prefix evidence.
data ShadowLookAudit = ShadowLookAudit
  { slaCohortIndex                 :: Int
  , slaPrefixId                    :: String
  , slaCandidateId                 :: String
  , slaBoundaryCandidateId         :: String
  , slaPolicyKind                  :: String
  , slaFavorableResamples          :: Maybe Int
  , slaTotalResamples              :: Maybe Int
  , slaRank                        :: Maybe Int
  , slaPriorSurvivorCount          :: Maybe Int
  , slaTargetSurvivorCount         :: Maybe Int
  , slaNewlyEliminated             :: Maybe Bool
  , slaDisposition                 :: ShadowDisposition
  , slaEarlyMeanDifference         :: Double
  , slaMaximumMeanDifference       :: Double
  , slaFinalReachesRecordedBoundary :: Bool
  , slaStrata                      :: [StratumAudit]
  }
  deriving (Eq, Show)

-- | One candidate's complete shadow-decision path.
data CandidatePathAudit = CandidatePathAudit
  { cpaCohortIndex             :: Int
  , cpaCandidateId             :: String
  , cpaProtected               :: Bool
  , cpaFinalTopSet             :: Bool
  , cpaLooks                   :: [ShadowLookAudit]
  , cpaFirstEliminationPrefixId :: Maybe String
  , cpaAvoidedUniquePairs      :: Int
  , cpaAvoidedCompute          :: PhaseCompute
  }
  deriving (Eq, Show)

-- | A calibration bin for bootstrap promotion probabilities.
data CalibrationBin = CalibrationBin
  { cbLower               :: Double
  , cbUpper               :: Double
  , cbCount               :: Int
  , cbMeanPrediction      :: Double
  , cbObservedSuccessRate :: Double
  }
  deriving (Eq, Show)

-- | Per-stratum reversal summary.
data StratumSummary = StratumSummary
  { ssStratumId           :: String
  , ssLooks               :: Int
  , ssReversals           :: Int
  , ssEliminationReversals :: Int
  }
  deriving (Eq, Show)

-- | The complete evidence-only counterfactual audit for recorded shadow races.
data ShadowAudit = ShadowAudit
  { shPaths                                :: [CandidatePathAudit]
  , shCalibrationBins                      :: [CalibrationBin]
  , shStrata                               :: [StratumSummary]
  , shCounterfactualEliminations           :: Int
  , shEligibleTopSetPaths                  :: Int
  , shTopSetFalseEliminations              :: Int
  , shTrueTrashEliminations                :: Int
  , shBrierScore                           :: Maybe Double
  , shRecordedComputeAfterFirstElimination :: PhaseCompute
  , shSupersededRosterLooks                :: Int
  , shBoundaryReversals                    :: Int
  , shRuleTieEvictions                     :: Int
  , shPerStratumDangerousFlips             :: Int
  }
  deriving (Eq, Show)

-- | Label immutable shadow decisions with maximum-prefix tuning evidence.
buildShadowAudit :: Manifest -> ReplayState -> [EvidenceEvent] -> ShadowAudit
buildShadowAudit = undefined
